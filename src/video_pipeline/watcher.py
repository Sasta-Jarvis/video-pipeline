"""Continuous folder watching with FIFO pairing.

Uses watchdog for near-instant wake-ups on filesystem activity, but all the
actual decision-making (is a file done being written? is it already
claimed? what pairs with what) happens in a centralized poll pass rather
than inside watchdog's own event callbacks -- this avoids races between
"a file just appeared" and "a file is safe to touch" (someone might still
be copying a multi-GB recording into the folder).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import Config
from .db import JobDB, make_source_key


class _WakeHandler(FileSystemEventHandler):
    def __init__(self, wake_event: threading.Event):
        self._wake_event = wake_event

    def on_any_event(self, event):
        if not event.is_directory:
            self._wake_event.set()


@dataclass
class _TrackedFile:
    size: int
    last_size_change: float


class FolderWatcher:
    """Watches speech_dir/racing_dir, tracks per-file size stability, and
    finds FIFO-paired (speech, racing) candidates once both sides have at
    least one stable, unclaimed file."""

    def __init__(self, config: Config, db: JobDB, logger):
        self.config = config
        self.db = db
        self.logger = logger
        self._wake_event = threading.Event()
        self._tracked: dict[Path, _TrackedFile] = {}

        self._observer = Observer()
        handler = _WakeHandler(self._wake_event)
        self._observer.schedule(handler, str(config.paths.speech_dir), recursive=False)
        self._observer.schedule(handler, str(config.paths.racing_dir), recursive=False)

    def start(self) -> None:
        self._observer.start()
        self.logger.info(
            "Watching %s and %s", self.config.paths.speech_dir, self.config.paths.racing_dir
        )

    def stop(self) -> None:
        self._wake_event.set()
        self._observer.stop()
        self._observer.join(timeout=5)

    def _scan_dir(self, directory: Path) -> list[Path]:
        exts = self.config.watcher.video_extensions
        try:
            files = [
                p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts
            ]
        except FileNotFoundError:
            return []
        return sorted(files, key=lambda p: p.stat().st_mtime)

    def _update_stability(self, directory: Path, paths: list[Path]) -> None:
        # _tracked is shared across both watched directories, so cleanup
        # must only drop entries that belong to *this* directory scan --
        # otherwise scanning racing/ would wipe out speech/'s tracked
        # files (and vice versa) on every single call, permanently
        # resetting stability to zero and starving find_next_pair().
        now = time.time()
        seen = set(paths)
        for path in paths:
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue
            existing = self._tracked.get(path)
            if existing is None or existing.size != size:
                self._tracked[path] = _TrackedFile(size=size, last_size_change=now)
        for tracked_path in list(self._tracked.keys()):
            if tracked_path.parent == directory and tracked_path not in seen:
                del self._tracked[tracked_path]

    def _stable_candidates(self, directory: Path) -> list[Path]:
        paths = self._scan_dir(directory)
        self._update_stability(directory, paths)
        now = time.time()
        return [
            p
            for p in paths
            if p in self._tracked
            and (now - self._tracked[p].last_size_change) >= self.config.watcher.stable_wait_seconds
        ]

    def _first_unclaimed(self, candidates: list[Path], kind: str) -> Path | None:
        for path in candidates:
            try:
                key = make_source_key(kind, path)
            except FileNotFoundError:
                continue
            if not self.db.is_claimed(key):
                return path
        return None

    def find_next_pair(self) -> tuple[Path, Path] | None:
        """FIFO: oldest unclaimed stable speech file + oldest unclaimed
        stable racing file. Returns None if either side has nothing ready."""
        speech_path = self._first_unclaimed(
            self._stable_candidates(self.config.paths.speech_dir), "speech"
        )
        racing_path = self._first_unclaimed(
            self._stable_candidates(self.config.paths.racing_dir), "racing"
        )
        if speech_path is None or racing_path is None:
            return None
        return speech_path, racing_path

    def wait_for_activity(self, timeout: float) -> None:
        """Blocks until a filesystem event wakes us, or timeout elapses
        (the periodic-poll fallback)."""
        self._wake_event.wait(timeout=timeout)
        self._wake_event.clear()
