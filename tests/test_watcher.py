"""Tests FIFO pairing, file-stability detection, and the no-double-claim
guarantee, using real temp directories and real file timestamps (no mocks)
so the actual stat()-based logic is exercised.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from video_pipeline.config import load_config  # noqa: E402
from video_pipeline.db import JobDB, make_source_key  # noqa: E402
from video_pipeline.watcher import FolderWatcher  # noqa: E402


class _NullLogger:
    def __getattr__(self, name):
        return lambda *a, **k: None


class TestWatcher(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vp_watcher_test_"))
        cfg = load_config(config_path=None, base_dir=self.tmp)
        cfg.paths.root = self.tmp
        cfg.paths.speech_dir = self.tmp / "speech"
        cfg.paths.racing_dir = self.tmp / "racing"
        cfg.paths.output_dir = self.tmp / "output"
        cfg.paths.failed_dir = self.tmp / "failed"
        cfg.paths.processing_dir = self.tmp / "processing"
        cfg.paths.state_db = self.tmp / "state.db"
        cfg.watcher.stable_wait_seconds = 0  # instant "stable" for fast tests
        for d in cfg.paths.all_dirs():
            d.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.db = JobDB(cfg.paths.state_db)
        self.watcher = FolderWatcher(cfg, self.db, _NullLogger())

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _touch(self, directory: Path, name: str, content: bytes = b"x" * 1024) -> Path:
        p = directory / name
        p.write_bytes(content)
        return p

    def test_no_pair_when_one_side_empty(self):
        self._touch(self.cfg.paths.speech_dir, "speech1.mp4")
        self.assertIsNone(self.watcher.find_next_pair())

    def test_pairs_when_both_sides_present(self):
        self._touch(self.cfg.paths.speech_dir, "speech1.mp4")
        self._touch(self.cfg.paths.racing_dir, "racing1.mp4")
        pair = self.watcher.find_next_pair()
        self.assertIsNotNone(pair)
        speech, racing = pair
        self.assertEqual(speech.name, "speech1.mp4")
        self.assertEqual(racing.name, "racing1.mp4")

    def test_fifo_order(self):
        s1 = self._touch(self.cfg.paths.speech_dir, "speech_older.mp4")
        time.sleep(0.05)
        s2 = self._touch(self.cfg.paths.speech_dir, "speech_newer.mp4")
        r1 = self._touch(self.cfg.paths.racing_dir, "racing_only.mp4")

        pair = self.watcher.find_next_pair()
        self.assertEqual(pair[0].name, "speech_older.mp4")

    def test_claimed_files_are_not_repaired(self):
        speech = self._touch(self.cfg.paths.speech_dir, "speech1.mp4")
        racing = self._touch(self.cfg.paths.racing_dir, "racing1.mp4")

        speech_key = make_source_key("speech", speech)
        racing_key = make_source_key("racing", racing)
        self.db.create_job(speech_key, racing_key, speech.name, racing.name)

        # Files still physically present (simulating: not yet moved), but
        # claimed in the DB -> must not be paired again.
        self.assertIsNone(self.watcher.find_next_pair())

    def test_unstable_file_not_paired_yet(self):
        self.cfg.watcher.stable_wait_seconds = 100  # effectively "never stable" in this test
        self._touch(self.cfg.paths.speech_dir, "speech1.mp4")
        self._touch(self.cfg.paths.racing_dir, "racing1.mp4")
        self.assertIsNone(self.watcher.find_next_pair())

    def test_stability_survives_interleaved_scans_of_both_dirs(self):
        # Regression test: stability tracking is a single dict shared by
        # both watched directories. Repeatedly scanning one directory must
        # not reset stability timers for files in the other directory.
        self.cfg.watcher.stable_wait_seconds = 0.3
        self._touch(self.cfg.paths.speech_dir, "speech1.mp4")
        self._touch(self.cfg.paths.racing_dir, "racing1.mp4")

        deadline = time.time() + 3.0
        found = None
        while time.time() < deadline and found is None:
            found = self.watcher.find_next_pair()
            time.sleep(0.1)

        self.assertIsNotNone(
            found, "pair should eventually become stable even with interleaved dir scans"
        )


if __name__ == "__main__":
    unittest.main()
