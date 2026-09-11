"""SQLite-backed job state.

This is the thing that makes restarts safe: a file is only ever "claimed"
into a job once (recorded here before it's physically moved), and every
job's lifecycle (pending -> processing -> completed/failed) is durable
across process restarts. Combined with the physical move of source files
into processing/, this gives two independent guards against double work.

Single-writer usage pattern: the watcher thread enqueues candidate pairs,
but only the single pipeline-processing thread ever writes to this DB, so
we don't need a connection pool — one connection + a lock is enough.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS claimed_sources (
    source_key TEXT PRIMARY KEY,   -- fingerprint: kind:name:size:mtime
    kind TEXT NOT NULL,            -- 'speech' | 'racing'
    original_path TEXT NOT NULL,
    job_id TEXT NOT NULL,
    claimed_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    speech_source_key TEXT NOT NULL,
    racing_source_key TEXT NOT NULL,
    speech_original_name TEXT NOT NULL,
    racing_original_name TEXT NOT NULL,
    status TEXT NOT NULL,          -- pending | processing | completed | failed
    stage TEXT,                    -- current/last pipeline stage, for diagnostics
    output_path TEXT,
    error_message TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


@dataclass
class Job:
    job_id: str
    speech_source_key: str
    racing_source_key: str
    speech_original_name: str
    racing_original_name: str
    status: str
    stage: str | None
    output_path: str | None
    error_message: str | None
    created_at: float
    updated_at: float


def make_source_key(kind: str, path: Path) -> str:
    st = path.stat()
    return f"{kind}:{path.name}:{st.st_size}:{int(st.st_mtime)}"


class JobDB:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @contextmanager
    def _cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def is_claimed(self, source_key: str) -> bool:
        with self._cursor() as cur:
            cur.execute(
                "SELECT 1 FROM claimed_sources WHERE source_key = ?", (source_key,)
            )
            return cur.fetchone() is not None

    def create_job(
        self,
        speech_source_key: str,
        racing_source_key: str,
        speech_original_name: str,
        racing_original_name: str,
    ) -> Job:
        """Atomically claims both source files and creates a pending job.
        Raises sqlite3.IntegrityError if either source was already claimed
        (race between watcher scans) — caller should treat that as "someone
        else took it" and move on.
        """
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO claimed_sources (source_key, kind, original_path, job_id, claimed_at) "
                "VALUES (?, 'speech', ?, ?, ?)",
                (speech_source_key, speech_original_name, job_id, now),
            )
            cur.execute(
                "INSERT INTO claimed_sources (source_key, kind, original_path, job_id, claimed_at) "
                "VALUES (?, 'racing', ?, ?, ?)",
                (racing_source_key, racing_original_name, job_id, now),
            )
            cur.execute(
                "INSERT INTO jobs (job_id, speech_source_key, racing_source_key, "
                "speech_original_name, racing_original_name, status, stage, "
                "output_path, error_message, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', NULL, NULL, NULL, ?, ?)",
                (
                    job_id,
                    speech_source_key,
                    racing_source_key,
                    speech_original_name,
                    racing_original_name,
                    now,
                    now,
                ),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Job | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            return Job(**dict(row)) if row else None

    def update_status(
        self,
        job_id: str,
        status: str,
        stage: str | None = None,
        output_path: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = ?, stage = COALESCE(?, stage), "
                "output_path = COALESCE(?, output_path), "
                "error_message = ?, updated_at = ? WHERE job_id = ?",
                (status, stage, output_path, error_message, time.time(), job_id),
            )

    def set_stage(self, job_id: str, stage: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE jobs SET stage = ?, updated_at = ? WHERE job_id = ?",
                (stage, time.time(), job_id),
            )

    def jobs_in_status(self, status: str) -> list[Job]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE status = ?", (status,))
            return [Job(**dict(row)) for row in cur.fetchall()]

    def all_jobs(self) -> list[Job]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            return [Job(**dict(row)) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()
