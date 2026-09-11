"""Entrypoint: `python -m video_pipeline.main [--config config.yaml]`

Wires everything together: load config -> ensure folders exist -> recover
any jobs interrupted by a previous crash -> start watching -> loop forever,
pairing FIFO and running jobs one at a time. A failure in one job is caught
inside pipeline.run_job and never propagates here, so the process keeps
running indefinitely until the user stops it.
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from . import db as db_module
from . import pipeline
from .config import Config, ensure_directories, load_config
from .logging_setup import setup_logging
from .watcher import FolderWatcher

_STOP = False


def _handle_signal(signum, frame):
    global _STOP
    _STOP = True


def run(config: Config) -> int:
    logger = setup_logging(config.paths.log_dir, config.logging)

    logger.info("=== video-pipeline starting ===")
    logger.info(config.hardware.summary())
    for w in config.hardware.warnings:
        logger.warning(w)

    if not config.hardware.ffmpeg_path:
        logger.error(
            "ffmpeg is required and was not found on PATH. Install ffmpeg and "
            "restart. See README.md for instructions."
        )
        return 1

    ensure_directories(config)
    logger.info("Folders ready:")
    logger.info("  speech:     %s", config.paths.speech_dir)
    logger.info("  racing:     %s", config.paths.racing_dir)
    logger.info("  output:     %s", config.paths.output_dir)
    logger.info("  failed:     %s", config.paths.failed_dir)
    logger.info("  processing: %s", config.paths.processing_dir)

    db = db_module.JobDB(config.paths.state_db)
    pipeline.recover_interrupted_jobs(config, db, logger)

    watcher = FolderWatcher(config, db, logger)
    watcher.start()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("Ready. Drop a speech video into speech/ and a racing video into racing/.")

    try:
        while not _STOP:
            pair = watcher.find_next_pair()
            if pair is None:
                watcher.wait_for_activity(timeout=config.watcher.poll_interval_seconds)
                continue

            speech_path, racing_path = pair
            try:
                speech_key = db_module.make_source_key("speech", speech_path)
                racing_key = db_module.make_source_key("racing", racing_path)
                job = db.create_job(
                    speech_source_key=speech_key,
                    racing_source_key=racing_key,
                    speech_original_name=speech_path.name,
                    racing_original_name=racing_path.name,
                )
            except FileNotFoundError:
                # File vanished between selection and claiming (e.g. user
                # moved it out); just retry on the next loop iteration.
                continue
            except Exception:
                logger.exception(
                    "Failed to claim pair (%s, %s); will retry next scan.",
                    speech_path,
                    racing_path,
                )
                continue

            logger.info(
                "Paired job %s: speech=%s racing=%s", job.job_id, speech_path.name, racing_path.name
            )
            pipeline.run_job(job, speech_path, racing_path, config, db, logger)
    finally:
        logger.info("Shutting down...")
        watcher.stop()
        db.close()
        logger.info("=== video-pipeline stopped ===")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Automated speech+racing video pipeline")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: ./config.yaml; falls back to built-in defaults if missing)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config, base_dir=Path.cwd())
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
