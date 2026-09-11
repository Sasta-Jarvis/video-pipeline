"""Logging setup: one rotating app-wide log, plus a helper to create a
dedicated per-job log file (kept next to the job's files in processing/ or
failed/ so a failure is fully self-explanatory without cross-referencing the
main log).
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import LoggingConfig

APP_LOGGER_NAME = "video_pipeline"


def setup_logging(log_dir: Path, config: LoggingConfig) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        log_dir / "video_pipeline.log",
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_job_logger(job_id: str, job_log_path: Path) -> logging.Logger:
    """A logger dedicated to one job, writing to a file that travels with
    the job (into failed/<job_id>/error.log on failure)."""
    job_log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{APP_LOGGER_NAME}.job.{job_id}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = True  # also flows to the main app log

    handler = logging.FileHandler(job_log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(handler)
    return logger


def close_job_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
