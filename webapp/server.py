"""Web UI for the video pipeline: upload speech/racing videos from a
browser instead of dragging files into folders on disk. Runs the exact
same watcher/pipeline code as the CLI (video_pipeline.main) in a
background thread -- this is a thin presentation layer on top of the
already-tested pipeline, not a reimplementation of it.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from video_pipeline import db as db_module
from video_pipeline import pipeline
from video_pipeline.config import ensure_directories, load_config
from video_pipeline.logging_setup import setup_logging
from video_pipeline.watcher import FolderWatcher

CONFIG_PATH = os.environ.get("VIDEO_PIPELINE_CONFIG", "/app/config.yaml")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024 * 1024  # 20 GB, generous for raw recordings

_config = load_config(CONFIG_PATH, base_dir=Path(CONFIG_PATH).parent)
ensure_directories(_config)
_logger = setup_logging(_config.paths.log_dir, _config.logging)
_db = db_module.JobDB(_config.paths.state_db)
pipeline.recover_interrupted_jobs(_config, _db, _logger)
_watcher = FolderWatcher(_config, _db, _logger)


def _worker_loop() -> None:
    _watcher.start()
    _logger.info("Background pipeline worker started.")
    while True:
        pair = _watcher.find_next_pair()
        if pair is None:
            _watcher.wait_for_activity(timeout=_config.watcher.poll_interval_seconds)
            continue
        speech_path, racing_path = pair
        try:
            speech_key = db_module.make_source_key("speech", speech_path)
            racing_key = db_module.make_source_key("racing", racing_path)
            job = _db.create_job(speech_key, racing_key, speech_path.name, racing_path.name)
        except FileNotFoundError:
            continue
        except Exception:
            _logger.exception("Failed to claim pair (%s, %s)", speech_path, racing_path)
            continue
        _logger.info("Paired job %s", job.job_id)
        pipeline.run_job(job, speech_path, racing_path, _config, _db, _logger)


def _unique_dest(directory: Path, filename: str) -> Path:
    safe_name = secure_filename(filename) or "upload.mp4"
    dest = directory / safe_name
    if dest.exists():
        stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
        dest = directory / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
    return dest


@app.route("/")
def index():
    return render_template("index.html", hardware=_config.hardware)


@app.route("/api/jobs")
def api_jobs():
    jobs = _db.all_jobs()
    return jsonify(
        [
            {
                "job_id": j.job_id,
                "status": j.status,
                "stage": j.stage,
                "speech": j.speech_original_name,
                "racing": j.racing_original_name,
                "output_path": (Path(j.output_path).name if j.output_path else None),
                "error_message": j.error_message,
                "created_at": j.created_at,
                "updated_at": j.updated_at,
            }
            for j in jobs
        ]
    )


@app.route("/api/pending")
def api_pending():
    """Files sitting in speech/ or racing/ waiting for their match, so the
    UI can show "waiting for racing footage" instead of nothing at all."""
    def _list(directory: Path) -> list[str]:
        exts = _config.watcher.video_extensions
        try:
            return sorted(p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)
        except FileNotFoundError:
            return []

    return jsonify({
        "speech_waiting": _list(_config.paths.speech_dir),
        "racing_waiting": _list(_config.paths.racing_dir),
    })


@app.route("/upload", methods=["POST"])
def upload():
    speech_file = request.files.get("speech")
    racing_file = request.files.get("racing")

    if speech_file and speech_file.filename:
        dest = _unique_dest(_config.paths.speech_dir, speech_file.filename)
        speech_file.save(dest)
        _logger.info("Uploaded speech file: %s", dest)

    if racing_file and racing_file.filename:
        dest = _unique_dest(_config.paths.racing_dir, racing_file.filename)
        racing_file.save(dest)
        _logger.info("Uploaded racing file: %s", dest)

    _watcher_wake()
    return redirect("/")


def _watcher_wake():
    try:
        _watcher._wake_event.set()
    except Exception:
        pass


@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(_config.paths.output_dir, filename, as_attachment=True)


def start_background_worker() -> None:
    t = threading.Thread(target=_worker_loop, daemon=True)
    t.start()


start_background_worker()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
