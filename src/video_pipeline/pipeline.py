"""Pipeline orchestration: turns a claimed (speech, racing) pair into a
finished, subtitled MP4 in output/.

Implements the job's 11 steps end to end, recording progress in the job DB
at each stage and keeping a full per-job log alongside the job's files so a
failure is self-explanatory without digging through the main app log.

This function is designed to never raise: any failure is caught, logged in
detail, and the job is moved to failed/ -- the calling watch loop just moves
on to the next job. That's the "don't crash the whole application" contract.
"""
from __future__ import annotations

import logging
import shutil
import traceback
from pathlib import Path

from . import media, transcribe
from .config import Config
from .db import Job, JobDB
from .logging_setup import close_job_logger, get_job_logger


class PipelineError(RuntimeError):
    pass


def _safe_move(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def run_job(
    job: Job,
    speech_src: Path,
    racing_src: Path,
    config: Config,
    db: JobDB,
    app_logger: logging.Logger,
) -> bool:
    """Runs one job end-to-end. Returns True on success, False on failure."""
    job_id = job.job_id
    processing_dir = config.paths.processing_dir / job_id
    work_dir = processing_dir / "work"
    processing_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    job_log_path = processing_dir / "job.log"
    job_logger = get_job_logger(job_id, job_log_path)

    try:
        job_logger.info("Starting job %s", job_id)
        job_logger.info("Speech source: %s", speech_src)
        job_logger.info("Racing source: %s", racing_src)

        # Step 2: move files into processing/ -- this is also what makes the
        # job durable against a crash: once moved, the watcher can never see
        # these source files again, so they can't be double-claimed.
        db.update_status(job_id, "processing", stage="moving_inputs")
        speech_path = _safe_move(speech_src, processing_dir / speech_src.name)
        racing_path = _safe_move(racing_src, processing_dir / racing_src.name)
        job_logger.info("Moved inputs into %s", processing_dir)

        # Step 3: extract speech audio (video discarded).
        db.set_stage(job_id, "extracting_audio")
        job_logger.info("Extracting speech audio (discarding speech video)...")
        speech_audio = media.extract_audio(
            speech_path, work_dir / "speech_audio.wav", logger=job_logger
        )

        # Step 4: determine exact duration from the extracted audio.
        db.set_stage(job_id, "measuring_duration")
        duration = media.probe_duration_seconds(speech_audio, logger=job_logger)
        job_logger.info("Speech duration: %.2fs", duration)
        if duration <= 0:
            raise PipelineError("Speech audio has zero or negative duration")

        # Step 5: prepare racing video (trim or loop to cover duration;
        # audio discarded entirely since it's never mapped into the output).
        db.set_stage(job_id, "preparing_racing_video")
        job_logger.info("Preparing racing video to cover %.2fs...", duration)
        racing_prepared = media.prepare_racing_video(
            racing_path, duration, work_dir, logger=job_logger
        )

        # Steps 6-7: transcription + subtitle generation.
        srt_path = None
        if config.subtitles.enabled:
            db.set_stage(job_id, "transcribing")
            job_logger.info("Transcribing speech audio locally...")
            srt_path = transcribe.transcribe_to_srt(
                speech_audio, work_dir / f"{job_id}.srt", config, logger=job_logger
            )

        # Steps 8-9: combine racing video + speech audio, burn subtitles.
        db.set_stage(job_id, "combining")
        job_logger.info(
            "Combining racing video + speech audio%s...",
            " and burning subtitles" if srt_path else "",
        )
        final_name = config.output.filename_template.format(job_id=job_id)
        final_tmp = work_dir / final_name
        media.combine_final(
            racing_video=racing_prepared,
            speech_audio=speech_audio,
            srt_path=srt_path if (config.subtitles.enabled and config.subtitles.burn_in) else None,
            duration_seconds=duration,
            out_path=final_tmp,
            config=config,
            logger=job_logger,
        )

        # Step 10: move final MP4 (and SRT) into output/.
        db.set_stage(job_id, "finalizing_output")
        final_out = config.paths.output_dir / final_name
        _safe_move(final_tmp, final_out)
        job_logger.info("Wrote final video to %s", final_out)

        if srt_path and config.subtitles.keep_srt and config.output.keep_srt_in_output:
            srt_out = config.paths.output_dir / f"{Path(final_name).stem}.srt"
            shutil.copy2(srt_path, srt_out)
            job_logger.info("Copied SRT to %s", srt_out)

        # Step 11: mark completed.
        db.update_status(job_id, "completed", stage="done", output_path=str(final_out))
        job_logger.info("Job %s completed successfully.", job_id)
        close_job_logger(job_logger)

        shutil.rmtree(processing_dir, ignore_errors=True)
        return True

    except Exception as e:
        job_logger.error("Job failed: %s", e)
        job_logger.error(traceback.format_exc())
        close_job_logger(job_logger)

        db.update_status(job_id, "failed", stage="failed", error_message=str(e))
        app_logger.error("Job %s failed: %s (see %s)", job_id, e, job_log_path)

        try:
            failed_dir = config.paths.failed_dir / job_id
            if processing_dir.exists():
                if failed_dir.exists():
                    shutil.rmtree(failed_dir, ignore_errors=True)
                shutil.move(str(processing_dir), str(failed_dir))
            else:
                failed_dir.mkdir(parents=True, exist_ok=True)

            error_summary = failed_dir / "ERROR.txt"
            error_summary.write_text(
                f"Job {job_id} failed.\n\n"
                f"Speech source: {speech_src}\n"
                f"Racing source: {racing_src}\n\n"
                f"Error: {e}\n\n"
                f"{traceback.format_exc()}",
                encoding="utf-8",
            )
        except Exception:
            app_logger.exception("Also failed while moving job %s into failed/", job_id)

        return False


def recover_interrupted_jobs(config: Config, db: JobDB, app_logger: logging.Logger) -> None:
    """Startup recovery: any job left in 'processing' status (the app died
    mid-job) is moved to failed/ with an explanatory log rather than being
    silently retried or silently lost. This is what makes restarts safe."""
    stuck = db.jobs_in_status("processing")
    for job in stuck:
        app_logger.warning(
            "Found interrupted job %s from a previous run; moving to failed/.",
            job.job_id,
        )
        processing_dir = config.paths.processing_dir / job.job_id
        failed_dir = config.paths.failed_dir / job.job_id
        try:
            if processing_dir.exists():
                if failed_dir.exists():
                    shutil.rmtree(failed_dir, ignore_errors=True)
                shutil.move(str(processing_dir), str(failed_dir))
            else:
                failed_dir.mkdir(parents=True, exist_ok=True)
            (failed_dir / "ERROR.txt").write_text(
                f"Job {job.job_id} was interrupted by an application restart "
                f"while in stage '{job.stage}'.\n"
                "It was moved here automatically on startup rather than being "
                "silently reprocessed or lost. If the source files are intact "
                "inside this folder, you can move them back into speech/ and "
                "racing/ to retry.\n",
                encoding="utf-8",
            )
        except Exception:
            app_logger.exception("Failed to recover interrupted job %s", job.job_id)
        finally:
            db.update_status(
                job.job_id,
                "failed",
                stage="interrupted",
                error_message="Interrupted by application restart",
            )
