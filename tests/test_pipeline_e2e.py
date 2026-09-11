"""End-to-end pipeline tests using real ffmpeg on generated sample media.

The whisper transcription call is mocked (network access to download a
model isn't assumed to be available in every test environment -- see
README) but every other step -- audio extraction, racing video looping,
the final combine+subtitle-burn ffmpeg encode, file moves, and DB state --
runs for real.
"""
from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_sample_media import generate_racing_video, generate_speech_video  # noqa: E402
from video_pipeline import media, pipeline  # noqa: E402
from video_pipeline.config import load_config  # noqa: E402
from video_pipeline.db import JobDB, make_source_key  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@dataclass
class FakeWord:
    start: float
    end: float
    word: str


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    words: list


def _fake_whisper_model(duration: float):
    """A fake WhisperModel producing plausible word timestamps spread over
    the given duration, so burned subtitles have real content to render."""
    script = "this is a test recording for the automated pipeline end to end".split()
    n = len(script)
    words = []
    for i, w in enumerate(script):
        start = duration * i / n
        end = duration * (i + 0.7) / n
        words.append(FakeWord(start=start, end=end, word=f" {w}"))
    segment = FakeSegment(start=0.0, end=duration, text=" ".join(script), words=words)

    model = MagicMock()
    model.transcribe.return_value = ([segment], MagicMock(language="en"))
    return model


class TestPipelineE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FIXTURES.mkdir(parents=True, exist_ok=True)
        cls.speech_fixture = FIXTURES / "e2e_speech.mp4"
        cls.racing_short_fixture = FIXTURES / "e2e_racing_short.mp4"
        cls.racing_long_fixture = FIXTURES / "e2e_racing_long.mp4"
        if not cls.speech_fixture.exists():
            generate_speech_video(cls.speech_fixture)
        if not cls.racing_short_fixture.exists():
            generate_racing_video(cls.racing_short_fixture, duration_seconds=3.0)
        if not cls.racing_long_fixture.exists():
            generate_racing_video(cls.racing_long_fixture, duration_seconds=20.0)

        cls.speech_duration = media.probe_duration_seconds(cls.speech_fixture)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vp_e2e_test_"))
        self.cfg = load_config(config_path=None, base_dir=self.tmp)
        self.cfg.paths.root = self.tmp
        self.cfg.paths.speech_dir = self.tmp / "speech"
        self.cfg.paths.racing_dir = self.tmp / "racing"
        self.cfg.paths.output_dir = self.tmp / "output"
        self.cfg.paths.failed_dir = self.tmp / "failed"
        self.cfg.paths.processing_dir = self.tmp / "processing"
        self.cfg.paths.state_db = self.tmp / "state.db"
        self.cfg.paths.log_dir = self.tmp / "logs"
        for d in self.cfg.paths.all_dirs():
            d.mkdir(parents=True, exist_ok=True)

        # Keep the test fast: small resolution, ultrafast preset.
        self.cfg.video.preset = "ultrafast"
        self.cfg.video.resolution = "320x240"
        self.cfg.video.fps = 24

        self.db = JobDB(self.cfg.paths.state_db)
        self.logger = logging.getLogger("test_pipeline_e2e")
        self.logger.addHandler(logging.NullHandler())

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _drop_files(self, speech_src: Path, racing_src: Path):
        speech_dst = self.cfg.paths.speech_dir / speech_src.name
        racing_dst = self.cfg.paths.racing_dir / racing_src.name
        shutil.copy2(speech_src, speech_dst)
        shutil.copy2(racing_src, racing_dst)
        return speech_dst, racing_dst

    def _claim(self, speech_path: Path, racing_path: Path):
        speech_key = make_source_key("speech", speech_path)
        racing_key = make_source_key("racing", racing_path)
        return self.db.create_job(speech_key, racing_key, speech_path.name, racing_path.name)

    def test_full_pipeline_with_short_racing_clip_loops_and_burns_subtitles(self):
        speech_path, racing_path = self._drop_files(self.speech_fixture, self.racing_short_fixture)
        job = self._claim(speech_path, racing_path)

        fake_model = _fake_whisper_model(self.speech_duration)
        with patch("video_pipeline.transcribe.get_model", return_value=fake_model):
            ok = pipeline.run_job(job, speech_path, racing_path, self.cfg, self.db, self.logger)

        self.assertTrue(ok)

        completed = self.db.get_job(job.job_id)
        self.assertEqual(completed.status, "completed")
        self.assertIsNotNone(completed.output_path)

        output_mp4 = Path(completed.output_path)
        self.assertTrue(output_mp4.exists())
        self.assertTrue(output_mp4.is_relative_to(self.cfg.paths.output_dir))

        result = media.probe(output_mp4)
        self.assertTrue(result.has_video)
        self.assertTrue(result.has_audio)
        self.assertAlmostEqual(result.duration_seconds, self.speech_duration, delta=0.5)
        self.assertEqual(result.width, 320)
        self.assertEqual(result.height, 240)

        srt_out = output_mp4.with_suffix(".srt")
        self.assertTrue(srt_out.exists())
        self.assertIn("test", srt_out.read_text(encoding="utf-8").lower())

        # Inputs must be gone from the watched folders (no double-processing risk).
        self.assertFalse(speech_path.exists())
        self.assertFalse(racing_path.exists())
        # processing/ is cleaned up after success.
        self.assertFalse((self.cfg.paths.processing_dir / job.job_id).exists())

    def test_full_pipeline_with_long_racing_clip_trims(self):
        speech_path, racing_path = self._drop_files(self.speech_fixture, self.racing_long_fixture)
        job = self._claim(speech_path, racing_path)

        fake_model = _fake_whisper_model(self.speech_duration)
        with patch("video_pipeline.transcribe.get_model", return_value=fake_model):
            ok = pipeline.run_job(job, speech_path, racing_path, self.cfg, self.db, self.logger)

        self.assertTrue(ok)
        completed = self.db.get_job(job.job_id)
        output_mp4 = Path(completed.output_path)
        result = media.probe(output_mp4)
        self.assertAlmostEqual(result.duration_seconds, self.speech_duration, delta=0.5)

    def test_pipeline_without_subtitles(self):
        self.cfg.subtitles.enabled = False
        speech_path, racing_path = self._drop_files(self.speech_fixture, self.racing_long_fixture)
        job = self._claim(speech_path, racing_path)

        ok = pipeline.run_job(job, speech_path, racing_path, self.cfg, self.db, self.logger)
        self.assertTrue(ok)

        completed = self.db.get_job(job.job_id)
        output_mp4 = Path(completed.output_path)
        self.assertTrue(output_mp4.exists())
        srt_out = output_mp4.with_suffix(".srt")
        self.assertFalse(srt_out.exists())

    def test_failed_job_moves_to_failed_with_error_log(self):
        # Corrupt "racing" file -> ffprobe/ffmpeg will fail deterministically.
        speech_path, _ = self._drop_files(self.speech_fixture, self.racing_long_fixture)
        bad_racing = self.cfg.paths.racing_dir / "corrupt.mp4"
        bad_racing.write_bytes(b"this is not a real video file")

        job = self._claim(speech_path, bad_racing)
        ok = pipeline.run_job(job, speech_path, bad_racing, self.cfg, self.db, self.logger)

        self.assertFalse(ok)
        failed = self.db.get_job(job.job_id)
        self.assertEqual(failed.status, "failed")
        self.assertIsNotNone(failed.error_message)

        failed_dir = self.cfg.paths.failed_dir / job.job_id
        self.assertTrue(failed_dir.exists())
        self.assertTrue((failed_dir / "ERROR.txt").exists())
        error_text = (failed_dir / "ERROR.txt").read_text(encoding="utf-8")
        self.assertIn("Error:", error_text)

        # Not left behind in processing/.
        self.assertFalse((self.cfg.paths.processing_dir / job.job_id).exists())

    def test_cannot_claim_same_source_twice(self):
        speech_path, racing_path = self._drop_files(self.speech_fixture, self.racing_long_fixture)
        self._claim(speech_path, racing_path)

        speech_key = make_source_key("speech", speech_path)
        racing_key = make_source_key("racing", racing_path)
        self.assertTrue(self.db.is_claimed(speech_key))
        self.assertTrue(self.db.is_claimed(racing_key))

        with self.assertRaises(Exception):
            self.db.create_job(speech_key, racing_key, speech_path.name, racing_path.name)

    def test_restart_recovery_moves_stuck_job_to_failed(self):
        speech_path, racing_path = self._drop_files(self.speech_fixture, self.racing_long_fixture)
        job = self._claim(speech_path, racing_path)

        # Simulate a crash mid-job: files moved into processing/, status
        # left at "processing", process dies before completing.
        proc_dir = self.cfg.paths.processing_dir / job.job_id
        proc_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(speech_path), str(proc_dir / speech_path.name))
        shutil.move(str(racing_path), str(proc_dir / racing_path.name))
        self.db.update_status(job.job_id, "processing", stage="combining")

        pipeline.recover_interrupted_jobs(self.cfg, self.db, self.logger)

        recovered = self.db.get_job(job.job_id)
        self.assertEqual(recovered.status, "failed")
        self.assertFalse(proc_dir.exists())
        failed_dir = self.cfg.paths.failed_dir / job.job_id
        self.assertTrue(failed_dir.exists())
        self.assertTrue((failed_dir / "ERROR.txt").exists())


if __name__ == "__main__":
    unittest.main()
