"""Unit tests that actually shell out to ffmpeg/ffprobe -- no mocking --
because the whole point is to verify the real commands work, not that our
Python glues strings together correctly.
"""
from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from video_pipeline import media  # noqa: E402
from video_pipeline.config import Config, load_config  # noqa: E402
from generate_sample_media import generate_racing_video, generate_speech_video  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
WORK = Path(__file__).resolve().parent / "_media_test_work"


def _cfg() -> Config:
    return load_config(config_path=None)


class TestMedia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FIXTURES.mkdir(parents=True, exist_ok=True)
        cls.speech = FIXTURES / "media_test_speech.mp4"
        cls.racing_short = FIXTURES / "media_test_racing_short.mp4"
        cls.racing_long = FIXTURES / "media_test_racing_long.mp4"
        if not cls.speech.exists():
            generate_speech_video(cls.speech)
        if not cls.racing_short.exists():
            generate_racing_video(cls.racing_short, duration_seconds=3.0)
        if not cls.racing_long.exists():
            generate_racing_video(cls.racing_long, duration_seconds=30.0)

    def setUp(self):
        if WORK.exists():
            shutil.rmtree(WORK)
        WORK.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(WORK, ignore_errors=True)

    def test_probe_duration(self):
        result = media.probe(self.speech)
        self.assertGreater(result.duration_seconds, 0)
        self.assertTrue(result.has_video)
        self.assertTrue(result.has_audio)

    def test_probe_missing_file_raises(self):
        with self.assertRaises(media.MediaError):
            media.probe(FIXTURES / "does_not_exist.mp4")

    def test_extract_audio_discards_video(self):
        out_wav = WORK / "audio.wav"
        result = media.extract_audio(self.speech, out_wav)
        self.assertTrue(result.exists())
        probed = media.probe(result)
        self.assertTrue(probed.has_audio)
        self.assertFalse(probed.has_video)
        # duration should match the source's audio duration closely
        source_duration = media.probe_duration_seconds(self.speech)
        self.assertAlmostEqual(probed.duration_seconds, source_duration, delta=0.2)

    def test_prepare_racing_video_uses_original_when_long_enough(self):
        target = media.probe_duration_seconds(self.racing_long) - 5
        result = media.prepare_racing_video(self.racing_long, target, WORK)
        self.assertEqual(result, self.racing_long, "should reuse original file, no copy/re-encode")

    def test_prepare_racing_video_loops_when_short(self):
        source_duration = media.probe_duration_seconds(self.racing_short)
        target = source_duration * 2.5
        result = media.prepare_racing_video(self.racing_short, target, WORK)
        self.assertNotEqual(result, self.racing_short)
        self.assertTrue(result.exists())
        looped_duration = media.probe_duration_seconds(result)
        self.assertGreaterEqual(looped_duration, target)

    def test_combine_final_duration_matches_speech_and_drops_racing_audio(self):
        cfg = _cfg()
        cfg.subtitles.enabled = False  # isolate the combine step from whisper
        speech_audio = media.extract_audio(self.speech, WORK / "speech_audio.wav")
        duration = media.probe_duration_seconds(speech_audio)
        racing_prepared = media.prepare_racing_video(self.racing_long, duration, WORK)

        out_path = WORK / "final.mp4"
        media.combine_final(
            racing_video=racing_prepared,
            speech_audio=speech_audio,
            srt_path=None,
            duration_seconds=duration,
            out_path=out_path,
            config=cfg,
        )

        self.assertTrue(out_path.exists())
        result = media.probe(out_path)
        self.assertTrue(result.has_video)
        self.assertTrue(result.has_audio)
        self.assertAlmostEqual(result.duration_seconds, duration, delta=0.3)


if __name__ == "__main__":
    unittest.main()
