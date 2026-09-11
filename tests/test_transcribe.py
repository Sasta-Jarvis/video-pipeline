"""Tests the whisper-output -> SRT glue code without needing network access
to download a real model (faster-whisper's WhisperModel is mocked out).
Real end-to-end transcription is exercised manually / in an environment
with internet access -- see README's testing notes.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from video_pipeline import transcribe  # noqa: E402
from video_pipeline.config import load_config  # noqa: E402

WORK = Path(__file__).resolve().parent / "_transcribe_test_work"


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


class TestTranscribe(unittest.TestCase):
    def setUp(self):
        WORK.mkdir(parents=True, exist_ok=True)
        transcribe._model_cache.clear()

    def tearDown(self):
        import shutil

        shutil.rmtree(WORK, ignore_errors=True)

    def test_transcribe_to_srt_writes_expected_content(self):
        words = [
            FakeWord(0.0, 0.4, " This"),
            FakeWord(0.4, 0.7, " is"),
            FakeWord(0.7, 1.0, " a"),
            FakeWord(1.0, 1.5, " test."),
        ]
        fake_segment = FakeSegment(start=0.0, end=1.5, text=" This is a test.", words=words)
        fake_model = MagicMock()
        fake_model.transcribe.return_value = ([fake_segment], MagicMock(language="en"))

        with patch.object(transcribe, "get_model", return_value=fake_model):
            cfg = load_config(config_path=None)
            out_srt = WORK / "out.srt"
            result = transcribe.transcribe_to_srt(WORK / "fake_audio.wav", out_srt, cfg)

        self.assertEqual(result, out_srt)
        content = out_srt.read_text(encoding="utf-8")
        self.assertIn("This is a test.", content)
        self.assertIn("00:00:00,000 -->", content)
        fake_model.transcribe.assert_called_once()
        _, kwargs = fake_model.transcribe.call_args
        self.assertTrue(kwargs["word_timestamps"])

    def test_transcribe_to_srt_raises_on_empty_audio(self):
        fake_model = MagicMock()
        fake_model.transcribe.return_value = ([], MagicMock(language="en"))
        with patch.object(transcribe, "get_model", return_value=fake_model):
            cfg = load_config(config_path=None)
            with self.assertRaises(transcribe.TranscriptionError):
                transcribe.transcribe_to_srt(WORK / "silent.wav", WORK / "out.srt", cfg)

    def test_get_model_caches_by_config(self):
        with patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value = MagicMock()
            cfg = load_config(config_path=None)
            cfg.whisper.model = "tiny"
            cfg.whisper.device = "cpu"
            cfg.whisper.compute_type = "int8"
            m1 = transcribe.get_model(cfg)
            m2 = transcribe.get_model(cfg)
            self.assertIs(m1, m2)
            MockModel.assert_called_once()


if __name__ == "__main__":
    unittest.main()
