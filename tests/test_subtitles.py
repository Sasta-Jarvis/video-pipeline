"""Pure-logic tests for subtitle cue formatting -- no ffmpeg/whisper needed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from video_pipeline.subtitles import (  # noqa: E402
    Word,
    cues_to_srt,
    format_timestamp,
    words_to_cues,
)


class TestSubtitles(unittest.TestCase):
    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(0), "00:00:00,000")
        self.assertEqual(format_timestamp(61.5), "00:01:01,500")
        self.assertEqual(format_timestamp(3661.234), "01:01:01,234")

    def test_words_to_cues_respects_line_budget(self):
        words = [
            Word(start=i * 0.3, end=i * 0.3 + 0.25, text=w)
            for i, w in enumerate(
                "the quick brown fox jumps over the lazy dog and keeps running".split()
            )
        ]
        cues = words_to_cues(
            words, max_chars_per_line=20, max_lines_per_cue=2, max_cue_duration_seconds=100
        )
        self.assertGreater(len(cues), 0)
        for cue in cues:
            self.assertLessEqual(len(cue.lines), 2)
            for line in cue.lines:
                self.assertLessEqual(len(line), max(20, len(max(line.split(), key=len, default=""))))

    def test_words_to_cues_splits_on_pause(self):
        words = [
            Word(start=0.0, end=0.3, text="hello"),
            Word(start=0.3, end=0.6, text="world"),
            Word(start=5.0, end=5.3, text="new"),
            Word(start=5.3, end=5.6, text="sentence"),
        ]
        cues = words_to_cues(
            words, max_chars_per_line=42, max_lines_per_cue=2, max_cue_duration_seconds=100,
            pause_break_seconds=0.6,
        )
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].lines, ["hello world"])
        self.assertEqual(cues[1].lines, ["new sentence"])

    def test_words_to_cues_splits_on_max_duration(self):
        words = [Word(start=i * 1.0, end=i * 1.0 + 0.5, text=f"w{i}") for i in range(10)]
        cues = words_to_cues(
            words, max_chars_per_line=100, max_lines_per_cue=5, max_cue_duration_seconds=3.0
        )
        for cue in cues:
            self.assertLessEqual(cue.end - cue.start, 3.0 + 1e-6)

    def test_cues_to_srt_format(self):
        words = [Word(start=0.0, end=1.0, text="hello"), Word(start=1.0, end=2.0, text="world")]
        cues = words_to_cues(words, 42, 2, 10)
        srt = cues_to_srt(cues)
        self.assertIn("1\n00:00:00,000 --> 00:00:02,000\nhello world", srt)

    def test_empty_words_produces_empty_cues(self):
        self.assertEqual(words_to_cues([], 42, 2, 6), [])


if __name__ == "__main__":
    unittest.main()
