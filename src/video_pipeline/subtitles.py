"""Word-timestamp -> readable SRT cue formatting.

Whisper gives per-word timestamps; this module regroups those words into
subtitle cues that are actually readable: capped line length, capped number
of lines, capped cue duration, and cue breaks preferred at natural pauses in
speech rather than mid-thought.
"""
from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class SubtitleCue:
    start: float
    end: float
    lines: list[str]


def format_timestamp(seconds: float) -> str:
    """SRT timestamp: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, ms = divmod(total_ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _wrap_text(text: str, max_chars_per_line: int, max_lines_per_cue: int) -> list[str]:
    wrapped = textwrap.wrap(text, width=max_chars_per_line) or [text]
    if len(wrapped) <= max_lines_per_cue:
        return wrapped
    # Too many lines for the budget: rewrap wider so it fits in max_lines_per_cue
    # lines (still as readable as the content allows).
    wider = max(max_chars_per_line, math.ceil(len(text) / max_lines_per_cue) + 1)
    wrapped = textwrap.wrap(text, width=wider)
    return wrapped[:max_lines_per_cue]


def words_to_cues(
    words: list[Word],
    max_chars_per_line: int,
    max_lines_per_cue: int,
    max_cue_duration_seconds: float,
    pause_break_seconds: float = 0.6,
) -> list[SubtitleCue]:
    """Greedily groups words into cues, breaking when the cue would exceed
    its character budget, its max duration, or when a natural pause (gap
    between words) is detected."""
    cues: list[SubtitleCue] = []
    current: list[Word] = []
    char_budget = max_chars_per_line * max_lines_per_cue

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = " ".join(w.text for w in current if w.text)
        if text:
            lines = _wrap_text(text, max_chars_per_line, max_lines_per_cue)
            cues.append(SubtitleCue(start=current[0].start, end=current[-1].end, lines=lines))
        current = []

    for w in words:
        text = w.text.strip()
        if not text:
            continue
        word = Word(start=w.start, end=w.end, text=text)

        if current:
            prospective_len = sum(len(x.text) for x in current) + len(current) + len(word.text)
            prospective_duration = word.end - current[0].start
            gap = word.start - current[-1].end
            if (
                prospective_len > char_budget
                or prospective_duration > max_cue_duration_seconds
                or gap > pause_break_seconds
            ):
                flush()

        current.append(word)

    flush()
    return cues


def cues_to_srt(cues: list[SubtitleCue]) -> str:
    lines: list[str] = []
    for i, cue in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}")
        lines.extend(cue.lines)
        lines.append("")
    return "\n".join(lines)
