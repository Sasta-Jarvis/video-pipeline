#!/usr/bin/env python3
"""Generates synthetic test media entirely locally, no internet/API needed:

- A "speech" video: a plain color video track + real synthesized speech
  audio (via espeak-ng, so Whisper has actual words to transcribe -- this
  lets tests assert real transcription content, not just "did it run").
- A "racing" video: a moving-pattern video (ffmpeg lavfi testsrc/mandelbrot)
  with a distinct audio tone track, so tests can assert the tone never
  makes it into the final output.

Run standalone to produce files under tests/fixtures/, or import
`generate_speech_video` / `generate_racing_video` from test code.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

SPEECH_TEXT = (
    "This is a test recording for the automated video pipeline. "
    "The quick brown fox jumps over the lazy dog. "
    "Racing gameplay footage will be combined with this narration automatically."
)


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise RuntimeError(f"{binary} is required to generate test media but was not found.")
    return path


def generate_speech_video(out_path: Path, duration_hint_text: str = SPEECH_TEXT) -> Path:
    """Real synthesized speech (espeak-ng) muxed with a plain color video,
    so it looks like a genuine screen-recorded commentary clip."""
    espeak = _require("espeak-ng")
    ffmpeg = _require("ffmpeg")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_wav = out_path.with_suffix(".rawspeech.wav")
    subprocess.run(
        [espeak, "-s", "150", "-w", str(raw_wav), duration_hint_text],
        check=True,
        capture_output=True,
    )

    subprocess.run(
        [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", "color=c=gray:s=640x360:r=30",
            "-i", str(raw_wav),
            "-shortest",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    raw_wav.unlink(missing_ok=True)
    return out_path


def generate_racing_video(out_path: Path, duration_seconds: float = 6.0) -> Path:
    """A moving test pattern (stands in for gameplay footage) with a pure
    tone audio track that must NOT survive into the final combined video."""
    ffmpeg = _require("ffmpeg")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", f"testsrc=size=640x360:rate=30:duration={duration_seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=880:duration={duration_seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return out_path


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    speech_out = FIXTURES_DIR / "sample_speech.mp4"
    racing_out = FIXTURES_DIR / "sample_racing.mp4"
    print(f"Generating {speech_out} ...")
    generate_speech_video(speech_out)
    print(f"Generating {racing_out} ...")
    generate_racing_video(racing_out, duration_seconds=6.0)
    print("Done.")
    sys.exit(0)
