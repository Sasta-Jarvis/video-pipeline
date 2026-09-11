#!/usr/bin/env python3
"""Standalone environment inspection: run this before anything else to see
what hardware/software was detected and which Whisper model / video codec
the app will default to. Also runnable as part of setup to sanity-check a
new machine.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from video_pipeline import hw_detect  # noqa: E402

if __name__ == "__main__":
    report = hw_detect.detect_hardware()
    print(report.summary())
    if not report.ffmpeg_path:
        print("\nffmpeg is required. Install it and re-run this script.")
        sys.exit(1)
    sys.exit(0)
