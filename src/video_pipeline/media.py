"""ffmpeg/ffprobe wrappers: probing, audio extraction, racing video
preparation (trim/loop), and the final combine+subtitle-burn encode.

Design goal called out by the user: don't unnecessarily re-encode the
racing footage multiple times. The strategy here is:

  1. If the racing clip is already >= the speech duration, it's used
     as-is -- zero processing, not even a copy.
  2. If it's shorter, it's looped via the concat demuxer with `-c copy`
     (stream copy: no re-encode, just remuxing the same bytes N times).
  3. Exactly one real re-encode happens: the final combine pass, which
     simultaneously trims to the exact speech duration, muxes in the
     speech audio, burns subtitles, and applies the configured
     codec/resolution/fps. Burning subtitles requires decoding+encoding
     video no matter what, so this is the minimum possible work.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Config


class MediaError(RuntimeError):
    """Raised when an ffmpeg/ffprobe invocation fails. Carries the full
    command and captured stderr so job failure logs are self-explanatory."""

    def __init__(self, message: str, cmd: list[str] | None = None, stderr: str = ""):
        self.cmd = cmd or []
        self.stderr = stderr
        full = message
        if cmd:
            full += f"\nCommand: {' '.join(cmd)}"
        if stderr:
            tail = "\n".join(stderr.strip().splitlines()[-40:])
            full += f"\n--- ffmpeg/ffprobe stderr (tail) ---\n{tail}"
        super().__init__(full)


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(
            f"'{name}' was not found on PATH. Install ffmpeg (which provides "
            f"both ffmpeg and ffprobe) and make sure it's on your PATH."
        )
    return path


def run(cmd: list[str], logger: logging.Logger | None = None, timeout: float = 3600.0) -> str:
    """Run a subprocess command, raising MediaError with full context on
    non-zero exit. Returns stdout on success."""
    if logger:
        logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError as e:
        raise MediaError(f"Executable not found: {cmd[0]}", cmd=cmd) from e
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"Command timed out after {timeout}s", cmd=cmd) from e

    if result.returncode != 0:
        raise MediaError(
            f"Command exited with code {result.returncode}",
            cmd=cmd,
            stderr=result.stderr or "",
        )
    return result.stdout


@dataclass
class ProbeResult:
    duration_seconds: float
    has_video: bool
    has_audio: bool
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None


def probe(path: Path, logger: logging.Logger | None = None) -> ProbeResult:
    ffprobe = _require_binary("ffprobe")
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate",
        "-of", "json",
        str(path),
    ]
    out = run(cmd, logger=logger, timeout=60.0)
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        raise MediaError(f"Could not parse ffprobe output for {path}", cmd=cmd) from e

    duration = None
    fmt = data.get("format", {})
    if fmt.get("duration"):
        try:
            duration = float(fmt["duration"])
        except ValueError:
            duration = None

    has_video = has_audio = False
    width = height = None
    fps = None
    video_codec = audio_codec = None
    stream_duration = None

    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and not has_video:
            has_video = True
            width = s.get("width")
            height = s.get("height")
            video_codec = s.get("codec_name")
            rate = s.get("r_frame_rate")
            if rate and "/" in rate:
                num, den = rate.split("/")
                try:
                    if float(den) != 0:
                        fps = float(num) / float(den)
                except ValueError:
                    fps = None
        elif s.get("codec_type") == "audio":
            has_audio = True
            audio_codec = s.get("codec_name")

    if duration is None:
        raise MediaError(f"ffprobe returned no duration for {path}", cmd=cmd)

    return ProbeResult(
        duration_seconds=duration,
        has_video=has_video,
        has_audio=has_audio,
        width=width,
        height=height,
        fps=fps,
        video_codec=video_codec,
        audio_codec=audio_codec,
    )


def probe_duration_seconds(path: Path, logger: logging.Logger | None = None) -> float:
    return probe(path, logger=logger).duration_seconds


def extract_audio(
    source_video: Path, out_wav: Path, logger: logging.Logger | None = None
) -> Path:
    """Extracts the audio track losslessly to PCM WAV, discarding video
    entirely. This file is used both as the whisper transcription input and
    as the final audio source for muxing."""
    ffmpeg = _require_binary("ffmpeg")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-i", str(source_video),
        "-vn",
        "-acodec", "pcm_s16le",
        str(out_wav),
    ]
    run(cmd, logger=logger)
    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise MediaError(f"Audio extraction produced no output for {source_video}")
    return out_wav


def _build_concat_list(source: Path, loops: int, list_file: Path) -> None:
    # ffmpeg concat demuxer requires the file's absolute path, single-quoted,
    # with internal single quotes escaped.
    escaped = str(source.resolve()).replace("'", r"'\''")
    with open(list_file, "w", encoding="utf-8") as f:
        for _ in range(loops):
            f.write(f"file '{escaped}'\n")


def prepare_racing_video(
    racing_path: Path,
    target_duration: float,
    work_dir: Path,
    logger: logging.Logger | None = None,
) -> Path:
    """Returns a path to a racing video whose duration is >= target_duration.
    If the source is already long enough, returns it unchanged (no copy, no
    re-encode). Otherwise loops it via stream-copy concat."""
    racing_probe = probe(racing_path, logger=logger)
    if not racing_probe.has_video:
        raise MediaError(f"Racing file has no video stream: {racing_path}")

    if racing_probe.duration_seconds >= target_duration:
        if logger:
            logger.info(
                "Racing clip (%.2fs) already covers speech duration (%.2fs); using as-is.",
                racing_probe.duration_seconds,
                target_duration,
            )
        return racing_path

    loops = math.ceil(target_duration / racing_probe.duration_seconds)
    if logger:
        logger.info(
            "Racing clip (%.2fs) shorter than speech (%.2fs); looping x%d via stream copy.",
            racing_probe.duration_seconds,
            target_duration,
            loops,
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    list_file = work_dir / "racing_concat_list.txt"
    looped_out = work_dir / "racing_looped.mp4"
    _build_concat_list(racing_path, loops, list_file)

    ffmpeg = _require_binary("ffmpeg")
    cmd = [
        ffmpeg, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(looped_out),
    ]
    run(cmd, logger=logger)

    if not looped_out.exists() or looped_out.stat().st_size == 0:
        raise MediaError("Looping racing video produced no output", cmd=cmd)
    return looped_out


def _escape_for_ffmpeg_filter_path(path: Path) -> str:
    """Escape a filesystem path for embedding inside an ffmpeg filtergraph
    string (e.g. subtitles=<path>). Handles Windows drive-letter colons and
    backslashes as well as POSIX paths with special characters."""
    s = str(path.resolve())
    s = s.replace("\\", "/")       # normalize Windows backslashes to forward slashes
    s = s.replace(":", r"\:")      # escape drive-letter / any colon
    s = s.replace("'", r"\'")
    return s


def _build_subtitle_style(config: Config) -> str:
    sc = config.subtitles
    parts = [
        f"FontName={sc.font_name}",
        f"FontSize={sc.font_size}",
        f"PrimaryColour={sc.primary_color}",
        f"OutlineColour={sc.outline_color}",
        f"BackColour={sc.back_color}",
        f"Bold={1 if sc.bold else 0}",
        f"Outline={sc.outline_width}",
        f"Shadow={sc.shadow}",
        f"Alignment={sc.alignment}",
        f"MarginV={sc.margin_v}",
    ]
    return ",".join(parts)


def combine_final(
    racing_video: Path,
    speech_audio: Path,
    srt_path: Path | None,
    duration_seconds: float,
    out_path: Path,
    config: Config,
    logger: logging.Logger | None = None,
) -> Path:
    """The single re-encode pass: trims racing video to duration_seconds,
    muxes in the speech audio, optionally burns subtitles, applies the
    configured codec/resolution/fps/audio settings, and writes an
    upload-ready MP4."""
    ffmpeg = _require_binary("ffmpeg")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vf_parts: list[str] = []
    if config.video.resolution:
        try:
            w, h = config.video.resolution.lower().split("x")
            vf_parts.append(
                f"scale={int(w)}:{int(h)}:force_original_aspect_ratio=increase,"
                f"crop={int(w)}:{int(h)}"
            )
        except ValueError:
            raise MediaError(
                f"Invalid video.resolution '{config.video.resolution}', expected WIDTHxHEIGHT"
            )

    if config.subtitles.enabled and config.subtitles.burn_in and srt_path is not None:
        escaped = _escape_for_ffmpeg_filter_path(srt_path)
        style = _build_subtitle_style(config)
        vf_parts.append(f"subtitles='{escaped}':force_style='{style}'")

    cmd = [
        ffmpeg, "-y",
        "-i", str(racing_video),
        "-i", str(speech_audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
    ]

    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    codec = config.video.codec
    cmd += ["-c:v", codec]
    if codec == "h264_nvenc":
        cmd += ["-preset", _nvenc_preset(config.video.preset), "-cq", str(config.video.crf)]
    else:
        cmd += ["-preset", config.video.preset, "-crf", str(config.video.crf)]

    cmd += [
        "-pix_fmt", config.video.pixel_format,
        "-r", str(config.video.fps),
        "-c:a", config.audio.codec,
        "-b:a", config.audio.bitrate,
        "-ar", str(config.audio.sample_rate),
        "-t", f"{duration_seconds:.3f}",
        "-movflags", "+faststart",
        str(out_path),
    ]

    run(cmd, logger=logger, timeout=7200.0)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise MediaError("Final combine produced no output", cmd=cmd)
    return out_path


def _nvenc_preset(software_preset: str) -> str:
    """Map a libx264-style preset name to the closest nvenc preset so the
    same config value works regardless of which encoder auto-selection
    picked."""
    mapping = {
        "ultrafast": "p1", "superfast": "p1", "veryfast": "p2",
        "faster": "p3", "fast": "p4", "medium": "p4",
        "slow": "p5", "slower": "p6", "veryslow": "p7",
    }
    return mapping.get(software_preset, "p4")
