"""Configuration loading.

Loads config.yaml (falling back to built-in defaults for anything missing
or set to "auto"), resolves all paths to absolute Path objects, and expands
"auto" fields using hw_detect. Everything downstream works off a single
Config object so there's one source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import hw_detect


@dataclass
class PathsConfig:
    root: Path
    speech_dir: Path
    racing_dir: Path
    output_dir: Path
    failed_dir: Path
    processing_dir: Path
    state_db: Path
    log_dir: Path

    def all_dirs(self) -> list[Path]:
        return [
            self.speech_dir,
            self.racing_dir,
            self.output_dir,
            self.failed_dir,
            self.processing_dir,
            self.log_dir,
        ]


@dataclass
class WatcherConfig:
    poll_interval_seconds: float = 5.0
    stable_wait_seconds: float = 5.0
    video_extensions: tuple[str, ...] = (
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"
    )


@dataclass
class WhisperConfig:
    model: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True


@dataclass
class SubtitlesConfig:
    enabled: bool = True
    burn_in: bool = True
    keep_srt: bool = True
    max_chars_per_line: int = 42
    max_lines_per_cue: int = 2
    max_cue_duration_seconds: float = 6.0
    font_name: str = "Arial"
    font_size: int = 28
    primary_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    back_color: str = "&H80000000"
    bold: bool = True
    outline_width: int = 2
    shadow: int = 0
    alignment: int = 2
    margin_v: int = 60


@dataclass
class VideoConfig:
    codec: str = "libx264"
    preset: str = "veryfast"
    crf: int = 20
    resolution: str | None = "1920x1080"
    fps: int = 60
    pixel_format: str = "yuv420p"


@dataclass
class AudioConfig:
    codec: str = "aac"
    bitrate: str = "192k"
    sample_rate: int = 48000


@dataclass
class RacingConfig:
    loop_method: str = "concat_loop"


@dataclass
class OutputConfig:
    container: str = "mp4"
    filename_template: str = "{job_id}.mp4"
    keep_srt_in_output: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 5


@dataclass
class Config:
    paths: PathsConfig
    watcher: WatcherConfig
    whisper: WhisperConfig
    subtitles: SubtitlesConfig
    video: VideoConfig
    audio: AudioConfig
    racing: RacingConfig
    output: OutputConfig
    logging: LoggingConfig
    hardware: hw_detect.HardwareReport = field(repr=False)


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def load_config(config_path: str | Path | None, base_dir: str | Path = ".") -> Config:
    base_dir = Path(base_dir).resolve()
    raw: dict[str, Any] = {}
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

    hardware = hw_detect.detect_hardware()

    # --- paths ---
    root = Path(_get(raw, "paths", "root", default="./video-pipeline"))
    if not root.is_absolute():
        root = (base_dir / root).resolve()

    def resolve_dir(key: str, default_name: str) -> Path:
        val = _get(raw, "paths", key)
        p = Path(val) if val else (root / default_name)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        return p

    state_db_val = _get(raw, "paths", "state_db")
    state_db = Path(state_db_val) if state_db_val else (root / "state.db")
    if not state_db.is_absolute():
        state_db = (base_dir / state_db).resolve()

    paths = PathsConfig(
        root=root,
        speech_dir=resolve_dir("speech_dir", "speech"),
        racing_dir=resolve_dir("racing_dir", "racing"),
        output_dir=resolve_dir("output_dir", "output"),
        failed_dir=resolve_dir("failed_dir", "failed"),
        processing_dir=resolve_dir("processing_dir", "processing"),
        state_db=state_db,
        log_dir=resolve_dir("log_dir", "logs"),
    )

    # --- watcher ---
    watcher = WatcherConfig(
        poll_interval_seconds=_get(raw, "watcher", "poll_interval_seconds", default=5.0),
        stable_wait_seconds=_get(raw, "watcher", "stable_wait_seconds", default=5.0),
        video_extensions=tuple(
            _get(
                raw,
                "watcher",
                "video_extensions",
                default=[".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"],
            )
        ),
    )

    # --- whisper (resolve "auto") ---
    w_model = _get(raw, "whisper", "model", default="auto")
    w_device = _get(raw, "whisper", "device", default="auto")
    w_compute = _get(raw, "whisper", "compute_type", default="auto")
    if w_model == "auto":
        w_model = hardware.recommended_whisper_model
    if w_device == "auto":
        w_device = hardware.recommended_whisper_device
    if w_compute == "auto":
        w_compute = hardware.recommended_whisper_compute_type
    whisper = WhisperConfig(
        model=w_model,
        device=w_device,
        compute_type=w_compute,
        language=_get(raw, "whisper", "language"),
        beam_size=_get(raw, "whisper", "beam_size", default=5),
        vad_filter=_get(raw, "whisper", "vad_filter", default=True),
    )

    # --- subtitles ---
    sub_defaults = SubtitlesConfig()
    subtitles = SubtitlesConfig(
        enabled=_get(raw, "subtitles", "enabled", default=sub_defaults.enabled),
        burn_in=_get(raw, "subtitles", "burn_in", default=sub_defaults.burn_in),
        keep_srt=_get(raw, "subtitles", "keep_srt", default=sub_defaults.keep_srt),
        max_chars_per_line=_get(
            raw, "subtitles", "max_chars_per_line", default=sub_defaults.max_chars_per_line
        ),
        max_lines_per_cue=_get(
            raw, "subtitles", "max_lines_per_cue", default=sub_defaults.max_lines_per_cue
        ),
        max_cue_duration_seconds=_get(
            raw,
            "subtitles",
            "max_cue_duration_seconds",
            default=sub_defaults.max_cue_duration_seconds,
        ),
        font_name=_get(raw, "subtitles", "font_name", default=sub_defaults.font_name),
        font_size=_get(raw, "subtitles", "font_size", default=sub_defaults.font_size),
        primary_color=_get(
            raw, "subtitles", "primary_color", default=sub_defaults.primary_color
        ),
        outline_color=_get(
            raw, "subtitles", "outline_color", default=sub_defaults.outline_color
        ),
        back_color=_get(raw, "subtitles", "back_color", default=sub_defaults.back_color),
        bold=_get(raw, "subtitles", "bold", default=sub_defaults.bold),
        outline_width=_get(
            raw, "subtitles", "outline_width", default=sub_defaults.outline_width
        ),
        shadow=_get(raw, "subtitles", "shadow", default=sub_defaults.shadow),
        alignment=_get(raw, "subtitles", "alignment", default=sub_defaults.alignment),
        margin_v=_get(raw, "subtitles", "margin_v", default=sub_defaults.margin_v),
    )

    # --- video (resolve "auto" codec) ---
    v_codec = _get(raw, "video", "codec", default="auto")
    if v_codec == "auto":
        v_codec = hardware.recommended_video_codec
    v_defaults = VideoConfig()
    video = VideoConfig(
        codec=v_codec,
        preset=_get(raw, "video", "preset", default=v_defaults.preset),
        crf=_get(raw, "video", "crf", default=v_defaults.crf),
        resolution=_get(raw, "video", "resolution", default=v_defaults.resolution),
        fps=_get(raw, "video", "fps", default=v_defaults.fps),
        pixel_format=_get(raw, "video", "pixel_format", default=v_defaults.pixel_format),
    )

    # --- audio ---
    a_defaults = AudioConfig()
    audio = AudioConfig(
        codec=_get(raw, "audio", "codec", default=a_defaults.codec),
        bitrate=_get(raw, "audio", "bitrate", default=a_defaults.bitrate),
        sample_rate=_get(raw, "audio", "sample_rate", default=a_defaults.sample_rate),
    )

    racing = RacingConfig(
        loop_method=_get(raw, "racing", "loop_method", default="concat_loop"),
    )

    o_defaults = OutputConfig()
    output = OutputConfig(
        container=_get(raw, "output", "container", default=o_defaults.container),
        filename_template=_get(
            raw, "output", "filename_template", default=o_defaults.filename_template
        ),
        keep_srt_in_output=_get(
            raw, "output", "keep_srt_in_output", default=o_defaults.keep_srt_in_output
        ),
    )

    l_defaults = LoggingConfig()
    logging_cfg = LoggingConfig(
        level=_get(raw, "logging", "level", default=l_defaults.level),
        max_bytes=_get(raw, "logging", "max_bytes", default=l_defaults.max_bytes),
        backup_count=_get(raw, "logging", "backup_count", default=l_defaults.backup_count),
    )

    return Config(
        paths=paths,
        watcher=watcher,
        whisper=whisper,
        subtitles=subtitles,
        video=video,
        audio=audio,
        racing=racing,
        output=output,
        logging=logging_cfg,
        hardware=hardware,
    )


def ensure_directories(config: Config) -> None:
    for d in config.paths.all_dirs():
        d.mkdir(parents=True, exist_ok=True)
