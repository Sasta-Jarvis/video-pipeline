"""Environment/hardware inspection used to pick sensible defaults.

This runs at startup (and can be run standalone via `scripts/inspect_env.py`)
to answer: what OS/Python are we on, is ffmpeg installed and what can it
encode with, is there an NVIDIA GPU with CUDA + how much VRAM, and therefore
which Whisper model / video encoder should we default to.

Nothing here is fatal-on-failure: every probe degrades to a safe default
instead of raising, because this module's whole job is to make a best-effort
guess, not to gate startup.
"""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class HardwareReport:
    os_name: str
    os_version: str
    python_version: str
    architecture: str
    cpu_count: int
    total_ram_gb: float

    ffmpeg_path: str | None
    ffmpeg_version: str | None
    ffprobe_path: str | None
    ffmpeg_encoders: list[str] = field(default_factory=list)

    gpu_vendor: str | None = None       # "nvidia" | "amd" | "apple" | None
    gpu_name: str | None = None
    vram_gb: float | None = None
    cuda_available: bool = False
    cuda_version: str | None = None

    recommended_whisper_model: str = "base"
    recommended_whisper_device: str = "cpu"
    recommended_whisper_compute_type: str = "int8"
    recommended_video_codec: str = "libx264"

    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Environment Report ===",
            f"OS:              {self.os_name} {self.os_version} ({self.architecture})",
            f"Python:          {self.python_version}",
            f"CPU cores:       {self.cpu_count}",
            f"RAM:             {self.total_ram_gb:.1f} GB",
            f"ffmpeg:          {self.ffmpeg_path or 'NOT FOUND'}"
            + (f"  ({self.ffmpeg_version})" if self.ffmpeg_version else ""),
            f"ffprobe:         {self.ffprobe_path or 'NOT FOUND'}",
            f"GPU:             {self.gpu_name or 'none detected'}"
            + (f"  [{self.vram_gb:.1f} GB VRAM]" if self.vram_gb else ""),
            f"CUDA available:  {self.cuda_available}"
            + (f"  ({self.cuda_version})" if self.cuda_version else ""),
            f"h264_nvenc:      {'yes' if 'h264_nvenc' in self.ffmpeg_encoders else 'no'}",
            "--- Recommendations ---",
            f"Whisper model:   {self.recommended_whisper_model}",
            f"Whisper device:  {self.recommended_whisper_device} "
            f"(compute_type={self.recommended_whisper_compute_type})",
            f"Video codec:     {self.recommended_video_codec}",
        ]
        if self.warnings:
            lines.append("--- Warnings ---")
            lines.extend(f"  ! {w}" for w in self.warnings)
        return "\n".join(lines)


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return result.stdout + result.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _detect_ffmpeg() -> tuple[str | None, str | None, str | None, list[str]]:
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    version = None
    encoders: list[str] = []

    if ffmpeg_path:
        out = _run([ffmpeg_path, "-version"])
        if out:
            first_line = out.splitlines()[0] if out.splitlines() else ""
            m = re.search(r"ffmpeg version (\S+)", first_line)
            version = m.group(1) if m else first_line.strip()

        enc_out = _run([ffmpeg_path, "-hide_banner", "-encoders"])
        if enc_out:
            for line in enc_out.splitlines():
                line = line.strip()
                m = re.match(r"^[VAS][F.][S.][X.][B.][D.]\s+(\S+)", line)
                if m:
                    encoders.append(m.group(1))

    return ffmpeg_path, version, ffprobe_path, encoders


def _detect_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (FileNotFoundError, ValueError, IndexError):
        pass
    try:
        import os

        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / (1024**3)
    except (ValueError, AttributeError, OSError):
        return 8.0  # conservative fallback


def _detect_nvidia_gpu() -> tuple[str | None, float | None, bool, str | None]:
    """Returns (gpu_name, vram_gb, cuda_available, cuda_version)."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None, None, False, None

    out = _run(
        [
            nvidia_smi,
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out or not out.strip():
        return None, None, False, None

    first_line = out.strip().splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    gpu_name = parts[0] if parts else None
    vram_gb = None
    if len(parts) > 1:
        try:
            vram_gb = float(parts[1]) / 1024.0  # MiB -> GiB
        except ValueError:
            vram_gb = None

    cuda_version = None
    version_out = _run([nvidia_smi])
    if version_out:
        m = re.search(r"CUDA Version:\s*(\S+)", version_out)
        if m:
            cuda_version = m.group(1)

    # A working nvidia-smi + driver is a strong proxy for CUDA being usable.
    # We confirm more precisely in torch/ctranslate2 if available, but don't
    # require torch just to answer this question.
    cuda_available = gpu_name is not None
    return gpu_name, vram_gb, cuda_available, cuda_version


def _recommend_whisper(
    cuda_available: bool, vram_gb: float | None, ram_gb: float
) -> tuple[str, str, str]:
    """Returns (model, device, compute_type)."""
    if cuda_available and vram_gb:
        if vram_gb >= 10:
            return "large-v3", "cuda", "float16"
        if vram_gb >= 6:
            return "medium", "cuda", "float16"
        if vram_gb >= 4:
            return "small", "cuda", "float16"
        return "base", "cuda", "float16"

    # CPU-only. faster-whisper on CPU with int8 is the reliable choice;
    # bigger models get slow fast on CPU, so weight RAM conservatively.
    # Thresholds are set a bit below the round number (15 not 16, 7 not 8)
    # because a machine advertised as "16GB" typically reports more like
    # 15.5-15.9 GB to the OS once reserved memory is subtracted.
    if ram_gb >= 15:
        return "small", "cpu", "int8"
    if ram_gb >= 7:
        return "base", "cpu", "int8"
    return "tiny", "cpu", "int8"


def detect_hardware() -> HardwareReport:
    warnings: list[str] = []

    os_name = platform.system()
    os_version = platform.release()
    python_version = platform.python_version()
    architecture = platform.machine()
    cpu_count = __import__("os").cpu_count() or 1
    ram_gb = _detect_ram_gb()

    ffmpeg_path, ffmpeg_version, ffprobe_path, encoders = _detect_ffmpeg()
    if not ffmpeg_path:
        warnings.append(
            "ffmpeg not found on PATH. Install it before running the pipeline "
            "(see README)."
        )
    if not ffprobe_path:
        warnings.append(
            "ffprobe not found on PATH. It normally ships alongside ffmpeg."
        )

    gpu_name, vram_gb, cuda_available, cuda_version = _detect_nvidia_gpu()

    if os_name == "Darwin" and "videotoolbox" in encoders:
        gpu_name = gpu_name or "Apple Silicon (VideoToolbox)"

    whisper_model, whisper_device, whisper_compute = _recommend_whisper(
        cuda_available, vram_gb, ram_gb
    )

    video_codec = "libx264"
    if cuda_available and "h264_nvenc" in encoders:
        video_codec = "h264_nvenc"
    elif os_name == "Darwin" and "h264_videotoolbox" in encoders:
        video_codec = "h264_videotoolbox"

    if cuda_available and "h264_nvenc" not in encoders and ffmpeg_path:
        warnings.append(
            "NVIDIA GPU detected but this ffmpeg build lacks h264_nvenc; "
            "falling back to libx264 (software encode, still works fine)."
        )

    return HardwareReport(
        os_name=os_name,
        os_version=os_version,
        python_version=python_version,
        architecture=architecture,
        cpu_count=cpu_count,
        total_ram_gb=ram_gb,
        ffmpeg_path=ffmpeg_path,
        ffmpeg_version=ffmpeg_version,
        ffprobe_path=ffprobe_path,
        ffmpeg_encoders=encoders,
        gpu_vendor="nvidia" if cuda_available else None,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        recommended_whisper_model=whisper_model,
        recommended_whisper_device=whisper_device,
        recommended_whisper_compute_type=whisper_compute,
        recommended_video_codec=video_codec,
        warnings=warnings,
    )


if __name__ == "__main__":
    report = detect_hardware()
    print(report.summary())
    if not report.ffmpeg_path:
        sys.exit(1)
