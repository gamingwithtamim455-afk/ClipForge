"""FFmpeg wrapper.

Bundled FFmpeg (see tools/download_ffmpeg.py) is used when present;
otherwise the system ``ffmpeg``/``ffprobe`` on PATH is used. All rendering
goes through this module so the binary location is resolved in one place.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "FFmpegError",
    "ProbeResult",
    "bundled_ffmpeg_dir",
    "encoder_is_usable",
    "encoder_quality_args",
    "extract_audio",
    "find_ffmpeg",
    "find_ffprobe",
    "list_video_encoders",
    "probe",
    "run_ffmpeg",
    "select_video_encoder",
]


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg/ffprobe call fails (rendering layer)."""


@dataclass
class ProbeResult:
    """Metadata returned by ``ffprobe`` for a media file."""

    path: str
    duration: float  # seconds
    width: int
    height: int
    fps: float
    has_audio: bool


def bundled_ffmpeg_dir() -> Path:
    """Directory holding the bundled ffmpeg/ffprobe binaries (if any)."""
    return Path(__file__).resolve().parent.parent / "bin"


def find_ffmpeg() -> str | None:
    """Path to an ffmpeg binary: bundled first, then system PATH."""
    bundled = bundled_ffmpeg_dir() / ("ffmpeg.exe" if _is_windows() else "ffmpeg")
    if bundled.exists():
        return str(bundled)
    return shutil.which("ffmpeg")


def find_ffprobe() -> str | None:
    """Path to an ffprobe binary: bundled first, then system PATH."""
    bundled = bundled_ffmpeg_dir() / ("ffprobe.exe" if _is_windows() else "ffprobe")
    if bundled.exists():
        return str(bundled)
    return shutil.which("ffprobe")


def _is_windows() -> bool:
    import sys

    return sys.platform.startswith("win")


def probe(path: str) -> ProbeResult:
    """Probe a media file with ffprobe (duration, size, fps, audio)."""
    ffprobe = find_ffprobe()
    if not ffprobe:
        raise RuntimeError("ffprobe not found — bundle FFmpeg or install it")
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=width,height,r_frame_rate,codec_type",
        "-of", "json",
        path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed for {Path(path).name}: {exc.stderr.strip()[:200]}") from exc
    data = json.loads(out.stdout or "{}")
    duration = float(data.get("format", {}).get("duration") or 0.0)
    width = height = 0
    fps = 0.0
    has_audio = False
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and width == 0:
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            rate = str(stream.get("r_frame_rate") or "0/1")
            try:
                num, den = rate.split("/")
                fps = float(num) / float(den) if float(den) else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0
        elif stream.get("codec_type") == "audio":
            has_audio = True
    return ProbeResult(
        path=path, duration=duration, width=width, height=height, fps=fps,
        has_audio=has_audio,
    )


def extract_audio(
    video_path: str, out_path: str, *, sample_rate: int = 16000,
    start: float | None = None, duration: float | None = None,
) -> str:
    """Extract the audio track as mono PCM16 WAV (for transcription).

    ``start``/``duration`` (seconds) extract only that slice of the
    audio — used for manual-range transcription so a 2 h podcast only
    pays for the chosen clips. The seek is an input seek: ffmpeg drops
    frames until the exact timestamp, so the slice starts
    sample-accurately at ``start``. Returns ``out_path``. Raises
    :class:`RuntimeError` when ffmpeg is missing or the extraction fails.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found — bundle FFmpeg or install it")
    cmd = [ffmpeg, "-y", "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(max(0.0, start))]
    cmd += ["-i", video_path]
    if duration is not None:
        cmd += ["-t", str(max(0.0, duration))]
    cmd += [
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        out_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Audio extraction failed for {Path(video_path).name}: "
            f"{exc.stderr.strip()[:200]}"
        ) from exc
    return out_path


def run_ffmpeg(args: list[str], *, timeout: float = 3600.0) -> subprocess.CompletedProcess:
    """Run ffmpeg with the given arguments; raise :class:`FFmpegError` on failure."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise FFmpegError("ffmpeg not found (no bundled binary, none on PATH)")
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-y", *args],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise FFmpegError(
            "ffmpeg failed (exit "
            f"{proc.returncode}): {proc.stderr.strip()[-1500:]}"
        )
    return proc


#: GPU H.264 encoders, in preference order (NVENC > QSV > AMF).
GPU_ENCODERS = ("h264_nvenc", "h264_qsv", "h264_amf")

#: Cache of encoder usability probes (encoder name -> can open).
_ENCODER_USABLE_CACHE: dict[str, bool] = {}


def encoder_is_usable(encoder: str) -> bool:
    """Check an encoder can actually open (not just be listed).

    A listed encoder may still fail at open time — e.g. ``h264_nvenc``
    is built into many ffmpeg binaries but needs an NVIDIA GPU + CUDA
    driver. A 1-frame null encode proves it works. Results are cached.
    """
    if encoder in _ENCODER_USABLE_CACHE:
        return _ENCODER_USABLE_CACHE[encoder]
    ffmpeg = find_ffmpeg()
    usable = False
    if ffmpeg:
        proc = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-v", "error",
                "-f", "lavfi", "-i", "testsrc=size=64x64:rate=1:duration=0.1",
                "-frames:v", "1", "-c:v", encoder, "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=60,
        )
        usable = proc.returncode == 0
    _ENCODER_USABLE_CACHE[encoder] = usable
    return usable


def list_video_encoders() -> set[str]:
    """Video encoder names this ffmpeg build supports."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return set()
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True, text=True, timeout=60,
    )
    encoders: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith(("V", "D")):
            encoders.add(parts[1])
    return encoders


def select_video_encoder(preference: str = "Auto") -> tuple[str, bool]:
    """Pick an H.264 encoder.

    ``preference`` is ``"Auto"`` (GPU when available, else CPU),
    ``"GPU"`` (fail if no GPU encoder exists) or ``"CPU"``.
    Returns ``(encoder_name, is_gpu)``. GPU candidates are probed with
    a 1-frame encode — a listed encoder that cannot open (no GPU/driver)
    is skipped.
    """
    available = list_video_encoders()
    gpu = next(
        (name for name in GPU_ENCODERS
         if name in available and encoder_is_usable(name)),
        None,
    )
    if preference == "CPU":
        if "libx264" not in available:
            raise FFmpegError("libx264 encoder is not available in this ffmpeg build")
        return "libx264", False
    if preference == "GPU":
        if gpu is None:
            raise FFmpegError("no GPU H.264 encoder available in this ffmpeg build")
        return gpu, True
    # Auto.
    if gpu is not None:
        return gpu, True
    if "libx264" not in available:
        raise FFmpegError("no H.264 encoder available in this ffmpeg build")
    return "libx264", False


def encoder_quality_args(encoder: str, crf: int) -> list[str]:
    """Quality flags for an encoder, equivalent to libx264 CRF ``crf``.

    libx264 takes ``-crf`` directly. NVENC/QSV/AMF use their own
    constant-quality modes; the same numeric value keeps the quality
    target comparable (14–16 = visually near-lossless for clips).
    """
    if encoder == "libx264":
        return ["-crf", str(crf)]
    if encoder == "h264_nvenc":
        return ["-rc", "vbr", "-cq", str(crf), "-preset", "slow"]
    if encoder == "h264_qsv":
        return ["-global_quality", str(crf), "-preset", "slow"]
    if encoder == "h264_amf":
        return ["-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf), "-qp_b", str(crf)]
    return ["-crf", str(crf)]
