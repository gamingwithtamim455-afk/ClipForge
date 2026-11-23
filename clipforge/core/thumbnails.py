"""Video thumbnails for the Home projects grid.

One frame grabbed with ffmpeg, scaled to a small JPEG. Thumbnails are
best-effort: when extraction fails (missing ffmpeg, unreadable file) the
caller keeps the generated placeholder art instead of crashing.
"""

from __future__ import annotations

from pathlib import Path

from .ffmpeg import FFmpegError, find_ffmpeg, run_ffmpeg

__all__ = ["generate_thumbnail"]


def generate_thumbnail(
    video_path: str,
    out_path: str | Path,
    *,
    at_seconds: float = 1.0,
    width: int = 320,
) -> str:
    """Grab one frame as a JPEG; returns ``out_path`` or ``""`` on failure."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return ""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_ffmpeg(
            [
                "-ss", f"{max(0.0, at_seconds):.3f}",
                "-i", video_path,
                "-frames:v", "1",
                "-vf", f"scale={width}:-1",
                "-q:v", "4",
                str(out),
            ],
            timeout=60.0,
        )
    except (FFmpegError, OSError):
        return ""
    return str(out) if out.exists() else ""
