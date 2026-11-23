"""Caption preview stills for the Edit step.

Renders ONE frame of a clip the way the final render will look: the
source frame at a chosen time, reframed by the clip's crop window,
scaled to the output size, with the caption line burned in from the
SAME :func:`~clipforge.core.captions.build_ass` document the render
uses (same preset, same anchor, same face-safe placement). The Edit
step shows this still under the player, so picking a caption style
shows the caption ON the video, in its real position — the overview
the reference app gives — instead of a style name in a list.

One frame, on demand: the caller runs this in a background job and
only re-renders when the clip, the preset or the caption line
changes — never per video frame.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .captions import (
    CaptionColors,
    CaptionLayout,
    GradientConfig,
    build_ass,
    fonts_dir,
    resolve_custom_font,
)
from .face_tracking import CropPath
from .ffmpeg import escape_filter_arg as _escape_filter_arg, find_ffmpeg, run_ffmpeg
from .transcription import Word

__all__ = ["render_caption_still"]


def render_caption_still(
    video_path: str,
    *,
    at_time: float,
    clip_start: float = 0.0,
    crop_rect: tuple[float, float, float, float] | None,
    words: list[Word],
    language: str = "en",
    preset_name: str = "Hormozi",
    width: int = 1080,
    height: int = 1920,
    face_rects: list[tuple[float, float, float, float]] | None = None,
    layout: CaptionLayout | None = None,
    colors: CaptionColors | None = None,
    gradient: GradientConfig | None = None,
    keyword_scale: float = 0.0,
    custom_font: str = "",
    caption_font: str = "",
    custom_fonts_dir: str = "",
    out_path: str | Path | None = None,
) -> Path:
    """Render one captioned still frame; returns the PNG path.

    ``at_time`` is the SOURCE-video time of the frame and
    ``clip_start`` the clip's start in the source, so the frame's
    clip-relative time is ``at_time - clip_start`` (the caption line
    active then is what shows). ``crop_rect`` is the clip's crop
    window in native source pixels ``(x, y, w, h)``; ``None`` = the
    renderer's no-path fallback (centre crop to the output aspect).
    ``words`` are clip-relative, exactly as the render receives them.
    ``face_rects`` (output pixels) feed the same face-safe placement
    the render uses.

    Timestamp handling: the encode seeks fast (input ``-ss``), then a
    ``trim`` + ``setpts`` moves the single frame to its clip-relative
    time before the ``ass`` filter — otherwise libass would evaluate
    every frame as t=0 and always show the first caption line.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found (no bundled binary, none on PATH)")
    if out_path is None:
        handle, name = tempfile.mkstemp(prefix="clipforge-preview-", suffix=".png")
        os.close(handle)
        Path(name).unlink(missing_ok=True)
        out_path = Path(name)
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Font resolution mirrors render_clip: an installed custom font
    # wins, then the picked caption font; the ASS style carries the
    # family and libass reads the matching fontsdir.
    custom_family: str | None = None
    picked_family: str | None = None
    fontsdir: Path | None = None
    base_dir = Path(custom_fonts_dir) if custom_fonts_dir else None
    if custom_font:
        family, fdir, _error = resolve_custom_font(custom_font, base_dir=base_dir)
        if family is not None:
            custom_family, fontsdir = family, fdir
    elif caption_font:
        family, fdir, _error = resolve_custom_font(caption_font, base_dir=base_dir)
        if family is not None:
            picked_family = family
            if fdir is not None and fdir != fonts_dir():
                fontsdir = fdir

    with tempfile.TemporaryDirectory(prefix="clipforge-preview-") as tmp:
        ass_text = build_ass(
            words, language=language, preset_name=preset_name,
            width=width, height=height,
            custom_font=custom_family, font=picked_family,
            colors=colors, keyword_scale=keyword_scale,
            face_rects=face_rects, layout=layout, gradient=gradient,
        )
        ass_path = Path(tmp) / "captions.ass"
        ass_path.write_text(ass_text, encoding="utf-8")
        fonts = fontsdir if fontsdir is not None else fonts_dir()
        ass_filter = (
            f"ass={_escape_filter_arg(str(ass_path))}"
            f":fontsdir={_escape_filter_arg(str(fonts))}"
        )
        # Fast input seek to just before the frame, then trim to it
        # and stamp it with its clip-relative time for the ass filter.
        seek = max(0.0, at_time - 2.0)
        trim_start = at_time - seek
        target = max(0.0, at_time - clip_start)
        timeline = (
            f"trim=start={trim_start:.3f}:duration=0.5,"
            f"setpts=PTS-STARTPTS+{target:.3f}/TB,"
        )
        if crop_rect is not None:
            cx, cy, cw, ch = (int(round(v)) for v in crop_rect)
            graph = (
                f"{timeline}crop={cw}:{ch}:{cx}:{cy},"
                f"scale={width}:{height}:flags=lanczos,setsar=1,{ass_filter}"
            )
        else:
            graph = (
                f"{timeline}scale={width}:{height}:"
                f"force_original_aspect_ratio=increase,"
                f"crop={width}:{height},setsar=1,{ass_filter}"
            )
        run_ffmpeg(
            [
                "-ss", f"{seek:.3f}",
                "-i", video_path,
                "-vf", graph,
                "-frames:v", "1",
                "-y", str(out_path),
            ],
            timeout=120.0,
        )
    return out_path


def face_rects_at(
    crop_path: CropPath,
    *,
    clip_time: float,
    source_w: int,
    source_h: int,
    out_w: int,
    out_h: int,
) -> list[tuple[float, float, float, float]]:
    """Output-pixel face rects for the crop sample nearest ``clip_time``.

    Lets the preview still use the render's face-safe caption
    placement for the moment being previewed. Split paths return no
    rects: the still shows the first pane's crop, while the face
    mapping is defined on the full two-pane frame — the render itself
    still applies the full mapping.
    """
    if crop_path.mode == "split" or not crop_path.times:
        return []
    from .render import _output_face_rects  # noqa: PLC0415 - avoid a cycle

    index = min(
        range(len(crop_path.times)),
        key=lambda i: abs(crop_path.times[i] - clip_time),
    )
    single = CropPath(
        times=[crop_path.times[index]],
        boxes=[crop_path.boxes[index]],
        faces=[crop_path.faces[index]],
        mode="track",
    )
    return _output_face_rects(
        single, source_w=source_w, source_h=source_h,
        out_w=out_w, out_h=out_h,
    )
