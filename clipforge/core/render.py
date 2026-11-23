"""Rendering: one encode pass per clip, burned captions, TikTok captions.

Output spec (build spec §6):
- 1080x1920 default; 1440x2560 and 2160x3840 selectable (upscaling is
  flagged when the source is smaller).
- H.264, CRF 14-16, preset slow, yuv420p, bt709, AAC 256k, faststart,
  perfect A/V sync.
- The tracked crop path is driven with ffmpeg ``sendcmd`` (one command
  per option per timestamp — the crop filter takes a single argument per
  command), so the crop follows the face in a single encode pass.
- Burned captions with word highlight in the TikTok safe zone, using the
  Fonts tab presets; FFmpeg libass (HarfBuzz) for correct Bengali shaping
  and Arabic/Urdu right-to-left rendering.
- After rendering, DeepSeek writes per clip a TikTok caption + EXACTLY 4
  hashtags in the clip's language, saved as ``01_label.txt`` next to the
  video. When the DeepSeek provider is unavailable the label is left
  unwritten and flagged (never faked).
- After rendering the duration is verified with ffprobe (within 1 frame).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field, replace
from math import gcd
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .captions import (
    CaptionColors,
    CaptionLayout,
    CaptionOverlay,
    GradientConfig,
    build_ass,
    choose_caption_placement,
    fonts_dir,
    layout_font_size,
    layout_margin_v,
    resolve_custom_font,
    write_ass_file,
)
from .face_tracking import CropPath, track_faces
from .ffmpeg import (
    FFmpegError,
    encoder_quality_args,
    find_ffmpeg,
    probe,
    run_ffmpeg,
    select_video_encoder,
)
from .transcription import Word

if TYPE_CHECKING:
    from ..providers.deepseek import DeepSeekClient

__all__ = [
    "ClipResult",
    "RenderError",
    "RenderJob",
    "RenderSettings",
    "attach_label_to_result",
    "cleanup_clipforge_temp",
    "generate_clip_label",
    "render_clip",
    "render_clips",
    "write_label_file",
]


class RenderError(RuntimeError):
    """Raised when a render fails or its verification fails."""


@dataclass
class RenderSettings:
    """Output settings for one clip render."""

    width: int = 1080
    height: int = 1920
    crf: int = 15
    preset: str = "slow"
    audio_bitrate: str = "256k"
    captions: str = "burn"  # "burn" | "none"
    quality_label: str = "1080p"
    encoder: str = "Auto"  # "Auto" | "GPU" | "CPU"
    caption_preset: str = "Hormozi"
    fps: float | None = None  # None = source fps
    volume: float = 1.0  # 0.0 = muted, 1.0 = original level
    custom_font: str = ""  # installed/bundled font file name ("" = auto)
    custom_fonts_dir: str = ""  # "" = user fonts folder
    caption_font: str = ""  # picked caption font file name ("" = preset default)
    keyword_scale: float = 0.0  # 0.0 = preset default, else keyword size factor
    caption_colors: CaptionColors | None = None  # None = preset colours
    caption_layout: CaptionLayout | None = None  # None = preset layout
    caption_gradient: GradientConfig | None = None  # None = classic 2-colour blend


@dataclass
class ClipResult:
    """One finished clip."""

    index: int
    video_path: str
    label_path: str  # 01_label.txt next to the video ("" if not written)
    duration: float
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    upscaled: bool = False
    encoder: str = ""
    label_written: bool = False
    flags: list[str] = field(default_factory=list)


@dataclass
class RenderJob:
    """One clip to render (used by the Render step's batch runner)."""

    video_path: str
    start: float
    end: float
    index: int = 1
    settings: RenderSettings = field(default_factory=RenderSettings)
    crop_path: CropPath | None = None
    words: list[Word] | None = None
    language: str = "en"
    output_path: str | None = None
    label: tuple[str, list[str]] | None = None
    overlays: list[CaptionOverlay] = field(default_factory=list)


# ----------------------------------------------------------------------
# Filter-graph helpers
# ----------------------------------------------------------------------

def _escape_filter_arg(value: str) -> str:
    """Quote a filter argument for a filtergraph (handles :, ', \\)."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _sendcmd_file(
    path: Path, *, crop_path: CropPath, crop_w: int, crop_h: int
) -> None:
    """Write a sendcmd file driving the crop filter along the crop path.

    One command per option per timestamp (the crop filter accepts a
    single argument per command). Timestamps are clip-relative because
    the input is sought to the clip start before decoding.
    """
    lines: list[str] = []
    for t, box in zip(crop_path.times, crop_path.boxes):
        ts = max(0.0, t)
        lines.append(f"{ts:.3f} crop x {int(round(box.x))};")
        lines.append(f"{ts:.3f} crop y {int(round(box.y))};")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _filter_graph(
    *,
    crop_path: CropPath | None,
    crop_w: int,
    crop_h: int,
    width: int,
    height: int,
    ass_path: str | None,
    fontsdir: Path | None = None,
) -> str:
    """Build the video filtergraph for one clip."""
    ass = ""
    if ass_path:
        fonts = fontsdir if fontsdir is not None else fonts_dir()
        ass = (
            f",ass={_escape_filter_arg(ass_path)}"
            f":fontsdir={_escape_filter_arg(str(fonts))}"
        )
    if crop_path is not None and crop_path.mode == "blur-fit":
        # No face: blurred cover background + fitted foreground.
        return (
            f"[0:v]split=2[bg_in][fg_in];"
            f"[bg_in]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},gblur=sigma=40[bg];"
            f"[fg_in]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2{ass},format=yuv420p[vout]"
        )
    if crop_path is not None:
        # Tracked crop: sendcmd drives x/y along the smoothed path.
        return (
            "[0:v]sendcmd=f={cmds},"
            f"crop=w={crop_w}:h={crop_h}:x=0:y=0,"
            f"scale={width}:{height}:flags=lanczos,setsar=1"
            f"{ass},format=yuv420p[vout]"
        )
    # No crop path at all: plain scale (center crop to aspect first).
    return (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1{ass},format=yuv420p[vout]"
    )


def _output_face_rects(
    crop_path: CropPath,
    *,
    source_w: int,
    source_h: int,
    out_w: int,
    out_h: int,
) -> list[tuple[float, float, float, float]]:
    """Tracked face boxes in output pixels (for face-safe captions).

    ``crop_path.faces`` are normalized in the source frame; ``boxes``
    are the crop rectangles in native pixels. A face outside the crop
    maps outside the output frame and is skipped.
    """
    rects: list[tuple[float, float, float, float]] = []
    for crop, face in zip(crop_path.boxes, crop_path.faces):
        if face is None or crop.width <= 0 or crop.height <= 0:
            continue
        fx = face.x * source_w
        fy = face.y * source_h
        fw = face.width * source_w
        fh = face.height * source_h
        sx = out_w / crop.width
        sy = out_h / crop.height
        x = (fx - crop.x) * sx
        y = (fy - crop.y) * sy
        if x + fw <= 0 or y + fh <= 0 or x >= out_w or y >= out_h:
            continue
        rects.append((x, y, fw * sx, fh * sy))
    return rects


# ----------------------------------------------------------------------
# TikTok caption + hashtags (DeepSeek)
# ----------------------------------------------------------------------

_HASHTAG_RE = re.compile(r"#([\w\u0980-\u09FF\u0600-\u06FF\u0900-\u097F]+)")

_LABEL_PROMPT = (
    "Write a short TikTok caption for this video clip, in {language_name}. "
    "The caption must be 1-2 sentences, punchy, in the clip's language, "
    "and must NOT contain any hashtags. After the caption, on a new line, "
    "write EXACTLY 4 relevant hashtags in the same language, separated by "
    "spaces. Nothing else.\n\n"
    "Transcript excerpt:\n{transcript}"
)

_LANGUAGE_NAMES = {
    "auto": "the clip's language",
    "bn": "Bengali",
    "en": "English",
    "ar": "Arabic",
    "ur": "Urdu",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "tr": "Turkish",
    "id": "Indonesian",
    "pt": "Portuguese",
    "ru": "Russian",
}


def generate_clip_label(
    client: "DeepSeekClient",
    *,
    transcript_text: str,
    language: str,
) -> tuple[str, list[str]] | None:
    """Ask DeepSeek for a TikTok caption + EXACTLY 4 hashtags.

    Returns ``(caption, hashtags)`` or ``None`` when the provider is
    unavailable, the call fails, or the model did not return exactly 4
    hashtags. Never raises for provider failures — the caller leaves the
    ``.txt`` unwritten and flags the clip instead.
    """
    if not transcript_text.strip():
        return None
    try:
        response = client.chat_completions(
            [
                {
                    "role": "user",
                    "content": _LABEL_PROMPT.format(
                        language_name=_LANGUAGE_NAMES.get(language, "English"),
                        transcript=transcript_text.strip()[:2000],
                    ),
                }
            ],
            thinking=False,
        )
        content = response["choices"][0]["message"]["content"]
    except Exception:
        return None
    hashtags = _HASHTAG_RE.findall(content)
    # De-duplicate, preserving order.
    seen: set[str] = set()
    unique = [h for h in hashtags if not (h in seen or seen.add(h))]
    if len(unique) != 4:
        return None
    caption = _HASHTAG_RE.sub("", content).strip()
    caption = re.sub(r"\n{3,}", "\n\n", caption).strip()
    if not caption:
        return None
    return caption, ["#" + h for h in unique]


def write_label_file(
    video_path: str | Path, index: int, caption: str, hashtags: list[str]
) -> Path:
    """Write ``NN_label.txt`` next to the rendered video."""
    video = Path(video_path)
    target = video.parent / f"{index:02d}_label.txt"
    target.write_text(
        caption.strip() + "\n\n" + " ".join(hashtags) + "\n",
        encoding="utf-8",
    )
    return target


def attach_label_to_result(
    result: ClipResult, label: tuple[str, list[str]]
) -> ClipResult:
    """Write ``NN_label.txt`` for a finished clip and fill in its fields.

    Shared by the pipeline render stage and the Render step so a label is
    written exactly one way everywhere.
    """
    caption, hashtags = label
    label_file = write_label_file(result.video_path, result.index, caption, hashtags)
    result.caption = caption
    result.hashtags = list(hashtags)
    result.label_path = str(label_file)
    result.label_written = True
    # The renderer flags "label pending" when no label was supplied at
    # render time; the label is attached now, so drop that stale flag.
    result.flags = [f for f in result.flags if "label pending" not in f]
    return result


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def _crop_size(
    source_w: int, source_h: int, aspect: tuple[int, int] = (9, 16)
) -> tuple[int, int]:
    """Native crop size for the target aspect (even, for yuv420p)."""
    aw, ah = aspect
    crop_w = int(source_h * aw / ah)
    crop_h = source_h
    if crop_w > source_w:
        crop_w = source_w
        crop_h = int(source_w * ah / aw)
    return crop_w - crop_w % 2, crop_h - crop_h % 2


def render_clip(
    video_path: str,
    *,
    start: float,
    end: float,
    settings: RenderSettings | None = None,
    index: int = 1,
    crop_path: CropPath | None = None,
    words: list[Word] | None = None,
    language: str = "en",
    output_path: str | None = None,
    label: tuple[str, list[str]] | None = None,
    overlays: list[CaptionOverlay] | None = None,
    on_progress: Callable[[int, str], None] | None = None,
) -> ClipResult:
    """Render one clip in a single FFmpeg encode pass.

    Frame-accurate cuts: input seeking (``-ss`` before ``-i``) + ``-t``
    with re-encoding starts the first output frame exactly at ``start``.
    After the encode, ffprobe verifies the duration is within 1 frame of
    ``end - start``; otherwise :class:`RenderError` is raised.
    """
    settings = settings or RenderSettings()
    if end <= start:
        raise RenderError("clip end must be after start")
    if find_ffmpeg() is None:
        raise RenderError("ffmpeg not found (no bundled binary, none on PATH)")
    try:
        info = probe(video_path)
    except Exception as exc:
        raise RenderError(f"cannot probe {video_path}: {exc}") from exc
    if info.duration <= 0:
        raise RenderError(f"cannot probe duration of {video_path}")
    if end > info.duration + 0.05:
        raise RenderError(
            f"clip end {end:.2f}s is past the video length ({info.duration:.2f}s)"
        )
    expected = end - start
    fps = settings.fps or info.fps or 30.0
    frame_duration = 1.0 / fps
    upscaled = settings.width > info.width or settings.height > info.height

    def _report(percent: int, message: str) -> None:
        if on_progress is not None:
            on_progress(percent, message)

    _report(5, "Preparing render…")
    _divisor = gcd(settings.width, settings.height)
    _aspect = (settings.width // _divisor, settings.height // _divisor)
    crop_w, crop_h = _crop_size(info.width, info.height, aspect=_aspect)
    if crop_path is None:
        try:
            crop_path = track_faces(
                video_path, start=start, end=end, aspect=_aspect,
            )
        except Exception:
            crop_path = None  # fall back to a plain centered crop
    # Keep the crop size consistent with the tracked path.
    if crop_path is not None and crop_path.boxes:
        crop_w = int(round(crop_path.boxes[0].width))
        crop_h = int(round(crop_path.boxes[0].height))

    out = Path(output_path) if output_path else Path(video_path).parent / (
        f"{Path(video_path).stem}_{index:02d}_clip.mp4"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    # Caption font + face-safe placement, resolved before the encode.
    custom_family: str | None = None
    picked_family: str | None = None
    custom_fontsdir: Path | None = None
    custom_font_warning = ""
    if settings.custom_font:
        family, fdir, error = resolve_custom_font(
            settings.custom_font,
            base_dir=Path(settings.custom_fonts_dir)
            if settings.custom_fonts_dir
            else None,
        )
        if family is not None and fdir is not None:
            custom_family, custom_fontsdir = family, fdir
        else:
            custom_font_warning = error or "custom font unavailable"
    elif settings.caption_font:
        # Picked caption font (bundled viral font or installed custom):
        # its family goes to the ASS style; an installed font also
        # supplies the fontsdir (bundled fonts were copied in there).
        family, fdir, error = resolve_custom_font(
            settings.caption_font,
            base_dir=Path(settings.custom_fonts_dir)
            if settings.custom_fonts_dir
            else None,
        )
        if family is not None:
            picked_family = family
            if fdir is not None and fdir != fonts_dir():
                custom_fontsdir = fdir
        else:
            custom_font_warning = error or "caption font unavailable"
    face_rects: list[tuple[float, float, float, float]] = []
    captions_moved = False
    if crop_path is not None and crop_path.faces:
        from .captions import (  # noqa: PLC0415 - local import, same package
            PRESETS,
            CaptionPlacement,
            caption_zone_clear,
        )

        face_rects = _output_face_rects(
            crop_path,
            source_w=info.width,
            source_h=info.height,
            out_w=settings.width,
            out_h=settings.height,
        )
        if face_rects:
            preset = PRESETS.get(settings.caption_preset, PRESETS["Hormozi"])
            layout = settings.caption_layout or CaptionLayout()
            font_size = layout_font_size(preset, settings.height, layout)
            margin_v = layout_margin_v(settings.height, layout)
            bottom = CaptionPlacement(
                alignment=2, margin_v=margin_v, font_size=font_size
            )
            captions_moved = not caption_zone_clear(
                face_rects, bottom, width=settings.width, height=settings.height
            )

    # Encode to a temp file first: a cancelled, crashed or failed render
    # must never leave a partial file at the finished-clip path. Only
    # after ffprobe verifies the temp file is it moved into place.
    tmp_out = out.with_suffix(".part" + out.suffix)
    tmp_out.unlink(missing_ok=True)  # stale partial from a killed run
    try:
        with tempfile.TemporaryDirectory(prefix="clipforge-render-") as tmp:
            tmp_path = Path(tmp)
            ass_path: str | None = None
            if settings.captions == "burn" and words:
                ass_text = build_ass(
                    words, language=language, preset_name=settings.caption_preset,
                    width=settings.width, height=settings.height,
                    custom_font=custom_family, font=picked_family,
                    colors=settings.caption_colors,
                    keyword_scale=settings.keyword_scale,
                    face_rects=face_rects or None, overlays=overlays,
                    layout=settings.caption_layout,
                    gradient=settings.caption_gradient,
                )
                ass_path = str(write_ass_file(ass_text, tmp_path / "captions.ass"))
            graph = _filter_graph(
                crop_path=crop_path, crop_w=crop_w, crop_h=crop_h,
                width=settings.width, height=settings.height, ass_path=ass_path,
                fontsdir=custom_fontsdir,
            )
            if crop_path is not None and crop_path.mode == "track":
                _sendcmd_file(tmp_path / "crop.cmds",
                              crop_path=crop_path, crop_w=crop_w, crop_h=crop_h)
                graph = graph.replace(
                    "{cmds}", _escape_filter_arg(str(tmp_path / "crop.cmds"))
                )

            encoder, is_gpu = select_video_encoder(settings.encoder)
            args: list[str] = [
                "-ss", f"{start:.3f}",
                "-i", video_path,
                "-t", f"{expected:.3f}",
                "-filter_complex", graph,
                "-map", "[vout]",
                "-map", "0:a?",
                "-c:v", encoder,
                *encoder_quality_args(encoder, settings.crf),
            ]
            if encoder == "libx264":
                args += ["-preset", settings.preset]
            if info.has_audio and settings.volume != 1.0:
                args += ["-af", f"volume={settings.volume:g}"]
            args += [
                "-r", f"{fps:g}",
                "-pix_fmt", "yuv420p",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-colorspace", "bt709",
                "-c:a", "aac",
                "-b:a", settings.audio_bitrate,
                "-movflags", "+faststart",
                str(tmp_out),
            ]
            _report(20, f"Encoding with {encoder}…")
            try:
                run_ffmpeg(args, timeout=max(600.0, expected * 30.0))
            except FFmpegError as exc:
                raise RenderError(f"encode failed: {exc}") from exc
            _report(85, "Verifying output…")

            # Verify with ffprobe: duration within 1 frame, exact resolution.
            try:
                result_probe = probe(str(tmp_out))
            except Exception as exc:
                raise RenderError(
                    f"cannot probe rendered clip {tmp_out}: {exc}"
                ) from exc
            if abs(result_probe.duration - expected) > frame_duration + 0.02:
                raise RenderError(
                    f"rendered duration {result_probe.duration:.3f}s differs from "
                    f"expected {expected:.3f}s by more than 1 frame "
                    f"({frame_duration:.3f}s)"
                )
            if (result_probe.width, result_probe.height) != (settings.width, settings.height):
                raise RenderError(
                    f"rendered size {result_probe.width}x{result_probe.height} != "
                    f"{settings.width}x{settings.height}"
                )
            os.replace(tmp_out, out)
    except Exception:
        tmp_out.unlink(missing_ok=True)
        raise

    flags: list[str] = []
    if upscaled:
        flags.append(
            f"upscaled: source is {info.width}x{info.height}, output is "
            f"{settings.width}x{settings.height}"
        )
    if crop_path is not None and crop_path.mode == "blur-fit":
        flags.append("no face found — used blur-fit background")
    if crop_path is not None and crop_path.needs_review:
        flags.append("face tracking QA failed — review framing in the Edit step")
    if settings.captions == "burn" and not words:
        flags.append("captions requested but no transcript words were available")
    if custom_font_warning:
        flags.append(custom_font_warning)
    if captions_moved:
        flags.append("captions moved to the top to keep the face clear")
    if settings.volume == 0.0:
        flags.append("muted")

    label_path = ""
    label_written = False
    caption = ""
    hashtags: list[str] = []
    if label is not None:
        caption, hashtags = label
        label_file = write_label_file(out, index, caption, hashtags)
        label_path = str(label_file)
        label_written = True
    elif settings.captions == "burn":
        flags.append("label pending: DeepSeek caption not generated yet")

    _report(100, "Done")
    return ClipResult(
        index=index,
        video_path=str(out),
        label_path=label_path,
        duration=result_probe.duration,
        caption=caption,
        hashtags=hashtags,
        upscaled=upscaled,
        encoder=encoder,
        label_written=label_written,
        flags=flags,
    )


def render_clips(
    jobs: list[RenderJob],
    *,
    on_progress: Callable[[int, str], None] | None = None,
) -> list[ClipResult]:
    """Render a batch of clips (used by the Render step)."""
    results: list[ClipResult] = []
    total = max(len(jobs), 1)
    for i, job in enumerate(jobs):
        def _job_progress(percent: int, message: str, _i=i) -> None:
            if on_progress is not None:
                overall = int((_i + percent / 100.0) / total * 100)
                on_progress(overall, f"Clip {job.index}: {message}")

        results.append(render_clip(
            job.video_path, start=job.start, end=job.end,
            settings=job.settings, index=job.index,
            crop_path=job.crop_path, words=job.words,
            language=job.language, output_path=job.output_path,
            label=job.label, overlays=job.overlays, on_progress=_job_progress,
        ))
    return results


def cleanup_clipforge_temp(max_age_days: float = 7.0) -> int:
    """Delete stale ``clipforge-*`` temp dirs; returns the count removed."""
    temp_root = Path(tempfile.gettempdir())
    now = time.time()
    removed = 0
    for entry in temp_root.iterdir():
        if not entry.name.startswith("clipforge-"):
            continue
        try:
            age_days = (now - entry.stat().st_mtime) / 86400.0
            if age_days > max_age_days:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed
