"""End-to-end pipeline: probe -> transcribe -> ranges -> face tracking -> render.

This module wires the pieces that already exist (transcription, discovery,
face tracking, rendering, labels) into one flow driven by a background
:class:`~clipforge.core.jobs.Job`. It does not change provider behaviour —
it only calls the adapters with the settings the user chose.

Stages (each reports progress through ``progress(percent, message)``):

1. Probe the source video (duration, resolution, fps, audio).
2. Transcribe it (Deepgram cloud, or local faster-whisper when the engine
   is set to local or Deepgram cannot serve the audio). With manual
   ranges, only those ranges are transcribed — a 2 h podcast with ten
   40 s ranges costs ~7 minutes of transcription, not 2 hours.
3. Clip ranges: manual timeline ranges are used exactly as entered and
   AI discovery is skipped; otherwise DeepSeek discovery + review propose
   the requested number of clips.
4. Face tracking per range (smoothed crop path + QA flag). A tracking
   failure for one range never kills the run — that clip renders with a
   plain centered crop instead and is flagged for review.
5. Build :class:`~clipforge.core.render.RenderJob` objects (rendering
   itself happens in the Render step via :func:`render_pipeline_jobs`).

Errors are raised as :class:`PipelineError` with a friendly, user-facing
message: no internet, a missing/invalid key (pointing at Settings), a bad
video file — never a raw traceback and never a leaked key.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from ..providers._http import ProviderError
from ..providers.deepseek import DeepSeekClient
from ..providers.deepgram import DeepgramClient
from ..settings import Settings
from .discovery import ClipCandidate, discover_clips, generate_caption, review_clips
from .captions import (
    CaptionColors,
    CaptionLayout,
    CaptionOverlay,
    GradientConfig,
)
from .face_tracking import CropPath, track_faces
from .ffmpeg import probe
from .keys import get_key
from .ranges import ClipRange, validate_ranges
from .render import (
    RenderJob,
    RenderSettings,
    attach_label_to_result,
    generate_clip_label,
    render_clip,
)
from .transcription import Transcript, Word, transcribe_ranges, transcribe_video

__all__ = [
    "PipelineConfig",
    "PipelineError",
    "PipelineResult",
    "build_render_jobs",
    "deepseek_client_from_settings",
    "friendly_error",
    "render_pipeline_jobs",
    "run_pipeline",
]

#: Quality label -> (width, height) output resolution.
QUALITY_SIZES: dict[str, tuple[int, int]] = {
    "1080p": (1080, 1920),
    "1440p": (1440, 2560),
    "4K": (2160, 3840),
}

#: Clips requested when the user leaves Clips on "Auto".
DEFAULT_AUTO_CLIPS = 5


class PipelineError(RuntimeError):
    """A pipeline failure with a friendly, user-facing message."""


def friendly_error(exc: BaseException) -> str:
    """Turn any failure into the message the status line / dialog shows."""
    if isinstance(exc, PipelineError):
        return str(exc)
    if isinstance(exc, ProviderError):
        if exc.kind == "no_internet":
            return (
                "No internet connection — check your network and try again. "
                "Local Whisper transcription still works offline "
                "(Settings > Transcription)."
            )
        if exc.kind == "invalid_key":
            return (
                f"{exc.message} — check the API key in Settings "
                "(Transcription for Deepgram, AI Provider for DeepSeek)."
            )
        if exc.kind == "no_balance":
            return f"{exc.message} — top up the provider account and try again."
        if exc.kind == "not_found":
            return (
                f"{exc.message} — check the base URL and model in "
                "Settings > AI Provider."
            )
        if exc.kind == "cancelled":
            return "Cancelled."
        if exc.kind == "no_audio":
            return exc.message
        return f"{exc.message} — try again shortly."
    if isinstance(exc, FileNotFoundError):
        return f"Video file not found: {exc.filename or exc}"
    return str(exc) or exc.__class__.__name__


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Everything one pipeline run needs (UI state, already resolved)."""

    video_path: str
    language: str = "auto"  # BCP-47 code, or "auto"
    engine: str = "deepgram"  # "deepgram" | "whisper_local"
    clip_length: float = 30.0
    clips_requested: int | None = None  # None = Auto -> DEFAULT_AUTO_CLIPS
    manual_ranges: list[ClipRange] = field(default_factory=list)
    captions: str = "burn"  # "burn" | "none" ("Cutout Only" = no burn-in)
    quality: str = "1080p"  # "1080p" | "1440p" | "4K"
    aspect: tuple[int, int] = (9, 16)
    accel: str = "Auto"  # "Auto" | "GPU" | "CPU"
    caption_preset: str = "Hormozi"
    custom_font: str = ""  # caption font file name ("" = auto per language)
    caption_font: str = ""  # picked caption font file name ("" = preset default)
    keyword_scale: float = 0.0  # 0.0 = preset default
    caption_colors: CaptionColors | None = None  # None = preset colours
    caption_layout: CaptionLayout | None = None  # None = preset layout
    caption_gradient: GradientConfig | None = None  # None = classic blend
    export_dir: str = ""
    project_name: str = ""
    discovery_template: str | None = None
    review_template: str | None = None
    # Optional overrides (used by tests to render small and fast).
    render_settings: RenderSettings | None = None

    @property
    def output_dir(self) -> Path:
        """Where this project's clips + labels are written."""
        base = Path(self.export_dir) if self.export_dir else Path.cwd()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.project_name or "ClipForge")
        return base / safe

    def render_settings_for(self) -> RenderSettings:
        """Output settings for this run (quality + captions + encoder)."""
        if self.render_settings is not None:
            return self.render_settings
        width, height = QUALITY_SIZES.get(self.quality, QUALITY_SIZES["1080p"])
        return RenderSettings(
            width=width,
            height=height,
            captions=self.captions,
            quality_label=self.quality,
            encoder=self.accel,
            caption_preset=self.caption_preset,
            custom_font=self.custom_font,
            caption_font=self.caption_font,
            keyword_scale=self.keyword_scale,
            caption_colors=self.caption_colors,
            caption_layout=self.caption_layout,
            caption_gradient=self.caption_gradient,
        )


@dataclass
class PipelineResult:
    """What a pipeline run produced (before rendering)."""

    video_path: str
    duration: float
    width: int
    height: int
    fps: float
    transcript: Transcript
    ranges: list[ClipRange]
    crop_paths: dict[int, CropPath | None] = field(default_factory=dict)
    qa_flags: dict[int, bool] = field(default_factory=dict)  # index -> needs review
    render_jobs: list[RenderJob] = field(default_factory=list)
    output_dir: Path = field(default_factory=Path)
    used_manual_ranges: bool = False


# ---------------------------------------------------------------------------
# Provider clients from stored settings/keys
# ---------------------------------------------------------------------------

def deepseek_client_from_settings(settings: Settings) -> DeepSeekClient | None:
    """DeepSeek client from Settings + keyring, or None when no key is stored."""
    key = get_key("deepseek")
    if not key:
        return None
    return DeepSeekClient(
        base_url=str(settings.get("ai_base_url")),
        api_key=key,
        model=str(settings.get("ai_model") or "deepseek-flash"),
    )


def deepgram_key() -> str | None:
    """Stored Deepgram key, or None."""
    return get_key("deepgram")


# ---------------------------------------------------------------------------
# Render-job construction (shared by the pipeline and the Edit step)
# ---------------------------------------------------------------------------

def build_render_jobs(
    *,
    video_path: str,
    ranges: list[ClipRange],
    words: list,
    crop_paths: dict[int, CropPath | None] | None = None,
    language: str = "en",
    settings: RenderSettings | None = None,
    output_dir: Path | None = None,
    label_source: str = "",
    words_by_index: dict[int, list[Word]] | None = None,
    overlays_by_index: dict[int, list[CaptionOverlay]] | None = None,
    volumes: dict[int, float] | None = None,
    captions_enabled: dict[int, bool] | None = None,
) -> list[RenderJob]:
    """Build one :class:`RenderJob` per range.

    ``words`` are the transcript words in source-video time; each job gets
    only the words inside its range, re-timed to the clip's own timeline
    (``start`` becomes 0) so the burned captions line up with the seeked
    output. ``crop_paths`` maps a range index to its tracked path (``None``
    = let the renderer track it, or use a plain centered crop). Output
    files land in ``output_dir`` as ``NN_clip.mp4`` (labels are written
    next to them as ``NN_label.txt``).

    Edit-step overrides (all keyed by range index): ``words_by_index``
    replaces the words with already clip-relative edited words;
    ``overlays_by_index`` adds custom text lines; ``volumes`` sets the
    per-clip volume (0.0 = muted); ``captions_enabled`` turns captions
    off for a single clip.
    """
    settings = settings or RenderSettings()
    paths = crop_paths or {}
    jobs: list[RenderJob] = []
    out_dir = Path(output_dir) if output_dir else Path(video_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, rng in enumerate(ranges):
        if words_by_index is not None and index in words_by_index:
            clip_words = list(words_by_index[index])
        else:
            # Clip-relative times: the renderer seeks to rng.start, so
            # caption words must be re-timed (start -> 0). Without this,
            # captions only appeared in the last (end - start) seconds
            # of the clip.
            clip_words = [
                Word(
                    text=w.text,
                    start=w.start - rng.start,
                    end=w.end - rng.start,
                    confidence=w.confidence,
                )
                for w in words
                if rng.start <= w.start < rng.end
            ]
        job_settings = settings
        if volumes is not None and index in volumes:
            job_settings = replace(job_settings, volume=volumes[index])
        if captions_enabled is not None and not captions_enabled.get(index, True):
            job_settings = replace(job_settings, captions="none")
        jobs.append(
            RenderJob(
                video_path=video_path,
                start=rng.start,
                end=rng.end,
                index=index + 1,
                settings=job_settings,
                crop_path=paths.get(index),
                words=clip_words,
                language=language,
                output_path=str(out_dir / f"{index + 1:02d}_clip.mp4"),
                overlays=list((overlays_by_index or {}).get(index, [])),
            )
        )
    return jobs


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, str], None]
CancelCheck = Callable[[], bool]


def _check_cancel(cancel: CancelCheck | None) -> None:
    if cancel is not None and cancel():
        raise PipelineError("Cancelled.")


def _format_audio_seconds(seconds: float) -> str:
    """Human length for a status line: 400 -> "6m 40s", 45 -> "45s"."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def run_pipeline(
    config: PipelineConfig,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
    deepgram_client: DeepgramClient | None = None,
    deepseek_client: DeepSeekClient | None = None,
    track_faces_fn: Callable[..., CropPath] | None = None,
) -> PipelineResult:
    """Run probe -> transcribe -> ranges -> face tracking; return the result.

    Rendering is intentionally not done here — the Render step runs
    :func:`render_pipeline_jobs` on ``result.render_jobs`` so the user can
    review the clips in the Edit step first.
    """
    def _report(percent: int, message: str) -> None:
        if progress is not None:
            progress(percent, message)

    if not config.video_path or not os.path.exists(config.video_path):
        raise PipelineError(f"Video file not found: {config.video_path}")

    # -- 1. Probe ---------------------------------------------------------
    _report(2, "Reading video…")
    try:
        info = probe(config.video_path)
    except Exception as exc:
        raise PipelineError(f"Could not read this video file: {exc}") from exc
    if info.duration <= 0:
        raise PipelineError("Could not read this video file (no duration).")
    _check_cancel(cancel)

    # -- 2. Transcribe ----------------------------------------------------
    # Manual ranges are validated FIRST (a bad range fails fast) and
    # transcription is then limited to exactly those ranges: a 2 h
    # podcast with ten 40 s ranges transcribes ~7 minutes of audio,
    # not 2 hours. Auto mode still transcribes the whole video —
    # discovery needs the full transcript.
    manual_validated: list[ClipRange] | None = None
    if config.manual_ranges:
        validation = validate_ranges(config.manual_ranges, info.duration)
        if not validation.is_valid:
            raise PipelineError(
                "Manual ranges are not valid:\n" + "\n".join(validation.errors)
            )
        manual_validated = [
            ClipRange(
                start=r.start, end=r.end, source="manual",
                label=r.label or f"Clip {i + 1}",
            )
            for i, r in enumerate(config.manual_ranges)
        ]
    if config.engine == "deepgram" and deepgram_client is None:
        if not deepgram_key():
            raise PipelineError(
                "Deepgram API key is missing — add it in "
                "Settings > Transcription, or switch to Local Whisper (Offline)."
            )
    try:
        if manual_validated is not None:
            total_audio = sum(r.end - r.start for r in manual_validated)
            _report(
                5,
                f"Transcribing your {len(manual_validated)} range(s) "
                f"({_format_audio_seconds(total_audio)} of audio)…",
            )
            transcript = transcribe_ranges(
                config.video_path,
                [(r.start, r.end) for r in manual_validated],
                duration=info.duration,
                language=config.language,
                engine=config.engine,
                deepgram_key=deepgram_key() or "",
                deepgram_client=deepgram_client,
                progress=lambda p, m: _report(5 + int(p * 0.35), m),
                cancel=cancel,
            )
        else:
            _report(5, "Transcribing…")
            transcript = transcribe_video(
                config.video_path,
                language=config.language,
                engine=config.engine,
                deepgram_key=deepgram_key() or "",
                deepgram_client=deepgram_client,
                progress=lambda p, m: _report(5 + int(p * 0.35), m),
                cancel=cancel,
            )
    except ProviderError as exc:
        raise PipelineError(friendly_error(exc)) from exc
    except Exception as exc:
        raise PipelineError(friendly_error(exc)) from exc
    _check_cancel(cancel)
    if transcript.note:
        _report(40, transcript.note)

    # -- 3. Ranges: manual or AI discovery -------------------------------
    if manual_validated is not None:
        ranges = manual_validated
        used_manual = True
        _report(55, f"Using {len(ranges)} manual range(s) — AI discovery skipped.")
    else:
        if deepseek_client is None:
            raise PipelineError(
                "DeepSeek API key is missing — add it in Settings > AI Provider "
                "to find viral clips, or add manual ranges on the Timeline."
            )
        target = config.clips_requested or DEFAULT_AUTO_CLIPS
        _report(42, "Finding viral clips…")
        try:
            candidates = discover_clips(
                deepseek_client,
                transcript,
                clip_length=config.clip_length,
                target_clips=target,
                duration=info.duration,
                template=config.discovery_template,
            )
            if len(candidates) > target:
                candidates = review_clips(
                    deepseek_client,
                    candidates,
                    transcript,
                    target_count=target,
                    template=config.review_template,
                )
        except ProviderError as exc:
            raise PipelineError(friendly_error(exc)) from exc
        if not candidates:
            raise PipelineError(
                "No viral clips found in this video — try a different clip "
                "length, or add manual ranges on the Timeline."
            )
        ranges = [
            ClipRange(
                start=c.start, end=c.end, source="ai",
                label=c.title or f"Clip {i + 1}",
            )
            for i, c in enumerate(candidates)
        ]
        used_manual = False
        _report(55, f"Found {len(ranges)} clip(s).")
    _check_cancel(cancel)

    # -- 4. Face tracking per range ---------------------------------------
    tracker = track_faces_fn or track_faces
    crop_paths: dict[int, CropPath | None] = {}
    qa_flags: dict[int, bool] = {}
    for index, rng in enumerate(ranges):
        _check_cancel(cancel)
        _report(
            55 + int(20 * index / max(len(ranges), 1)),
            f"Tracking faces in clip {index + 1} of {len(ranges)}…",
        )
        try:
            path = tracker(
                config.video_path, start=rng.start, end=rng.end,
                aspect=config.aspect,
            )
        except Exception:
            # One bad range must not kill the run: render it with a plain
            # centered crop and flag it for review in the Edit step.
            path = None
        crop_paths[index] = path
        qa_flags[index] = bool(path is not None and path.needs_review)
    _check_cancel(cancel)

    # -- 5. Render jobs ----------------------------------------------------
    settings = config.render_settings_for()
    jobs = build_render_jobs(
        video_path=config.video_path,
        ranges=ranges,
        words=transcript.words,
        crop_paths=crop_paths,
        language=config.language,
        settings=settings,
        output_dir=config.output_dir,
    )
    _report(80, f"{len(jobs)} clip(s) ready — review them in the Edit step.")

    return PipelineResult(
        video_path=config.video_path,
        duration=info.duration,
        width=info.width,
        height=info.height,
        fps=info.fps,
        transcript=transcript,
        ranges=ranges,
        crop_paths=crop_paths,
        qa_flags=qa_flags,
        render_jobs=jobs,
        output_dir=config.output_dir,
        used_manual_ranges=used_manual,
    )


# ---------------------------------------------------------------------------
# Render stage (used by the Render step)
# ---------------------------------------------------------------------------

def render_pipeline_jobs(
    jobs: list[RenderJob],
    *,
    label_client: DeepSeekClient | None = None,
    language: str = "en",
    transcript: Transcript | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
) -> list:
    """Render each job, then attach a TikTok caption + 4 hashtags per clip.

    Labels use :func:`core.render.generate_clip_label` (DeepSeek, exactly 4
    hashtags); when the model does not return exactly 4, the discovery
    caption generator is used as a fallback (it truncates to 4 and never
    invents hashtags). A clip whose label cannot be generated is flagged
    "label pending" — no ``.txt`` is ever faked.
    """
    from .render import ClipResult  # noqa: PLC0415 - avoid a cycle at import

    results: list[ClipResult] = []
    total = max(len(jobs), 1)
    for i, job in enumerate(jobs):
        if cancel is not None and cancel():
            break
        base = int(100 * i / total)

        def _job_progress(percent: int, message: str, _base=base) -> None:
            if progress is not None:
                progress(_base + int(percent / total), f"Clip {job.index}: {message}")

        result = render_clip(
            job.video_path, start=job.start, end=job.end,
            settings=job.settings, index=job.index,
            crop_path=job.crop_path, words=job.words,
            language=job.language, output_path=job.output_path,
            label=job.label, overlays=job.overlays, on_progress=_job_progress,
        )
        results.append(result)
        if result.label_written or label_client is None:
            continue
        if cancel is not None and cancel():
            break
        label = _generate_label(label_client, job, language, transcript)
        if label is not None:
            attach_label_to_result(result, label)
        else:
            result.flags.append("label pending: DeepSeek caption not generated")
    if progress is not None:
        progress(100, f"Done — {len(results)} clip(s) rendered.")
    return results


def _generate_label(
    client: DeepSeekClient,
    job: RenderJob,
    language: str,
    transcript: Transcript | None,
) -> tuple[str, list[str]] | None:
    """Caption + exactly 4 hashtags for one rendered clip."""
    words = job.words or []
    if transcript is not None:
        words = [w for w in transcript.words if job.start <= w.start < job.end]
    transcript_text = " ".join(w.text for w in words)
    if not transcript_text.strip():
        return None
    label = generate_clip_label(
        client, transcript_text=transcript_text, language=language,
    )
    if label is not None:
        return label
    # Fallback: the discovery caption generator truncates to 4 hashtags
    # and never invents them; only use it when it produced all 4.
    candidate = ClipCandidate(start=job.start, end=job.end)
    transcript_for_caption = transcript or Transcript(
        language=language, words=list(words)
    )
    generated = generate_caption(
        client, candidate, transcript_for_caption, language=language
    )
    if len(generated.get("hashtags", [])) == 4 and generated.get("caption"):
        return generated["caption"], generated["hashtags"]
    return None
