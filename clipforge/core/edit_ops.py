"""Basic editing operations for the Edit step (pure functions).

One :class:`EditableClip` holds everything the user can change about a
clip: its range (trim/split), its caption words (edit/delete), custom
text overlays, captions on/off, and volume. Words are stored in
**source-video time**; the render re-times them to clip-relative time,
so trimming or splitting a clip never needs to re-time words by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .caption_edit import delete_line, delete_word, edit_line_text, line_texts
from .captions import CaptionOverlay
from .ranges import ClipRange, validate_ranges
from .transcription import Word

__all__ = [
    "EditableClip",
    "add_overlay",
    "clip_words_relative",
    "delete_clip",
    "delete_clip_line",
    "delete_clip_word",
    "edit_clip_line",
    "move_clip",
    "split_clip",
    "to_range",
    "trim_clip",
]


@dataclass
class EditableClip:
    """One clip in the Edit step, with all user edits applied."""

    start: float
    end: float
    source: str = "manual"  # "manual" | "ai"
    label: str = ""
    words: list[Word] | None = None  # caption words, SOURCE-video time
    captions_enabled: bool = True
    overlays: list[CaptionOverlay] = field(default_factory=list)  # source time
    volume: float = 1.0  # 0.0 = muted, 1.0 = original level
    kept: bool = True

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_range(self) -> ClipRange:
        return ClipRange(
            start=self.start, end=self.end, source=self.source, label=self.label
        )


def to_range(clip: EditableClip) -> ClipRange:
    return clip.to_range()


def _clip_words(clip: EditableClip) -> list[Word]:
    """Words inside the clip, clip-relative (for the caption editor)."""
    if not clip.words:
        return []
    return [
        Word(w.text, w.start - clip.start, w.end - clip.start)
        for w in clip.words
        if clip.start <= w.start < clip.end
    ]


def clip_words_relative(clip: EditableClip) -> list[Word]:
    """Clip-relative caption words (public; used by the render wiring)."""
    return _clip_words(clip)


def _restore_words(clip: EditableClip, rel_words: list[Word]) -> list[Word]:
    """Clip-relative words back to source time, merged with the rest."""
    kept = [
        w for w in (clip.words or [])
        if not (clip.start <= w.start < clip.end)
    ]
    restored = [
        Word(w.text, w.start + clip.start, w.end + clip.start)
        for w in rel_words
    ]
    return sorted(kept + restored, key=lambda w: w.start)


def trim_clip(
    clip: EditableClip,
    new_start: float,
    new_end: float,
    *,
    video_duration: float,
    siblings: list[EditableClip],
) -> EditableClip:
    """Trim a clip; same validation as the manual timeline.

    Raises :class:`ValueError` with a friendly message when the new
    range is invalid (end <= start, outside the video, overlaps a
    sibling). Words/overlays outside the new range are dropped.
    """
    trimmed = EditableClip(
        start=new_start, end=new_end, source=clip.source, label=clip.label,
        words=clip.words, captions_enabled=clip.captions_enabled,
        overlays=list(clip.overlays), volume=clip.volume, kept=clip.kept,
    )
    ranges = [c.to_range() for c in siblings if c is not clip]
    ranges.append(trimmed.to_range())
    validation = validate_ranges(ranges, video_duration)
    if not validation.is_valid:
        raise ValueError("; ".join(validation.errors))
    if trimmed.words is not None:
        trimmed.words = [
            w for w in trimmed.words if new_start <= w.start < new_end
        ]
    trimmed.overlays = [
        o for o in trimmed.overlays
        if new_start <= o.start < new_end
    ]
    return trimmed


def split_clip(clip: EditableClip, at: float) -> tuple[EditableClip, EditableClip]:
    """Split a clip into two at ``at`` (source-video seconds).

    Words/overlays go to the half containing their start; volume and
    the captions toggle are copied to both halves.
    """
    if not clip.start < at < clip.end:
        raise ValueError(
            f"Split time {at:.2f}s must be inside the clip "
            f"({clip.start:.2f}s – {clip.end:.2f}s)."
        )
    first = EditableClip(
        start=clip.start, end=at, source=clip.source, label=clip.label,
        words=[w for w in (clip.words or []) if w.start < at],
        captions_enabled=clip.captions_enabled,
        overlays=[o for o in clip.overlays if o.start < at],
        volume=clip.volume, kept=clip.kept,
    )
    second = EditableClip(
        start=at, end=clip.end, source=clip.source, label=clip.label,
        words=[w for w in (clip.words or []) if w.start >= at],
        captions_enabled=clip.captions_enabled,
        overlays=[o for o in clip.overlays if o.start >= at],
        volume=clip.volume, kept=clip.kept,
    )
    return first, second


def move_clip(
    clips: list[EditableClip], from_index: int, to_index: int
) -> list[EditableClip]:
    """Move a clip up/down in render order; returns a new list."""
    if not 0 <= from_index < len(clips) or not 0 <= to_index < len(clips):
        raise IndexError("clip index out of range")
    out = list(clips)
    item = out.pop(from_index)
    out.insert(to_index, item)
    return out


def delete_clip(
    clips: list[EditableClip], index: int
) -> list[EditableClip]:
    """Remove a clip from the list entirely (vs un-keeping it)."""
    if not 0 <= index < len(clips):
        raise IndexError("clip index out of range")
    return [c for i, c in enumerate(clips) if i != index]


# -- caption edits (via the caption editor, source-time aware) -----------

def clip_line_texts(clip: EditableClip) -> list[str]:
    """Caption line texts for the editor, clip-relative grouping."""
    return line_texts(_clip_words(clip))


def edit_clip_line(
    clip: EditableClip, line_index: int, new_text: str
) -> EditableClip:
    """Replace one caption line's text (empty text deletes the line)."""
    rel = _clip_words(clip)
    edited = edit_line_text(rel, line_index, new_text)
    return EditableClip(
        start=clip.start, end=clip.end, source=clip.source, label=clip.label,
        words=_restore_words(clip, edited),
        captions_enabled=clip.captions_enabled,
        overlays=list(clip.overlays), volume=clip.volume, kept=clip.kept,
    )


def delete_clip_word(clip: EditableClip, word_index: int) -> EditableClip:
    """Delete one caption word (flat index across the clip's lines)."""
    rel = _clip_words(clip)
    edited = delete_word(rel, word_index)
    return EditableClip(
        start=clip.start, end=clip.end, source=clip.source, label=clip.label,
        words=_restore_words(clip, edited),
        captions_enabled=clip.captions_enabled,
        overlays=list(clip.overlays), volume=clip.volume, kept=clip.kept,
    )


def delete_clip_line(clip: EditableClip, line_index: int) -> EditableClip:
    """Delete a whole caption line."""
    rel = _clip_words(clip)
    edited = delete_line(rel, line_index)
    return EditableClip(
        start=clip.start, end=clip.end, source=clip.source, label=clip.label,
        words=_restore_words(clip, edited),
        captions_enabled=clip.captions_enabled,
        overlays=list(clip.overlays), volume=clip.volume, kept=clip.kept,
    )


def add_overlay(
    clip: EditableClip, text: str, start: float, end: float
) -> EditableClip:
    """Add a custom text line at a chosen time (source-video seconds)."""
    text = text.strip()
    if not text:
        raise ValueError("Custom text cannot be empty.")
    if end <= start:
        raise ValueError("Custom text end must be after its start.")
    if start < clip.start or end > clip.end:
        raise ValueError(
            f"Custom text ({start:.2f}s – {end:.2f}s) must be inside the clip "
            f"({clip.start:.2f}s – {clip.end:.2f}s)."
        )
    return EditableClip(
        start=clip.start, end=clip.end, source=clip.source, label=clip.label,
        words=clip.words, captions_enabled=clip.captions_enabled,
        overlays=[*clip.overlays, CaptionOverlay(text=text, start=start, end=end)],
        volume=clip.volume, kept=clip.kept,
    )
