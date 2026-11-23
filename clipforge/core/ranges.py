"""Manual clip ranges: data model + pure validation.

A range is one (start, end) pair in seconds. Validation rules (build
spec §1):
- end > start
- inside the video length (0 <= start, end <= duration)
- no overlaps between ranges
- warn when a range is longer than 60 s
- at most 20 ranges
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "MAX_RANGES",
    "WARN_SECONDS",
    "ClipRange",
    "RangeValidation",
    "validate_ranges",
]

MAX_RANGES = 20
WARN_SECONDS = 60.0


@dataclass
class ClipRange:
    """One clip range, in seconds (manual or AI-discovered)."""

    start: float
    end: float
    source: str = "manual"  # "manual" | "ai"
    label: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class RangeValidation:
    """Outcome of :func:`validate_ranges`."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _label(rng: ClipRange, index: int) -> str:
    return rng.label or f"Clip {index + 1}"


def validate_ranges(
    ranges: list[ClipRange],
    duration: float,
    *,
    max_ranges: int = MAX_RANGES,
    warn_seconds: float = WARN_SECONDS,
) -> RangeValidation:
    """Validate a list of clip ranges against the video length."""
    result = RangeValidation()
    if len(ranges) > max_ranges:
        result.errors.append(
            f"Too many ranges: {len(ranges)} (max {max_ranges})."
        )
    for index, rng in enumerate(ranges):
        name = _label(rng, index)
        if rng.start < 0:
            result.errors.append(f"{name}: start is before 0.")
        if rng.end > duration:
            result.errors.append(
                f"{name}: end {rng.end:.2f}s is past the video length "
                f"({duration:.2f}s)."
            )
        if rng.end <= rng.start:
            result.errors.append(
                f"{name}: end must be after start "
                f"({rng.start:.2f}s – {rng.end:.2f}s)."
            )
        elif rng.duration > warn_seconds:
            result.warnings.append(
                f"{name}: {rng.duration:.0f}s is longer than "
                f"{warn_seconds:.0f}s — long for a viral clip."
            )
    ordered = sorted(enumerate(ranges), key=lambda item: item[1].start)
    for (i1, r1), (i2, r2) in zip(ordered, ordered[1:]):
        if r2.start < r1.end:
            result.errors.append(
                f"{_label(r1, i1)} overlaps {_label(r2, i2)} "
                f"({r1.start:.2f}s–{r1.end:.2f}s vs "
                f"{r2.start:.2f}s–{r2.end:.2f}s)."
            )
    return result
