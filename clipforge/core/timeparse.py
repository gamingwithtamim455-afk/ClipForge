"""Timecode parsing and formatting for manual clip ranges.

Accepted input forms (all case/whitespace tolerant):
- plain seconds: ``"90"``, ``"90.5"``
- minutes:seconds: ``"1:30"``, ``"01:30"``, ``"1:30.25"``
- hours:minutes:seconds: ``"1:02:03"``, ``"1:02:03.5"``

Seconds/minutes in colon form must be < 60; values must be >= 0.
"""

from __future__ import annotations

import re

__all__ = ["TimecodeError", "parse_timecode", "format_timecode"]


class TimecodeError(ValueError):
    """Raised when a timecode string cannot be parsed."""


_INT_RE = re.compile(r"^\d+$")
_SECONDS_RE = re.compile(r"^\d+(?:\.\d+)?$")


def _parse_int(part: str, text: str) -> int:
    if not _INT_RE.match(part):
        raise TimecodeError(f"invalid timecode: {text!r}")
    return int(part)


def parse_timecode(text: str) -> float:
    """Parse a timecode string into seconds.

    >>> parse_timecode("90")
    90.0
    >>> parse_timecode("1:30")
    90.0
    >>> parse_timecode("01:30")
    90.0
    >>> parse_timecode("1:02:03")
    3723.0
    >>> parse_timecode("90.5")
    90.5
    >>> parse_timecode("1:30.25")
    90.25
    """
    if text is None:
        raise TimecodeError("empty timecode")
    raw = str(text).strip()
    if not raw:
        raise TimecodeError("empty timecode")
    parts = raw.split(":")
    if len(parts) > 3:
        raise TimecodeError(f"invalid timecode: {text!r}")
    if not _SECONDS_RE.match(parts[-1]):
        raise TimecodeError(f"invalid timecode: {text!r}")
    seconds = float(parts[-1])
    if len(parts) == 1:
        return seconds
    minutes = _parse_int(parts[-2], text)
    if seconds >= 60:
        raise TimecodeError(f"seconds must be < 60 in {text!r}")
    if len(parts) == 2:
        return minutes * 60 + seconds
    hours = _parse_int(parts[-3], text)
    if minutes >= 60:
        raise TimecodeError(f"minutes must be < 60 in {text!r}")
    return hours * 3600 + minutes * 60 + seconds


def format_timecode(seconds: float, *, with_hours: bool = False) -> str:
    """Format seconds back to a timecode string.

    Integral values render without a fractional part; fractional values
    render with up to two decimals (trailing zeros stripped).

    >>> format_timecode(90.0)
    '01:30'
    >>> format_timecode(3723.0, with_hours=True)
    '01:02:03'
    >>> format_timecode(90.25)
    '01:30.25'
    """
    if seconds < 0:
        raise TimecodeError("cannot format a negative timecode")
    total = round(float(seconds), 2)
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total - hours * 3600 - minutes * 60
    if secs == int(secs):
        secs_s = f"{int(secs):02d}"
    else:
        secs_s = f"{secs:05.2f}".rstrip("0")
    if with_hours or hours:
        return f"{hours:02d}:{minutes:02d}:{secs_s}"
    return f"{minutes:02d}:{secs_s}"
