"""Tests: timecode parsing/formatting and clip-range validation."""

import pytest

from clipforge.core.ranges import MAX_RANGES, ClipRange, validate_ranges
from clipforge.core.timeparse import TimecodeError, format_timecode, parse_timecode


# -- parsing -----------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("90", 90.0),
        ("0", 0.0),
        ("90.5", 90.5),
        ("0.25", 0.25),
        ("1:30", 90.0),
        ("01:30", 90.0),
        ("1:30.25", 90.25),
        ("01:30.5", 90.5),
        ("1:02:03", 3723.0),
        ("01:02:03", 3723.0),
        ("1:02:03.5", 3723.5),
        ("  1:30  ", 90.0),
        ("0:00", 0.0),
        ("0:00.5", 0.5),
        ("59:59", 3599.0),
        ("2:00:00", 7200.0),
    ],
)
def test_parse_timecode_valid(text: str, expected: float) -> None:
    assert parse_timecode(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    ["", "   ", "abc", "1:60", "1:30:60", "1:2:3:4", "-5", "1:-30", "1:30.",
     "1.30.5", ":30", "1:", "1:30:"],
)
def test_parse_timecode_invalid(text: str) -> None:
    with pytest.raises(TimecodeError):
        parse_timecode(text)


# -- formatting --------------------------------------------------------
@pytest.mark.parametrize(
    ("seconds", "kwargs", "expected"),
    [
        (90.0, {}, "01:30"),
        (0.0, {}, "00:00"),
        (3723.0, {"with_hours": True}, "01:02:03"),
        (3723.0, {}, "01:02:03"),  # hours shown when non-zero anyway
        (90.25, {}, "01:30.25"),
        (90.5, {}, "01:30.5"),
        (5.0, {}, "00:05"),
        (3599.0, {}, "59:59"),
    ],
)
def test_format_timecode(seconds: float, kwargs: dict, expected: str) -> None:
    assert format_timecode(seconds, **kwargs) == expected


def test_format_roundtrip() -> None:
    for value in (0.0, 5.5, 90.0, 90.25, 3723.5, 7200.0):
        assert parse_timecode(format_timecode(value)) == pytest.approx(value)


def test_format_negative_raises() -> None:
    with pytest.raises(TimecodeError):
        format_timecode(-1.0)


# -- range validation --------------------------------------------------
def test_validate_ranges_ok() -> None:
    ranges = [ClipRange(30.0, 70.0), ClipRange(100.0, 140.0)]
    result = validate_ranges(ranges, duration=300.0)
    assert result.is_valid
    assert result.warnings == []


def test_validate_ranges_end_before_start() -> None:
    result = validate_ranges([ClipRange(70.0, 30.0)], duration=300.0)
    assert not result.is_valid
    assert "end must be after start" in result.errors[0]


def test_validate_ranges_zero_length() -> None:
    result = validate_ranges([ClipRange(30.0, 30.0)], duration=300.0)
    assert not result.is_valid


def test_validate_ranges_out_of_bounds() -> None:
    result = validate_ranges([ClipRange(290.0, 310.0)], duration=300.0)
    assert not result.is_valid
    assert any("past the video length" in e for e in result.errors)
    result2 = validate_ranges([ClipRange(-5.0, 10.0)], duration=300.0)
    assert not result2.is_valid


def test_validate_ranges_overlap() -> None:
    ranges = [ClipRange(30.0, 70.0), ClipRange(60.0, 100.0)]
    result = validate_ranges(ranges, duration=300.0)
    assert not result.is_valid
    assert any("overlaps" in e for e in result.errors)


def test_validate_ranges_touching_is_ok() -> None:
    # Adjacent ranges (end == next start) do not overlap.
    ranges = [ClipRange(30.0, 70.0), ClipRange(70.0, 100.0)]
    assert validate_ranges(ranges, duration=300.0).is_valid


def test_validate_ranges_long_warning() -> None:
    ranges = [ClipRange(0.0, 61.0)]
    result = validate_ranges(ranges, duration=300.0)
    assert result.is_valid  # warning only, not an error
    assert len(result.warnings) == 1
    assert "longer than 60s" in result.warnings[0]


def test_validate_ranges_max_count() -> None:
    ranges = [ClipRange(float(i * 10), float(i * 10 + 5)) for i in range(MAX_RANGES + 1)]
    result = validate_ranges(ranges, duration=1000.0)
    assert not result.is_valid
    assert any("Too many ranges" in e for e in result.errors)
    ok = [ClipRange(float(i * 10), float(i * 10 + 5)) for i in range(MAX_RANGES)]
    assert validate_ranges(ok, duration=1000.0).is_valid


def test_validate_ranges_unsorted_input_still_detects_overlap() -> None:
    ranges = [ClipRange(100.0, 140.0), ClipRange(30.0, 110.0)]
    result = validate_ranges(ranges, duration=300.0)
    assert not result.is_valid
    assert any("overlaps" in e for e in result.errors)
