"""Tests: face-tracking math (smoothing, QA) + pipeline on a faceless clip."""

import subprocess

import numpy as np
import pytest

from clipforge.core.face_tracking import (
    FaceBox,
    face_inside_crop_fraction,
    smooth_path_1d,
    track_faces,
)
from clipforge.core.ffmpeg import find_ffmpeg


# -- smoothing ---------------------------------------------------------
def test_smooth_constant_path_unchanged() -> None:
    targets = [100.0] * 10
    times = [i * 0.2 for i in range(10)]
    out = smooth_path_1d(targets, times)
    assert out == pytest.approx(targets)


def test_smooth_reduces_acceleration() -> None:
    # A jittery target: alternating jumps.
    targets = [100.0 if i % 2 == 0 else 130.0 for i in range(20)]
    times = [i * 0.2 for i in range(20)]
    raw_acc = float(np.sum(np.abs(np.diff(targets, n=2))))
    out = smooth_path_1d(targets, times, lam=8.0)
    smooth_acc = float(np.sum(np.abs(np.diff(out, n=2))))
    assert smooth_acc < raw_acc
    # Smoothed path stays near the target band.
    assert min(out) >= 95.0 and max(out) <= 135.0


def test_smooth_follows_step_with_lag() -> None:
    targets = [100.0] * 10 + [300.0] * 10
    times = [i * 0.2 for i in range(20)]
    out = smooth_path_1d(targets, times, lam=8.0)
    assert out[0] == pytest.approx(100.0, abs=5.0)
    assert out[-1] == pytest.approx(300.0, abs=5.0)
    # Nearly monotone: the second-difference penalty may dip slightly
    # below the step before rising (anticipation), but never far.
    assert min(out) >= 90.0 and max(out) <= 310.0


def test_smooth_velocity_clamp() -> None:
    targets = [0.0, 1000.0, 0.0, 1000.0]
    times = [0.0, 0.2, 0.4, 0.6]
    out = smooth_path_1d(targets, times, lam=0.0, max_speed=100.0)
    # Max step per 0.2 s sample is 20 px at 100 px/s.
    for a, b in zip(out, out[1:]):
        assert abs(b - a) <= 20.0 + 1e-6


def test_smooth_short_input() -> None:
    assert smooth_path_1d([], []) == []
    assert smooth_path_1d([5.0], [0.0]) == [5.0]
    assert smooth_path_1d([5.0, 7.0], [0.0, 0.2]) == [5.0, 7.0]


# -- QA ----------------------------------------------------------------
def test_qa_face_inside() -> None:
    faces = [FaceBox(0.4, 0.2, 0.2, 0.3)]
    crops = [FaceBox(0.3, 0.1, 0.4, 0.8)]
    assert face_inside_crop_fraction(faces, crops) == 1.0


def test_qa_face_outside() -> None:
    faces = [FaceBox(0.4, 0.2, 0.2, 0.3)]
    crops = [FaceBox(0.0, 0.1, 0.3, 0.8)]  # face sticks out to the right
    assert face_inside_crop_fraction(faces, crops) == 0.0


def test_qa_skips_missing_faces() -> None:
    faces: list[FaceBox | None] = [None, FaceBox(0.4, 0.2, 0.2, 0.3)]
    crops = [FaceBox(0.0, 0.0, 0.5, 0.5), FaceBox(0.3, 0.1, 0.4, 0.8)]
    assert face_inside_crop_fraction(faces, crops) == 1.0
    assert face_inside_crop_fraction([None, None], crops) == 1.0


# -- pipeline on a faceless synthetic clip ----------------------------
def _make_faceless_clip(tmp_path) -> str:
    path = str(tmp_path / "noface.mp4")
    subprocess.run(
        [
            find_ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", path,
        ],
        check=True,
    )
    return path


def test_track_faces_faceless_clip_falls_back_to_centred_fill(tmp_path) -> None:
    """No face anywhere -> centred FILL crop (v0.2.1 B).

    The old "blur-fit" fallback fitted the whole 16:9 picture inside
    the 9:16 canvas over a blurred background (letterboxed output);
    the fallback is now a plain centred crop path the renderer scales
    to fill the frame, flagged for review in the Edit step.
    """
    path = _make_faceless_clip(tmp_path)
    result = track_faces(path, start=0.0, end=3.0)
    assert result.mode == "track"
    assert result.fallback_fraction == 1.0
    assert result.qa_passed
    assert result.needs_review
    assert len(result.times) == 15  # 3 s at 5 fps
    # 9:16 crop of a 640x360 frame: width 202, height 360, centered.
    box = result.boxes[0]
    assert box.width == pytest.approx(202, abs=2)
    assert box.height == 360
    assert box.x == pytest.approx((640 - box.width) / 2, abs=2)
    assert box.y == 0


def test_track_faces_rejects_bad_range(tmp_path) -> None:
    path = _make_faceless_clip(tmp_path)
    with pytest.raises(ValueError):
        track_faces(path, start=2.0, end=2.0)


def test_track_faces_cancel_aborts_scan(tmp_path) -> None:
    """D: a cancel during the scan aborts it instead of finishing it."""
    path = _make_faceless_clip(tmp_path)
    with pytest.raises(RuntimeError, match="cancelled"):
        track_faces(path, start=0.0, end=3.0, cancel=lambda: True)


def test_track_faces_reports_progress(tmp_path) -> None:
    """E: the scan reports a rising fraction so the bar keeps moving."""
    path = _make_faceless_clip(tmp_path)
    seen: list[float] = []
    result = track_faces(
        path, start=0.0, end=3.0, progress=lambda frac: seen.append(frac)
    )
    assert result is not None
    assert seen, "no progress reported"
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)
