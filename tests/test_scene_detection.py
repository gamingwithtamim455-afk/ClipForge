"""Tests for PySceneDetect scene-cut detection (with frame-diff fallback).

A synthetic two-scene clip (solid red for 2 s, then solid blue for 2 s)
has exactly one hard cut; PySceneDetect must find it, and the fallback
must still flag it when PySceneDetect is unavailable.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from clipforge.core.face_tracking import (
    _frame_diff_scene_cuts,
    detect_scene_cuts,
    scene_cut_flags,
)
from clipforge.core.ffmpeg import find_ffmpeg


def _make_two_scene_clip(tmp_path) -> str:
    """4 s clip: 2 s solid red, then 2 s solid blue (one hard cut at 2 s)."""
    path = str(tmp_path / "two_scene.mp4")
    subprocess.run(
        [
            find_ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=red:size=640x360:rate=30:duration=2",
            "-f", "lavfi", "-i", "color=c=blue:size=640x360:rate=30:duration=2",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
        ],
        check=True,
    )
    return path


def test_detect_scene_cuts_two_scene_clip(tmp_path) -> None:
    path = _make_two_scene_clip(tmp_path)
    cuts = detect_scene_cuts(path)
    assert len(cuts) == 1, f"expected exactly one cut, got {cuts}"
    assert cuts[0] == pytest.approx(2.0, abs=0.5)


def test_detect_scene_cuts_respects_range(tmp_path) -> None:
    path = _make_two_scene_clip(tmp_path)
    # The only cut is at 2 s; a window ending before it has no cuts.
    assert detect_scene_cuts(path, start=0.0, end=1.0) == []
    assert len(detect_scene_cuts(path, start=1.5, end=4.0)) == 1


def test_detect_scene_cuts_missing_video_returns_empty() -> None:
    assert detect_scene_cuts("/nonexistent/nope.mp4") == []


def test_detect_scene_cuts_without_pyscenedetect(monkeypatch) -> None:
    """PySceneDetect unavailable -> [] (caller falls back to frame-diff)."""
    monkeypatch.setitem(sys.modules, "scenedetect", None)
    monkeypatch.setitem(sys.modules, "scenedetect.detectors", None)
    # detect_scene_cuts imports scenedetect inside the function; a None
    # entry makes that import raise, which is caught -> [].
    assert detect_scene_cuts("whatever.mp4") == []


def test_scene_cut_flags_mapping() -> None:
    times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # Cut at 0.5 s lands in the 0.6 s sample (first frame after the cut).
    assert scene_cut_flags(times, [0.5]) == [False, False, False, True, False, False]
    # Cut exactly on a sample time flags that sample.
    assert scene_cut_flags(times, [0.4]) == [False, False, True, False, False, False]
    # No cuts -> no flags; multiple cuts map independently.
    assert scene_cut_flags(times, []) == [False] * 6
    assert scene_cut_flags(times, [0.1, 0.9]) == [
        False, True, False, False, False, True,
    ]


def test_frame_diff_fallback_detects_two_scene_cut(tmp_path) -> None:
    """The frame-diff fallback flags the same hard cut (red -> blue)."""
    import cv2

    path = _make_two_scene_clip(tmp_path)
    cap = cv2.VideoCapture(path)
    frames = []
    for i in range(20):  # 4 s at 5 fps: seek to each sample time
        cap.set(cv2.CAP_PROP_POS_MSEC, i * 200.0)
        ok, frame = cap.read()
        frames.append(frame if ok else None)
    cap.release()
    flags = _frame_diff_scene_cuts(frames)
    assert len(flags) == 20
    # The cut is at 2 s -> sample index 10 (5 fps) is the first blue frame.
    assert flags[10] is True
    # Solid-colour scenes have no internal cuts.
    assert sum(flags) == 1
