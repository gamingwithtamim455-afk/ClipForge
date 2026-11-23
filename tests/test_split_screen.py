"""Tests: multi-person split screen (Settings > Experimental).

Selection logic is unit-tested on fabricated detections; ``track_faces``
mode selection is tested with the detector stubbed (MediaPipe itself is
exercised for real in test_face_tracking.py and the face E2E); rendering
is real ffmpeg, verified with ffprobe and by reading back output frames.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from clipforge.core import face_tracking
from clipforge.core.face_tracking import (
    CropPath,
    FaceBox,
    dominant_identity_pair,
    track_faces,
)
from clipforge.core.ffmpeg import find_ffmpeg, probe
from clipforge.core.render import RenderSettings, _output_face_rects, render_clip
from clipforge.settings import DEFAULTS


def _face(identity: int, x: float = 0.1) -> FaceBox:
    return FaceBox(x=x, y=0.30, width=0.25, height=0.30, identity=identity)


# -- pair selection (pure) ----------------------------------------------

def test_pair_two_shot_both_present_throughout() -> None:
    detected = [[_face(1, 0.08), _face(2, 0.62)] for _ in range(20)]
    assert dominant_identity_pair(detected) == (1, 2)


def test_pair_alternating_speakers() -> None:
    detected = [[_face(1)] for _ in range(10)] + [[_face(2)] for _ in range(10)]
    assert dominant_identity_pair(detected) == (1, 2)


def test_pair_single_person_returns_none() -> None:
    detected = [[_face(1)] for _ in range(20)]
    assert dominant_identity_pair(detected) is None


def test_pair_passer_by_returns_none() -> None:
    # Second face in only 3 of 20 face samples (15% < 30%): not a co-host.
    detected = [[_face(1)] for _ in range(17)]
    detected += [[_face(1, 0.08), _face(2, 0.62)] for _ in range(3)]
    assert dominant_identity_pair(detected) is None


def test_pair_too_few_samples_returns_none() -> None:
    detected = [[_face(1, 0.08), _face(2, 0.62)] for _ in range(4)]
    assert dominant_identity_pair(detected) is None
    assert dominant_identity_pair([]) is None
    assert dominant_identity_pair([[] for _ in range(10)]) is None


# -- track_faces mode selection (detector stubbed) ------------------------

def _make_clip(tmp_path: Path) -> str:
    path = str(tmp_path / "clip.mp4")
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


def _stub_detector(monkeypatch, *, two_faces: bool) -> None:
    def fake(frames, frame_w, frame_h, *, min_detection_confidence=0.5,
             scene_cuts=None, progress=None, cancel=None):
        detected = []
        for _ in frames:
            boxes = [_face(1, 0.08)]
            if two_faces:
                boxes.append(_face(2, 0.62))
            detected.append(boxes)
        flags = (
            list(scene_cuts)
            if scene_cuts is not None and len(scene_cuts) == len(frames)
            else [False] * len(frames)
        )
        return detected, flags

    monkeypatch.setattr(face_tracking, "_detect_faces_in_frames", fake)


def test_track_faces_two_faces_selects_split(tmp_path, monkeypatch) -> None:
    _stub_detector(monkeypatch, two_faces=True)
    result = track_faces(_make_clip(tmp_path), start=0.0, end=3.0)
    assert result.mode == "split"
    assert result.qa_passed
    assert len(result.boxes) == len(result.times) == 15  # 3 s at 5 fps
    assert len(result.boxes_b) == 15
    assert all(f is not None and f.identity == 2 for f in result.faces_b)
    # Pane crops have the output aspect at half height: 2*9 : 16.
    box = result.boxes[0]
    assert box.width / box.height == pytest.approx(18 / 16, abs=0.01)
    assert result.boxes_b[0].width == box.width


def test_track_faces_one_face_stays_tracked_crop(tmp_path, monkeypatch) -> None:
    _stub_detector(monkeypatch, two_faces=False)
    result = track_faces(_make_clip(tmp_path), start=0.0, end=3.0)
    assert result.mode == "track"
    assert result.boxes_b == []


def test_track_faces_split_disabled_by_setting(tmp_path, monkeypatch) -> None:
    _stub_detector(monkeypatch, two_faces=True)
    result = track_faces(
        _make_clip(tmp_path), start=0.0, end=3.0, split_screen=False
    )
    assert result.mode == "track"
    assert result.boxes_b == []


# -- rendering (real ffmpeg) ----------------------------------------------

def _make_two_tone_clip(tmp_path: Path) -> str:
    """640x360, left half red / right half blue, 4 s."""
    path = str(tmp_path / "twotone.mp4")
    subprocess.run(
        [
            find_ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=red:size=320x360:rate=30:duration=4",
            "-f", "lavfi", "-i", "color=blue:size=320x360:rate=30:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
            "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", path,
        ],
        check=True,
    )
    return path


def _split_path(*, moving_top: bool = False) -> CropPath:
    times = [i * 0.2 for i in range(15)]
    boxes = []
    for i in range(15):
        x = 236.0 * i / 14 if moving_top else 0.0
        boxes.append(FaceBox(x=x, y=0.0, width=404.0, height=360.0))
    boxes_b = [FaceBox(x=236.0, y=0.0, width=404.0, height=360.0) for _ in times]
    faces = [_face(1, 0.08) for _ in times]
    faces_b = [_face(2, 0.62) for _ in times]
    return CropPath(
        times=times, boxes=boxes, faces=faces, mode="split",
        boxes_b=boxes_b, faces_b=faces_b,
    )


def _frame_colours(video: str, tmp_path: Path, at: float) -> tuple[tuple, tuple]:
    """Mean (B, G, R) of the top and bottom halves of one output frame."""
    import cv2

    frame = str(tmp_path / f"frame_{at}.png")
    subprocess.run(
        [find_ffmpeg(), "-y", "-v", "error", "-ss", str(at), "-i", video,
         "-frames:v", "1", frame],
        check=True,
    )
    img = cv2.imread(frame)
    half = img.shape[0] // 2
    top = img[:half].reshape(-1, 3).mean(axis=0)
    bottom = img[half:].reshape(-1, 3).mean(axis=0)
    return tuple(top), tuple(bottom)


def test_render_split_output_exact_dimensions(tmp_path) -> None:
    video = _make_two_tone_clip(tmp_path)
    out = str(tmp_path / "split_out.mp4")
    result = render_clip(
        video, start=0.0, end=3.0,
        settings=RenderSettings(width=360, height=640, crf=18,
                                preset="veryfast", captions="none"),
        crop_path=_split_path(), output_path=out,
    )
    info = probe(result.video_path)
    assert (info.width, info.height) == (360, 640)
    assert abs(info.duration - 3.0) <= 1 / 30 + 0.02
    assert "two-person split screen" in result.flags
    # Top pane follows the left (red) crop, bottom the right (blue) one.
    (tb, tg, tr), (bb, bg, br) = _frame_colours(result.video_path, tmp_path, 1.0)
    assert tr > tb  # top pane red-dominant (BGR: red channel wins)
    assert bb > br  # bottom pane blue-dominant


def test_render_split_panes_move_independently(tmp_path) -> None:
    """The top pane's sendcmd path must not leak into the bottom pane."""
    video = _make_two_tone_clip(tmp_path)
    out = str(tmp_path / "split_move.mp4")
    result = render_clip(
        video, start=0.0, end=3.0,
        settings=RenderSettings(width=360, height=640, crf=18,
                                preset="veryfast", captions="none"),
        crop_path=_split_path(moving_top=True), output_path=out,
    )
    # Early: top crop is on the red half. Late: it has panned to blue.
    (_, _, early_r), _ = _frame_colours(result.video_path, tmp_path, 0.1)
    (late_b, _, late_r), (bot_b, _, bot_r) = _frame_colours(
        result.video_path, tmp_path, 2.9
    )
    assert early_r > 100
    assert late_b > late_r  # top pane now blue-dominant
    assert bot_b > bot_r  # bottom pane stayed on the blue crop throughout


def test_split_face_rects_map_into_their_panes() -> None:
    rects = _output_face_rects(
        _split_path(), source_w=640, source_h=360, out_w=360, out_h=640
    )
    assert len(rects) == 30  # 15 samples x 2 people
    top = [r for r in rects if r[1] < 320]
    bottom = [r for r in rects if r[1] >= 320]
    assert len(top) == 15 and len(bottom) == 15
    assert all(0 <= r[1] and r[1] + r[3] <= 320 for r in top)
    assert all(320 <= r[1] and r[1] + r[3] <= 640 for r in bottom)


# -- settings + pipeline plumbing -----------------------------------------

def test_split_screen_setting_defaults_on() -> None:
    assert DEFAULTS["split_screen"] is True


def test_experimental_tab_toggle_persists(app_window) -> None:
    from clipforge.ui.settings_dialog import SettingsDialog

    settings = app_window._settings
    original = bool(settings.get("split_screen"))
    dialog = SettingsDialog(settings, app_window)
    try:
        assert dialog._nav.count() == 8
        dialog._nav.setCurrentRow(6)  # Experimental
        switch = dialog._split_switch
        assert switch.isChecked() == original
        switch.click()
        assert bool(settings.get("split_screen")) == (not original)
    finally:
        settings.set("split_screen", original)
        dialog.close()


def test_pipeline_passes_split_screen_to_tracker(tmp_path, monkeypatch) -> None:
    from clipforge.core.pipeline import PipelineConfig, run_pipeline
    from clipforge.core.ranges import ClipRange
    from clipforge.core.transcription import Transcript, Word

    seen: dict = {}

    def fake_tracker(video_path, **kw):
        seen.update(kw)
        times = [0.0, 1.0]
        return CropPath(
            times=times,
            boxes=[FaceBox(x=0.0, y=0.0, width=202.0, height=360.0)] * 2,
            faces=[None, None],
        )

    transcript = Transcript(
        language="en",
        words=[Word(text="hello", start=0.5, end=0.9),
               Word(text="world", start=1.0, end=1.4)],
    )
    monkeypatch.setattr(
        "clipforge.core.pipeline.transcribe_ranges",
        lambda *a, **kw: transcript,
    )
    monkeypatch.setattr("clipforge.core.pipeline.deepgram_key", lambda: "k")

    config = PipelineConfig(
        video_path=_make_clip(tmp_path),
        language="en",
        manual_ranges=[ClipRange(start=0.0, end=3.0)],
        split_screen=False,
        export_dir=str(tmp_path / "exports"),
        render_settings=RenderSettings(width=360, height=640, captions="none"),
    )
    result = run_pipeline(config, track_faces_fn=fake_tracker)
    assert seen["split_screen"] is False
    assert result.crop_paths[0] is not None
