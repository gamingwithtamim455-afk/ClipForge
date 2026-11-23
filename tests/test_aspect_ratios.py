"""Tests (v0.2.2): the Clip & Framing aspect selector (9:16 / 1:1 /
16:9) drives render dimensions, face-track crop geometry, split
orientation, the Edit crop preview and per-project persistence."""

from __future__ import annotations

import pytest

from clipforge.core import face_tracking
from clipforge.core.face_tracking import (
    CropPath,
    FaceBox,
    split_layout_for_aspect,
    track_faces,
)
from clipforge.core.ffmpeg import probe
from clipforge.core.pipeline import (
    ASPECT_RATIOS,
    PipelineConfig,
    aspect_label,
    output_size,
)
from clipforge.core.projects import Project, ProjectStore
from clipforge.core.render import RenderSettings, render_clip
from clipforge.core.transcription import Word

# ---------------------------------------------------------------------------
# Render dimensions: every ratio x every quality, exact
# ---------------------------------------------------------------------------

EXPECTED_SIZES = {
    ("1080p", (9, 16)): (1080, 1920),
    ("1440p", (9, 16)): (1440, 2560),
    ("4K", (9, 16)): (2160, 3840),
    ("1080p", (1, 1)): (1080, 1080),
    ("1440p", (1, 1)): (1440, 1440),
    ("4K", (1, 1)): (2160, 2160),
    ("1080p", (16, 9)): (1920, 1080),
    ("1440p", (16, 9)): (2560, 1440),
    ("4K", (16, 9)): (3840, 2160),
}


@pytest.mark.parametrize(
    "quality,aspect", sorted(EXPECTED_SIZES), ids=str
)
def test_output_size_matrix(quality, aspect) -> None:
    assert output_size(quality, aspect) == EXPECTED_SIZES[(quality, aspect)]


@pytest.mark.parametrize(
    "quality,aspect", sorted(EXPECTED_SIZES), ids=str
)
def test_render_settings_carry_the_chosen_ratio(quality, aspect) -> None:
    config = PipelineConfig(
        video_path="v.mp4", quality=quality, aspect=aspect,
    )
    settings = config.render_settings_for()
    assert (settings.width, settings.height) == EXPECTED_SIZES[
        (quality, aspect)
    ]


def test_aspect_labels_round_trip() -> None:
    assert set(ASPECT_RATIOS) == {"9:16", "1:1", "16:9"}
    for label, pair in ASPECT_RATIOS.items():
        assert aspect_label(pair) == label
    # Unknown pairs format as-is; the wizard's set_aspect falls back
    # to 9:16 for labels it does not know.
    assert aspect_label((3, 4)) == "3:4"


# ---------------------------------------------------------------------------
# Crop geometry per ratio (stubbed detector, like test_split_screen)
# ---------------------------------------------------------------------------

def _stub_detector(monkeypatch, boxes_fn):
    """Replace the detector: boxes_fn(frame_index) -> list[FaceBox]
    in NORMALISED (0..1) coordinates, like the real detector."""

    def fake(frames, frame_w, frame_h, *, min_detection_confidence=0.5,
             scene_cuts=None, progress=None, cancel=None):
        detected = [boxes_fn(i) for i in range(len(frames))]
        return detected, [False] * len(frames)

    monkeypatch.setattr(face_tracking, "_detect_faces_in_frames", fake)


def _make_clip(tmp_path, name="src.mp4", seconds=3):
    from clipforge.core.ffmpeg import find_ffmpeg, run_ffmpeg

    out = tmp_path / name
    run_ffmpeg([
        "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc2=size=1920x1080:rate=24:d={seconds}",
        "-f", "lavfi", "-i", "sine=frequency=440",
        "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(out),
    ])
    return str(out)


def _face(x=0.40, identity=0):
    """A face in normalised detector coordinates."""
    return FaceBox(x=x, y=0.30, width=0.20, height=0.35, identity=identity)


@pytest.mark.parametrize(
    "aspect", [(9, 16), (1, 1), (16, 9)], ids=["9:16", "1:1", "16:9"]
)
def test_crop_geometry_matches_chosen_ratio(tmp_path, monkeypatch, aspect):
    """The tracked crop window has exactly the chosen ratio's shape."""
    src = _make_clip(tmp_path)
    _stub_detector(monkeypatch, lambda i: [_face()])
    path = track_faces(src, start=0.0, end=3.0, aspect=aspect, sample_fps=2.0)
    assert path.mode == "track"
    assert path.boxes
    for box in path.boxes:
        assert box.width / box.height == pytest.approx(
            aspect[0] / aspect[1], rel=0.02
        )
        # Crop-to-fill: the window stays inside the source frame.
        assert box.x >= -1 and box.y >= -1
        assert box.x + box.width <= 1921
        assert box.y + box.height <= 1081


def test_centred_fallback_also_follows_ratio(tmp_path, monkeypatch):
    """No face found: the centred fallback crop uses the ratio too."""
    src = _make_clip(tmp_path)
    _stub_detector(monkeypatch, lambda i: [])
    path = track_faces(src, start=0.0, end=3.0, aspect=(1, 1), sample_fps=2.0)
    assert path.mode == "track" and path.needs_review
    box = path.boxes[0]
    assert box.width == pytest.approx(box.height, abs=2)


# ---------------------------------------------------------------------------
# Split orientation per ratio
# ---------------------------------------------------------------------------

def test_split_layout_choice() -> None:
    assert split_layout_for_aspect((9, 16)) == "stacked"
    assert split_layout_for_aspect((16, 9)) == "side-by-side"
    assert split_layout_for_aspect((1, 1)) == "side-by-side"


def _two_faces(i):
    return [_face(x=0.08, identity=0), _face(x=0.68, identity=1)]


@pytest.mark.parametrize(
    "aspect,layout,pane_ratio",
    [
        ((9, 16), "stacked", 18 / 16),      # pane = 9:8 of the frame
        ((16, 9), "side-by-side", 16 / 18),  # pane = 8:9 of the frame
        ((1, 1), "side-by-side", 1 / 2),     # pane = 1:2 of the frame
    ],
    ids=["9:16", "16:9", "1:1"],
)
def test_split_panes_follow_ratio(
    tmp_path, monkeypatch, aspect, layout, pane_ratio
):
    src = _make_clip(tmp_path)
    _stub_detector(monkeypatch, _two_faces)
    path = track_faces(
        src, start=0.0, end=3.0, aspect=aspect, sample_fps=2.0,
        split_screen=True,
    )
    assert path.mode == "split"
    assert path.split_layout == layout
    for box in path.boxes:
        assert box.width / box.height == pytest.approx(pane_ratio, rel=0.02)


def test_split_render_side_by_side_places_people_left_right(tmp_path):
    """A real 16:9 split render: the left pane shows the LEFT source
    half, the right pane the RIGHT half (red/blue halves)."""
    import subprocess

    import cv2
    import numpy as np

    from clipforge.core.ffmpeg import find_ffmpeg

    ffmpeg = find_ffmpeg()
    src = tmp_path / "halves.mp4"
    subprocess.run(
        [
            ffmpeg, "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=red:s=960x1080:r=24:d=2",
            "-f", "lavfi", "-i", "color=c=blue:s=960x1080:r=24:d=2",
            "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
            "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(src),
        ],
        check=True,
    )
    pane_a = FaceBox(x=240, y=300, width=480, height=540)   # left half
    pane_b = FaceBox(x=1200, y=300, width=480, height=540)  # right half
    path = CropPath(
        times=[0.0], boxes=[pane_a], boxes_b=[pane_b], mode="split",
        split_layout="side-by-side",
    )
    out = tmp_path / "split169.mp4"
    result = render_clip(
        str(src), start=0.0, end=2.0,
        settings=RenderSettings(width=1920, height=1080, captions="none"),
        crop_path=path, output_path=str(out),
    )
    info = probe(result.video_path)
    assert (info.width, info.height) == (1920, 1080)
    frame_path = tmp_path / "split169.png"
    subprocess.run(
        [
            ffmpeg, "-y", "-v", "error", "-ss", "1", "-i", str(out),
            "-frames:v", "1", str(frame_path),
        ],
        check=True,
    )
    frame = cv2.imread(str(frame_path))
    left = frame[:, :480].reshape(-1, 3).mean(axis=0)   # BGR
    right = frame[:, 1440:].reshape(-1, 3).mean(axis=0)
    assert left[2] > 150 and left[0] < 80    # left pane is red
    assert right[0] > 150 and right[2] < 80  # right pane is blue
    assert np.abs(frame).sum() > 0


# ---------------------------------------------------------------------------
# Real renders at each ratio come out at the exact dimensions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "aspect,size",
    [((9, 16), (1080, 1920)), ((1, 1), (1080, 1080)), ((16, 9), (1920, 1080))],
    ids=["9:16", "1:1", "16:9"],
)
def test_render_clip_exact_dimensions_per_ratio(tmp_path, aspect, size):
    src = _make_clip(tmp_path, seconds=2)
    config = PipelineConfig(video_path=src, quality="1080p", aspect=aspect)
    settings = config.render_settings_for()
    out = tmp_path / "clip.mp4"
    result = render_clip(
        src, start=0.0, end=2.0, settings=settings, output_path=str(out),
    )
    info = probe(result.video_path)
    assert (info.width, info.height) == size


def test_render_clip_1440p_dimensions(tmp_path):
    src = _make_clip(tmp_path, seconds=2)
    config = PipelineConfig(video_path=src, quality="1440p", aspect=(1, 1))
    out = tmp_path / "clip1440.mp4"
    result = render_clip(
        src, start=0.0, end=2.0,
        settings=config.render_settings_for(), output_path=str(out),
    )
    info = probe(result.video_path)
    assert (info.width, info.height) == (1440, 1440)


def test_render_with_captions_at_16_9(tmp_path):
    """Captions burn correctly on a landscape frame too."""
    src = _make_clip(tmp_path, seconds=2)
    words = [Word("Hello", 0.0, 0.5), Word("there", 0.5, 1.0),
             Word("world", 1.0, 1.6)]
    out = tmp_path / "cap169.mp4"
    result = render_clip(
        src, start=0.0, end=2.0,
        settings=RenderSettings(
            width=1920, height=1080, captions="burn",
            caption_preset="Keyword Pop",
        ),
        words=words, output_path=str(out),
    )
    info = probe(result.video_path)
    assert (info.width, info.height) == (1920, 1080)


# ---------------------------------------------------------------------------
# Persistence + wizard restore + combo contents
# ---------------------------------------------------------------------------

def test_project_aspect_quality_round_trip(tmp_path) -> None:
    path = tmp_path / "projects.json"
    store = ProjectStore(path)
    project = Project(name="Ratios", source_path="v.mp4")
    store.add(project)
    project.aspect = "16:9"
    project.quality = "1440p"
    store.save()
    loaded = ProjectStore(path).projects[0]
    assert loaded.aspect == "16:9"
    assert loaded.quality == "1440p"
    # Defaults for projects saved before v0.2.2.
    fresh = Project(name="Old")
    assert fresh.aspect == "9:16" and fresh.quality == "1080p"


def test_wizard_restores_project_ratio(app_window, tmp_path) -> None:
    wizard = app_window._wizard
    project = Project(
        name="Restore", source_path=str(tmp_path / "v.mp4"),
        aspect="1:1", quality="4K",
    )
    wizard.set_project(project)
    assert wizard._viral.aspect == (1, 1)
    assert wizard._viral.aspect_label == "1:1"
    assert wizard._viral.quality == "4K"
    # A project without saved settings falls back to the defaults.
    wizard.set_project(Project(name="Plain"))
    assert wizard._viral.aspect == (9, 16)
    assert wizard._viral.quality == "1080p"


def test_aspect_combo_has_exactly_the_three_ratios(app_window) -> None:
    combo = app_window._wizard._viral._aspect
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "9:16", "1:1", "16:9",
    ]
