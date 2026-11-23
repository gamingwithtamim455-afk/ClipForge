"""Regression E2E (v0.2.1 B): the 9:16 output must FILL the frame.

The user's real 16:9 video produced a 9:16 canvas with the 16:9
picture sitting in the middle (empty space above/below). Root cause:
when face tracking found no face it returned mode="blur-fit", whose
render graph fitted the whole frame inside the canvas over a blurred
background. That graph is gone; the no-face fallback is a centred
FILL crop, and tracked crops fill by construction.

These tests render a 16:9 source (real face patch from
``docs/caption-style-references`` composited off-centre over a
patterned background) in BOTH caption modes and assert:
- output is exactly 1080x1920,
- the top and bottom 5% bands are real picture (not uniform, not
  near-black, not blurred-flat),
- the face is inside the rendered frame (tracking QA).

A faceless 16:9 source (the user's exact case) gets the same band
assertions through the centred fill fallback.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from clipforge.core.face_tracking import track_faces
from clipforge.core.ffmpeg import find_ffmpeg, probe
from clipforge.core.render import RenderSettings, _filter_graph, render_clip
from clipforge.core.transcription import Word

REF_DIR = Path(__file__).resolve().parent.parent / "docs" / "caption-style-references"
FACE_REF = REF_DIR / "01-pyaar-magenta-keyword.png"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _make_face_source(tmp_path: Path) -> str:
    """1920x1080 patterned source with a real face patch off-centre."""
    ffmpeg = find_ffmpeg()
    patch = str(tmp_path / "patch.png")
    _run([
        ffmpeg, "-y", "-v", "error", "-i", str(FACE_REF),
        "-vf", "crop=480:620:140:370", patch,
    ])
    out = str(tmp_path / "face_source.mp4")
    _run([
        ffmpeg, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=4",
        "-loop", "1", "-i", patch,
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-filter_complex", "[1:v]scale=480:620[patch];[0:v][patch]overlay=1100:230[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", out,
    ])
    return out


def _make_faceless_source(tmp_path: Path) -> str:
    out = str(tmp_path / "faceless.mp4")
    _run([
        find_ffmpeg(), "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", out,
    ])
    return out


def _band_stats(video: str, tmp_path: Path, tag: str) -> tuple[float, float, float, float]:
    """(top_std, bottom_std, top_mean, bottom_mean) of a mid frame.

    Bands are the top/bottom 5% of the 1080x1920 frame. A letterboxed
    or blurred-backdrop render fails these: its bands are uniform,
    near-black, or blurred flat (per-row std near zero).
    """
    import cv2
    import numpy as np

    frame_png = str(tmp_path / f"frame_{tag}.png")
    _run([
        find_ffmpeg(), "-y", "-v", "error", "-ss", "2",
        "-i", video, "-frames:v", "1", frame_png,
    ])
    img = cv2.imread(frame_png, cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape == (1920, 1080)
    band = img.shape[0] // 20  # 5%
    top = img[:band]
    bottom = img[-band:]

    def _std(block) -> float:
        return float(np.mean(np.std(block.astype(np.float64), axis=1)))

    return (
        _std(top), _std(bottom),
        float(np.mean(top)), float(np.mean(bottom)),
    )


def _assert_filled(video: str, tmp_path: Path, tag: str) -> None:
    info = probe(video)
    assert (info.width, info.height) == (1080, 1920)
    top_std, bottom_std, top_mean, bottom_mean = _band_stats(video, tmp_path, tag)
    # Real picture in both bands: textured (not uniform/blurred flat)
    # and not near-black empty space.
    assert top_std > 6.0, f"top band looks empty/blurred (row std {top_std:.2f})"
    assert bottom_std > 6.0, f"bottom band looks empty/blurred (row std {bottom_std:.2f})"
    assert top_mean > 12.0, f"top band near-black (mean {top_mean:.1f})"
    assert bottom_mean > 12.0, f"bottom band near-black (mean {bottom_mean:.1f})"


def test_filter_graph_never_fits_or_pads() -> None:
    """No render graph may contain the fit-to-canvas construct."""
    from clipforge.core.face_tracking import CropPath, FaceBox

    path = CropPath(
        times=[0.0], boxes=[FaceBox(100, 0, 606, 1080)],
        faces=[None], mode="track", fallback_fraction=1.0,
    )
    for crop_path in (path, None):
        graph = _filter_graph(
            crop_path=crop_path, crop_w=606, crop_h=1080,
            width=1080, height=1920, ass_path=None, fontsdir=None,
        )
        assert "force_original_aspect_ratio=decrease" not in graph
        assert "gblur" not in graph


@pytest.mark.skipif(not FACE_REF.exists(), reason="face reference image missing")
def test_face_clip_fills_frame_in_both_caption_modes(tmp_path) -> None:
    src = _make_face_source(tmp_path)
    path = track_faces(src, start=0.0, end=4.0)
    assert path.mode == "track"
    assert path.fallback_fraction == 0.0
    assert path.qa_face_inside_fraction >= 0.99  # face inside the crop

    words = [Word("hello", 0.2, 0.6), Word("world", 0.7, 1.1),
             Word("again", 1.5, 1.9), Word("today", 2.2, 2.6)]
    for captions, tag in (("burn", "burn"), ("none", "cutout")):
        result = render_clip(
            src, start=0.0, end=4.0,
            settings=RenderSettings(
                width=1080, height=1920, captions=captions,
                preset="veryfast", crf=23,
            ),
            index=1, crop_path=path, words=words, language="en",
            output_path=str(tmp_path / f"clip_{tag}.mp4"),
        )
        _assert_filled(result.video_path, tmp_path, tag)


def test_faceless_clip_fills_frame_via_centred_crop(tmp_path) -> None:
    """The user's exact case: no face found -> fill, never letterbox."""
    src = _make_faceless_source(tmp_path)
    result = render_clip(
        src, start=0.0, end=3.0,
        settings=RenderSettings(
            width=1080, height=1920, captions="none",
            preset="veryfast", crf=23,
        ),
        index=1, words=[], language="en",
        output_path=str(tmp_path / "clip_faceless.mp4"),
    )
    assert any("centred fill crop" in f for f in result.flags)
    _assert_filled(result.video_path, tmp_path, "faceless")
