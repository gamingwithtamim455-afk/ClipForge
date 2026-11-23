"""Tests (v0.2.2): the Edit step's caption preview — a still of the
selected clip with the current caption line burned in at its real
position (the reference app's on-video caption preview)."""

from __future__ import annotations

import subprocess
import time

import cv2
import numpy as np
import pytest

from clipforge.core.ffmpeg import find_ffmpeg
from clipforge.core.pipeline import ClipRange
from clipforge.core.preview import render_caption_still
from clipforge.core.transcription import Word

WORDS = [
    Word("Hello", 0.0, 0.5), Word("brave", 0.5, 1.0),
    Word("new", 1.0, 1.4), Word("world", 1.4, 2.0),
    Word("this", 2.2, 2.6), Word("is", 2.6, 2.9),
    Word("caption", 2.9, 3.6), Word("preview", 3.6, 4.2),
]


def _make_source(tmp_path, name="src.mp4"):
    """A 4 s grey 1920x1080 source (uniform, so caption pixels stand
    out against the background)."""
    out = tmp_path / name
    subprocess.run(
        [
            find_ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=0x808080:s=1920x1080:r=24:d=5",
            "-f", "lavfi", "-i", "sine=frequency=440:d=5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(out),
        ],
        check=True,
    )
    return str(out)


def _frame(path):
    frame = cv2.imread(str(path))
    assert frame is not None
    return frame


def test_caption_still_draws_caption_in_lower_area(tmp_path) -> None:
    """The still differs from the raw frame ONLY around the caption:
    caption-band pixels change, the top of the frame does not."""
    src = _make_source(tmp_path)
    raw = tmp_path / "raw.png"
    subprocess.run(
        [
            find_ffmpeg(), "-y", "-v", "error", "-ss", "1.2",
            "-i", src, "-frames:v", "1", str(raw),
        ],
        check=True,
    )
    out = tmp_path / "still.png"
    render_caption_still(
        src, at_time=1.2, clip_start=0.0, crop_rect=None,
        words=WORDS, preset_name="Keyword Pop",
        width=432, height=768, out_path=out,
    )
    still = _frame(out)
    assert still.shape[:2] == (768, 432)
    # Reference frame cropped the same way the still is (centred
    # 9:16 crop of the 1920x1080 source, scaled to 432x768).
    ref = _frame(raw)
    ref = cv2.resize(ref[:, 660:1260], (432, 768))
    diff = np.abs(still.astype(int) - ref.astype(int)).max(axis=2)
    changed = diff > 30
    assert changed.sum() > 200, "caption pixels were drawn"
    rows = np.where(changed.any(axis=1))[0]
    # The caption sits in the lower-middle/lower area (the preset's
    # v=70 anchor) — never in the top third of the frame.
    assert rows.min() > 768 * 0.35
    assert rows.max() < 768 * 0.85
    # The top band is untouched video.
    assert not changed[: int(768 * 0.3)].any()


def test_caption_still_uses_the_selected_line(tmp_path) -> None:
    """A still taken inside the SECOND line shows different pixels
    than one inside the first (the preview follows the line)."""
    src = _make_source(tmp_path)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    render_caption_still(
        src, at_time=1.2, clip_start=0.0, crop_rect=None,
        words=WORDS, preset_name="Keyword Pop",
        width=432, height=768, out_path=first,
    )
    render_caption_still(
        src, at_time=3.0, clip_start=0.0, crop_rect=None,
        words=WORDS, preset_name="Keyword Pop",
        width=432, height=768, out_path=second,
    )
    a, b = _frame(first), _frame(second)
    band_a = a[int(768 * 0.4):]
    band_b = b[int(768 * 0.4):]
    assert np.abs(band_a.astype(int) - band_b.astype(int)).max() > 30


def test_caption_still_respects_crop_window(tmp_path) -> None:
    """With a crop window (the tracked face crop), the still shows
    that window — a red left-half source crops to a red still."""
    src = tmp_path / "halves.mp4"
    subprocess.run(
        [
            find_ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=red:s=960x1080:r=24:d=5",
            "-f", "lavfi", "-i", "color=c=blue:s=960x1080:r=24:d=5",
            "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
            "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(src),
        ],
        check=True,
    )
    out = tmp_path / "crop.png"
    render_caption_still(
        str(src), at_time=1.2, clip_start=0.0,
        crop_rect=(100.0, 200.0, 540.0, 960.0),
        words=WORDS, preset_name="Keyword Pop",
        width=432, height=768, out_path=out,
    )
    frame = _frame(out)
    top = frame[:100].reshape(-1, 3).mean(axis=0)  # BGR, above captions
    assert top[2] > 150 and top[0] < 90  # red half


# ---------------------------------------------------------------------------
# Edit-step wiring
# ---------------------------------------------------------------------------

def _wait_for_preview(edit_page, timeout=30.0) -> bool:
    from PySide6.QtWidgets import QApplication

    deadline = time.time() + timeout
    while time.time() < deadline:
        QApplication.processEvents()
        pixmap = edit_page._caption_preview.pixmap()
        if pixmap is not None and not pixmap.isNull():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture()
def edit_page(app_window, tmp_path):
    page = app_window._wizard._edit
    src = _make_source(tmp_path)
    page.set_video_source(src)
    page.set_clips(
        [ClipRange(start=0.0, end=4.2, source="manual")],
        words=list(WORDS), video_duration=5.0,
    )
    page.set_output_size(1080, 1920)
    yield page
    page.set_clips([])


def test_edit_page_preview_renders_on_clip_select(edit_page) -> None:
    """Selecting a clip produces a caption preview pixmap."""
    assert _wait_for_preview(edit_page), "preview still never arrived"
    pixmap = edit_page._caption_preview.pixmap()
    # The still keeps the output ratio (9:16), scaled to the label.
    assert pixmap.width() * 16 == pytest.approx(pixmap.height() * 9, abs=16)


def test_edit_page_preview_clears_without_clips(edit_page) -> None:
    edit_page.set_clips([])
    pixmap = edit_page._caption_preview.pixmap()
    assert pixmap is None or pixmap.isNull()
    assert "Pick a clip" in edit_page._caption_preview.text()


def test_edit_page_preset_change_emits_and_repreviews(edit_page) -> None:
    """Changing the style in the Edit step now reaches the render
    jobs (clips_changed) and re-renders the preview."""
    assert _wait_for_preview(edit_page)
    seen = []
    edit_page.clips_changed.connect(lambda: seen.append(True))
    index = edit_page._preset.findData("Stacked")
    assert index >= 0
    edit_page._preset.setCurrentIndex(index)
    assert edit_page.caption_preset == "Stacked"
    assert seen, "preset change did not emit clips_changed"
    assert _wait_for_preview(edit_page)
