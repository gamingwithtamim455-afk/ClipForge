"""Edit-step crop preview (v0.2.1 B): the player shows the reframed
9:16 crop window, not the raw letterboxed 16:9 source."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from clipforge.core.face_tracking import CropPath, FaceBox
from clipforge.core.ranges import ClipRange
from clipforge.ui.widgets import VideoPlayerWidget


@pytest.fixture(scope="module")
def qapp(app_window):
    yield QApplication.instance()


def test_player_crop_preview_scales_and_offsets_surface(qapp) -> None:
    player = VideoPlayerWidget()  # fixed 270x480 area
    player.show()
    QApplication.processEvents()
    container = player._video_container
    assert container.width() == 270 and container.height() == 480
    # Full frame: the surface exactly fills the container.
    assert player._video_widget.geometry().getRect() == (0, 0, 270, 480)

    # Crop = right-hand 9:16 window of a 16:9 frame: normalised
    # x=0.5, w=0.3164 (606/1920 of the width), full height.
    player.set_crop_preview((0.5, 0.0, 0.3164, 1.0))
    QApplication.processEvents()
    geo = player._video_widget.geometry()
    # Surface scaled up so the crop fills the width, offset left by
    # the crop's x share; height matches (crop aspect == area aspect).
    assert geo.width() == pytest.approx(270 / 0.3164, abs=3)
    assert geo.x() == pytest.approx(-0.5 * 270 / 0.3164, abs=3)
    assert geo.height() == 480 and geo.y() == 0

    # Clearing restores the full frame.
    player.set_crop_preview(None)
    QApplication.processEvents()
    assert player._video_widget.geometry().getRect() == (0, 0, 270, 480)
    player.close()


def test_edit_page_applies_crop_preview_on_selection(app_window) -> None:
    edit = app_window._wizard._edit
    edit.set_clips(
        [ClipRange(0.0, 5.0, label="Clip 1"), ClipRange(10.0, 15.0, label="Clip 2")],
        video_duration=30.0,
    )
    # 1920x1080 source; clip 1 tracked with a right-side crop window,
    # clip 2 has no path -> centred fill crop preview.
    path = CropPath(
        times=[0.0, 2.5, 5.0],
        boxes=[
            FaceBox(1000, 0, 606, 1080),
            FaceBox(1100, 0, 606, 1080),
            FaceBox(1200, 0, 606, 1080),
        ],
        faces=[None, None, None], mode="track",
    )
    edit.set_crop_paths({0: path, 1: None}, 1920, 1080)
    edit._clip_list.setCurrentRow(0)
    rect = edit._player_widget._crop_preview
    assert rect is not None
    # Median box (1100, 0, 606, 1080) normalised.
    assert rect[0] == pytest.approx(1100 / 1920)
    assert rect[2] == pytest.approx(606 / 1920)
    edit._clip_list.setCurrentRow(1)
    centred = edit._player_widget._crop_preview
    assert centred is not None
    assert centred[0] == pytest.approx((1920 - 606) / 2 / 1920, abs=0.01)
    assert centred[2] == pytest.approx(606 / 1920, abs=0.01)
