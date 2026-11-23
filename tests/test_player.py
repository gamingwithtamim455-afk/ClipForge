"""Tests: the video player actually plays (QMediaPlayer-backed).

Regression suite for the #1 user complaint — the original "player" was a
static pixmap whose play button and seek slider were never connected, so
no video ever played anywhere in the app. These tests load a real file
into :class:`VideoPlayerWidget` and assert the media loads, plays
(position advances), seeks, and pauses.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from clipforge.core.ffmpeg import find_ffmpeg
from clipforge.ui.widgets import VideoPlayerWidget


@pytest.fixture(scope="module")
def qapp(app_window):
    yield QApplication.instance()


def _make_video(tmp_path: Path, *, duration: float = 4.0) -> str:
    path = str(tmp_path / "playable.mp4")
    subprocess.run(
        [
            find_ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i",
            f"testsrc2=size=320x240:rate=30:duration={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", path,
        ],
        check=True,
    )
    return path


def _pump_until(predicate, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_player_loads_plays_seeks_and_pauses(qapp, tmp_path) -> None:
    video = _make_video(tmp_path)
    player = VideoPlayerWidget()
    player.show()
    player.set_source(video)
    assert _pump_until(lambda: player.duration_seconds > 3.0), (
        "media never loaded (duration stayed 0)"
    )
    assert player._play_btn.isEnabled()
    assert player._seek.isEnabled()

    player.play()
    assert _pump_until(lambda: player.is_playing and player.position_seconds > 0.2), (
        "playback never started / position never advanced"
    )

    player.seek_seconds(2.5)
    assert _pump_until(lambda: player.position_seconds >= 2.3)
    player.pause()
    assert _pump_until(lambda: not player.is_playing)
    player.close()


def test_player_play_range_stops_at_clip_end(qapp, tmp_path) -> None:
    video = _make_video(tmp_path)
    player = VideoPlayerWidget()
    player.show()
    player.set_source(video)
    assert _pump_until(lambda: player.duration_seconds > 3.0)
    player.set_play_range(0.5, 1.2)
    player.play()
    # Playback must pause at the range end (1.2 s), not run to 4 s.
    assert _pump_until(
        lambda: not player.is_playing and player.position_seconds >= 1.1,
        timeout_s=6.0,
    )
    assert player.position_seconds < 2.0
    player.close()


def test_viral_page_player_is_real(qapp, tmp_path) -> None:
    """The Viral Clips page's player loads and plays the source video."""
    from clipforge.ui.viral_clips import ViralClipsPage

    video = _make_video(tmp_path)
    page = ViralClipsPage()
    page.show()
    page.set_source(video)
    player = page._player_widget
    assert _pump_until(lambda: player.duration_seconds > 3.0)
    player.play()
    assert _pump_until(lambda: player.is_playing)
    page.close()
