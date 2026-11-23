"""Tests for the animated intro splash, sidebar branding, WhatsApp
share button (folder-open only, no API) and the Open Muse link."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QAbstractAnimation, Qt, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QPushButton

from clipforge.core.branding import (
    branding_photo_path,
    default_branding_photo_path,
    load_branding_photo,
)
from clipforge.core.render import ClipResult
from clipforge.ui.render import RenderPage, whatsapp_share_folder
from clipforge.ui.splash import SidebarBranding, SplashScreen, _TileItem

# The bundled photo is Boss's blue-crystal portrait (JPEG bytes, 1672x941).
EXPECTED_PHOTO_SIZE = (1672, 941)


def test_branding_photo_file_loads(app_window) -> None:
    path = branding_photo_path()
    assert path == default_branding_photo_path()
    if not path.exists():
        # The bundled portrait is Boss's private photo: it is NOT shipped
        # in the public repo (added at install only), so CI has no file.
        pytest.skip("bundled branding photo is not shipped in the public repo")
    pixmap = load_branding_photo()
    assert pixmap is not None and not pixmap.isNull()
    assert (pixmap.width(), pixmap.height()) == EXPECTED_PHOTO_SIZE


def test_branding_photo_override_env(app_window, monkeypatch, tmp_path) -> None:
    """A configured path wins over the bundled file (swappable photo)."""
    from PySide6.QtGui import QPixmap

    fake = tmp_path / "other.png"
    pixmap = QPixmap(64, 48)
    assert pixmap.save(str(fake))
    monkeypatch.setenv("CLIPFORGE_BRANDING_PHOTO", str(fake))
    assert branding_photo_path() == fake
    loaded = load_branding_photo()
    assert loaded is not None and not loaded.isNull()


def test_splash_constructs_with_photo_tiles(app_window) -> None:
    if load_branding_photo() is None:
        pytest.skip("bundled branding photo is not shipped in the public repo")
    splash = SplashScreen()
    splash.show()
    try:
        assert splash.isVisible()
        assert splash.width() == 640 and splash.height() == 420
        tiles = [i for i in splash._scene.items() if isinstance(i, _TileItem)]
        assert len(tiles) == 6 * 4, "photo should split into 24 shards"
        assert splash._animations.state() == QAbstractAnimation.State.Running
    finally:
        splash.close()


def test_splash_dismisses_on_key(app_window) -> None:
    splash = SplashScreen()
    splash.show()
    QTest.keyClick(splash, Qt.Key.Key_Escape)
    assert not splash.isVisible()


def test_splash_without_photo_falls_back(app_window, monkeypatch) -> None:
    """A missing/unreadable photo must never break launch."""
    import clipforge.ui.splash as splash_mod

    monkeypatch.setattr(splash_mod, "load_branding_photo", lambda: None)
    splash = SplashScreen()
    splash.show()
    try:
        assert splash.isVisible()
        tiles = [i for i in splash._scene.items() if isinstance(i, _TileItem)]
        assert tiles == []
    finally:
        splash.close()


def test_sidebar_branding_pulses(app_window) -> None:
    if load_branding_photo() is None:
        pytest.skip("bundled branding photo is not shipped in the public repo")
    branding = SidebarBranding()
    # The photo label exists with the portrait loaded, and the glow
    # animation runs forever (slow pulse).
    labels = branding.findChildren(QLabel)
    assert labels, "sidebar branding should show the photo"
    assert not labels[0].pixmap().isNull()
    assert branding._glow_anim.state() == QAbstractAnimation.State.Running
    assert branding._glow_anim.loopCount() == -1


def test_whatsapp_share_folder(tmp_path) -> None:
    video = tmp_path / "clips" / "01_clip.mp4"
    assert whatsapp_share_folder(str(video)) == str(tmp_path / "clips")


def test_render_share_button_opens_clip_folder(app_window, monkeypatch, tmp_path) -> None:
    """Share via WhatsApp: opens the clip folder, no WhatsApp login/API."""
    opened: list[QUrl] = []
    import PySide6.QtGui as qt_gui

    monkeypatch.setattr(
        qt_gui.QDesktopServices, "openUrl", lambda url: opened.append(url) or True
    )
    page = RenderPage()
    assert not page._share_btn.isEnabled()
    clips_dir = tmp_path / "clips"
    result = ClipResult(
        index=1,
        video_path=str(clips_dir / "01_clip.mp4"),
        label_path=str(clips_dir / "01_label.txt"),
        duration=40.0,
        caption="Test caption",
        hashtags=["#a", "#b", "#c", "#d"],
        label_written=True,
    )
    page.set_results([result])
    assert page._share_btn.isEnabled()
    page._share_btn.click()
    assert len(opened) == 1
    assert Path(opened[0].toLocalFile()) == clips_dir


def test_open_muse_button(app_window, monkeypatch, tmp_path) -> None:
    """Sidebar 'Open Muse' opens https://muse.ai in the browser."""
    from clipforge.core.projects import ProjectStore
    from clipforge.ui.home import HomePage

    opened: list[QUrl] = []
    import PySide6.QtGui as qt_gui

    monkeypatch.setattr(
        qt_gui.QDesktopServices, "openUrl", lambda url: opened.append(url) or True
    )
    home = HomePage(ProjectStore(tmp_path / "projects.json"))
    buttons = [b for b in home.findChildren(QPushButton) if b.text() == "Open Muse"]
    assert len(buttons) == 1
    buttons[0].click()
    assert len(opened) == 1
    assert opened[0].toString() == "https://muse.ai"

