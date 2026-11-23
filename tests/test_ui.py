"""UI tests (offscreen): app constructs, Home <-> wizard navigation works,
settings dialog opens, language switch applies."""

import pytest
from PySide6.QtWidgets import QApplication

from clipforge.app import create_app
from clipforge.ui.settings_dialog import SettingsDialog


@pytest.fixture(scope="module")
def window(app_window):
    # Shared session QApplication (conftest.app_window): Qt allows only one
    # QApplication singleton per process.
    yield app_window


def test_app_constructs(window) -> None:
    assert window.windowTitle() == "ClipForge"
    assert window._home is not None
    assert window._wizard is not None
    # A fresh store starts EMPTY — the old seeded sample projects (no
    # source file, no thumbnail) were removed: they were dead cards that
    # showed placeholders and led to a file picker when clicked.


def test_home_to_wizard_navigation(window) -> None:
    from clipforge.core.projects import Project

    project = Project(name="nav-test")
    window._open_project(project)
    assert window._stack.currentWidget() is window._wizard
    window._wizard.go_to_step(1)
    assert window._wizard._step == 1
    window._wizard._on_back()
    assert window._wizard._step == 0
    window._wizard._on_back()
    assert window._stack.currentWidget() is window._home


def test_start_clipping_disabled_until_ready(window) -> None:
    from clipforge.core.projects import Project

    window._open_project(Project(name="not-ready-test"))
    window._wizard.go_to_step(1)
    # No video selected and no language chosen -> disabled.
    assert not window._wizard._start_btn.isEnabled()
    window._show_home()


def test_window_fits_small_screen(window) -> None:
    # Regression: at 150% Windows scaling a 1080p screen offers only
    # ~1280x720 logical px; the window used to demand 1280x800 minimum
    # and its right side (Clip & Framing, timeline controls) was cut
    # off. At 1024x700 the resize must stick and every Viral Clips
    # panel must lie fully inside the window.
    from clipforge.core.projects import Project

    window._open_project(Project(name="small-screen-test"))
    window._wizard.go_to_step(1)
    window.resize(1024, 700)
    window.show()
    QApplication.processEvents()
    try:
        assert window.width() <= 1024
        assert window.height() <= 700
        viral = window._wizard._viral
        bounds = window.rect()
        panels = []
        layout = viral.layout()
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() is not None:
                panels.append(item.widget())
            elif item.layout() is not None:
                for j in range(item.layout().count()):
                    child = item.layout().itemAt(j).widget()
                    if child is not None:
                        panels.append(child)
        # Left options, player, timeline, right Clip & Framing.
        assert len(panels) == 4
        for panel in panels:
            rect = panel.rect().translated(
                panel.mapTo(window, panel.rect().topLeft())
            )
            assert bounds.contains(rect), f"{panel!r} overflows: {rect} vs {bounds}"
    finally:
        window._show_home()


def test_settings_dialog_opens(window) -> None:
    dialog = SettingsDialog(window._settings, window)
    assert dialog._stack.count() == 8  # incl. the Experimental tab
    assert dialog._nav.count() == 8
    dialog.close()


def test_language_switch_bengali(window) -> None:
    from clipforge.i18n import set_language, t

    window._on_language_changed("bn")
    assert t("home.projects") == "প্রজেক্ট"
    window._on_language_changed("en")
    assert t("home.projects") == "Projects"


def test_opening_a_project_resets_wizard_state(window, tmp_path) -> None:
    """Regression: video + language from one project must not leak into the next.

    Previously the shared session window kept the earlier video/language,
    so Start clipping was enabled for a new project with nothing chosen —
    and this test failed whenever test_wiring ran before test_ui.
    """
    import subprocess

    from clipforge.core.ffmpeg import find_ffmpeg
    from clipforge.core.projects import Project

    video = str(tmp_path / "source.mp4")
    subprocess.run(
        [find_ffmpeg(), "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=5",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", video],
        check=True,
    )
    # First project: choose a video and a language — Start enables.
    window._open_project(Project(name="first", source_path=video))
    window._wizard._select_video.set_video(video)
    window._wizard._viral._language.setCurrentIndex(1)
    assert window._wizard._start_btn.isEnabled()
    # Second project: everything resets, Start is disabled again.
    window._open_project(Project(name="second", source_path=""))
    assert window._wizard.video_path == ""
    assert window._wizard._viral.language_code == ""
    assert not window._wizard._viral.is_ready
    assert not window._wizard._start_btn.isEnabled()
    window._show_home()


def _pump_until(predicate, timeout_s: float = 10.0) -> bool:
    """Process Qt events until ``predicate()`` or the deadline passes."""
    import time

    from PySide6.QtWidgets import QApplication

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_home_grid_columns_follow_window_width(window, tmp_path) -> None:
    """Grid reflow regression: no hard-coded 5 columns / clipped cards."""
    from pathlib import Path

    from clipforge.core.projects import Project, ProjectStore
    from clipforge.ui.home import CARD_MIN_WIDTH, HomePage

    store = ProjectStore(Path(tmp_path) / "projects.json")
    for i in range(6):
        store.add(Project(name=f"grid-{i}"))
    page = HomePage(store)
    page.refresh()
    page.resize(1280, 800)
    page.show()
    QApplication.processEvents()
    wide_columns = page._columns
    assert wide_columns >= 4
    # Every card sits inside the computed column count.
    for index, card in enumerate(page._cards):
        row, col, _rs, _cs = page._grid_layout.getItemPosition(
            page._grid_layout.indexOf(card)
        )
        assert col < wide_columns
        assert row == index // wide_columns
    page.resize(430, 800)
    QApplication.processEvents()
    narrow_columns = page._columns
    assert 1 <= narrow_columns < wide_columns
    # Column width never drops below the card minimum (no clipped strips).
    viewport = page._scroll.viewport().width()
    assert viewport // narrow_columns >= CARD_MIN_WIDTH - 20
    page.close()


def test_home_repairs_missing_thumbnail(window, tmp_path) -> None:
    """A real project without a thumbnail gets one in the background."""
    import subprocess
    from pathlib import Path

    from clipforge.core.ffmpeg import find_ffmpeg
    from clipforge.core.projects import Project, ProjectStore
    from clipforge.ui.home import HomePage

    video = str(tmp_path / "source.mp4")
    subprocess.run(
        [find_ffmpeg(), "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=5",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", video],
        check=True,
    )
    store = ProjectStore(Path(tmp_path) / "projects.json")
    project = Project(name="thumb-test", source_path=video)
    store.add(project)
    page = HomePage(store)
    page.refresh()
    page.show()
    QApplication.processEvents()
    assert _pump_until(lambda: bool(project.thumbnail
                                    and Path(project.thumbnail).exists()))
    page.close()
