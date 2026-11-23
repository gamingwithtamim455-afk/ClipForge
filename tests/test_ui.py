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
    # Sample projects seeded from the reference recording.
    assert len(window._store.projects) == 12


def test_home_to_wizard_navigation(window) -> None:
    project = window._store.projects[0]
    window._open_project(project)
    assert window._stack.currentWidget() is window._wizard
    window._wizard.go_to_step(1)
    assert window._wizard._step == 1
    window._wizard._on_back()
    assert window._wizard._step == 0
    window._wizard._on_back()
    assert window._stack.currentWidget() is window._home


def test_start_clipping_disabled_until_ready(window) -> None:
    window._open_project(window._store.projects[0])
    window._wizard.go_to_step(1)
    # No video selected and no language chosen -> disabled.
    assert not window._wizard._start_btn.isEnabled()
    window._show_home()


def test_settings_dialog_opens(window) -> None:
    dialog = SettingsDialog(window._settings, window)
    assert dialog._stack.count() == 7  # no Experimental tab
    assert dialog._nav.count() == 7
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
