"""ClipForge entry point.

Run:  python main.py            (repo root shim)
      python -m clipforge.main  (package entry)
    CLIPFORGE_SMOKE_TEST=1 QT_QPA_PLATFORM=offscreen python main.py
This shows the window, screenshots Home and the wizard, prints the paths
and exits with code 0.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .app import create_app


def _smoke_test(app, window) -> int:
    """Show the window, screenshot Home + wizard, exit 0."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    out_dir = Path(os.environ.get("CLIPFORGE_SCREENSHOT_DIR", "/tmp/clipforge-shots"))
    out_dir.mkdir(parents=True, exist_ok=True)

    def grab(name: str) -> None:
        pixmap = window.grab()
        path = out_dir / name
        pixmap.save(str(path))
        print(f"SMOKE screenshot: {path}")

    def step_wizard() -> None:
        # Open the first project in the wizard and screenshot it.
        projects = window._store.projects
        if projects:
            window._open_project(projects[0])
            window.show()
            QTimer.singleShot(400, lambda: (grab("wizard.png"), step_viral()))
        else:
            print("SMOKE: no projects found")
            app.quit()

    def step_viral() -> None:
        window._wizard.go_to_step(1)
        QTimer.singleShot(400, lambda: (grab("viral_clips.png"), step_settings()))

    def step_settings() -> None:
        from clipforge.ui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(window._settings, window)
        dialog.show()
        QTimer.singleShot(400, lambda: (dialog.grab().save(str(out_dir / "settings.png")),
                                        print(f"SMOKE screenshot: {out_dir / 'settings.png'}"),
                                        dialog.close(), app.quit()))

    window.show()
    QTimer.singleShot(400, lambda: (grab("home.png"), step_wizard()))
    return app.exec()


def _check_updates_on_launch(window) -> None:
    """Check the GitHub Releases feed once, shortly after launch.

    Silent when up to date or offline; only asks when a newer release
    with a Windows installer is actually available.
    """
    import logging

    from PySide6.QtWidgets import QMessageBox

    from clipforge.core.jobs import run_job
    from clipforge.core.updates import check_for_updates

    log = logging.getLogger(__name__)
    if not window._settings.get("check_updates_on_launch"):
        return

    def _on_checked(check) -> None:
        if not getattr(check, "available", False):
            log.info("Update check: %s", check.message)
            return
        info = check.info
        if info is None or not info.installer_url:
            log.info("Update check: %s", check.message)
            return
        answer = QMessageBox.question(
            window,
            "ClipForge update",
            f"{check.message}\n\nOpen Settings > Updates to download it?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            from clipforge.ui.settings_dialog import SettingsDialog

            dialog = SettingsDialog(window._settings, window)
            dialog.exec()

    run_job(
        lambda ctx: check_for_updates(),
        on_finished=_on_checked,
        on_failed=lambda err: log.info("Update check failed: %s", err),
        parent=window,
    )


def main() -> int:
    # Show the animated intro first (skipped in smoke tests / CI): it runs
    # while the main window is being built, so it costs no launch time.
    splash = None
    if os.environ.get("CLIPFORGE_SMOKE_TEST") != "1" and os.environ.get(
        "CLIPFORGE_NO_SPLASH"
    ) != "1":
        from PySide6.QtWidgets import QApplication as _QApp

        from .core.branding import branding_photo_path as _photo_path
        from .settings import Settings as _Settings
        from .ui.splash import SplashScreen as _Splash

        _QApp.instance() or _QApp(sys.argv)
        splash = _Splash(photo_path=_photo_path(_Settings()))
        splash.show()
        _QApp.processEvents()

    app, window = create_app()
    if os.environ.get("CLIPFORGE_SMOKE_TEST") == "1":
        return _smoke_test(app, window)
    window.show()
    if splash is not None:
        splash.finish()
    from PySide6.QtCore import QTimer  # noqa: PLC0415 - local, keeps import light

    QTimer.singleShot(1500, lambda: _check_updates_on_launch(window))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
