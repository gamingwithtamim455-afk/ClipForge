"""Shared filesystem paths for the app (one place, no duplication).

The app data directory is ``QStandardPaths.AppDataLocation / ClipForge``
(``%APPDATA%/ClipForge`` on Windows, ``~/.local/share/ClipForge`` on
Linux). It holds ``projects.json``, ``logs/`` and ``backups/``.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QStandardPaths

from .. import __app_name__

__all__ = ["app_data_dir", "default_video_dir"]


def app_data_dir() -> Path:
    """The app data directory (created on demand by callers)."""
    base = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation
    )
    return Path(base or f"~/.local/share/{__app_name__}") / __app_name__


def default_video_dir() -> str:
    """Where "Select Video" file pickers should start.

    The user's Videos folder (falling back to their home folder) — never
    the process working directory: a frozen app launched from its install
    folder would otherwise open the picker inside the install directory
    (``…\\AppData\\Local\\Programs\\ClipForge``), which confused users into
    thinking the app itself was a file browser.
    """
    movies = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.MoviesLocation
    )
    if movies and Path(movies).exists():
        return movies
    return str(Path.home())
