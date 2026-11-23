"""Shared filesystem paths for the app (one place, no duplication).

The app data directory is ``QStandardPaths.AppDataLocation / ClipForge``
(``%APPDATA%/ClipForge`` on Windows, ``~/.local/share/ClipForge`` on
Linux). It holds ``projects.json``, ``logs/`` and ``backups/``.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QStandardPaths

from .. import __app_name__

__all__ = ["app_data_dir"]


def app_data_dir() -> Path:
    """The app data directory (created on demand by callers)."""
    base = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation
    )
    return Path(base or f"~/.local/share/{__app_name__}") / __app_name__
