"""App logging: rotating log files.

Logs live in ``<app data>/logs/clipforge.log`` — 1 MB per file, 7 files
kept (roughly the last 7 days of runs). API keys are never logged
anywhere in the app; the diagnostics exporter scrubs once more.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

__all__ = ["setup_logging"]

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_MAX_BYTES = 1_000_000
_BACKUP_COUNT = 7


def setup_logging(data_dir: Path) -> Path:
    """Configure rotating file logging; returns the main log file path."""
    log_dir = Path(data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "clipforge.log"
    handler = RotatingFileHandler(
        log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Don't add a second file handler when create_app runs twice in tests.
    if not any(
        isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_path)
        for h in root.handlers
    ):
        root.addHandler(handler)
    return log_path
