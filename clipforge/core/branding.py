"""Branding photo: the user's portrait used by the splash intro and the
Home sidebar.

The photo is swappable with no code change — replace the file and the
app picks it up on the next launch:

- Default (bundled): ``clipforge/assets/branding/user-photo.png``
  (Boss's blue-crystal portrait; note the file holds JPEG bytes under a
  ``.png`` name — loading is by content, never by extension).
- Override: Settings key ``branding_photo`` (a file path), or the
  ``CLIPFORGE_BRANDING_PHOTO`` environment variable (for tests/dev).

Loading never raises: a missing or unreadable photo returns ``None``
and the splash falls back to a generated gradient + title, so a bad
photo can never break app launch.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["branding_photo_path", "default_branding_photo_path", "load_branding_photo"]


def default_branding_photo_path() -> Path:
    """The bundled photo shipped with the app."""
    return Path(__file__).resolve().parent.parent / "assets" / "branding" / "user-photo.png"


def branding_photo_path(settings=None) -> Path:
    """Resolve the photo to use: settings override > env var > bundled file."""
    if settings is not None:
        try:
            configured = str(settings.get("branding_photo") or "").strip()
        except Exception:
            configured = ""
        if configured and Path(configured).exists():
            return Path(configured)
    env_path = os.environ.get("CLIPFORGE_BRANDING_PHOTO", "").strip()
    if env_path and Path(env_path).exists():
        return Path(env_path)
    return default_branding_photo_path()


def load_branding_photo(settings=None):
    """Load the branding photo as a QPixmap, or ``None`` when unavailable.

    Uses QImageReader (content-sniffed), so the JPEG bytes in the
    ``.png``-named file load fine. Imported lazily — this module must
    stay importable without Qt (tests import the path helpers plainly).
    """
    from PySide6.QtGui import QPixmap

    path = branding_photo_path(settings)
    if not path.exists():
        return None
    pixmap = QPixmap(str(path))
    return pixmap if not pixmap.isNull() else None
