"""i18n: JSON-backed translations with RTL support for Arabic/Urdu.

``set_language`` loads ``<code>.json`` from this package (falls back to
English for missing keys). Arabic (``ar``) and Urdu (``ur``) use a
right-to-left layout direction; the app applies it at startup and when the
setting changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt

SUPPORTED_LANGUAGES = {
    "en": "English",
    "bn": "Bengali",
    "ar": "Arabic",
    "ur": "Urdu",
}

RTL_LANGUAGES = {"ar", "ur"}

_current_code = "en"
_current_strings: dict[str, str] = {}
_fallback_strings: dict[str, str] = {}


def _load(code: str) -> dict[str, str]:
    path = Path(__file__).with_name(f"{code}.json")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def set_language(code: str) -> None:
    """Load translations for ``code``; unknown codes fall back to English."""
    global _current_code, _current_strings, _fallback_strings
    _fallback_strings = _load("en")
    _current_code = code if code in SUPPORTED_LANGUAGES else "en"
    _current_strings = _load(_current_code)


def current_language() -> str:
    return _current_code


def is_rtl(code: str | None = None) -> bool:
    """True when the (current) language needs a right-to-left layout."""
    return (code or _current_code) in RTL_LANGUAGES


def layout_direction(code: str | None = None) -> Qt.LayoutDirection:
    return Qt.LayoutDirection.RightToLeft if is_rtl(code) else Qt.LayoutDirection.LeftToRight


def t(key: str) -> str:
    """Translate ``key``; fall back to English, then to the key itself."""
    return _current_strings.get(key) or _fallback_strings.get(key) or key


# Load English by default so ``t()`` works before any explicit call.
set_language("en")
