"""Encrypted API-key storage + validation entry points.

Keys live in the system keyring — Windows DPAPI on Windows (via the
``keyring`` package), Secret Service on Linux. Keys are never logged, never
written to a file, never shown again after saving, and never included in
any exception or status message.

This machine (the Linux dev VM) has NO keyring backend, so saving raises
:class:`KeyringUnavailable` here; that is expected and the Settings dialog
shows it instead of crashing. On the user's Windows PC the DPAPI backend
works normally.

Validation runs the real provider checks (``core`` re-exports them for the
Settings dialog): DeepSeek ``GET {base}/models``, Deepgram
``GET /v1/projects`` with the 1-second test-transcription fallback.
"""

from __future__ import annotations

import keyring

from ..providers.deepgram import DeepgramClient, DeepgramStatus
from ..providers.deepseek import DeepSeekClient, DeepSeekStatus

SERVICE = "ClipForge"
PROVIDERS = ("deepseek", "deepgram")


class KeyringUnavailable(Exception):
    """Secure storage is not available on this system."""


def save_key(provider: str, key: str) -> None:
    """Store a key encrypted in the system keyring.

    Raises :class:`KeyringUnavailable` when no backend exists.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    try:
        keyring.set_password(SERVICE, provider, key)
    except Exception as exc:
        raise KeyringUnavailable(
            "Secure storage is unavailable on this system — the key was not saved"
        ) from exc


def get_key(provider: str) -> str | None:
    """Load a stored key, or ``None`` when none is stored / storage is down."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    try:
        return keyring.get_password(SERVICE, provider)
    except Exception:
        return None


def delete_key(provider: str) -> None:
    """Remove a stored key (no error when there is nothing to remove)."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    try:
        keyring.delete_password(SERVICE, provider)
    except Exception:
        pass


def has_key(provider: str) -> bool:
    """Whether a key is stored for ``provider``."""
    return bool(get_key(provider))


def validate_deepseek(
    base_url: str, key: str, session: object | None = None
) -> DeepSeekStatus:
    """Validate a DeepSeek key via ``GET {base}/models``."""
    client = DeepSeekClient(base_url=base_url, api_key=key, session=session)
    return client.validate_key()


def validate_deepgram(key: str, session: object | None = None) -> DeepgramStatus:
    """Validate a Deepgram key via ``GET /v1/projects`` (test-listen fallback)."""
    client = DeepgramClient(api_key=key, session=session)
    return client.validate_key()
