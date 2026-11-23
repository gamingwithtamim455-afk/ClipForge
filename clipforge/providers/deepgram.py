"""Deepgram adapter (cloud speech-to-text).

Rules (docs/provider-research.md §1, verified 2026-11-22):
- ``POST https://api.deepgram.com/v1/listen`` with
  ``Authorization: Token <key>`` (never ``Bearer`` for plain API keys).
- Model ``nova-3`` for ALL 12 app languages (multilingual); no per-language
  model switch and no faster-whisper fallback is triggered by these 12.
- Params: ``smart_format=true``, ``punctuate=true``, ``utterances=true``.
  There is NO ``word_timestamps`` parameter — word-level start/end are
  returned by default in ``results.channels[0].alternatives[0].words``.
- Auto-detect: ``detect_language=true`` and omit ``language``.
- Key validation: ``GET https://api.deepgram.com/v1/projects`` (200 = valid).
  A 403 there may only mean the key lacks management scope — fall back to a
  1-second test transcription instead of reporting the key invalid.
- App-side chunking: audio only (mono 16 kHz), chunks of at most 8 minutes
  with 5 s overlap; merging happens in ``core.transcription``.

The API key is only ever sent in the ``Authorization`` header and is never
logged, never included in an exception message, and never returned by any
method here. Pass a fake ``session`` in tests to avoid real network calls.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import requests

from . import load_presets
from ._http import ProviderError, request_with_retry

LISTEN_URL = "https://api.deepgram.com/v1/listen"
PROJECTS_URL = "https://api.deepgram.com/v1/projects"
DEFAULT_MODEL = "nova-3"

#: App language code -> BCP-47 code sent to Deepgram (``None`` = auto-detect).
LANGUAGE_CODES: dict[str, str | None] = {
    "auto": None,
    "bn": "bn",
    "en": "en",
    "ar": "ar",
    "ur": "ur",
    "hi": "hi",
    "es": "es",
    "fr": "fr",
    "tr": "tr",
    "id": "id",
    "pt": "pt",
    "ru": "ru",
}

CHUNK_SECONDS = 480  # at most 8 minutes per request
CHUNK_OVERLAP_SECONDS = 5


def model_for_language(language: str) -> str:
    """Deepgram model for an app language code (nova-3 for all 12)."""
    presets = load_presets()["deepgram"]["languages"]
    return str(presets.get(language, DEFAULT_MODEL))


def build_listen_params(language: str) -> dict:
    """Query parameters for ``POST /v1/listen`` for an app language code.

    >>> build_listen_params("bn")["language"]
    'bn'
    >>> build_listen_params("auto")
    {'model': 'nova-3', 'detect_language': 'true', 'smart_format': 'true', 'punctuate': 'true', 'utterances': 'true'}
    """
    presets = load_presets()["deepgram"]
    params: dict[str, str] = {
        "model": model_for_language(language),
        "smart_format": "true" if presets["params"]["smart_format"] else "false",
        "punctuate": "true" if presets["params"]["punctuate"] else "false",
        "utterances": "true" if presets["params"]["utterances"] else "false",
    }
    code = LANGUAGE_CODES.get(language)
    if code is None:
        params["detect_language"] = "true"
    else:
        params["language"] = code
    return params


def one_second_silence_wav(sample_rate: int = 16000) -> bytes:
    """1 s of silent mono 16 kHz PCM16 WAV, for the key-validation fallback."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * sample_rate)
    return buffer.getvalue()


@dataclass
class DeepgramStatus:
    """Result of a key validation (``GET /v1/projects`` or test listen)."""

    connected: bool
    message: str  # e.g. "Connected" or "Invalid API key (401)"
    validated_by: str = "projects"  # "projects" or "test_transcription"


@dataclass
class DeepgramClient:
    """Cloud transcription client."""

    api_key: str = ""  # never logged; stored encrypted via keyring
    model: str = DEFAULT_MODEL
    session: object | None = None  # injectable for tests
    max_retries: int = 3
    backoff_base: float = 1.0

    def __post_init__(self) -> None:
        self._session = self.session if self.session is not None else requests.Session()

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_key}"}

    def validate_key(self, timeout: float = 10.0) -> DeepgramStatus:
        """Validate the key via ``GET /v1/projects``.

        A 403 there does NOT mean an invalid key — a listen-only scoped key
        is denied on the management endpoint, so fall back to a 1-second
        test transcription and judge by that instead (research §1.4).
        Never raises for a bad key; the reason is in the status.
        """
        try:
            request_with_retry(
                self._session,
                "GET",
                PROJECTS_URL,
                headers=self._auth_headers(),
                timeout=timeout,
                max_retries=self.max_retries,
                backoff_base=self.backoff_base,
            )
        except ProviderError as exc:
            if exc.kind == "forbidden":
                return self._validate_by_test_listen(timeout)
            return DeepgramStatus(connected=False, message=exc.message)
        return DeepgramStatus(connected=True, message="Connected")

    def _validate_by_test_listen(self, timeout: float) -> DeepgramStatus:
        """Fallback validation: transcribe 1 s of silence.

        Costs a negligible amount of credit; used only when the projects
        endpoint answered 403 (listen-only key scope).
        """
        try:
            request_with_retry(
                self._session,
                "POST",
                LISTEN_URL,
                headers={**self._auth_headers(), "Content-Type": "audio/wav"},
                params=build_listen_params("en"),
                data=one_second_silence_wav(),
                timeout=timeout,
                max_retries=self.max_retries,
                backoff_base=self.backoff_base,
            )
        except ProviderError as exc:
            return DeepgramStatus(
                connected=False, message=exc.message, validated_by="test_transcription"
            )
        return DeepgramStatus(
            connected=True, message="Connected", validated_by="test_transcription"
        )

    def transcribe_chunk(
        self, wav_bytes: bytes, *, language: str = "auto", timeout: float = 300.0
    ) -> dict:
        """Transcribe one audio chunk (mono 16 kHz WAV bytes).

        Returns the raw Deepgram ``/v1/listen`` JSON response. Chunking and
        timestamp merging live in ``core.transcription``. Raises
        :class:`ProviderError` on failure.
        """
        response = request_with_retry(
            self._session,
            "POST",
            LISTEN_URL,
            headers={**self._auth_headers(), "Content-Type": "audio/wav"},
            params=build_listen_params(language),
            data=wav_bytes,
            timeout=timeout,
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
        )
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(
                "bad_response", "Unexpected response from Deepgram"
            ) from exc


def default_presets() -> dict:
    """Deepgram section of the provider presets."""
    return load_presets()["deepgram"]
