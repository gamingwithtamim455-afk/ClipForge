"""DeepSeek adapter (OpenAI-compatible chat completions).

Rules (docs/provider-research.md §2, verified 2026-11-22):
- Default model ``deepseek-flash``; second option ``deepseek-v4-pro``.
  ``deepseek-v4-flash`` is a retired legacy alias — never the default.
  ``deepseek-chat`` / ``deepseek-reasoner`` are retired.
- Base URL has NO ``/v1`` segment; normalize by stripping a trailing ``/``
  and a trailing ``/v1``.
- Key validation + model dropdown source: ``GET {base}/models`` with
  ``Authorization: Bearer <key>``.
- Chat: ``POST {base}/chat/completions``. Thinking is enabled by default
  (``reasoning_effort: "high"``) for clip discovery; disabled for fast
  per-clip caption generation (see ``presets.json``).
- Error mapping: 401 invalid key, 402 no balance, 404 wrong base URL/model,
  429 rate limit, 5xx/timeout retry with exponential backoff.

The API key is only ever sent in the ``Authorization`` header and is never
logged, never included in an exception message, and never returned by any
method here. Pass a fake ``session`` in tests to avoid real network calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import requests

from . import load_presets
from ._http import ProviderError, request_with_retry

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
FALLBACK_MODEL = "deepseek-v4-pro"

#: Legacy aliases the server still maps; accepted, never presented as current.
LEGACY_ALIASES = {"deepseek-v4-flash": "deepseek-flash"}


def is_deepseek_host(base_url: str) -> bool:
    """True when the base URL points at DeepSeek's own API host."""
    return "deepseek.com" in base_url


def normalize_base_url(url: str) -> str:
    """Normalize an AI provider base URL.

    Trim whitespace and strip a trailing ``/``. A trailing ``/v1`` is
    stripped ONLY for DeepSeek (its paths have no ``/v1`` segment) —
    OpenAI's ``https://api.openai.com/v1`` and Gemini's compatible
    endpoint REQUIRE their versioned path, so stripping it there broke
    every request with a 404.

    >>> normalize_base_url("https://api.deepseek.com")
    'https://api.deepseek.com'
    >>> normalize_base_url("https://api.deepseek.com/")
    'https://api.deepseek.com'
    >>> normalize_base_url("https://api.deepseek.com/v1")
    'https://api.deepseek.com'
    >>> normalize_base_url(" https://api.deepseek.com/v1/ ")
    'https://api.deepseek.com'
    >>> normalize_base_url("https://api.openai.com/v1")
    'https://api.openai.com/v1'
    >>> normalize_base_url("https://generativelanguage.googleapis.com/v1beta/openai/")
    'https://generativelanguage.googleapis.com/v1beta/openai'
    """
    base = url.strip().rstrip("/")
    if is_deepseek_host(base) and base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base.rstrip("/")


def resolve_model_id(model_id: str) -> str:
    """Map a legacy alias to its current model ID; pass others through."""
    return LEGACY_ALIASES.get(model_id.strip(), model_id.strip())


@dataclass
class DeepSeekStatus:
    """Result of a key validation (``GET {base}/models``)."""

    connected: bool
    message: str  # e.g. "Connected" or "Invalid API key (401)"
    models: list[str] = field(default_factory=list)


@dataclass
class DeepSeekClient:
    """OpenAI-compatible chat client for a DeepSeek-compatible endpoint."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""  # never logged; stored encrypted via keyring
    model: str = DEFAULT_MODEL
    session: object | None = None  # injectable for tests
    max_retries: int = 3
    backoff_base: float = 1.0

    def __post_init__(self) -> None:
        self.base_url = normalize_base_url(self.base_url)
        self.model = resolve_model_id(self.model)
        self._session = self.session if self.session is not None else requests.Session()
        # DeepSeek-specific request shaping (thinking params, /v1-less
        # paths) applies only to DeepSeek's own host; the same client
        # serves OpenAI / Gemini / custom OpenAI-compatible endpoints.
        self._deepseek_host = is_deepseek_host(self.base_url)

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/models"

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _thinking_body(self, thinking: bool) -> dict:
        """Thinking config from presets: discovery on/high, caption off.

        DeepSeek-only: other providers (OpenAI, Gemini) reject or
        mishandle DeepSeek's ``thinking``/``reasoning_effort`` fields,
        so for them the request body stays plain.
        """
        if not self._deepseek_host:
            return {}
        presets = load_presets()["deepseek"]["thinking"]
        return dict(presets["discovery" if thinking else "caption"])

    def list_models(self, timeout: float = 10.0) -> list[str]:
        """Model IDs from ``GET {base}/models``; raises :class:`ProviderError`."""
        response = request_with_retry(
            self._session,
            "GET",
            self.models_url,
            headers=self._auth_headers(),
            timeout=timeout,
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
        )
        try:
            data = response.json()
            return [str(item["id"]) for item in data.get("data", [])]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError(
                "bad_response", "Unexpected response from the AI provider"
            ) from exc

    def validate_key(self, timeout: float = 10.0) -> DeepSeekStatus:
        """Validate the key via ``GET {base}/models``.

        Never raises for a bad key — the failure is reported in the status
        so the Settings dialog can show the exact reason inline.
        """
        try:
            models = self.list_models(timeout=timeout)
        except ProviderError as exc:
            return DeepSeekStatus(connected=False, message=exc.message)
        return DeepSeekStatus(connected=True, message="Connected", models=models)

    def chat_completions(
        self,
        messages: list[dict],
        *,
        thinking: bool = True,
        model: str | None = None,
        timeout: float = 180.0,
    ) -> dict:
        """Call ``POST {base}/chat/completions``; returns the parsed JSON.

        ``thinking=True`` (clip discovery) sends thinking enabled at high
        effort; ``thinking=False`` (caption generation) disables it.
        Raises :class:`ProviderError` on failure.
        """
        body: dict = {
            "model": resolve_model_id(model or self.model),
            "messages": messages,
            "stream": False,
        }
        body.update(self._thinking_body(thinking))
        response = request_with_retry(
            self._session,
            "POST",
            self.chat_url,
            headers={**self._auth_headers(), "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
        )
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(
                "bad_response", "Unexpected response from the AI provider"
            ) from exc

    def chat_text(
        self,
        messages: list[dict],
        *,
        thinking: bool = True,
        model: str | None = None,
        timeout: float = 180.0,
    ) -> str:
        """Chat call returning just the assistant's text content."""
        data = self.chat_completions(
            messages, thinking=thinking, model=model, timeout=timeout
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "bad_response", "Unexpected response from the AI provider"
            ) from exc
        return content or ""


def default_presets() -> dict:
    """DeepSeek section of the provider presets."""
    return load_presets()["deepseek"]
