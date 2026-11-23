"""Shared HTTP helper for the provider adapters.

One place for retry + error mapping so DeepSeek and Deepgram behave the
same way (docs/provider-research.md §1.4, §2.6):

- Retry with exponential backoff on 429 / 5xx / timeouts only.
- A ``ConnectionError`` means no internet: fail fast with a friendly
  message, never retry (retrying a dead connection helps nobody).
- 401/402/403/404/400 fail immediately with an exact, user-facing reason.
- Error messages never contain an API key (keys are only ever sent in the
  ``Authorization`` header, never in a message, URL or log line).

``session`` is injectable so tests can pass a fake session; production code
passes a real ``requests.Session``.
"""

from __future__ import annotations

import random
import time

import requests

#: Statuses worth one more attempt (with backoff) before giving up.
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class ProviderError(Exception):
    """Typed provider failure with a user-facing message.

    ``kind`` is a machine-readable tag (``invalid_key``, ``no_balance``,
    ``not_found``, ``forbidden``, ``rate_limit``, ``server``, ``timeout``,
    ``no_internet``, ``bad_request``, ``bad_response``, ``unknown``);
    ``message`` is what the status line shows. ``status`` is the HTTP
    status when one was received.
    """

    def __init__(self, kind: str, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status


def status_message(status: int) -> tuple[str, str]:
    """Map an HTTP status to ``(kind, user-facing message)``."""
    if status == 400:
        return "bad_request", "Request rejected by the server (400)"
    if status == 401:
        return "invalid_key", "Invalid API key (401)"
    if status == 402:
        return "no_balance", "No balance — top up your account (402)"
    if status == 403:
        return "forbidden", "Key lacks permission for this request (403)"
    if status == 404:
        return "not_found", "Wrong base URL or model (404)"
    if status == 429:
        return "rate_limit", "Rate limit — try again shortly (429)"
    if 500 <= status < 600:
        return "server", f"Server error ({status}) — try again shortly"
    return "unknown", f"Unexpected response ({status})"


def request_with_retry(
    session,
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    sleep=time.sleep,
    **kwargs,
) -> requests.Response:
    """One HTTP request with exponential backoff on 429/5xx/timeouts.

    Returns the response for any 2xx/3xx status. Raises :class:`ProviderError`
    for anything else (or for a transport failure) after retries are spent.
    ``backoff_base`` is the first delay in seconds; each retry doubles it
    (plus a small jitter). Tests pass a tiny ``backoff_base`` and a fake
    ``sleep`` so retries cost nothing.
    """
    last_error: ProviderError | None = None
    for attempt in range(max_retries + 1):
        try:
            response = session.request(method, url, **kwargs)
        except requests.exceptions.ConnectionError as exc:
            # No route to the server at all — friendly offline error, no retry.
            raise ProviderError(
                "no_internet", "No internet connection — check your network"
            ) from exc
        except requests.exceptions.Timeout as exc:
            last_error = ProviderError("timeout", "Request timed out — try again")
        else:
            status = response.status_code
            if status in RETRYABLE_STATUSES:
                kind, message = status_message(status)
                last_error = ProviderError(kind, message, status=status)
            elif status >= 400:
                kind, message = status_message(status)
                raise ProviderError(kind, message, status=status)
            else:
                return response
        if attempt < max_retries:
            delay = backoff_base * (2**attempt) + random.uniform(0, 0.25)
            sleep(delay)
    # Retries spent (429/5xx/timeout every time).
    assert last_error is not None  # guaranteed: the loop always sets one
    raise last_error
