"""Retry + error-mapping tests for providers._http."""

import requests

from clipforge.providers._http import ProviderError, request_with_retry, status_message
from conftest import FakeResponse, FakeSession


def test_status_messages_exact() -> None:
    assert status_message(401) == ("invalid_key", "Invalid API key (401)")
    assert status_message(402) == ("no_balance", "No balance — top up your account (402)")
    assert status_message(404) == ("not_found", "Wrong base URL or model (404)")
    assert status_message(429)[0] == "rate_limit"
    assert status_message(503)[0] == "server"


def test_retry_on_429_then_success() -> None:
    session = FakeSession(
        {("GET", "/models"): [FakeResponse(429), FakeResponse(200, {"data": []})]}
    )
    response = request_with_retry(
        session, "GET", "https://api.deepseek.com/models",
        max_retries=3, backoff_base=0, sleep=lambda s: None,
    )
    assert response.status_code == 200
    assert len(session.calls) == 2


def test_retry_on_500_then_success() -> None:
    session = FakeSession(
        {("GET", "/x"): [FakeResponse(500), FakeResponse(502), FakeResponse(200)]}
    )
    response = request_with_retry(
        session, "GET", "https://example.com/x",
        max_retries=3, backoff_base=0, sleep=lambda s: None,
    )
    assert response.status_code == 200
    assert len(session.calls) == 3


def test_retry_exhausted_raises_last_error() -> None:
    session = FakeSession({("GET", "/x"): FakeResponse(429)})
    try:
        request_with_retry(
            session, "GET", "https://example.com/x",
            max_retries=2, backoff_base=0, sleep=lambda s: None,
        )
    except ProviderError as exc:
        assert exc.kind == "rate_limit"
    else:  # pragma: no cover
        raise AssertionError("expected ProviderError")
    assert len(session.calls) == 3  # 1 try + 2 retries


def test_no_retry_on_401() -> None:
    session = FakeSession({("GET", "/x"): FakeResponse(401)})
    try:
        request_with_retry(session, "GET", "https://example.com/x", backoff_base=0)
    except ProviderError as exc:
        assert exc.kind == "invalid_key"
    assert len(session.calls) == 1


def test_exponential_backoff_delays() -> None:
    delays: list[float] = []
    session = FakeSession({("GET", "/x"): FakeResponse(503)})
    try:
        request_with_retry(
            session, "GET", "https://example.com/x",
            max_retries=3, backoff_base=1.0, sleep=delays.append,
        )
    except ProviderError:
        pass
    # Delays double each retry (1, 2, 4) plus up to 0.25 s jitter.
    assert len(delays) == 3
    assert 1.0 <= delays[0] < 1.25
    assert 2.0 <= delays[1] < 2.25
    assert 4.0 <= delays[2] < 4.25


def test_timeout_retries_then_raises() -> None:
    class FlakySession:
        def __init__(self):
            self.calls = 0

        def request(self, *args, **kwargs):
            self.calls += 1
            raise requests.exceptions.ReadTimeout("slow")

    session = FlakySession()
    try:
        request_with_retry(
            session, "GET", "https://example.com/x",
            max_retries=2, backoff_base=0, sleep=lambda s: None,
        )
    except ProviderError as exc:
        assert exc.kind == "timeout"
    assert session.calls == 3


def test_connection_error_is_no_internet_and_not_retried() -> None:
    class DeadSession:
        def __init__(self):
            self.calls = 0

        def request(self, *args, **kwargs):
            self.calls += 1
            raise requests.exceptions.ConnectionError("dns fail")

    session = DeadSession()
    try:
        request_with_retry(
            session, "GET", "https://example.com/x",
            max_retries=3, backoff_base=0, sleep=lambda s: None,
        )
    except ProviderError as exc:
        assert exc.kind == "no_internet"
        assert exc.message == "No internet connection — check your network"
    assert session.calls == 1
