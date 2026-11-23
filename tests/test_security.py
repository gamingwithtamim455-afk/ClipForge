"""Security sweep tests: no secret in code, keys never in logs,
diagnostics export scrubs credential-like values.

The repo-wide grep (run by hand during the hardening pass) found no
literal key/token/password values in ``clipforge/`` — keys only ever
enter the process from the system keyring and leave it inside an
``Authorization`` header. These tests pin the runtime behaviour.
"""

from __future__ import annotations

import logging
import zipfile

from clipforge.core.updates import export_diagnostics, scrub_text
from clipforge.providers.deepgram import DeepgramClient
from clipforge.providers.deepseek import DeepSeekClient

FAKE_KEY = "sk-UNIQUEFAKEKEY1234567890abcdef"


class _FakeResponse:
    def __init__(self, status_code: int, json_data=None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.text = text
        self.headers = {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


class _FakeSession:
    """Always answers 401 — the key must still never leak anywhere."""

    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return _FakeResponse(401)


def test_key_never_appears_in_logs_or_errors(caplog) -> None:
    """Validate with a bad key: logs and messages must not contain it."""
    caplog.set_level(logging.DEBUG)
    session = _FakeSession()
    deepseek = DeepSeekClient(api_key=FAKE_KEY, session=session)
    status = deepseek.validate_key()
    assert status.connected is False
    assert FAKE_KEY not in status.message
    deepgram = DeepgramClient(api_key=FAKE_KEY, session=session)
    status = deepgram.validate_key()
    assert status.connected is False
    assert FAKE_KEY not in status.message
    for record in caplog.records:
        assert FAKE_KEY not in record.getMessage()
    # The key only ever leaves inside an Authorization header.
    assert session.calls[0]["headers"]["Authorization"].endswith(FAKE_KEY)


def test_diagnostics_scrubs_api_key_assignment_lines(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "clipforge.log").write_text(
        "2026-11-23 INFO start\n"
        "api_key: abcdefgh12345678\n"
        "Authorization: Token abcdefgh12345678\n"
        "Authorization: Bearer sk-abcdefgh12345678\n"
        "normal line\n"
    )
    out = export_diagnostics(log_dir, tmp_path / "diag.zip")
    with zipfile.ZipFile(out) as zf:
        text = zf.read("clipforge.log").decode()
    assert "abcdefgh12345678" not in text
    assert "sk-abcdefgh12345678" not in text
    assert "[REDACTED]" in text
    assert "normal line" in text


def test_scrub_text_keeps_prefix_and_normal_text() -> None:
    # The whole Authorization header value goes (scheme word included).
    assert scrub_text("Authorization: Bearer sk-abcdefgh12345678") == (
        "Authorization: [REDACTED]"
    )
    # Bare scheme lines (no Authorization prefix) keep the scheme word.
    assert scrub_text("Bearer abcdefgh12345678") == "Bearer [REDACTED]"
    assert scrub_text("api_key=abcdefgh12345678") == "api_key=[REDACTED]"
    assert scrub_text("nothing secret here") == "nothing secret here"
