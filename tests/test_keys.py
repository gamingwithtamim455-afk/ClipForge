"""Key storage + validation tests (keyring mocked — no real backend here)."""

import pytest

from clipforge.core import keys
from clipforge.providers.deepseek import DeepSeekStatus
from conftest import FakeResponse, FakeSession


class FakeKeyring:
    """In-memory keyring backend."""

    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}

    def set_password(self, service, username, password):
        self.store[(service, username)] = password

    def get_password(self, service, username):
        return self.store.get((service, username))

    def delete_password(self, service, username):
        self.store.pop((service, username), None)


@pytest.fixture
def fake_keyring(monkeypatch):
    backend = FakeKeyring()
    monkeypatch.setattr(keys, "keyring", backend)
    return backend


def test_save_get_delete_roundtrip(fake_keyring) -> None:
    assert keys.get_key("deepseek") is None
    keys.save_key("deepseek", "sk-secret")
    assert keys.get_key("deepseek") == "sk-secret"
    assert keys.has_key("deepseek")
    keys.delete_key("deepseek")
    assert keys.get_key("deepseek") is None
    assert not keys.has_key("deepseek")


def test_save_unknown_provider_rejected(fake_keyring) -> None:
    with pytest.raises(ValueError):
        keys.save_key("openai", "sk-x")


def test_keyring_unavailable_raises_on_save(monkeypatch) -> None:
    class BrokenKeyring:
        def set_password(self, *args):
            raise RuntimeError("no backend")

        def get_password(self, *args):
            raise RuntimeError("no backend")

        def delete_password(self, *args):
            raise RuntimeError("no backend")

    monkeypatch.setattr(keys, "keyring", BrokenKeyring())
    with pytest.raises(keys.KeyringUnavailable):
        keys.save_key("deepseek", "sk-x")
    # Reads degrade to None instead of crashing.
    assert keys.get_key("deepseek") is None
    assert not keys.has_key("deepseek")


def test_validate_deepseek_wrong_key_message() -> None:
    session = FakeSession({("GET", "/models"): FakeResponse(401)})
    status = keys.validate_deepseek("https://api.deepseek.com", "sk-bad", session)
    assert isinstance(status, DeepSeekStatus)
    assert not status.connected
    assert status.message == "Invalid API key (401)"
    assert "sk-bad" not in status.message


def test_validate_deepseek_normalizes_base_url() -> None:
    session = FakeSession(
        {("GET", "/models"): FakeResponse(200, {"data": [{"id": "deepseek-flash"}]})}
    )
    status = keys.validate_deepseek("https://api.deepseek.com/v1/", "sk-ok", session)
    assert status.connected
    assert session.calls[0]["url"] == "https://api.deepseek.com/models"


def test_validate_deepgram_connected() -> None:
    session = FakeSession({("GET", "/v1/projects"): FakeResponse(200)})
    status = keys.validate_deepgram("dg-ok", session)
    assert status.connected
    assert status.message == "Connected"


def test_validate_deepgram_rate_limit_message() -> None:
    session = FakeSession({("GET", "/v1/projects"): FakeResponse(429)})
    status = keys.validate_deepgram("dg-ok", session)
    assert not status.connected
    assert status.message == "Rate limit — try again shortly (429)"
