"""Tests: updates (GitHub Releases feed), diagnostics, weekly backups."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

from clipforge.core.backups import maybe_weekly_backup, prune_backups
from clipforge.core.updates import (
    UpdateCheck,
    backup_app_state,
    check_for_updates,
    export_diagnostics,
    restore_app_state,
    scrub_text,
    verify_sha256,
)


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.text = text
        self.headers = {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


class _FakeSession:
    """Session stub: scripted GET responses by URL substring."""

    def __init__(self, responses):
        self._responses = responses

    def get(self, url, **kwargs):
        for needle, response in self._responses.items():
            if needle in url:
                if isinstance(response, Exception):
                    raise response
                return response
        return _FakeResponse(404)


def _release_payload(*, tag="v0.2.0", assets=None):
    return {
        "tag_name": tag,
        "body": "Bug fixes.",
        "assets": assets if assets is not None else [
            {"name": "ClipForge-Setup.exe",
             "browser_download_url": "https://example.com/ClipForge-Setup.exe"},
            {"name": "ClipForge-Setup.exe.sha256",
             "browser_download_url": "https://example.com/ClipForge-Setup.exe.sha256"},
        ],
    }


def test_check_for_updates_finds_newer_release() -> None:
    session = _FakeSession({
        "releases/latest": _FakeResponse(200, _release_payload()),
    })
    check = check_for_updates(repo="clipforge/clipforge", current_version="0.1.0",
                              session=session)
    assert check.available is True
    assert check.info is not None
    assert check.info.version == "0.2.0"
    assert check.info.installer_url == "https://example.com/ClipForge-Setup.exe"
    assert check.info.checksum_url.endswith(".sha256")


def test_check_for_updates_up_to_date() -> None:
    session = _FakeSession({
        "releases/latest": _FakeResponse(200, _release_payload(tag="v0.1.0")),
    })
    check = check_for_updates(current_version="0.1.0", session=session)
    assert check.available is False
    assert "up to date" in check.message


def test_check_for_updates_offline_never_crashes() -> None:
    session = _FakeSession({
        "releases/latest": requests.exceptions.ConnectionError("boom"),
    })
    check = check_for_updates(session=session)
    assert check.available is None
    assert "internet" in check.message


def test_check_for_updates_no_releases_yet() -> None:
    session = _FakeSession({"releases/latest": _FakeResponse(404)})
    check = check_for_updates(session=session)
    assert check.available is None
    assert "No releases" in check.message


def test_check_for_updates_no_installer_attached() -> None:
    session = _FakeSession({
        "releases/latest": _FakeResponse(200, _release_payload(assets=[])),
    })
    check = check_for_updates(current_version="0.1.0", session=session)
    assert check.available is True
    assert check.info is not None and check.info.installer_url == ""
    assert "no Windows installer" in check.message


def test_verify_sha256(tmp_path) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(b"clipforge")
    digest = hashlib.sha256(b"clipforge").hexdigest()
    assert verify_sha256(path, digest) is True
    assert verify_sha256(path, digest.upper()) is True
    assert verify_sha256(path, "0" * 64) is False


def test_backup_and_restore_roundtrip(tmp_path) -> None:
    projects = tmp_path / "projects.json"
    projects.write_text(json.dumps({"version": 2, "projects": []}))
    snapshot = {"appearance": "dark"}
    backup = backup_app_state(projects, snapshot, tmp_path / "backups")
    assert backup.exists()
    projects.write_text("corrupted")
    restored = restore_app_state(backup, projects)
    assert json.loads(projects.read_text())["version"] == 2
    assert restored == snapshot


def test_diagnostics_export_scrubs_keys(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "clipforge.log").write_text(
        "2026-11-23 INFO start\n"
        "key was sk-abc123SECRETKEYHERE in a line\n"
        "Authorization: Bearer sk-xyz987654321\n"
        "normal line\n"
    )
    out = export_diagnostics(log_dir, tmp_path / "diag.zip")
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        text = zf.read("clipforge.log").decode()
        assert "sk-abc123SECRETKEYHERE" not in text
        assert "sk-xyz987654321" not in text
        assert "[REDACTED]" in text
        assert "normal line" in text
        assert "system-info.txt" in zf.namelist()


def test_scrub_text_patterns() -> None:
    assert "sk-abc" not in scrub_text("token sk-abcdefghij done")
    assert scrub_text("nothing secret here") == "nothing secret here"


# ---------------------------------------------------------------------------
# Weekly backups
# ---------------------------------------------------------------------------

def _write_projects(tmp_path, name="projects.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"version": 2, "projects": []}))
    return path


def test_weekly_backup_creates_then_skips(tmp_path) -> None:
    projects = _write_projects(tmp_path)
    backups = tmp_path / "backups"
    first = maybe_weekly_backup(projects, {"a": 1}, backups)
    assert first is not None and first.exists()
    second = maybe_weekly_backup(projects, {"a": 1}, backups)
    assert second is None  # a backup from today already exists


def test_weekly_backup_after_a_week(tmp_path) -> None:
    projects = _write_projects(tmp_path)
    backups = tmp_path / "backups"
    now = datetime.now(timezone.utc)
    first = maybe_weekly_backup(projects, {}, backups, now=now)
    assert first is not None
    second = maybe_weekly_backup(projects, {}, backups, now=now + timedelta(days=8))
    assert second is not None and second != first


def test_prune_backups_keeps_newest_four(tmp_path) -> None:
    backups = tmp_path / "backups"
    backups.mkdir()
    for i in range(6):
        (backups / f"clipforge-backup-2026-01-0{i}-0000.zip").write_bytes(b"x")
    removed = prune_backups(backups, keep=4)
    assert removed == 2
    assert len(list(backups.glob("*.zip"))) == 4


# ---------------------------------------------------------------------------
# End-to-end: fake local update feed (no real GitHub repo exists yet)
# ---------------------------------------------------------------------------

import http.server
import threading

from clipforge.core.updates import (
    UpdateInfo,
    apply_update,
    check_for_updates,
)


class _FeedHandler(http.server.BaseHTTPRequestHandler):
    """Serves a fake release feed + installer + checksum on localhost."""

    installer_bytes = b"fake-clipforge-installer"
    serve_bad_checksum = False

    def log_message(self, *args):  # silence the test server
        pass

    def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server override
        host = f"http://127.0.0.1:{self.server.server_address[1]}"
        if self.path == "/releases/latest":
            payload = {
                "tag_name": "v0.2.0",
                "body": "Fake release.",
                "assets": [
                    {"name": "ClipForge-Setup.exe",
                     "browser_download_url": f"{host}/ClipForge-Setup.exe"},
                    {"name": "ClipForge-Setup.exe.sha256",
                     "browser_download_url": f"{host}/ClipForge-Setup.exe.sha256"},
                ],
            }
            self._send(200, json.dumps(payload).encode())
        elif self.path == "/ClipForge-Setup.exe":
            self._send(200, self.installer_bytes, "application/octet-stream")
        elif self.path == "/ClipForge-Setup.exe.sha256":
            digest = hashlib.sha256(self.installer_bytes).hexdigest()
            if self.serve_bad_checksum:
                digest = "0" * 64
            self._send(200, f"{digest}  ClipForge-Setup.exe\n".encode(), "text/plain")
        else:
            self._send(404, b"not found")


@pytest.fixture()
def fake_feed(monkeypatch):
    """Local HTTP update feed; ``check_for_updates`` reads from it."""
    import clipforge.core.updates as updates_module

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FeedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    monkeypatch.setattr(
        updates_module, "load_presets",
        lambda: {"updates": {
            "releases_api": f"http://127.0.0.1:{port}/releases/latest",
            "repo": "test/test",
        }},
    )
    yield server
    server.shutdown()
    server.server_close()


def _fake_info(fake_feed) -> UpdateInfo:
    port = fake_feed.server_address[1]
    host = f"http://127.0.0.1:{port}"
    digest = hashlib.sha256(_FeedHandler.installer_bytes).hexdigest()
    return UpdateInfo(
        version="0.2.0", tag="v0.2.0",
        installer_url=f"{host}/ClipForge-Setup.exe",
        installer_name="ClipForge-Setup.exe",
        checksum_url=f"{host}/ClipForge-Setup.exe.sha256",
        checksum=digest,
    )


def test_check_for_updates_against_fake_local_feed(fake_feed) -> None:
    check = check_for_updates(repo="test/test", current_version="0.1.0")
    assert check.available is True
    assert check.info is not None and check.info.version == "0.2.0"
    assert check.info.installer_url.endswith("ClipForge-Setup.exe")
    assert check.info.checksum_url.endswith(".sha256")


def test_apply_update_happy_path(fake_feed, tmp_path) -> None:
    projects = tmp_path / "projects.json"
    projects.write_text(json.dumps({"version": 2, "projects": []}))
    result = apply_update(
        _fake_info(fake_feed), app_dir=tmp_path,
        settings_snapshot={"appearance": "dark"}, projects_path=projects,
        install_fn=lambda path: "installer_downloaded",  # stub: never launch the fake .exe
    )
    assert result.checksum_verified is True
    assert result.outcome == "installer_downloaded"
    assert result.installer_path.read_bytes() == _FeedHandler.installer_bytes
    assert result.backup_path.exists()
    assert json.loads(projects.read_text())["version"] == 2
    with zipfile.ZipFile(result.backup_path) as zf:
        assert json.loads(zf.read("settings.json")) == {"appearance": "dark"}


def test_apply_update_bad_checksum_refuses(fake_feed, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_FeedHandler, "serve_bad_checksum", True)
    projects = tmp_path / "projects.json"
    projects.write_text(json.dumps({"version": 2, "projects": []}))
    with pytest.raises(RuntimeError, match="checksum"):
        apply_update(
            _fake_info(fake_feed), app_dir=tmp_path,
            settings_snapshot={}, projects_path=projects,
        )
    # Bad file deleted, nothing backed up, nothing installed.
    assert not (tmp_path / "updates" / "ClipForge-Setup.exe").exists()
    assert not (tmp_path / "backups").exists()
    assert json.loads(projects.read_text())["version"] == 2


def test_apply_update_install_failure_rolls_back(fake_feed, tmp_path) -> None:
    projects = tmp_path / "projects.json"
    original = json.dumps({"version": 2, "projects": ["keep-me"]})
    projects.write_text(original)

    def _failing_install(path):
        projects.write_text("corrupted by failed installer")
        raise RuntimeError("installer crashed")

    with pytest.raises(RuntimeError, match="previous version restored"):
        apply_update(
            _fake_info(fake_feed), app_dir=tmp_path,
            settings_snapshot={}, projects_path=projects,
            install_fn=_failing_install,
        )
    assert projects.read_text() == original

