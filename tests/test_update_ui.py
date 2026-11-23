"""Tests: one-click in-app update through the Settings > Updates tab.

End-to-end against a fake local update feed (no real GitHub repo exists
yet): Check for updates -> "Download and install it now?" -> download ->
checksum -> backup -> install, all through the real dialog code.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
from pathlib import Path

import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QMessageBox

import clipforge.core.paths as paths_module
import clipforge.ui.settings_dialog as dialog_module
import clipforge.core.updates as updates_module
from clipforge.core.updates import UpdateCheck, UpdateInfo
from clipforge.ui.settings_dialog import SettingsDialog

INSTALLER_BYTES = b"fake-clipforge-installer"


class _FeedHandler(http.server.BaseHTTPRequestHandler):
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
        if self.path == "/ClipForge-Setup.exe":
            self._send(200, INSTALLER_BYTES, "application/octet-stream")
        elif self.path == "/ClipForge-Setup.exe.sha256":
            digest = hashlib.sha256(INSTALLER_BYTES).hexdigest()
            self._send(200, f"{digest}  ClipForge-Setup.exe\n".encode(), "text/plain")
        else:
            self._send(404, b"not found")


@pytest.fixture()
def fake_feed():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FeedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _pump_until(predicate, timeout_ms: int = 10000) -> bool:
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    checker = QTimer()
    checker.setInterval(10)
    checker.timeout.connect(lambda: loop.quit() if predicate() else None)
    checker.start()
    loop.exec()
    checker.stop()
    return predicate()


def test_updates_tab_one_click_downloads_and_verifies(
    app_window, fake_feed, tmp_path, monkeypatch
) -> None:
    port = fake_feed.server_address[1]
    host = f"http://127.0.0.1:{port}"
    info = UpdateInfo(
        version="0.2.0", tag="v0.2.0",
        installer_url=f"{host}/ClipForge-Setup.exe",
        installer_name="ClipForge-Setup.exe",
        checksum_url=f"{host}/ClipForge-Setup.exe.sha256",
    )
    monkeypatch.setattr(
        dialog_module, "check_for_updates",
        lambda *a, **k: UpdateCheck(True, "ClipForge 0.2.0 is available.", info),
    )
    monkeypatch.setattr(
        updates_module, "install_update", lambda path: "installer_downloaded"
    )
    monkeypatch.setattr(
        QMessageBox, "question",
        classmethod(lambda cls, *a, **k: QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(paths_module, "app_data_dir", lambda: Path(tmp_path))
    (tmp_path / "projects.json").write_text(json.dumps({"version": 2, "projects": []}))

    dialog = SettingsDialog(app_window._settings, app_window)
    dialog._on_check_updates()
    ok = _pump_until(
        lambda: "Installer downloaded:" in dialog._update_status.text()
        or "Update failed:" in dialog._update_status.text(),
    )
    status = dialog._update_status.text()
    dialog.close()
    assert ok, f"update flow did not finish; status={status!r}"
    assert status.startswith("Installer downloaded:")
    assert (tmp_path / "updates" / "ClipForge-Setup.exe").read_bytes() == INSTALLER_BYTES
    assert list((tmp_path / "backups").glob("*.zip"))
    assert json.loads((tmp_path / "projects.json").read_text())["version"] == 2

