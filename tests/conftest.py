"""Pytest configuration: force the offscreen platform so UI tests run
headless on Linux CI and local machines without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import pytest


@pytest.fixture(scope="session")
def app_window():
    """One shared QApplication + main window for the whole test session.

    Qt allows exactly one QApplication singleton per process, so UI test
    modules must share it instead of each creating their own.
    """
    from PySide6.QtCore import QThread

    from clipforge.app import create_app

    app, win = create_app()
    yield win
    win.close()
    # Background jobs (thumbnail grabs, pipeline runs) live on QThreads
    # owned by the window's pages. If the process exits while one is
    # still running, Qt destroys the thread mid-run and aborts the
    # process on Windows (0xC0000409) — which also eats pytest's summary
    # output, hiding which tests failed. Wait for every child thread to
    # stop before quitting.
    deadline = time.time() + 30
    while time.time() < deadline:
        app.processEvents()
        running = [t for t in win.findChildren(QThread) if t.isRunning()]
        if not running:
            break
        for thread in running:
            thread.quit()
        time.sleep(0.05)
    win.deleteLater()
    app.processEvents()
    app.quit()


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code: int, json_data=None) -> None:
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}

    def json(self):
        return self._json_data


class FakeSession:
    """Fake ``requests.Session``: scripted responses + a call log.

    ``responses`` maps ``(method, url_substring)`` to a ``FakeResponse`` or
    a list of them (popped in order, so retries can be scripted). Any
    request with no scripted response returns a 404.
    """

    def __init__(self, responses: dict) -> None:
        self._responses = responses
        self.calls: list[dict] = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        for (resp_method, needle), response in self._responses.items():
            if resp_method == method and needle in url:
                if isinstance(response, list):
                    if response:
                        return response.pop(0)
                    return FakeResponse(404)
                return response
        return FakeResponse(404)
