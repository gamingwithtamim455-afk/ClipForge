"""Pytest configuration: force the offscreen platform so UI tests run
headless on Linux CI and local machines without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def app_window():
    """One shared QApplication + main window for the whole test session.

    Qt allows exactly one QApplication singleton per process, so UI test
    modules must share it instead of each creating their own.
    """
    from clipforge.app import create_app

    app, win = create_app()
    yield win
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
