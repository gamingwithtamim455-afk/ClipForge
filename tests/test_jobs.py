"""Background-job tests: work runs off the UI thread, signals report back."""

import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from clipforge.core.jobs import Job, run_job


@pytest.fixture(scope="module")
def qapp(app_window):
    # Share the session QApplication (conftest.app_window) — Qt allows only
    # one QApplication singleton per process.
    yield QApplication.instance()


def _pump_until(predicate, timeout_ms: int = 5000) -> bool:
    """Run the Qt event loop until ``predicate()`` or the timeout."""
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


def test_job_finished_signal_carries_result(qapp) -> None:
    seen: dict = {}

    def work(ctx):
        ctx.report(50, "halfway")
        return 42

    job = Job(work)
    job.finished.connect(lambda value: seen.setdefault("result", value))
    job.progress.connect(lambda pct, msg: seen.setdefault("progress", (pct, msg)))
    job.start()
    assert _pump_until(lambda: "result" in seen)
    assert seen["result"] == 42
    assert seen["progress"] == (50, "halfway")
    job.shutdown()


def test_job_failed_signal_carries_message(qapp) -> None:
    seen: dict = {}

    def work(ctx):
        raise ValueError("boom")

    job = Job(work)
    job.failed.connect(lambda err: seen.setdefault("error", err))
    job.start()
    assert _pump_until(lambda: "error" in seen)
    assert seen["error"] == "boom"
    job.shutdown()


def test_job_cancellation_flag(qapp) -> None:
    seen: dict = {}

    def work(ctx):
        ctx.report(0, "start")
        job.request_cancel()
        return "cancelled" if ctx.cancelled else "ran"

    job = Job(work)
    job.finished.connect(lambda value: seen.setdefault("result", value))
    job.start()
    assert _pump_until(lambda: "result" in seen)
    assert seen["result"] == "cancelled"
    job.shutdown()


def test_run_job_convenience(qapp) -> None:
    seen: dict = {}
    job = run_job(
        lambda ctx: "done",
        on_finished=lambda value: seen.setdefault("result", value),
    )
    assert _pump_until(lambda: "result" in seen)
    assert seen["result"] == "done"
    job.shutdown()
