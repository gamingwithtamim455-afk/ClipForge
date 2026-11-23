"""Background jobs: run API / transcription work off the UI thread.

All API work runs through :class:`Job` so the UI never freezes. A job runs
a plain callable in a ``QThread`` and reports back through Qt signals
(queued connections, so the slots run on the UI thread):

- ``progress(int, str)`` — percent + status message,
- ``finished(object)`` — the callable's return value,
- ``failed(str)`` — a user-facing error message (never contains a key).

The callable receives a :class:`JobContext` with ``report(percent, message)``
and a ``cancelled`` flag it should poll between chunks; the UI calls
``Job.request_cancel()`` to ask it to stop.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot


class JobContext:
    """Handle handed to the job function for progress + cancellation."""

    def __init__(self, job: "Job") -> None:
        self._job = job

    def report(self, percent: int, message: str) -> None:
        """Emit a progress update (safe to call from the worker thread)."""
        self._job.progress.emit(int(percent), message)

    @property
    def cancelled(self) -> bool:
        """True after the UI asked the job to stop."""
        return self._job._cancelled


class _Worker(QObject):
    """Runs the job function inside the worker thread."""

    done = Signal(object, bool, str)  # result, ok, error message

    def __init__(self, fn: Callable[[JobContext], Any], ctx: JobContext) -> None:
        super().__init__()
        self._fn = fn
        self._ctx = ctx

    @Slot()
    def run(self) -> None:
        try:
            result = self._fn(self._ctx)
        except Exception as exc:  # noqa: BLE001 — surfaced via `failed`
            self.done.emit(None, False, str(exc))
        else:
            self.done.emit(result, True, "")


class Job(QObject):
    """One background job: ``Job(fn).start()`` runs ``fn(ctx)`` off-thread."""

    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self, fn: Callable[[JobContext], Any], parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._fn = fn
        self._cancelled = False
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        if parent is not None:
            # A job must never outlive its owner: if the owning widget is
            # destroyed while the thread is still running, Qt destroys the
            # QThread mid-run, which aborts the whole process on Windows
            # (0xC0000409). Shut the thread down with the owner instead.
            parent.destroyed.connect(lambda *_: self.shutdown())

    def start(self) -> None:
        """Start the job in a new thread."""
        ctx = JobContext(self)
        self._worker = _Worker(self._fn, ctx)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    @Slot(object, bool, str)
    def _on_done(self, result: object, ok: bool, error: str) -> None:
        if ok:
            self.finished.emit(result)
        else:
            self.failed.emit(error)
        if self._thread is not None:
            self._thread.quit()

    @Slot()
    def _on_thread_finished(self) -> None:
        """The thread has stopped: release it and the worker for deletion."""
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def request_cancel(self) -> None:
        """Ask the running function to stop at its next checkpoint."""
        self._cancelled = True

    def shutdown(self, timeout_ms: int = 2000) -> None:
        """Stop the thread (best effort) — call when the owner is closing."""
        self.request_cancel()
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    self._thread.quit()
                    self._thread.wait(timeout_ms)
            except RuntimeError:
                pass  # thread object already deleted — nothing to stop


def run_job(
    fn: Callable[[JobContext], Any],
    *,
    on_finished: Callable[[object], None] | None = None,
    on_failed: Callable[[str], None] | None = None,
    on_progress: Callable[[int, str], None] | None = None,
    parent: QObject | None = None,
) -> Job:
    """Convenience: build a :class:`Job`, wire callbacks, start it."""
    job = Job(fn, parent)
    if on_finished is not None:
        job.finished.connect(on_finished)
    if on_failed is not None:
        job.failed.connect(on_failed)
    if on_progress is not None:
        job.progress.connect(on_progress)
    job.start()
    return job
