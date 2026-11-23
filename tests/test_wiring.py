"""Tests: MainWindow wiring — Start clipping -> Edit -> Render, Cancel.

The pipeline itself is faked here (its behaviour is covered by
test_pipeline.py); these tests prove the app connects the pieces: the
wizard advances, the Edit step lists the clips, keep/drop rebuilds the
render list, failures show a friendly message, and Cancel stops a run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

import clipforge.app as app_module
from clipforge.core.pipeline import PipelineError, PipelineResult
from clipforge.core.projects import Project
from clipforge.core.ranges import ClipRange
from clipforge.core.render import RenderJob
from clipforge.core.transcription import Transcript, Word


@pytest.fixture(scope="module")
def qapp(app_window):
    yield QApplication.instance()


def _pump_until(predicate, timeout_ms: int = 8000) -> bool:
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


def _make_video(tmp_path: Path, *, duration: float = 20.0) -> str:
    from clipforge.core.ffmpeg import find_ffmpeg

    path = str(tmp_path / "source.mp4")
    subprocess.run(
        [
            find_ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i",
            f"testsrc2=size=640x360:rate=30:duration={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", path,
        ],
        check=True,
    )
    return path


def _canned_result(video: str, tmp_path: Path) -> PipelineResult:
    words = [Word("hello", 2.5, 3.0), Word("world", 3.1, 3.6),
             Word("viral", 10.5, 11.0), Word("clips", 11.1, 11.6)]
    ranges = [
        ClipRange(start=2.0, end=6.0, source="manual", label="Clip 1"),
        ClipRange(start=10.0, end=14.0, source="manual", label="Clip 2"),
    ]
    return PipelineResult(
        video_path=video,
        duration=20.0,
        width=640,
        height=360,
        fps=30.0,
        transcript=Transcript(language="en", words=words),
        ranges=ranges,
        crop_paths={},
        qa_flags={0: False, 1: False},
        render_jobs=[
            RenderJob(video_path=video, start=2.0, end=6.0, index=1,
                      output_path=str(tmp_path / "01_clip.mp4")),
            RenderJob(video_path=video, start=10.0, end=14.0, index=2,
                      output_path=str(tmp_path / "02_clip.mp4")),
        ],
        output_dir=tmp_path,
        used_manual_ranges=True,
    )


def _open_wizard_with_video(window, tmp_path) -> str:
    """Open a project in the wizard and load a real (probed) video."""
    video = _make_video(tmp_path)
    project = Project(name="wiring-test", source_path=video)
    window._open_project(project)
    window._wizard._select_video.set_video(video)
    assert window._wizard.video_path == video
    assert window._wizard.video_duration > 0
    # Pick a language so Start clipping enables.
    window._wizard._viral._language.setCurrentIndex(1)  # Bengali
    assert window._wizard._start_btn.isEnabled()
    return video


def test_open_project_loads_source_and_restores_clips(qapp, app_window, tmp_path) -> None:
    """Project click -> in-app workspace (regression: spec item 12).

    Opening a project must load its source video and land on the Viral
    Clips step with the saved clips on the timeline — never on the
    Select Video step / a file browser.
    """
    window = app_window
    video = _make_video(tmp_path)
    project = Project(
        name="saved-project",
        source_path=video,
        clips=[{"start": 2.0, "end": 6.0, "source": "manual", "label": "Clip 1"}],
        status="clips_ready",
    )
    window._open_project(project)
    assert window._wizard.video_path == video
    assert window._wizard.video_duration > 0
    assert window._wizard._step == 1  # Viral Clips, not Select Video
    assert window._wizard._viral.timeline.ai_clips == [(2.0, 6.0)]
    window._show_home()


def test_open_project_with_missing_source_stays_on_select_step(qapp, app_window) -> None:
    """A project whose file is gone falls back to Select Video (only then)."""
    window = app_window
    project = Project(name="gone", source_path="/nonexistent/gone.mp4")
    window._open_project(project)
    assert window._wizard.video_path == ""
    assert window._wizard._step == 0
    window._show_home()


def test_start_clipping_happy_path(qapp, app_window, tmp_path, monkeypatch) -> None:
    window = app_window
    video = _open_wizard_with_video(window, tmp_path)
    monkeypatch.setattr(
        app_module, "run_pipeline",
        lambda config, **kw: _canned_result(video, tmp_path),
    )
    window._wizard._start_btn.click()
    assert _pump_until(lambda: window._wizard._step == 2, timeout_ms=8000)
    assert len(window._wizard._edit.kept_ranges()) == 2
    assert len(window._wizard._render._jobs) == 2
    project = window._wizard.project
    assert project is not None
    assert project.status == "clips_ready"
    assert len(project.clips) == 2
    window._show_home()


def test_edit_keep_drop_rebuilds_render_list(qapp, app_window, tmp_path, monkeypatch) -> None:
    window = app_window
    video = _open_wizard_with_video(window, tmp_path)
    monkeypatch.setattr(
        app_module, "run_pipeline",
        lambda config, **kw: _canned_result(video, tmp_path),
    )
    window._wizard._start_btn.click()
    assert _pump_until(lambda: window._wizard._step == 2, timeout_ms=8000)
    assert len(window._wizard._render._jobs) == 2
    # Uncheck the first clip in the Edit list -> it drops from rendering.
    from PySide6.QtCore import Qt

    item = window._wizard._edit._clip_list.item(0)
    item.setCheckState(Qt.CheckState.Unchecked)
    assert _pump_until(lambda: len(window._wizard._render._jobs) == 1,
                       timeout_ms=5000)
    assert window._wizard._render._jobs[0].start == 10.0
    window._show_home()


def test_start_clipping_failure_shows_friendly_message(
    qapp, app_window, tmp_path, monkeypatch
) -> None:
    window = app_window
    _open_wizard_with_video(window, tmp_path)
    seen: dict = {}
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda *args, **kwargs: seen.setdefault("shown", args[-1] if args else ""),
    )
    monkeypatch.setattr(
        app_module, "run_pipeline",
        lambda config, **kw: (_ for _ in ()).throw(
            PipelineError("Deepgram API key is missing — add it in Settings.")
        ),
    )
    window._wizard._start_btn.click()
    assert _pump_until(lambda: window._pipeline_job is None, timeout_ms=8000)
    assert "Settings" in seen.get("shown", "")
    assert window._wizard._start_btn.isEnabled()  # idle again, can retry
    window._show_home()


def test_cancel_stops_a_running_pipeline(qapp, app_window, tmp_path, monkeypatch) -> None:
    window = app_window
    _open_wizard_with_video(window, tmp_path)
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: None,
    )

    def _slow_pipeline(config, **kw):
        import time

        cancel = kw.get("cancel") or (lambda: False)
        for _ in range(100):
            if cancel():
                raise PipelineError("Cancelled.")
            time.sleep(0.05)
        raise PipelineError("should have been cancelled")

    monkeypatch.setattr(app_module, "run_pipeline", _slow_pipeline)
    window._wizard._start_btn.click()
    assert _pump_until(lambda: window._wizard._cancel_btn.isEnabled(),
                       timeout_ms=5000)
    window._wizard._cancel_btn.click()
    assert _pump_until(lambda: window._pipeline_job is None, timeout_ms=8000)
    assert window._wizard._start_btn.isEnabled()
    window._show_home()
