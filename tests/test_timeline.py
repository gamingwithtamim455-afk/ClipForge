"""UI tests (offscreen): manual timeline ranges, manual-mode disabling,
Edit-step ranges, Render-step captions + Copy."""

import pytest

from clipforge.app import create_app
from clipforge.core.ranges import ClipRange
from clipforge.core.render import ClipResult, RenderJob, RenderSettings
from clipforge.ui.edit import EditPage
from clipforge.ui.render import RenderPage
from clipforge.ui.timeline import TimelineWidget


@pytest.fixture(scope="module")
def window():
    app, win = create_app()
    yield win
    app.quit()


@pytest.fixture()
def timeline(window):
    widget = TimelineWidget(window)
    widget.set_duration(300.0, fps=30.0)
    return widget


# -- manual ranges -----------------------------------------------------
def test_add_range_creates_row(timeline) -> None:
    assert timeline.add_range()
    assert len(timeline.manual_ranges()) == 1
    assert timeline.has_manual_ranges()
    assert len(timeline._rows) == 1
    rng = timeline.manual_ranges()[0]
    assert rng.start == 0.0 and rng.end == 30.0


def test_add_range_requires_video() -> None:
    widget = TimelineWidget()
    assert not widget.add_range()  # duration 0
    assert widget.manual_ranges() == []


def test_add_range_max_twenty(timeline) -> None:
    for _ in range(20):
        assert timeline.add_range()
    assert not timeline.add_range()
    assert len(timeline.manual_ranges()) == 20


def test_manual_mode_signal(timeline) -> None:
    seen: list[bool] = []
    timeline.manual_mode_changed.connect(seen.append)
    timeline.add_range()
    assert seen == [True]
    timeline.clear_ranges()
    assert seen == [True, False]


def test_range_validation_shown_in_widget(timeline) -> None:
    timeline.set_ranges([ClipRange(30.0, 70.0), ClipRange(60.0, 100.0)])
    assert not timeline.validation.is_valid
    assert not timeline._error_label.isHidden()
    assert "overlaps" in timeline._error_label.text()
    timeline.set_ranges([ClipRange(30.0, 70.0)])
    assert timeline.validation.is_valid
    assert timeline._error_label.isHidden()


def test_range_long_warning_shown(timeline) -> None:
    timeline.set_ranges([ClipRange(0.0, 90.0)])
    assert timeline.validation.is_valid
    assert not timeline._warning_label.isHidden()
    assert "longer than 60s" in timeline._warning_label.text()


def test_delete_range(timeline) -> None:
    timeline.set_ranges([ClipRange(0.0, 30.0), ClipRange(60.0, 90.0)])
    timeline.delete_range(0)
    assert len(timeline.manual_ranges()) == 1
    assert timeline.manual_ranges()[0].start == 60.0


def test_snap_time_to_frame_grid(timeline) -> None:
    assert timeline.snap_time(1.0) == 1.0
    assert timeline.snap_time(1.01) == 1.0  # 30 fps grid: 0.033 steps
    assert timeline.snap_time(1.02) == 1.03 or timeline.snap_time(1.02) == 1.0


def test_set_ranges_replaces(timeline) -> None:
    timeline.set_ranges([ClipRange(10.0, 20.0)])
    assert len(timeline._rows) == 1
    timeline.set_ranges([])
    assert timeline._rows == []
    assert not timeline.has_manual_ranges()


# -- manual mode disables AI controls ----------------------------------
def test_manual_mode_disables_clip_controls(window) -> None:
    window._open_project(window._store.projects[0])
    window._wizard.go_to_step(1)
    viral = window._wizard._viral
    viral.timeline.set_duration(300.0, fps=30.0)
    assert viral._clip_length.isEnabled()
    assert viral._clips.isEnabled()
    viral.timeline.add_range()
    assert not viral._clip_length.isEnabled()
    assert not viral._clips.isEnabled()
    assert viral.has_manual_ranges
    assert len(viral.manual_ranges) == 1
    assert "Manual ranges" in viral._status.text()
    viral.timeline.clear_ranges()
    assert viral._clip_length.isEnabled()
    assert viral._clips.isEnabled()
    window._show_home()


# -- Edit step ---------------------------------------------------------
def test_edit_page_loads_clips_with_qa_flag(window) -> None:
    page = EditPage(window)
    page.set_clips(
        [ClipRange(0.0, 30.0, label="Clip 1"), ClipRange(40.0, 70.0, label="Clip 2")],
        qa_flags={1: True},
    )
    assert page._clip_list.count() == 2
    assert "review framing" in page._clip_list.item(1).text()
    assert page._preset.isEnabled()
    assert len(page.timeline.manual_ranges()) == 2


# -- Render step -------------------------------------------------------
def _result(index: int, *, label: bool = True) -> ClipResult:
    return ClipResult(
        index=index,
        video_path=f"/tmp/clip{index:02d}.mp4",
        label_path=f"/tmp/{index:02d}_label.txt" if label else "",
        duration=30.0,
        caption="Great clip!" if label else "",
        hashtags=["#a", "#b", "#c", "#d"] if label else [],
        label_written=label,
    )


def test_render_page_set_results_and_copy(window) -> None:
    from PySide6.QtGui import QGuiApplication

    page = RenderPage(window)
    page.set_results([_result(1), _result(2, label=False)])
    assert page._clip_list.count() == 2
    page._clip_list.setCurrentRow(0)
    assert "Great clip!" in page._caption_view.toPlainText()
    assert "#a #b #c #d" in page._caption_view.toPlainText()
    assert page._copy_btn.isEnabled()
    page._on_copy_clicked()
    assert QGuiApplication.clipboard().text() == page._caption_view.toPlainText()
    page._clip_list.setCurrentRow(1)
    assert not page._copy_btn.isEnabled()  # no label on clip 2


def test_render_page_pending_clips_enable_button(window) -> None:
    page = RenderPage(window)
    assert not page._render_btn.isEnabled()
    page.set_pending_clips([
        RenderJob("video.mp4", 0.0, 30.0, index=1, settings=RenderSettings())
    ])
    assert page._render_btn.isEnabled()
    assert "1080 × 1920" in page._summary_rows["Resolution"].text()
