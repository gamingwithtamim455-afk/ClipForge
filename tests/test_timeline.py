"""UI tests (offscreen): manual timeline ranges, manual-mode disabling,
Edit-step ranges, Render-step captions + Copy."""

import pytest

from clipforge.app import create_app
from clipforge.core.ranges import ClipRange
from clipforge.core.render import ClipResult, RenderJob, RenderSettings
from clipforge.ui.edit import EditPage
from clipforge.ui.render import RenderPage
from clipforge.ui.timeline import TimelineWidget
from clipforge.ui.viral_clips import ViralClipsPage


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
    if not window._store.projects:
        # A fresh store starts empty (fake sample seeding was removed);
        # this test creates its own project instead of relying on one.
        from clipforge.core.projects import Project

        window._store.add(Project(name="Timeline test", source_path="video.mp4"))
    window._open_project(window._store.projects[0])
    window._wizard.go_to_step(1)
    viral = window._wizard._viral
    viral.timeline.set_duration(300.0, fps=30.0)
    assert viral._clip_length.isEnabled()
    assert viral._clips.isEnabled()
    viral.timeline.add_range()
    # v0.2.1 G: the Clips count still disables in manual mode, but Clip
    # length now stays ENABLED — it mirrors the selected range (the
    # default range is 30 s) and resizes it when edited.
    assert viral._clip_length.isEnabled()
    assert viral._clip_length.value() == 30
    assert not viral._clips.isEnabled()
    assert viral.has_manual_ranges
    assert len(viral.manual_ranges) == 1
    assert "Manual ranges" in viral._status.text()
    viral.timeline.clear_ranges()
    assert viral._clip_length.isEnabled()
    assert viral._clips.isEnabled()
    window._show_home()


# -- add-range discoverability ------------------------------------------
def test_add_range_button_has_visible_text(timeline) -> None:
    # Regression: the add control used to be a 26 px flat "+" glyph with
    # only a tooltip — users could not find how to cut a manual range.
    assert "Add clip range" in timeline._add_btn.text()
    timeline._add_btn.click()
    assert len(timeline.manual_ranges()) == 1


def test_empty_state_hint_click_adds_range(timeline) -> None:
    # The "+ Add a clip or start viral detection" empty state is a real
    # button now: clicking it adds a range too.
    timeline._empty.click()
    assert len(timeline.manual_ranges()) == 1
    assert timeline._empty.isHidden()


# -- readiness: manual ranges need no language ---------------------------
def test_manual_range_readiness_without_language(window) -> None:
    page = ViralClipsPage(window)
    assert page.language_code == ""
    assert not page.is_ready
    page.timeline.set_duration(300.0, fps=30.0)
    assert page.timeline.add_range()
    # A manual cutout needs no transcription: ready with no language.
    assert page.is_ready
    page.timeline.clear_ranges()
    assert not page.is_ready
    # A chosen language on its own still readies the page (AI path).
    page._language.setCurrentIndex(1)
    assert page.is_ready


def test_start_button_enabled_by_manual_range_without_language(window) -> None:
    if not window._store.projects:
        from clipforge.core.projects import Project

        window._store.add(Project(name="Timeline test", source_path="video.mp4"))
    window._open_project(window._store.projects[0])
    window._wizard.go_to_step(1)
    viral = window._wizard._viral
    viral.reset()
    window._wizard._video_path = "video.mp4"
    viral.timeline.set_duration(300.0, fps=30.0)
    assert viral.language_code == ""
    assert not window._wizard._start_btn.isEnabled()
    viral.timeline.add_range()
    assert window._wizard._start_btn.isEnabled()
    viral.timeline.clear_ranges()
    assert not window._wizard._start_btn.isEnabled()
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


# -- v0.2.1 A: buttons must have real clickable size --------------------
def test_timeline_buttons_have_real_clickable_size(window) -> None:
    """No more 26 px flat glyphs that render as tiny dots (A).

    Zoom -/+, the header Delete, the range-row delete and the player
    play button all get a bordered, >=34 px-class hit area.
    """
    from PySide6.QtWidgets import QApplication, QPushButton

    if not window._store.projects:
        from clipforge.core.projects import Project

        window._store.add(Project(name="Buttons test", source_path="video.mp4"))
    window._open_project(window._store.projects[0])
    window._wizard.go_to_step(1)
    viral = window._wizard._viral
    window.show()
    QApplication.processEvents()
    timeline = viral.timeline

    assert timeline._add_btn.minimumHeight() >= 34
    assert not timeline._add_btn.isFlat()
    assert timeline._delete_btn.minimumHeight() >= 34
    assert "Delete" in timeline._delete_btn.text()
    assert not timeline._delete_btn.isFlat()

    zoom = [b for b in timeline.findChildren(QPushButton) if b.text() in ("−", "+")]
    assert len(zoom) == 2
    for btn in zoom:
        assert btn.width() >= 34 and btn.height() >= 34
        assert not btn.isFlat()

    timeline.set_duration(300.0, fps=30.0)
    timeline.add_range(10.0, 40.0)
    QApplication.processEvents()
    row = timeline._rows[0]
    row_delete = [
        b for b in row.findChildren(QPushButton)
        if b.toolTip() == "Delete this range"
    ][0]
    assert row_delete.width() >= 34 and row_delete.height() >= 30
    assert not row_delete.isFlat()

    play_btn = viral._player_widget._play_btn
    assert play_btn.minimumWidth() >= 40 and play_btn.minimumHeight() >= 34
    assert not play_btn.isFlat()
    window._show_home()


# -- v0.2.1 G: Clip length <-> selected range sync -----------------------
def test_clip_length_syncs_with_selected_range(window) -> None:
    if not window._store.projects:
        from clipforge.core.projects import Project

        window._store.add(Project(name="Sync test", source_path="video.mp4"))
    window._open_project(window._store.projects[0])
    window._wizard.go_to_step(1)
    viral = window._wizard._viral
    viral.timeline.set_duration(300.0, fps=30.0)

    # Create a 65s -> 125s range: Clip length must show its duration.
    viral.timeline.add_range(65.0, 125.0)
    assert viral._clip_length.value() == 60
    assert viral._clip_length.isEnabled()

    # Editing the range end moves Clip length with it.
    viral.timeline.set_range_bounds(0, 65.0, 140.0)
    assert viral._clip_length.value() == 75

    # Setting Clip length to 45 moves the range end to start + 45.
    viral._clip_length.setValue(45)
    rng = viral.timeline.manual_ranges()[0]
    assert rng.start == pytest.approx(65.0)
    assert rng.end == pytest.approx(110.0)

    # Auto mode: Clip length is free again (no range to drive it).
    viral.timeline.clear_ranges()
    viral._clip_length.setValue(41)
    assert viral._clip_length.value() == 41
    assert not viral.timeline.has_manual_ranges()
    window._show_home()
