"""Edit-step UI tests (offscreen): trim, split, reorder, delete, keep/drop,
caption edit/delete/toggle, volume, custom text, timeline sync — all wired
through the page's model so the render jobs see the edits."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton

from clipforge.core.caption_edit import line_texts
from clipforge.core.edit_ops import clip_words_relative
from clipforge.core.ranges import ClipRange
from clipforge.core.transcription import Word

WORDS = [
    Word("Hello", 0.0, 0.4), Word("world", 0.4, 0.8), Word("this", 0.8, 1.2),
    Word("is", 10.0, 10.4), Word("a", 10.4, 10.8), Word("test", 10.8, 11.2),
    Word("second", 20.0, 20.4), Word("clip", 20.4, 20.8),
]
RANGES = [
    ClipRange(0.0, 5.0, label="Clip 1"),
    ClipRange(10.0, 15.0, label="Clip 2"),
    ClipRange(20.0, 25.0, label="Clip 3"),
]


@pytest.fixture(scope="module")
def window(app_window):
    yield app_window


@pytest.fixture()
def page(window):
    """Fresh Edit page state for each test."""
    edit = window._wizard._edit
    edit.set_clips(list(RANGES), words=list(WORDS), video_duration=30.0)
    edit._clip_list.setCurrentRow(0)
    return edit


def _ranges(page) -> list[tuple[float, float]]:
    return [(r.start, r.end) for r in page.kept_ranges()]


def test_trim_valid(page) -> None:
    page._trim_start.setText("0:01")
    page._trim_end.setText("0:04")
    page._trim_btn.click()
    assert _ranges(page)[0] == (1.0, 4.0)
    assert page._error.text() == ""


def test_trim_invalid_overlap(page) -> None:
    page._trim_end.setText("0:12")  # overlaps Clip 2 (10–15)
    page._trim_btn.click()
    assert _ranges(page)[0] == (0.0, 5.0)  # unchanged
    assert "overlap" in page._error.text().lower()


def test_trim_invalid_end_before_start(page) -> None:
    page._trim_start.setText("0:04")
    page._trim_end.setText("0:01")
    page._trim_btn.click()
    assert _ranges(page)[0] == (0.0, 5.0)
    assert page._error.text() != ""


def test_split(page) -> None:
    page._split_at.setText("0:02")
    page._split_btn.click()
    assert _ranges(page) == [
        (0.0, 2.0), (2.0, 5.0), (10.0, 15.0), (20.0, 25.0),
    ]
    assert len(page._clips) == 4


def test_split_outside_clip_rejected(page) -> None:
    page._split_at.setText("0:07")  # outside Clip 1 (0–5)
    page._split_btn.click()
    assert len(page._clips) == 3
    assert page._error.text() != ""


def test_move_down_then_up(page) -> None:
    page._down_btn.click()
    assert _ranges(page) == [(10.0, 15.0), (0.0, 5.0), (20.0, 25.0)]
    page._up_btn.click()
    assert _ranges(page) == [(0.0, 5.0), (10.0, 15.0), (20.0, 25.0)]


def test_delete_clip(page) -> None:
    page._clip_list.setCurrentRow(1)
    page._delete_btn.click()
    assert _ranges(page) == [(0.0, 5.0), (20.0, 25.0)]


def test_keep_drop_checkbox(page) -> None:
    item = page._clip_list.item(0)
    item.setCheckState(Qt.CheckState.Unchecked)
    assert _ranges(page) == [(10.0, 15.0), (20.0, 25.0)]
    item.setCheckState(Qt.CheckState.Checked)
    assert _ranges(page) == [(0.0, 5.0), (10.0, 15.0), (20.0, 25.0)]


def test_caption_line_edit(page) -> None:
    assert page._lines.count() == 1
    assert page._lines.item(0).text() == "Hello world this"
    page._lines.item(0).setText("Hi there")
    assert page._lines.item(0).text() == "Hi there"
    words, _, _, _ = page.edited_overrides()
    assert [w.text for w in words[0]] == ["Hi", "there"]


def test_delete_caption_line(page) -> None:
    page._lines.setCurrentRow(0)
    page._delete_line_btn.click()
    assert page._lines.count() == 0
    words, _, _, _ = page.edited_overrides()
    assert words[0] == []


def test_delete_caption_word(page) -> None:
    page._lines.setCurrentRow(0)
    assert page._word_combo.count() == 3  # Hello / world / this
    page._word_combo.setCurrentIndex(0)  # "Hello"
    page._delete_word_btn.click()
    assert page._lines.item(0).text() == "world this"
    words, _, _, _ = page.edited_overrides()
    assert [w.text for w in words[0]] == ["world", "this"]


def test_remove_all_captions(page) -> None:
    page._remove_all_btn.click()
    assert page._lines.count() == 0
    words, _, _, _ = page.edited_overrides()
    assert words[0] == []


def test_captions_off_per_clip(page) -> None:
    page._captions_on.setChecked(False)
    _, _, _, enabled = page.edited_overrides()
    assert enabled[0] is False
    assert enabled[1] is True
    page._captions_on.setChecked(True)
    _, _, _, enabled = page.edited_overrides()
    assert enabled[0] is True


def test_volume_mute_and_level(page) -> None:
    page._mute.setChecked(True)
    _, _, volumes, _ = page.edited_overrides()
    assert volumes[0] == 0.0
    page._mute.setChecked(False)
    _, _, volumes, _ = page.edited_overrides()
    assert volumes[0] == 1.0
    page._volume.setValue(50)
    _, _, volumes, _ = page.edited_overrides()
    assert volumes[0] == 0.5


def test_add_custom_text(page) -> None:
    page._overlay_text.setText("MY TEXT")
    page._overlay_start.setText("0:01")
    page._overlay_end.setText("0:03")
    page._overlay_btn.click()
    assert page._error.text() == ""
    _, overlays, _, _ = page.edited_overrides()
    assert len(overlays[0]) == 1
    overlay = overlays[0][0]
    assert overlay.text == "MY TEXT"
    # Clip-relative: clip starts at 0.0, so 1.0–3.0 stays.
    assert overlay.start == 1.0 and overlay.end == 3.0
    assert "MY TEXT" in page._overlay_list.text()


def test_add_custom_text_outside_clip_rejected(page) -> None:
    page._overlay_text.setText("NOPE")
    page._overlay_start.setText("0:06")  # outside Clip 1 (0–5)
    page._overlay_end.setText("0:07")
    page._overlay_btn.click()
    _, overlays, _, _ = page.edited_overrides()
    assert overlays[0] == []
    assert page._error.text() != ""


def test_timeline_shows_updated_ranges(page) -> None:
    page._trim_start.setText("0:01")
    page._trim_end.setText("0:04")
    page._trim_btn.click()
    shown = [(r.start, r.end) for r in page.timeline.manual_ranges()]
    assert shown == [(1.0, 4.0), (10.0, 15.0), (20.0, 25.0)]


def test_overrides_follow_reorder(page) -> None:
    # Mute clip 1, then move it down: the override must travel with it.
    page._mute.setChecked(True)
    page._down_btn.click()
    _, _, volumes, _ = page.edited_overrides()
    assert volumes == {0: 1.0, 1: 0.0, 2: 1.0}
    assert _ranges(page)[1] == (0.0, 5.0)


def _line_row(page, row: int):
    """The corrected-caption row widget (timestamp + editor + ×)."""
    item = page._lines.item(row)
    assert item is not None
    return page._lines.itemWidget(item)


def test_caption_lines_show_timestamps(page) -> None:
    assert page._lines.count() == 1
    row = _line_row(page, 0)
    labels = row.findChildren(QLabel)
    assert len(labels) == 1
    # Clip 1's only line: Hello/world/this, 0.0–1.2s clip-relative.
    assert labels[0].text() == "00:00–00:01.2"
    edits = row.findChildren(QLineEdit)
    assert len(edits) == 1
    assert edits[0].text() == "Hello world this"


def test_caption_line_delete_button(page) -> None:
    row = _line_row(page, 0)
    buttons = row.findChildren(QPushButton)
    assert len(buttons) == 1
    assert buttons[0].text() == "×"
    buttons[0].click()
    assert page._lines.count() == 0
    words, _, _, _ = page.edited_overrides()
    assert words[0] == []


def test_caption_line_text_edit_via_row(page) -> None:
    row = _line_row(page, 0)
    edit = row.findChildren(QLineEdit)[0]
    edit.setText("Hi there")
    edit.editingFinished.emit()
    words, _, _, _ = page.edited_overrides()
    assert [w.text for w in words[0]] == ["Hi", "there"]
