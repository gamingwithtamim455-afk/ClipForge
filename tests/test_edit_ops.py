"""Tests: basic edit operations (trim / split / reorder / captions)."""

import pytest

from clipforge.core.caption_edit import (
    delete_line,
    delete_word,
    edit_line_text,
    line_spans,
)
from clipforge.core.edit_ops import (
    EditableClip,
    add_overlay,
    clip_line_texts,
    delete_clip,
    delete_clip_line,
    delete_clip_word,
    edit_clip_line,
    move_clip,
    split_clip,
    trim_clip,
)
from clipforge.core.transcription import Word


def _words() -> list[Word]:
    # Source-time words: 30-40 s (clip 30-70), two lines.
    return [
        Word("hello", 30.0, 30.4),
        Word("world", 30.4, 30.8),
        Word("this", 30.8, 31.2),
        Word("is", 31.2, 31.5),
        Word("a", 31.5, 31.6),
        Word("test", 31.6, 32.0),
        Word("later", 60.0, 60.5),
    ]


def _clip() -> EditableClip:
    return EditableClip(start=30.0, end=70.0, label="Clip 1", words=_words())


# -- caption editing (pure) ------------------------------------------------

def test_line_spans_match_line_texts() -> None:
    words = _words()
    spans = line_spans(words)
    # Same grouping as the render (1.5s line cap splits the 2s span).
    assert [text for _s, _e, text in spans] == [
        "hello world this is",
        "a test",
        "later",
    ]
    assert spans[0] == (pytest.approx(30.0), pytest.approx(31.5),
                        "hello world this is")
    assert spans[1] == (pytest.approx(31.5), pytest.approx(32.0), "a test")
    assert spans[2] == (pytest.approx(60.0), pytest.approx(60.5), "later")


def test_edit_line_text_keeps_span() -> None:
    words = [Word("hello", 0.0, 0.4), Word("world", 0.4, 0.8)]
    edited = edit_line_text(words, 0, "hi there friend")
    assert [w.text for w in edited] == ["hi", "there", "friend"]
    assert edited[0].start == pytest.approx(0.0)
    assert edited[-1].end == pytest.approx(0.8)
    assert all(edited[i].end <= edited[i + 1].start for i in range(2))


def test_edit_line_empty_text_deletes_line() -> None:
    words = [Word("a", 0.0, 0.3), Word("b", 0.3, 0.6), Word("c", 5.0, 5.3)]
    assert edit_line_text(words, 0, "   ") == [Word("c", 5.0, 5.3)]


def test_delete_word_and_line() -> None:
    words = [Word("a", 0.0, 0.3), Word("b", 0.3, 0.6), Word("c", 5.0, 5.3)]
    assert [w.text for w in delete_word(words, 1)] == ["a", "c"]
    assert delete_line(words, 0) == [Word("c", 5.0, 5.3)]
    with pytest.raises(IndexError):
        delete_word(words, 5)
    with pytest.raises(IndexError):
        delete_line(words, 3)


# -- trim / split / reorder / delete ---------------------------------------

def test_trim_clip_valid_and_invalid() -> None:
    clip = _clip()
    other = EditableClip(start=80.0, end=100.0)
    trimmed = trim_clip(clip, 30.0, 60.0, video_duration=300.0, siblings=[other])
    assert (trimmed.start, trimmed.end) == (30.0, 60.0)
    # Words inside the new range survive; the 60.0s word is out.
    assert [w.text for w in trimmed.words] == [
        "hello", "world", "this", "is", "a", "test",
    ]
    assert trim_clip(
        clip, 35.0, 60.0, video_duration=300.0, siblings=[other]
    ).words == []
    with pytest.raises(ValueError):
        trim_clip(clip, 60.0, 35.0, video_duration=300.0, siblings=[other])
    with pytest.raises(ValueError):
        trim_clip(clip, 20.0, 301.0, video_duration=300.0, siblings=[other])
    with pytest.raises(ValueError):  # overlaps the sibling
        trim_clip(clip, 30.0, 90.0, video_duration=300.0, siblings=[other])


def test_split_clip() -> None:
    clip = _clip()
    first, second = split_clip(clip, 50.0)
    assert (first.start, first.end) == (30.0, 50.0)
    assert (second.start, second.end) == (50.0, 70.0)
    assert [w.text for w in first.words] == [
        "hello", "world", "this", "is", "a", "test",
    ]
    assert [w.text for w in second.words] == ["later"]
    assert first.volume == second.volume == 1.0
    with pytest.raises(ValueError):
        split_clip(clip, 30.0)
    with pytest.raises(ValueError):
        split_clip(clip, 70.0)


def test_move_and_delete_clip() -> None:
    clips = [EditableClip(0.0, 10.0, label="A"),
             EditableClip(20.0, 30.0, label="B"),
             EditableClip(40.0, 50.0, label="C")]
    moved = move_clip(clips, 0, 2)
    assert [c.label for c in moved] == ["B", "C", "A"]
    assert [c.label for c in clips] == ["A", "B", "C"]  # input untouched
    assert [c.label for c in delete_clip(clips, 1)] == ["A", "C"]
    with pytest.raises(IndexError):
        move_clip(clips, 0, 5)


# -- caption edits on a clip (source-time aware) ---------------------------

def test_edit_clip_line_preserves_source_time() -> None:
    clip = _clip()
    edited = edit_clip_line(clip, 0, "hi there")
    texts = clip_line_texts(edited)
    assert texts[0] == "hi there"
    words = edited.words
    assert words[0].start == pytest.approx(30.0)
    # Line 0 is hello..is (1.5s span cap), so its span ends at 31.5.
    assert words[1].end == pytest.approx(31.5)
    assert words[-1].text == "later"  # untouched word survives


def test_delete_clip_word_and_line() -> None:
    clip = _clip()
    assert "world" not in [
        w.text for w in delete_clip_word(clip, 1).words
    ]
    # Lines are [hello..is], [a test], [later] (1.5s span cap splits first).
    assert clip_line_texts(delete_clip_line(clip, 0)) == ["a test", "later"]
    # The 60s word is its own line (gap > 1.5s groups separately).
    assert len(clip_line_texts(clip)) == 3


def test_add_overlay_validation() -> None:
    clip = _clip()
    out = add_overlay(clip, "HELLO", 35.0, 37.0)
    assert len(out.overlays) == 1
    assert out.overlays[0].text == "HELLO"
    with pytest.raises(ValueError):
        add_overlay(clip, "  ", 35.0, 37.0)
    with pytest.raises(ValueError):
        add_overlay(clip, "x", 37.0, 35.0)
    with pytest.raises(ValueError):  # outside the clip
        add_overlay(clip, "x", 10.0, 12.0)
