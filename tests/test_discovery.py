"""Discovery + review + caption tests (DeepSeek client mocked)."""

import json

import pytest

from clipforge.core.discovery import (
    ClipCandidate,
    _extract_json,
    _parse_clip_items,
    build_discovery_prompt,
    discover_clips,
    generate_caption,
    render_template,
    review_clips,
)
from clipforge.core.transcription import Transcript, Utterance, Word


class FakeDeepSeekClient:
    """Scripted chat_text: returns ``replies`` in order, records calls."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict] = []

    def chat_text(self, messages, *, thinking=True, model=None, timeout=180.0):
        self.calls.append({"messages": messages, "thinking": thinking})
        if not self._replies:
            raise AssertionError("FakeDeepSeekClient ran out of scripted replies")
        return self._replies.pop(0)


def sample_transcript() -> Transcript:
    return Transcript(
        language="en",
        words=[
            Word("hello", 0.0, 0.5), Word("world", 0.5, 1.0),
            Word("goodbye", 60.0, 60.5), Word("world", 60.5, 61.0),
        ],
        utterances=[
            Utterance("hello world", 0.0, 1.0),
            Utterance("goodbye world", 60.0, 61.0),
        ],
    )


# -- prompt building -------------------------------------------------------

def test_render_template_substitutes_known_variables() -> None:
    out = render_template("Find {{targetCount}} clips of {{targetDuration}}s.", {
        "targetCount": 5, "targetDuration": 40,
    })
    assert out == "Find 5 clips of 40s."
    # Unknown placeholders are left untouched for the user to fill in.
    assert "{{missing}}" in render_template("Keep {{missing}}.", {})


def test_discovery_prompt_has_clip_structure_and_variables() -> None:
    system, user = build_discovery_prompt(
        sample_transcript(), clip_length=40, target_clips=3
    )
    for section in (
        "Hook (0–3 s)", "Lock-in zone (3–7 s)", "Body Block 1",
        "Rehook 1", "Rehook 2", "Body Block 3", "CTA",
    ):
        assert section in system
    assert "40" in system  # target duration
    assert "34" in system and "46" in system  # 85–115% window for 40 s
    assert "[0.0s–1.0s] hello world" in user
    assert "[60.0s–61.0s] goodbye world" in user


def test_discovery_prompt_excludes_manual_ranges() -> None:
    system, _ = build_discovery_prompt(
        sample_transcript(), excluded_ranges=[(30.0, 70.0)]
    )
    assert "30.0–70.0s" in system


# -- JSON parsing ----------------------------------------------------------

def test_extract_json_plain_and_fenced() -> None:
    assert _extract_json('{"clips": []}') == {"clips": []}
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('Here you go:\n[{"start": 1}]') == [{"start": 1}]


def test_extract_json_raises_on_prose() -> None:
    from clipforge.providers._http import ProviderError

    with pytest.raises(ProviderError):
        _extract_json("No JSON here at all.")


def test_parse_clip_items_validates_and_clamps() -> None:
    data = {"clips": [
        {"start": 10, "end": 50, "title": "good"},
        {"start": 60, "end": 60, "title": "zero length"},  # dropped
        {"start": 70, "end": 65, "title": "backwards"},  # dropped
        {"start": -5, "end": 400, "title": "clamped"},  # -> 0..300
        {"start": "bad", "end": 10},  # dropped
    ]}
    clips = _parse_clip_items(data, duration=300.0)
    assert [(c.start, c.end) for c in clips] == [(0.0, 300.0), (10.0, 50.0)]
    assert clips[1].title == "good"


# -- discovery / review ----------------------------------------------------

DISCOVERY_REPLY = json.dumps({
    "clips": [
        {"start": 5, "end": 45, "title": "First", "hookText": "hello",
         "concludingText": "world", "score": 0.9},
        {"start": 60, "end": 100, "title": "Second", "hookText": "goodbye",
         "concludingText": "world", "score": 0.7},
    ]
})


def test_discover_clips_uses_thinking_and_parses() -> None:
    client = FakeDeepSeekClient([DISCOVERY_REPLY])
    clips = discover_clips(client, sample_transcript(), clip_length=40, target_clips=2)
    assert len(clips) == 2
    assert clips[0].title == "First"
    assert clips[0].duration == pytest.approx(40.0)
    assert clips[0].hook_text == "hello"
    assert client.calls[0]["thinking"] is True  # discovery = thinking on


def test_review_keeps_strongest_and_never_moves_timestamps() -> None:
    candidates = [
        ClipCandidate(5, 45, "First", score=0.9),
        ClipCandidate(60, 100, "Second", score=0.7),
        ClipCandidate(120, 160, "Third", score=0.8),
    ]
    # Review keeps only the first and third (reordered by the model).
    reply = json.dumps({"clips": [
        {"start": 120, "end": 160, "title": "Third"},
        {"start": 5, "end": 45, "title": "First"},
    ]})
    client = FakeDeepSeekClient([reply])
    kept = review_clips(client, candidates, sample_transcript(), target_count=2)
    assert [(c.start, c.end) for c in kept] == [(5.0, 45.0), (120.0, 160.0)]
    assert kept[0].title == "First"  # original candidate, timestamps intact


def test_review_falls_back_to_best_scores_on_bad_output() -> None:
    candidates = [
        ClipCandidate(5, 45, "First", score=0.9),
        ClipCandidate(60, 100, "Second", score=0.7),
        ClipCandidate(120, 160, "Third", score=0.8),
    ]
    client = FakeDeepSeekClient(["not json at all"])
    kept = review_clips(client, candidates, sample_transcript(), target_count=2)
    assert [(c.start, c.end) for c in kept] == [(5.0, 45.0), (120.0, 160.0)]


# -- non-overlap enforcement -----------------------------------------------

#: The exact overlapping pair from acceptance test 4 (2026-11-23).
OVERLAPPING_REPLY = json.dumps({
    "clips": [
        {"start": 0.0, "end": 40.7, "title": "First", "score": 0.9},
        {"start": 33.3, "end": 74.3, "title": "Second", "score": 0.8},
    ]
})


def _assert_no_overlap(clips: list[ClipCandidate]) -> None:
    ordered = sorted(clips, key=lambda c: c.start)
    for prev, nxt in zip(ordered, ordered[1:]):
        assert nxt.start - prev.end >= -0.5, (
            f"clips overlap: {prev.start}–{prev.end} and {nxt.start}–{nxt.end}"
        )


def test_discover_clips_never_returns_overlapping_clips() -> None:
    client = FakeDeepSeekClient([OVERLAPPING_REPLY])
    clips = discover_clips(client, sample_transcript(), clip_length=40, target_clips=2)
    _assert_no_overlap(clips)
    # The higher-scored first clip wins the clash.
    assert [(c.start, c.end) for c in clips] == [(0.0, 40.7)]


def test_discover_clips_keeps_count_when_spares_exist() -> None:
    reply = json.dumps({"clips": [
        {"start": 0, "end": 40, "title": "A", "score": 0.9},
        {"start": 35, "end": 75, "title": "B", "score": 0.8},  # overlaps A
        {"start": 80, "end": 120, "title": "C", "score": 0.7},
    ]})
    client = FakeDeepSeekClient([reply])
    clips = discover_clips(
        client, sample_transcript(), clip_length=40, target_clips=2, duration=300.0
    )
    _assert_no_overlap(clips)
    assert [(c.start, c.end) for c in clips] == [(0.0, 40.0), (80.0, 120.0)]


def test_review_drops_overlapping_clips() -> None:
    candidates = [
        ClipCandidate(0, 40.7, "First", score=0.9),
        ClipCandidate(33.3, 74.3, "Second", score=0.8),
        ClipCandidate(80, 120, "Third", score=0.7),
    ]
    reply = json.dumps({"clips": [
        {"start": 0, "end": 40.7, "title": "First"},
        {"start": 33.3, "end": 74.3, "title": "Second"},
    ]})
    client = FakeDeepSeekClient([reply])
    kept = review_clips(client, candidates, sample_transcript(), target_count=2)
    _assert_no_overlap(kept)
    assert [(c.start, c.end) for c in kept] == [(0.0, 40.7)]


def test_review_fallback_dedupes_before_ranking() -> None:
    candidates = [
        ClipCandidate(0, 40, "A", score=0.9),
        ClipCandidate(35, 75, "B", score=0.8),  # overlaps A
        ClipCandidate(80, 120, "C", score=0.7),
    ]
    client = FakeDeepSeekClient(["not json at all"])
    kept = review_clips(client, candidates, sample_transcript(), target_count=2)
    _assert_no_overlap(kept)
    assert [(c.start, c.end) for c in kept] == [(0.0, 40.0), (80.0, 120.0)]


def test_dedupe_overlapping_boundary_touch_is_not_overlap() -> None:
    from clipforge.core.discovery import dedupe_overlapping

    clips = dedupe_overlapping([
        ClipCandidate(0, 40, "A", score=0.9),
        ClipCandidate(40, 80, "B", score=0.8),  # touches exactly at 40
        ClipCandidate(80.4, 120, "C", score=0.7),  # 0.4s gap
    ])
    assert len(clips) == 3


# -- caption generation ----------------------------------------------------

def test_generate_caption_exactly_four_hashtags_from_json() -> None:
    reply = json.dumps({
        "caption": "Watch this!",
        "hashtags": ["#one", "#two", "#three", "#four", "#five"],
    })
    client = FakeDeepSeekClient([reply])
    result = generate_caption(
        client, ClipCandidate(0, 1, "T"), sample_transcript(), language="en"
    )
    assert result["caption"] == "Watch this!"
    assert result["hashtags"] == ["#one", "#two", "#three", "#four"]
    assert client.calls[0]["thinking"] is False  # captions = thinking off


def test_generate_caption_harvests_hashtags_from_prose() -> None:
    client = FakeDeepSeekClient(["Caption here #a #b #c #d and more text"])
    result = generate_caption(
        client, ClipCandidate(0, 1, "T"), sample_transcript(), language="bn"
    )
    assert result["hashtags"] == ["#a", "#b", "#c", "#d"]
    # The Bengali request is in the system prompt.
    assert "Bengali" in client.calls[0]["messages"][0]["content"]


def test_generate_caption_never_invents_a_fourth_hashtag() -> None:
    reply = json.dumps({"caption": "Short", "hashtags": ["#only", "#two"]})
    client = FakeDeepSeekClient([reply])
    result = generate_caption(
        client, ClipCandidate(0, 1, "T"), sample_transcript(), language="en"
    )
    assert result["hashtags"] == ["#only", "#two"]
