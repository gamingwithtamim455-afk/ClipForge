"""Transcription tests: chunk math, WAV splitting, merging, orchestration."""

import io
import struct
import wave
from pathlib import Path

import pytest

from clipforge.core.transcription import (
    Transcript,
    Utterance,
    Word,
    chunk_plan,
    merge_chunks,
    parse_deepgram_response,
    parse_whisper_result,
    split_wav,
    transcribe_audio,
    transcribe_ranges,
    transcribe_video,
)
from clipforge.providers._http import ProviderError


def make_wav(seconds: float, sample_rate: int = 16000, ramp: bool = False) -> bytes:
    """Mono 16 kHz PCM16 WAV bytes; ``ramp`` fills frames with 0,1,2,…."""
    n_frames = int(seconds * sample_rate)
    if ramp:
        frames = b"".join(struct.pack("<h", i % 32767) for i in range(n_frames))
    else:
        frames = b"\x00\x00" * n_frames
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)
    return buffer.getvalue()


def wav_duration(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


# -- chunk plan ------------------------------------------------------------

def test_chunk_plan_short_audio_is_one_chunk() -> None:
    assert chunk_plan(300) == [(0.0, 300.0)]
    assert chunk_plan(480) == [(0.0, 480.0)]
    assert chunk_plan(0) == []


def test_chunk_plan_long_audio_overlaps_by_5s() -> None:
    plan = chunk_plan(1000)
    assert plan == [(0.0, 480.0), (475.0, 955.0), (950.0, 1000.0)]
    for (s1, _), (s2, _) in zip(plan, plan[1:]):
        assert s2 - s1 == 475.0  # 480 s chunk minus 5 s overlap


def test_chunk_plan_last_chunk_ends_exactly_at_duration() -> None:
    plan = chunk_plan(10000)
    assert plan[-1][1] == 10000.0
    assert all(end - start <= 480.0 for start, end in plan)


# -- WAV splitting ---------------------------------------------------------

def test_split_wav_short_audio_is_one_chunk() -> None:
    chunks = split_wav(make_wav(10))
    assert len(chunks) == 1
    assert chunks[0][0] == 0.0
    assert wav_duration(chunks[0][1]) == pytest.approx(10.0)


def test_split_wav_chunk_boundaries_and_overlap() -> None:
    chunks = split_wav(make_wav(10, ramp=True), chunk_seconds=4, overlap_seconds=1)
    assert [start for start, _ in chunks] == [0.0, 3.0, 6.0]
    assert [wav_duration(b) for _, b in chunks] == [
        pytest.approx(4.0), pytest.approx(4.0), pytest.approx(4.0),
    ]
    # Chunk 2 starts at the 3 s frame (sample value 48000).
    with wave.open(io.BytesIO(chunks[1][1]), "rb") as wav:
        first = struct.unpack("<h", wav.readframes(1))[0]
    assert first == 48000 % 32767


def test_split_wav_rejects_non_mono() -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00\x00\x00" * 16000)
    with pytest.raises(ValueError):
        split_wav(buffer.getvalue())


# -- merging ---------------------------------------------------------------

def test_merge_chunks_dedupes_overlap_words() -> None:
    t1 = Transcript(
        language="en",
        words=[Word("hello", 0.0, 0.5), Word("world", 474.5, 475.0)],
        utterances=[Utterance("hello world", 0.0, 475.0)],
    )
    t2 = Transcript(
        language="en",
        words=[Word("world", 474.5, 475.0), Word("again", 476.0, 476.5)],
        utterances=[Utterance("world again", 474.5, 476.5)],
    )
    merged = merge_chunks([t1, t2])
    assert [w.text for w in merged.words] == ["hello", "world", "again"]
    assert [w.start for w in merged.words] == [0.0, 474.5, 476.0]
    assert len(merged.utterances) == 2


def test_merge_chunks_empty() -> None:
    merged = merge_chunks([])
    assert merged.words == []
    assert merged.duration == 0.0


def test_merge_chunks_keeps_repeated_words_outside_overlap() -> None:
    # "yes yes" 2 s apart is a real repetition, not an overlap duplicate.
    t = Transcript(
        language="en",
        words=[Word("yes", 10.0, 10.4), Word("yes", 12.0, 12.4)],
    )
    assert len(merge_chunks([t]).words) == 2


# -- response parsing ------------------------------------------------------

DEEPGRAM_PAYLOAD = {
    "results": {
        "channels": [
            {
                "detected_language": "en",
                "alternatives": [
                    {
                        "words": [
                            {"word": "hello", "start": 0.1, "end": 0.4, "confidence": 0.99},
                            {"word": "world", "start": 0.5, "end": 0.9, "confidence": 0.98},
                        ]
                    }
                ],
            }
        ],
        "utterances": [{"transcript": "hello world", "start": 0.1, "end": 0.9}],
    }
}


def test_parse_deepgram_response_applies_offset() -> None:
    transcript = parse_deepgram_response(DEEPGRAM_PAYLOAD, offset=475.0)
    assert transcript.detected_language == "en"
    assert [(w.text, w.start, w.end) for w in transcript.words] == [
        ("hello", 475.1, 475.4),
        ("world", 475.5, 475.9),
    ]
    assert transcript.words[0].confidence == 0.99
    assert transcript.utterances[0].start == 475.1


def test_parse_deepgram_response_empty_is_safe() -> None:
    transcript = parse_deepgram_response({"results": {}})
    assert transcript.words == []
    assert transcript.utterances == []


def test_parse_whisper_result() -> None:
    result = {
        "words": [{"text": " hi ", "start": 1.0, "end": 1.5, "confidence": 0.8}],
        "utterances": [{"text": "hi", "start": 1.0, "end": 1.5}],
        "detected_language": "bn",
    }
    transcript = parse_whisper_result(result)
    assert transcript.words[0].text == "hi"  # stripped
    assert transcript.engine == "whisper_local"
    assert transcript.detected_language == "bn"


# -- orchestration ---------------------------------------------------------

class FakeDeepgramClient:
    """Returns one word per chunk, at the chunk's own start + 0.1 s."""

    def __init__(self, fail_with: ProviderError | None = None) -> None:
        self.fail_with = fail_with
        self.calls: list[bytes] = []

    def transcribe_chunk(self, wav_bytes: bytes, *, language: str = "auto") -> dict:
        self.calls.append(wav_bytes)
        if self.fail_with is not None:
            raise self.fail_with
        index = len(self.calls) - 1
        return {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {"words": [{"word": f"w{index}", "start": 0.1, "end": 0.4}]}
                        ]
                    }
                ],
                "utterances": [],
            }
        }


class FakeWhisperClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def transcribe(self, audio_path: str, *, language=None,
                   download_progress=None) -> dict:
        return {
            "words": [{"text": "local", "start": 0.0, "end": 0.5, "confidence": 1.0}],
            "utterances": [{"text": "local", "start": 0.0, "end": 0.5}],
            "detected_language": "en",
        }


def test_transcribe_audio_chunks_and_merges(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "clipforge.core.transcription._chunk_settings", lambda: (8.0, 2.0)
    )
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(make_wav(20))
    client = FakeDeepgramClient()
    transcript = transcribe_audio(
        str(wav_path), language="en", engine="deepgram",
        deepgram_client=client,
    )
    # 20 s at 8 s chunks / 2 s overlap -> starts 0, 6, 12 -> 3 chunks.
    assert len(client.calls) == 3
    assert [w.text for w in transcript.words] == ["w0", "w1", "w2"]
    assert transcript.words[1].start == pytest.approx(6.1)
    assert transcript.engine == "deepgram"
    assert not transcript.used_fallback


def test_transcribe_audio_falls_back_to_whisper_when_offline(tmp_path, monkeypatch) -> None:
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(make_wav(5))
    monkeypatch.setattr(
        "clipforge.core.transcription.WhisperLocalClient", FakeWhisperClient
    )
    client = FakeDeepgramClient(
        fail_with=ProviderError("no_internet", "No internet connection — check your network")
    )
    transcript = transcribe_audio(
        str(wav_path), language="en", engine="deepgram", deepgram_client=client
    )
    assert transcript.used_fallback
    assert transcript.engine == "whisper_local"
    assert transcript.words[0].text == "local"
    assert "local Whisper" in transcript.note


def test_transcribe_audio_does_not_fall_back_on_invalid_key(tmp_path) -> None:
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(make_wav(5))
    client = FakeDeepgramClient(
        fail_with=ProviderError("invalid_key", "Invalid API key (401)")
    )
    with pytest.raises(ProviderError) as excinfo:
        transcribe_audio(
            str(wav_path), language="en", engine="deepgram", deepgram_client=client
        )
    assert excinfo.value.kind == "invalid_key"


def test_transcribe_audio_requires_a_key(tmp_path) -> None:
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(make_wav(5))
    with pytest.raises(ProviderError) as excinfo:
        transcribe_audio(str(wav_path), language="en", engine="deepgram")
    assert excinfo.value.kind == "invalid_key"


def test_transcribe_video_extracts_then_transcribes(tmp_path, monkeypatch) -> None:
    def fake_extract(video_path, out_path, **kwargs):
        Path(out_path).write_bytes(make_wav(5))
        return out_path

    monkeypatch.setattr("clipforge.core.transcription.extract_audio", fake_extract)
    client = FakeDeepgramClient()
    transcript = transcribe_video(
        "video.mp4", language="en", engine="deepgram", deepgram_client=client
    )
    assert transcript.words[0].text == "w0"


# -- range-only transcription (manual timeline) ---------------------------

def _fake_range_words(*, relative: list[tuple[str, float, float]]) -> Transcript:
    """Transcript as a per-range engine would return it (WAV-relative)."""
    return Transcript(
        language="en",
        words=[Word(text, start, end) for text, start, end in relative],
        engine="whisper_local",
    )


def test_transcribe_ranges_extracts_only_the_ranges(tmp_path, monkeypatch) -> None:
    """Manual mode: extraction + transcription happen per range, never full."""
    extract_calls: list[tuple[float | None, float | None]] = []

    def fake_extract(video_path, out_path, **kwargs):
        extract_calls.append((kwargs.get("start"), kwargs.get("duration")))
        Path(out_path).write_bytes(make_wav(5))
        return out_path

    monkeypatch.setattr("clipforge.core.transcription.extract_audio", fake_extract)
    transcribe_calls: list[str] = []

    def fake_transcribe(wav_path, **kwargs):
        transcribe_calls.append(wav_path)
        # Words relative to the extracted WAV: one straddling the range
        # start (padding), two inside, one straddling the range end.
        return _fake_range_words(relative=[
            ("before", 0.2, 0.4), ("hello", 0.6, 0.9),
            ("world", 40.0, 40.4), ("after", 40.8, 41.0),
        ])

    monkeypatch.setattr("clipforge.core.transcription.transcribe_audio", fake_transcribe)
    result = transcribe_ranges(
        "video.mp4", [(30.0, 70.0), (100.0, 140.0)],
        duration=300.0, language="en", engine="whisper_local",
    )
    # 0.5 s padding on each side of every range.
    assert extract_calls == [(29.5, 41.0), (99.5, 41.0)]
    assert len(transcribe_calls) == 2
    # Words are offset back to video time and kept only inside a range.
    assert [(w.text, round(w.start, 1)) for w in result.words] == [
        ("hello", 30.1), ("world", 69.5),
        ("hello", 100.1), ("world", 139.5),
    ]
    assert all(
        30.0 <= w.start < 70.0 or 100.0 <= w.start < 140.0
        for w in result.words
    )


def test_transcribe_ranges_dedupes_overlapping_ranges(tmp_path, monkeypatch) -> None:
    """The same word transcribed in two overlapping ranges appears once."""
    def fake_extract(video_path, out_path, **kwargs):
        Path(out_path).write_bytes(make_wav(5))
        return out_path

    monkeypatch.setattr("clipforge.core.transcription.extract_audio", fake_extract)

    def fake_transcribe(wav_path, **kwargs):
        # Both ranges cover video time 64.5 s: range 0's WAV starts at
        # 29.5 s, range 1's at 59.5 s — same word, same video time.
        if "range-0" in wav_path:
            return _fake_range_words(relative=[("shared", 35.0, 35.4)])
        return _fake_range_words(relative=[("shared", 5.0, 5.4)])

    monkeypatch.setattr("clipforge.core.transcription.transcribe_audio", fake_transcribe)
    result = transcribe_ranges(
        "video.mp4", [(30.0, 70.0), (60.0, 100.0)],
        duration=300.0, language="en", engine="whisper_local",
    )
    assert [w.text for w in result.words] == ["shared"]
    assert result.words[0].start == pytest.approx(64.5)


def test_transcribe_ranges_respects_cancel() -> None:
    """A cancelled run stops before transcribing any range."""
    with pytest.raises(ProviderError) as excinfo:
        transcribe_ranges(
            "video.mp4", [(30.0, 70.0)], duration=300.0,
            language="en", engine="whisper_local", cancel=lambda: True,
        )
    assert excinfo.value.kind == "cancelled"


def test_transcribe_ranges_empty_is_empty() -> None:
    result = transcribe_ranges(
        "video.mp4", [], duration=300.0, language="en", engine="whisper_local",
    )
    assert result.words == []
