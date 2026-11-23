"""Transcription data model and orchestration.

The actual transcription runs in the provider adapters
(``providers.deepgram`` / ``providers.whisper_local``). This module owns:

- the shared data model (``Word`` / ``Utterance`` / ``Transcript``),
- the Deepgram chunk plan: audio only (mono 16 kHz), chunks of at most
  8 minutes with a 5 s overlap (research §1.5),
- merging per-chunk transcripts (de-duplicating the overlap window),
- parsing Deepgram / faster-whisper responses into ``Transcript``,
- the orchestration that picks Deepgram vs local Whisper per language and
  availability, and falls back to local Whisper when Deepgram cannot serve
  the language or the machine is offline — saying so in the status note.

Chunk boundaries are handled in seconds; the WAV itself is split with the
stdlib ``wave`` module, so splitting is frame-accurate at 16 kHz.
"""

from __future__ import annotations

import io
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..providers import load_presets
from ..providers._http import ProviderError
from ..providers.deepgram import DeepgramClient
from ..providers.whisper_local import WhisperLocalClient
from .ffmpeg import extract_audio, probe

#: Fallback chunk sizes (presets.json is the source of truth).
DEFAULT_CHUNK_SECONDS = 480
DEFAULT_OVERLAP_SECONDS = 5

#: Deepgram failures that fall back to local Whisper instead of erroring.
#: A 401 (invalid key) is deliberately NOT here — a wrong key must be fixed,
#: not silently worked around.
FALLBACK_ERROR_KINDS = {"no_internet", "bad_request", "timeout", "server"}


@dataclass
class Word:
    """One transcribed word with timestamps (seconds)."""

    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class Utterance:
    """One speaker utterance (a run of words)."""

    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    """Full transcript of a source video."""

    language: str  # BCP-47 code, or "auto"
    detected_language: str | None = None
    words: list[Word] = field(default_factory=list)
    utterances: list[Utterance] = field(default_factory=list)
    engine: str = "deepgram"  # "deepgram" | "whisper_local"
    used_fallback: bool = False  # True when Whisper covered a Deepgram gap
    note: str = ""  # status-line message, e.g. why the fallback was used

    @property
    def duration(self) -> float:
        """End of the last word (0 when the transcript is empty)."""
        return self.words[-1].end if self.words else 0.0

    @property
    def text(self) -> str:
        """Plain transcript text (words joined with spaces)."""
        return " ".join(word.text for word in self.words)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_plan(
    duration: float,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> list[tuple[float, float]]:
    """Split ``duration`` into ``(start, end)`` chunks.

    Each chunk is at most ``chunk_seconds`` long; consecutive chunks overlap
    by ``overlap_seconds`` so no word is lost at a boundary. The last chunk
    ends exactly at ``duration``.

    >>> chunk_plan(300)
    [(0.0, 300.0)]
    >>> chunk_plan(1000)
    [(0.0, 480.0), (475.0, 955.0), (950.0, 1000.0)]
    >>> chunk_plan(480.5)
    [(0.0, 480.0), (475.0, 480.5)]
    """
    if duration <= 0:
        return []
    if duration <= chunk_seconds:
        return [(0.0, float(duration))]
    step = chunk_seconds - overlap_seconds
    plan: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        plan.append((round(start, 3), round(end, 3)))
        if end >= duration:
            break
        start += step
    return plan


def split_wav(
    wav_bytes: bytes,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> list[tuple[float, bytes]]:
    """Split mono WAV bytes into ``(chunk_start_seconds, wav_bytes)``.

    The input must already be mono 16 kHz PCM16 (what
    :func:`core.ffmpeg.extract_audio` produces). Splitting is done on frame
    boundaries, so chunk audio is sample-accurate.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        n_frames = wav.getnframes()
        frames = wav.readframes(n_frames)
    if channels != 1 or sampwidth != 2:
        raise ValueError("split_wav expects mono 16-bit PCM WAV audio")
    duration = n_frames / framerate if framerate else 0.0
    chunks: list[tuple[float, bytes]] = []
    frame_size = channels * sampwidth
    for start_s, end_s in chunk_plan(duration, chunk_seconds, overlap_seconds):
        start_frame = int(round(start_s * framerate))
        end_frame = min(int(round(end_s * framerate)), n_frames)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(sampwidth)
            out.setframerate(framerate)
            out.writeframes(frames[start_frame * frame_size:end_frame * frame_size])
        chunks.append((start_s, buffer.getvalue()))
    return chunks


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def _same_word(a: Word, b: Word, tolerance: float = 0.15) -> bool:
    return a.text == b.text and abs(a.start - b.start) <= tolerance


def merge_chunks(chunks: list[Transcript]) -> Transcript:
    """Merge per-chunk transcripts, de-duplicating the overlap window.

    Words from a later chunk that duplicate a word already merged (same
    text, start within 0.15 s) are dropped — those are the overlap repeats.
    Only words near the merge boundary are compared, so this stays fast on
    long transcripts. Utterances are concatenated and exact duplicates
    dropped the same way.
    """
    if not chunks:
        return Transcript(language="auto")
    merged = Transcript(
        language=chunks[0].language,
        detected_language=chunks[0].detected_language,
        engine=chunks[0].engine,
        used_fallback=any(c.used_fallback for c in chunks),
        note=chunks[0].note,
    )
    words: list[Word] = []
    for chunk in chunks:
        if chunk.detected_language and not merged.detected_language:
            merged.detected_language = chunk.detected_language
        for word in sorted(chunk.words, key=lambda w: w.start):
            # Only words within ~1 s of this one can be overlap duplicates.
            duplicate = False
            for existing in reversed(words):
                if existing.start < word.start - 1.0:
                    break
                if _same_word(existing, word):
                    duplicate = True
                    break
            if not duplicate:
                words.append(word)
    merged.words = sorted(words, key=lambda w: w.start)
    utterances: list[Utterance] = []
    for chunk in chunks:
        for utt in chunk.utterances:
            if not any(
                abs(u.start - utt.start) <= 0.15 and u.text == utt.text
                for u in utterances
            ):
                utterances.append(utt)
    merged.utterances = sorted(utterances, key=lambda u: u.start)
    return merged


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_deepgram_response(data: dict, offset: float = 0.0) -> Transcript:
    """Parse a Deepgram ``/v1/listen`` response into a ``Transcript``.

    ``offset`` shifts every timestamp (the chunk's start in the full audio).
    Word timestamps come back by default — there is no ``word_timestamps``
    parameter (research §1.2).
    """
    language = "auto"
    detected: str | None = None
    words: list[Word] = []
    utterances: list[Utterance] = []
    results = data.get("results", {}) if isinstance(data, dict) else {}
    channels = results.get("channels", []) or []
    if channels:
        channel = channels[0] or {}
        detected = channel.get("detected_language")
        alternatives = channel.get("alternatives", []) or []
        if alternatives:
            alt = alternatives[0] or {}
            for item in alt.get("words", []) or []:
                words.append(
                    Word(
                        text=str(item.get("word", "")).strip(),
                        start=float(item.get("start", 0.0)) + offset,
                        end=float(item.get("end", 0.0)) + offset,
                        confidence=float(item.get("confidence", 1.0)),
                    )
                )
    for item in results.get("utterances", []) or []:
        utterances.append(
            Utterance(
                text=str(item.get("transcript", "")).strip(),
                start=float(item.get("start", 0.0)) + offset,
                end=float(item.get("end", 0.0)) + offset,
            )
        )
    return Transcript(
        language=language, detected_language=detected,
        words=words, utterances=utterances, engine="deepgram",
    )


def parse_whisper_result(result: dict) -> Transcript:
    """Parse a ``WhisperLocalClient.transcribe`` dict into a ``Transcript``."""
    words = [
        Word(
            text=str(item.get("text", "")).strip(),
            start=float(item.get("start", 0.0)),
            end=float(item.get("end", 0.0)),
            confidence=float(item.get("confidence", 1.0)),
        )
        for item in result.get("words", []) or []
    ]
    utterances = [
        Utterance(
            text=str(item.get("text", "")).strip(),
            start=float(item.get("start", 0.0)),
            end=float(item.get("end", 0.0)),
        )
        for item in result.get("utterances", []) or []
    ]
    return Transcript(
        language=result.get("detected_language") or "auto",
        detected_language=result.get("detected_language"),
        words=words, utterances=utterances, engine="whisper_local",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, str], None]


def _chunk_settings() -> tuple[float, float]:
    presets = load_presets()["deepgram"]
    return (
        float(presets.get("chunk_seconds", DEFAULT_CHUNK_SECONDS)),
        float(presets.get("chunk_overlap_seconds", DEFAULT_OVERLAP_SECONDS)),
    )


def _transcribe_with_whisper(
    wav_path: str,
    *,
    language: str,
    model_size: str,
    progress: ProgressCallback | None,
    device: str = "auto",
    cancel: Callable[[], bool] | None = None,
) -> Transcript:
    """Run local faster-whisper over the whole WAV file."""
    if progress is not None:
        progress(0, "Transcribing locally with Whisper (offline)…")
    client = WhisperLocalClient(model_size=model_size, device=device)
    # The first-use model download occupies the first 30% of this phase;
    # the status line shows its real percent + MB. When the model is
    # already cached the download callback is never invoked (no bar) and
    # segment progress uses the full 0-100 range instead. Segment
    # progress + cancel are polled per segment by the client (v0.2.1:
    # previously this reported 0% then 100% and ignored cancel for the
    # whole file, which is why the bar froze at 5% and Cancel hung).
    download_ran = {"ran": False}

    def _download_progress(p: int, m: str) -> None:
        download_ran["ran"] = True
        if progress is not None:
            progress(int(p * 0.3), m)

    def _segment_progress(p: int, m: str) -> None:
        if progress is not None:
            if download_ran["ran"]:
                progress(30 + int(p * 0.7), m)
            else:
                progress(p, m)

    result = client.transcribe(
        wav_path, language=None if language == "auto" else language,
        download_progress=_download_progress if progress is not None else None,
        progress=_segment_progress if progress is not None else None,
        cancel=cancel,
    )
    transcript = parse_whisper_result(result)
    transcript.language = language
    transcript.note = "Transcribed locally with Whisper (offline)."
    if progress is not None:
        progress(100, "Local transcription finished.")
    return transcript


def _assert_has_audio(video_path: str) -> None:
    """Raise ``no_audio`` only when probing proves there is no audio track.

    A probe failure (missing or corrupt file) is left alone — the caller
    already probed the file and reports that failure itself; the
    extraction below would fail the same way.
    """
    try:
        info = probe(video_path)
    except Exception:
        return
    if not info.has_audio:
        raise ProviderError(
            "no_audio",
            "This video has no audio track, so there is nothing to "
            "transcribe — pick a video with spoken audio.",
        )


def transcribe_audio(
    wav_path: str,
    *,
    language: str = "auto",
    engine: str = "deepgram",
    deepgram_key: str = "",
    whisper_model: str = "small",
    deepgram_model: str | None = None,
    whisper_device: str = "auto",
    deepgram_client: DeepgramClient | None = None,
    progress: ProgressCallback | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Transcript:
    """Transcribe a mono 16 kHz WAV file, chunked for Deepgram.

    Deepgram is used when ``engine == "deepgram"`` and a key is present;
    otherwise (or when Deepgram fails with an offline/unsupported-language
    error) local faster-whisper runs instead and the transcript's ``note``
    says so. A 401 (invalid key) is never silently worked around.
    """
    wav_bytes = Path(wav_path).read_bytes()
    if engine == "whisper_local":
        return _transcribe_with_whisper(
            wav_path, language=language, model_size=whisper_model,
            progress=progress, device=whisper_device, cancel=cancel,
        )
    if not deepgram_key and deepgram_client is None:
        # No key at all is NOT an error: the app must fully work with zero
        # API keys, so local Whisper covers transcription (with a note).
        # A *wrong* key is different — Deepgram answers 401, which stays a
        # loud error below and is never silently worked around.
        fallback = _transcribe_with_whisper(
            wav_path, language=language, model_size=whisper_model,
            progress=progress, device=whisper_device, cancel=cancel,
        )
        fallback.used_fallback = True
        fallback.note = (
            "No Deepgram API key — used local Whisper instead. "
            "Add a key in Settings > Transcription for cloud transcription."
        )
        return fallback
    client = deepgram_client or DeepgramClient(
        api_key=deepgram_key,
        **({"model": deepgram_model} if deepgram_model else {}),
    )
    chunk_seconds, overlap_seconds = _chunk_settings()
    chunks = split_wav(wav_bytes, chunk_seconds, overlap_seconds)
    transcripts: list[Transcript] = []
    try:
        for index, (start_s, chunk_bytes) in enumerate(chunks):
            if cancel is not None and cancel():
                raise ProviderError("cancelled", "Transcription cancelled")
            if progress is not None:
                progress(
                    int(100 * index / max(len(chunks), 1)),
                    f"Transcribing with Deepgram (part {index + 1} of {len(chunks)})…",
                )
            data = client.transcribe_chunk(chunk_bytes, language=language)
            parsed = parse_deepgram_response(data, offset=start_s)
            parsed.language = language
            transcripts.append(parsed)
    except ProviderError as exc:
        if exc.kind in FALLBACK_ERROR_KINDS:
            # Deepgram cannot serve this audio (offline / unsupported
            # language / server trouble) — fall back to local Whisper.
            fallback = _transcribe_with_whisper(
                wav_path, language=language, model_size=whisper_model,
                progress=progress, device=whisper_device, cancel=cancel,
            )
            fallback.used_fallback = True
            fallback.note = (
                f"Deepgram unavailable ({exc.message}) — "
                "used local Whisper instead."
            )
            return fallback
        raise
    merged = merge_chunks(transcripts)
    merged.language = language
    if not merged.words and language != "auto":
        # Deepgram answered but produced nothing (e.g. language dropped) —
        # same fallback, same status-line note.
        fallback = _transcribe_with_whisper(
            wav_path, language=language, model_size=whisper_model,
            progress=progress, device=whisper_device, cancel=cancel,
        )
        fallback.used_fallback = True
        fallback.note = (
            "Deepgram returned no words for this language — "
            "used local Whisper instead."
        )
        return fallback
    if progress is not None:
        progress(100, "Transcription finished.")
    return merged


def transcribe_video(
    video_path: str,
    *,
    language: str = "auto",
    engine: str = "deepgram",
    deepgram_key: str = "",
    whisper_model: str = "small",
    deepgram_model: str | None = None,
    whisper_device: str = "auto",
    deepgram_client: DeepgramClient | None = None,
    progress: ProgressCallback | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Transcript:
    """Transcribe a video file end to end.

    Extracts mono 16 kHz audio to a temp file, then transcribes it via
    :func:`transcribe_audio`. The temp file is removed afterwards.
    A video with no audio track raises ``ProviderError("no_audio")``
    instead of failing inside ffmpeg with an unreadable error.
    """
    _assert_has_audio(video_path)
    with tempfile.TemporaryDirectory(prefix="clipforge-audio-") as tmp:
        wav_path = str(Path(tmp) / "audio.wav")
        extract_audio(video_path, wav_path)
        return transcribe_audio(
            wav_path,
            language=language,
            engine=engine,
            deepgram_key=deepgram_key,
            whisper_model=whisper_model,
            deepgram_model=deepgram_model,
            whisper_device=whisper_device,
            deepgram_client=deepgram_client,
            progress=progress,
            cancel=cancel,
        )


#: Seconds of audio kept on each side of a manual range when extracting
#: it for transcription, so a word straddling the boundary is still
#: transcribed whole (words are then filtered to the range itself).
RANGE_TRANSCRIPTION_PADDING = 0.5


def transcribe_ranges(
    video_path: str,
    ranges: list[tuple[float, float]],
    *,
    duration: float,
    language: str = "auto",
    engine: str = "deepgram",
    deepgram_key: str = "",
    whisper_model: str = "small",
    deepgram_model: str | None = None,
    whisper_device: str = "auto",
    deepgram_client: DeepgramClient | None = None,
    progress: ProgressCallback | None = None,
    cancel: Callable[[], bool] | None = None,
    padding: float = RANGE_TRANSCRIPTION_PADDING,
) -> Transcript:
    """Transcribe only the given ``ranges`` of a video.

    Each range's audio is extracted (with ``padding`` on both sides so
    boundary words are not cut mid-word), transcribed through the same
    engine path as :func:`transcribe_audio` (Deepgram, or local Whisper
    when the engine is local / Deepgram cannot serve it), then the words
    are shifted back to video time and kept only when they start inside
    their range. The per-range transcripts are merged
    (:func:`merge_chunks` de-duplicates overlapping ranges), so the
    result is a drop-in replacement for a full-video transcript —
    downstream code (render jobs, labels) filters words per range
    exactly as before.

    This is what makes manual clipping fast on a long video: a 2 h
    podcast with ten 40 s ranges transcribes ~7 minutes of audio,
    not 2 hours (and Deepgram only bills for those minutes).
    """
    if not ranges:
        return Transcript(language=language, engine=engine)
    _assert_has_audio(video_path)
    total_audio = sum(max(0.0, end - start) for start, end in ranges)
    transcripts: list[Transcript] = []
    done_audio = 0.0
    with tempfile.TemporaryDirectory(prefix="clipforge-range-audio-") as tmp:
        for index, (start, end) in enumerate(ranges):
            if cancel is not None and cancel():
                raise ProviderError("cancelled", "Transcription cancelled")
            extract_start = max(0.0, start - padding)
            extract_end = min(duration, end + padding)
            wav_path = str(Path(tmp) / f"range-{index}.wav")
            extract_audio(
                video_path, wav_path,
                start=extract_start, duration=extract_end - extract_start,
            )
            range_audio = max(0.0, end - start)

            def _range_progress(
                p: int, m: str, *, _done=done_audio, _span=range_audio,
            ) -> None:
                if progress is not None:
                    if total_audio > 0:
                        overall = (_done + p / 100.0 * _span) / total_audio * 100.0
                    else:
                        overall = 100.0
                    progress(max(0, min(100, int(overall))), m)

            ranged = transcribe_audio(
                wav_path,
                language=language,
                engine=engine,
                deepgram_key=deepgram_key,
                whisper_model=whisper_model,
                deepgram_model=deepgram_model,
                whisper_device=whisper_device,
                deepgram_client=deepgram_client,
                progress=_range_progress,
                cancel=cancel,
            )
            words = [
                Word(
                    text=w.text,
                    start=w.start + extract_start,
                    end=w.end + extract_start,
                    confidence=w.confidence,
                )
                for w in ranged.words
                if start <= w.start + extract_start < end
            ]
            utterances = [
                Utterance(
                    text=u.text,
                    start=u.start + extract_start,
                    end=u.end + extract_start,
                    words=[
                        Word(
                            text=w.text,
                            start=w.start + extract_start,
                            end=w.end + extract_start,
                            confidence=w.confidence,
                        )
                        for w in u.words
                        if start <= w.start + extract_start < end
                    ],
                )
                for u in ranged.utterances
                if start <= u.start + extract_start < end
            ]
            transcripts.append(Transcript(
                language=ranged.language,
                detected_language=ranged.detected_language,
                words=words,
                utterances=utterances,
                engine=ranged.engine,
                used_fallback=ranged.used_fallback,
                note=ranged.note,
            ))
            done_audio += range_audio
    merged = merge_chunks(transcripts)
    merged.language = language
    return merged
