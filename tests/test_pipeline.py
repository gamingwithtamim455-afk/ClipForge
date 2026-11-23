"""Integration tests: the full pipeline end to end (mocked providers).

A synthetic video is generated with ffmpeg; Deepgram/DeepSeek network
calls are replaced by fakes, so the suite runs offline. Face tracking is
replaced by a stub crop path (MediaPipe is exercised in
test_face_tracking.py). Rendering is real ffmpeg, verified with ffprobe.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from clipforge.core.discovery import ClipCandidate
from clipforge.core.face_tracking import CropPath, FaceBox
from clipforge.core.ffmpeg import find_ffmpeg, probe
from clipforge.core.pipeline import (
    PipelineConfig,
    PipelineError,
    build_render_jobs,
    friendly_error,
    render_pipeline_jobs,
    run_pipeline,
)
from clipforge.core.ranges import ClipRange
from clipforge.core.render import RenderSettings
from clipforge.core.transcription import Transcript, Word
from clipforge.providers._http import ProviderError

#: Words the fake Deepgram returns, spread over the first 20 s.
_FAKE_WORDS = [
    ("hello", 0.5, 0.9), ("world", 1.0, 1.4), ("this", 2.0, 2.3),
    ("is", 2.4, 2.6), ("a", 2.7, 2.8), ("test", 3.0, 3.5),
    ("about", 8.0, 8.4), ("viral", 8.5, 8.9), ("clips", 9.0, 9.4),
    ("and", 12.0, 12.2), ("faces", 12.3, 12.7), ("tracking", 13.0, 13.5),
]


def _make_video(tmp_path: Path, *, duration: float = 20.0) -> str:
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


class FakeDeepgram:
    """Stand-in for DeepgramClient: canned words for every chunk."""

    def transcribe_chunk(self, audio_bytes: bytes, language: str = "auto") -> dict:
        words = [
            {"word": text, "start": start, "end": end, "confidence": 0.99}
            for text, start, end in _FAKE_WORDS
        ]
        return {
            "results": {
                "channels": [{
                    "detected_language": "en",
                    "alternatives": [{"words": words}],
                }],
                "utterances": [{
                    "transcript": "hello world this is a test",
                    "start": 0.5, "end": 3.5,
                }],
            }
        }


class FakeDeepSeek:
    """Stand-in for DeepSeekClient: discovery + review + labels."""

    def __init__(self, *, fail_label: bool = False) -> None:
        self.fail_label = fail_label

    def chat_text(self, messages, *, thinking=True, **_kwargs) -> str:
        # Discovery and review both get the same two clips; the review
        # step matches them back to the proposed candidates.
        return json.dumps({
            "clips": [
                {"start": 2.0, "end": 6.0, "title": "Hello world",
                 "hookText": "hello world this", "concludingText": "is a test"},
                {"start": 10.0, "end": 14.0, "title": "Viral clips",
                 "hookText": "about viral clips", "concludingText": "faces tracking"},
            ]
        })

    def chat_completions(self, messages, *, thinking=True, **_kwargs) -> dict:
        if self.fail_label:
            raise ProviderError("server", "Server error (500) — try again shortly")
        return {
            "choices": [{
                "message": {
                    "content": "A quick test clip about going viral.\n\n"
                               "#test #clips #viral #clipforge"
                }
            }]
        }


def _fake_crop_path(*, duration: float = 4.0) -> CropPath:
    """Centered 9:16 crop for a 640x360 source (crop 202x360 at x=219)."""
    times = [float(t) for t in range(int(duration) + 1)]
    boxes = [FaceBox(x=219.0, y=0.0, width=202.0, height=360.0) for _ in times]
    return CropPath(times=times, boxes=boxes, faces=[None] * len(times))


def _small_settings() -> RenderSettings:
    """Fast test render settings (small frame, fast preset)."""
    return RenderSettings(width=360, height=640, crf=18, preset="veryfast",
                          captions="burn")


# ---------------------------------------------------------------------------
# Manual mode end to end
# ---------------------------------------------------------------------------

def test_manual_pipeline_end_to_end(tmp_path) -> None:
    """Manual ranges -> transcribe -> track -> render -> labels (real ffmpeg)."""
    video = _make_video(tmp_path)
    export_dir = tmp_path / "exports"
    config = PipelineConfig(
        video_path=video,
        language="en",
        engine="deepgram",
        manual_ranges=[ClipRange(start=2.0, end=6.0, label="Clip 1"),
                       ClipRange(start=10.0, end=14.0, label="Clip 2")],
        captions="burn",
        quality="1080p",
        export_dir=str(export_dir),
        project_name="test-project",
        render_settings=_small_settings(),
    )
    result = run_pipeline(
        config,
        deepgram_client=FakeDeepgram(),
        track_faces_fn=lambda video_path, **kw: _fake_crop_path(),
    )
    assert result.used_manual_ranges is True
    assert [(r.start, r.end) for r in result.ranges] == [(2.0, 6.0), (10.0, 14.0)]
    assert len(result.render_jobs) == 2
    assert result.render_jobs[0].output_path.endswith("01_clip.mp4")
    assert result.transcript.words, "transcript should have words"
    assert result.qa_flags == {0: False, 1: False}

    results = render_pipeline_jobs(
        result.render_jobs,
        label_client=FakeDeepSeek(),
        language="en",
        transcript=result.transcript,
    )
    assert len(results) == 2
    for res in results:
        assert Path(res.video_path).exists()
        info = probe(res.video_path)
        assert abs(info.duration - 4.0) <= 1 / 30 + 0.02  # within 1 frame
        assert (info.width, info.height) == (360, 640)
        assert res.label_written is True
        label_path = Path(res.label_path)
        assert label_path.exists()
        text = label_path.read_text(encoding="utf-8")
        hashtags = [w for w in text.split() if w.startswith("#")]
        assert len(hashtags) == 4, f"exactly 4 hashtags required, got: {text!r}"
        assert res.caption, "caption must not be empty"


def test_manual_mode_transcribes_only_ranges(tmp_path, monkeypatch) -> None:
    """Manual mode never asks for the full video's audio.

    ``transcribe_video`` (full-video) must not be called at all; the
    range-only path gets exactly the manual ranges, and the status line
    says so.
    """
    video = _make_video(tmp_path)
    ranges_calls: list[list[tuple[float, float]]] = []

    def fake_ranges(video_path, ranges, **kwargs):
        ranges_calls.append(list(ranges))
        return Transcript(
            language="en",
            words=[Word("hello", 2.5, 2.9), Word("world", 3.0, 3.4)],
        )

    full_calls: list[int] = []
    monkeypatch.setattr("clipforge.core.pipeline.transcribe_ranges", fake_ranges)
    monkeypatch.setattr(
        "clipforge.core.pipeline.transcribe_video",
        lambda *a, **k: full_calls.append(1),
    )
    messages: list[str] = []
    config = PipelineConfig(
        video_path=video,
        language="en",
        engine="deepgram",
        manual_ranges=[ClipRange(start=2.0, end=6.0)],
        export_dir=str(tmp_path / "exports"),
        project_name="range-only",
        render_settings=_small_settings(),
    )
    result = run_pipeline(
        config,
        deepgram_client=FakeDeepgram(),
        track_faces_fn=lambda video_path, **kw: _fake_crop_path(),
        progress=lambda p, m: messages.append(m),
    )
    assert full_calls == []
    assert ranges_calls == [[(2.0, 6.0)]]
    assert result.used_manual_ranges is True
    assert any("Transcribing your 1 range(s) (4s of audio)" in m for m in messages)
    assert any("AI discovery skipped" in m for m in messages)
    assert len(result.render_jobs) == 1
    assert [w.text for w in result.render_jobs[0].words] == ["hello", "world"]


# The 5-minute acceptance sample (real speech, real audio) — present on
# this dev machine only; CI skips it.
_SAMPLE_5MIN = Path("/home/hatch/.tmp/acceptance/sample5min.mp4")


@pytest.mark.skipif(not _SAMPLE_5MIN.exists(), reason="acceptance sample not present")
def test_manual_mode_real_whisper_transcribes_only_the_range(tmp_path) -> None:
    """Real faster-whisper, manual 30–70 s range: words stay in the range.

    Proves the range-only path end to end with a real engine: no word
    from outside 30–70 s may appear, and the render job is a 40 s clip
    with clip-relative caption words.
    """
    messages: list[str] = []
    config = PipelineConfig(
        video_path=str(_SAMPLE_5MIN),
        language="en",
        engine="whisper_local",
        manual_ranges=[ClipRange(start=30.0, end=70.0)],
        captions="burn",
        export_dir=str(tmp_path / "exports"),
        project_name="range-only-real",
        render_settings=_small_settings(),
    )
    result = run_pipeline(
        config,
        track_faces_fn=lambda video_path, **kw: _fake_crop_path(duration=40.0),
        progress=lambda p, m: messages.append(m),
    )
    assert result.used_manual_ranges is True
    assert result.transcript.words, "real Whisper should find words in 30–70 s"
    assert all(30.0 <= w.start < 70.0 for w in result.transcript.words), (
        "no word from outside the manual range may appear"
    )
    assert any("Transcribing your 1 range(s) (40s of audio)" in m for m in messages)
    assert len(result.render_jobs) == 1
    job = result.render_jobs[0]
    assert (job.start, job.end) == (30.0, 70.0)
    assert all(0.0 <= w.start < 40.0 for w in job.words)
    assert job.words[0].start == pytest.approx(
        result.transcript.words[0].start - 30.0
    )


def test_full_size_render_matches_spec(tmp_path) -> None:
    """Acceptance-test shape: 1080x1920, CRF 15, preset slow, 1-frame check."""
    video = _make_video(tmp_path, duration=10.0)
    config = PipelineConfig(
        video_path=video,
        language="en",
        engine="deepgram",
        manual_ranges=[ClipRange(start=2.0, end=4.0)],
        captions="burn",
        export_dir=str(tmp_path / "exports"),
        project_name="spec-check",
        render_settings=RenderSettings(width=1080, height=1920, crf=15,
                                       preset="slow", captions="burn"),
    )
    result = run_pipeline(
        config,
        deepgram_client=FakeDeepgram(),
        track_faces_fn=lambda video_path, **kw: _fake_crop_path(duration=2.0),
    )
    results = render_pipeline_jobs(result.render_jobs, label_client=None)
    assert len(results) == 1
    info = probe(results[0].video_path)
    assert (info.width, info.height) == (1080, 1920)
    assert abs(info.duration - 2.0) <= 1 / 30 + 0.02


# ---------------------------------------------------------------------------
# Auto (AI discovery) mode
# ---------------------------------------------------------------------------

def test_auto_pipeline_discovers_clips(tmp_path, monkeypatch) -> None:
    """No manual ranges -> DeepSeek discovery proposes the clips."""
    video = _make_video(tmp_path)
    monkeypatch.setattr(
        "clipforge.core.pipeline.get_key", lambda provider: "fake-key"
    )
    config = PipelineConfig(
        video_path=video,
        language="en",
        engine="deepgram",
        clip_length=30,
        clips_requested=2,
        export_dir=str(tmp_path / "exports"),
        project_name="auto",
        render_settings=_small_settings(),
    )
    result = run_pipeline(
        config,
        deepgram_client=FakeDeepgram(),
        deepseek_client=FakeDeepSeek(),
        track_faces_fn=lambda video_path, **kw: _fake_crop_path(),
    )
    assert result.used_manual_ranges is False
    assert [(r.start, r.end) for r in result.ranges] == [(2.0, 6.0), (10.0, 14.0)]
    assert all(r.source == "ai" for r in result.ranges)


# ---------------------------------------------------------------------------
# Failure paths: friendly errors, never a crash
# ---------------------------------------------------------------------------

class OfflineDeepgram:
    def transcribe_chunk(self, audio_bytes: bytes, language: str = "auto") -> dict:
        raise ProviderError("no_internet", "No internet connection — check your network")


def test_pipeline_offline_falls_back_to_whisper(tmp_path, monkeypatch) -> None:
    """Deepgram offline -> local Whisper fallback, with a status-line note.

    The real faster-whisper model download is stubbed out here (the
    fallback wiring is what's under test, not the model itself).
    """
    from clipforge.core.transcription import Transcript as _T
    from clipforge.core.transcription import Word as _W

    video = _make_video(tmp_path)
    fallback_transcript = _T(
        language="en",
        words=[_W("hello", 0.5, 0.9), _W("world", 1.0, 1.4)],
        engine="whisper_local",
        used_fallback=True,
        note="Deepgram unavailable — used local Whisper instead.",
    )
    monkeypatch.setattr(
        "clipforge.core.transcription._transcribe_with_whisper",
        lambda *a, **k: fallback_transcript,
    )
    config = PipelineConfig(
        video_path=video, language="en", engine="deepgram",
        manual_ranges=[ClipRange(start=2.0, end=6.0)],
        export_dir=str(tmp_path / "exports"),
        project_name="offline",
        render_settings=_small_settings(),
    )
    result = run_pipeline(
        config, deepgram_client=OfflineDeepgram(),
        track_faces_fn=lambda video_path, **kw: _fake_crop_path(),
    )
    assert result.transcript.engine == "whisper_local"
    assert result.transcript.used_fallback is True
    assert "Whisper" in (result.transcript.note or "")
    assert len(result.render_jobs) == 1


def test_pipeline_missing_deepgram_key_points_at_settings(tmp_path, monkeypatch) -> None:
    video = _make_video(tmp_path)
    monkeypatch.setattr("clipforge.core.pipeline.get_key", lambda provider: None)
    config = PipelineConfig(
        video_path=video, language="en", engine="deepgram",
        manual_ranges=[ClipRange(start=2.0, end=6.0)],
    )
    with pytest.raises(PipelineError, match="Settings"):
        run_pipeline(config, track_faces_fn=lambda video_path, **kw: _fake_crop_path())


def test_pipeline_missing_deepseek_key_in_auto_mode(tmp_path, monkeypatch) -> None:
    video = _make_video(tmp_path)
    monkeypatch.setattr("clipforge.core.pipeline.get_key", lambda provider: None)
    config = PipelineConfig(
        video_path=video, language="en", engine="deepgram",
        clips_requested=2,
    )
    with pytest.raises(PipelineError, match="Settings"):
        run_pipeline(config, deepgram_client=FakeDeepgram(),
                     track_faces_fn=lambda video_path, **kw: _fake_crop_path())


def test_pipeline_invalid_manual_ranges(tmp_path) -> None:
    video = _make_video(tmp_path)
    config = PipelineConfig(
        video_path=video, language="en", engine="deepgram",
        manual_ranges=[ClipRange(start=2.0, end=6.0),
                       ClipRange(start=5.0, end=9.0)],  # overlaps the first
    )
    with pytest.raises(PipelineError, match="not valid"):
        run_pipeline(config, deepgram_client=FakeDeepgram(),
                     track_faces_fn=lambda video_path, **kw: _fake_crop_path())


def test_pipeline_missing_video_file(tmp_path) -> None:
    config = PipelineConfig(video_path=str(tmp_path / "nope.mp4"))
    with pytest.raises(PipelineError, match="not found"):
        run_pipeline(config)


def test_face_tracking_failure_falls_back(tmp_path) -> None:
    """One bad range renders with a plain crop instead of killing the run."""
    video = _make_video(tmp_path)
    config = PipelineConfig(
        video_path=video, language="en", engine="deepgram",
        manual_ranges=[ClipRange(start=2.0, end=6.0)],
        export_dir=str(tmp_path / "exports"),
        project_name="fallback",
        render_settings=_small_settings(),
    )

    def _boom(video_path, **kw):
        raise RuntimeError("no faces here")

    result = run_pipeline(
        config, deepgram_client=FakeDeepgram(), track_faces_fn=_boom
    )
    assert result.crop_paths[0] is None
    results = render_pipeline_jobs(result.render_jobs, label_client=None)
    assert len(results) == 1
    assert Path(results[0].video_path).exists()


def test_label_failure_leaves_clip_flagged_not_faked(tmp_path) -> None:
    """DeepSeek label failure -> 'label pending' flag, no fake .txt."""
    video = _make_video(tmp_path)
    config = PipelineConfig(
        video_path=video, language="en", engine="deepgram",
        manual_ranges=[ClipRange(start=2.0, end=6.0)],
        export_dir=str(tmp_path / "exports"),
        project_name="no-label",
        render_settings=_small_settings(),
    )
    result = run_pipeline(
        config, deepgram_client=FakeDeepgram(),
        track_faces_fn=lambda video_path, **kw: _fake_crop_path(),
    )
    results = render_pipeline_jobs(
        result.render_jobs, label_client=FakeDeepSeek(fail_label=True),
        language="en", transcript=result.transcript,
    )
    assert results[0].label_written is False
    assert not (tmp_path / "exports" / "no-label" / "01_label.txt").exists()
    assert any("label pending" in flag for flag in results[0].flags)


# ---------------------------------------------------------------------------
# Friendly error mapping
# ---------------------------------------------------------------------------

def test_friendly_error_mapping() -> None:
    assert "internet" in friendly_error(
        ProviderError("no_internet", "No internet connection"))
    assert "Settings" in friendly_error(
        ProviderError("invalid_key", "Invalid API key (401)"))
    assert "Settings" in friendly_error(
        ProviderError("not_found", "Wrong base URL or model (404)"))
    assert "top up" in friendly_error(
        ProviderError("no_balance", "No balance — top up your account (402)"))
    assert friendly_error(PipelineError("Custom message")) == "Custom message"


def test_generate_caption_fallback_still_gives_four_hashtags() -> None:
    """When the label call returns != 4 tags, the caption fallback covers it."""
    from clipforge.core.pipeline import _generate_label
    from clipforge.core.render import RenderJob

    class ChattyDeepSeek(FakeDeepSeek):
        def chat_completions(self, messages, *, thinking=True, **_kwargs):
            # Label call: only 2 hashtags -> generate_clip_label returns None.
            return {"choices": [{"message": {"content": "Nice clip.\n\n#one #two"}}]}

        def chat_text(self, messages, *, thinking=True, **_kwargs):
            # Caption fallback: exactly 4 hashtags.
            return (
                '{"caption": "Nice clip.", '
                '"hashtags": ["#one", "#two", "#three", "#four"]}'
            )

    transcript = Transcript(
        language="en",
        words=[Word("hello", 2.0, 2.5), Word("world", 2.6, 3.0)],
    )
    job = RenderJob(video_path="x.mp4", start=2.0, end=6.0, words=transcript.words)
    label = _generate_label(ChattyDeepSeek(), job, "en", transcript)
    assert label is not None
    caption, hashtags = label
    assert caption
    assert len(hashtags) == 4


def test_render_jobs_retime_caption_words_to_clip(tmp_path) -> None:
    """Regression: caption words must be clip-relative, not source times.

    The renderer seeks to the range start, so a word at source 35.0s in a
    30-70s clip must appear at 5.0s in the output. Before the fix the words
    kept source times and captions only showed in the last seconds.
    """
    from clipforge.core.captions import build_ass

    words = [
        Word("before", 29.0, 29.5),
        Word("hello", 30.5, 31.0),
        Word("world", 35.0, 35.5),
        Word("after", 70.5, 71.0),
    ]
    jobs = build_render_jobs(
        video_path="x.mp4",
        ranges=[ClipRange(start=30.0, end=70.0, source="manual")],
        words=words,
        output_dir=tmp_path,
    )
    assert len(jobs) == 1
    job_words = jobs[0].words
    assert [w.text for w in job_words] == ["hello", "world"]
    assert job_words[0].start == pytest.approx(0.5)
    assert job_words[1].start == pytest.approx(5.0)
    assert job_words[1].end == pytest.approx(5.5)
    ass = build_ass(job_words, language="en")
    assert "0:00:00.50" in ass
    assert "0:00:05.50" in ass
    assert "0:00:35" not in ass
