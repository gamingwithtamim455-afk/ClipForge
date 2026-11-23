"""Robustness edge cases: no audio, empty transcript, corrupt files,
cancel mid-render, disk-full, long videos, missing fonts, temp cleanup.

Every test here proves the app fails *gracefully* (friendly error, no
crash, no partial file left behind as a finished clip).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from clipforge.core.ffmpeg import FFmpegError, ProbeResult, find_ffmpeg
from clipforge.core.pipeline import (
    PipelineConfig,
    PipelineError,
    render_pipeline_jobs,
    run_pipeline,
)
from clipforge.core.ranges import ClipRange
from clipforge.core.render import RenderError, RenderJob, RenderSettings, render_clip
from clipforge.core.transcription import Transcript, Word


def _make_video(tmp_path: Path, *, duration: float = 10.0, audio: bool = True) -> str:
    path = str(tmp_path / f"source-{'a' if audio else 'silent'}.mp4")
    audio_args = (
        ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
         "-c:a", "aac"]
        if audio else []
    )
    subprocess.run(
        [find_ffmpeg(), "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={duration}",
         *audio_args, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-shortest", path],
        check=True,
    )
    return path


def _fast_settings() -> RenderSettings:
    return RenderSettings(preset="veryfast", crf=23, width=640, height=360)


# -- no audio / corrupt / empty ---------------------------------------------

def test_pipeline_rejects_video_without_audio(tmp_path) -> None:
    video = _make_video(tmp_path, audio=False)
    with pytest.raises(PipelineError, match="no audio track"):
        run_pipeline(PipelineConfig(video_path=video, engine="whisper_local"))


def test_pipeline_rejects_corrupt_file(tmp_path) -> None:
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"this is not a video at all" * 100)
    with pytest.raises(PipelineError, match="Could not read this video"):
        run_pipeline(PipelineConfig(video_path=str(bad), engine="whisper_local"))


def test_pipeline_empty_transcript_reports_no_clips(tmp_path, monkeypatch) -> None:
    video = _make_video(tmp_path)

    class _FakeDeepSeek:
        def chat_text(self, messages, **kwargs):
            return json.dumps({"clips": []})

    monkeypatch.setattr(
        "clipforge.core.pipeline.transcribe_video",
        lambda *a, **k: Transcript(language="en", words=[]),
    )
    with pytest.raises(PipelineError, match="No viral clips found"):
        run_pipeline(
            PipelineConfig(video_path=video, engine="whisper_local"),
            deepseek_client=_FakeDeepSeek(),
        )


def test_pipeline_manual_range_overlap_is_a_friendly_error(tmp_path) -> None:
    video = _make_video(tmp_path, duration=20.0)
    ranges = [ClipRange(2.0, 8.0), ClipRange(6.0, 12.0)]  # overlap 6–8s
    with pytest.raises(PipelineError, match="overlap"):
        run_pipeline(
            PipelineConfig(
                video_path=video, engine="whisper_local", manual_ranges=ranges
            )
        )


def test_render_clip_without_audio_succeeds(tmp_path) -> None:
    video = _make_video(tmp_path, audio=False)
    out = str(tmp_path / "01_clip.mp4")
    result = render_clip(
        video, start=1.0, end=4.0, settings=_fast_settings(),
        index=1, words=[], language="en", output_path=out,
    )
    assert Path(result.video_path).exists()
    assert result.duration == pytest.approx(3.0, abs=0.1)


def test_render_clip_with_no_words_flags_it(tmp_path) -> None:
    video = _make_video(tmp_path)
    out = str(tmp_path / "01_clip.mp4")
    result = render_clip(
        video, start=1.0, end=4.0, settings=_fast_settings(),
        index=1, words=[], language="en", output_path=out,
    )
    assert any("no transcript words" in flag for flag in result.flags)


# -- cancel / disk-full / partial files --------------------------------------

def test_render_failure_leaves_no_partial_file(tmp_path, monkeypatch) -> None:
    video = _make_video(tmp_path)
    out = str(tmp_path / "01_clip.mp4")

    def _boom(*args, **kwargs):
        raise FFmpegError("ffmpeg failed (exit 1): No space left on device")

    monkeypatch.setattr("clipforge.core.render.run_ffmpeg", _boom)
    with pytest.raises(RenderError, match="encode failed"):
        render_clip(
            video, start=1.0, end=4.0, settings=_fast_settings(),
            index=1, words=[], language="en", output_path=out,
        )
    assert not Path(out).exists()
    assert not Path(str(out).replace(".mp4", ".part.mp4")).exists()
    assert not list(tmp_path.glob("*.part.mp4"))


def test_render_pipeline_jobs_cancel_between_clips(tmp_path) -> None:
    video = _make_video(tmp_path, duration=20.0)
    jobs = [
        RenderJob(video_path=video, start=1.0, end=4.0, index=1,
                  output_path=str(tmp_path / "01_clip.mp4"),
                  settings=_fast_settings()),
        RenderJob(video_path=video, start=5.0, end=8.0, index=2,
                  output_path=str(tmp_path / "02_clip.mp4"),
                  settings=_fast_settings()),
    ]
    def _cancel():
        # Cancel once the first clip is fully rendered. (Before v0.2.1
        # this was a call counter; cancel is now also polled DURING an
        # encode, so the flag must flip on state, not on call count.)
        return (tmp_path / "01_clip.mp4").exists()

    results = render_pipeline_jobs(jobs, cancel=_cancel)
    assert len(results) == 1
    assert Path(tmp_path / "01_clip.mp4").exists()
    assert not Path(tmp_path / "02_clip.mp4").exists()


def test_render_pipeline_jobs_failure_leaves_no_partial(tmp_path, monkeypatch) -> None:
    video = _make_video(tmp_path, duration=20.0)
    jobs = [
        RenderJob(video_path=video, start=1.0, end=4.0, index=1,
                  output_path=str(tmp_path / "01_clip.mp4"),
                  settings=_fast_settings()),
        RenderJob(video_path=video, start=5.0, end=8.0, index=2,
                  output_path=str(tmp_path / "02_clip.mp4"),
                  settings=_fast_settings()),
    ]
    real_run = __import__("clipforge.core.render", fromlist=["run_ffmpeg"]).run_ffmpeg
    state = {"calls": 0}

    def _fail_on_second(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] > 1:
            raise FFmpegError("ffmpeg failed (exit 1): No space left on device")
        return real_run(*args, **kwargs)

    monkeypatch.setattr("clipforge.core.render.run_ffmpeg", _fail_on_second)
    with pytest.raises(RenderError):
        render_pipeline_jobs(jobs)
    assert Path(tmp_path / "01_clip.mp4").exists()
    assert not Path(tmp_path / "02_clip.mp4").exists()
    assert not list(tmp_path.glob("*.part.mp4"))


# -- temp cleanup --------------------------------------------------------------

def _temp_clipforge_dirs(prefix: str) -> set[str]:
    return {
        p.name for p in Path(tempfile.gettempdir()).iterdir()
        if p.name.startswith(prefix)
    }


def test_transcribe_video_cleans_temp_on_failure(tmp_path, monkeypatch) -> None:
    from clipforge.core import transcription as tx

    video = _make_video(tmp_path)
    before = _temp_clipforge_dirs("clipforge-audio-")

    def _boom(*args, **kwargs):
        raise RuntimeError("whisper crashed")

    monkeypatch.setattr(tx, "transcribe_audio", _boom)
    with pytest.raises(RuntimeError, match="whisper crashed"):
        tx.transcribe_video(video, language="en", engine="whisper_local")
    assert _temp_clipforge_dirs("clipforge-audio-") == before


def test_transcribe_ranges_cleans_temp_on_cancel(tmp_path) -> None:
    from clipforge.core import transcription as tx

    video = _make_video(tmp_path, duration=20.0)
    before = _temp_clipforge_dirs("clipforge-range-audio-")
    calls = {"n": 0}

    def _cancel():
        calls["n"] += 1
        return calls["n"] > 1  # cancel before the second range

    with pytest.raises(Exception, match="[Cc]ancelled"):
        tx.transcribe_ranges(
            video, [(1.0, 4.0), (5.0, 8.0)], duration=20.0,
            language="en", engine="whisper_local", cancel=_cancel,
        )
    assert _temp_clipforge_dirs("clipforge-range-audio-") == before


# -- long video / missing fonts ----------------------------------------------

def test_long_video_duration_flows_through_pipeline(tmp_path, monkeypatch) -> None:
    """A 2 h probe duration must not break range handling or offsets."""
    video = _make_video(tmp_path, duration=20.0)
    real_probe = __import__("clipforge.core.pipeline", fromlist=["probe"]).probe

    def _long_probe(path: str) -> ProbeResult:
        info = real_probe(path)
        return ProbeResult(
            path=info.path, duration=7200.0, width=info.width,
            height=info.height, fps=info.fps, has_audio=info.has_audio,
        )

    monkeypatch.setattr("clipforge.core.pipeline.probe", _long_probe)
    result = run_pipeline(
        PipelineConfig(
            video_path=video, engine="whisper_local",
            manual_ranges=[ClipRange(5.0, 9.0)],
        )
    )
    assert result.used_manual_ranges is True
    assert [(r.start, r.end) for r in result.ranges] == [(5.0, 9.0)]
    assert len(result.render_jobs) == 1
    assert result.render_jobs[0].start == 5.0


def test_missing_fonts_dir_still_renders(tmp_path, monkeypatch) -> None:
    """Bundled fonts dir deleted: libass falls back to system fonts, no crash."""
    import clipforge.core.render as render_module

    video = _make_video(tmp_path)
    monkeypatch.setattr(
        render_module, "fonts_dir", lambda: tmp_path / "no-such-fonts-dir"
    )
    out = str(tmp_path / "01_clip.mp4")
    result = render_clip(
        video, start=1.0, end=4.0, settings=_fast_settings(),
        index=1,
        words=[Word("hello", 0.0, 0.5), Word("world", 0.5, 1.0)],
        language="en", output_path=out,
    )
    assert Path(result.video_path).exists()


# ---------------------------------------------------------------------------
# v0.2.1 D: cancel must complete within seconds, mid-stage
# ---------------------------------------------------------------------------

def test_run_ffmpeg_cancel_terminates_promptly(tmp_path) -> None:
    """A running encode is terminated, not waited out (D)."""
    import time

    from clipforge.core.ffmpeg import run_ffmpeg

    out = str(tmp_path / "long.mp4")
    started = time.monotonic()
    with pytest.raises(FFmpegError, match="cancelled"):
        run_ffmpeg(
            [
                "-f", "lavfi", "-i",
                "testsrc2=size=1920x1080:rate=30:duration=120",
                "-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p",
                out,
            ],
            timeout=600.0,
            cancel=lambda: time.monotonic() - started > 0.6,
        )
    assert time.monotonic() - started < 15.0


def test_render_clip_cancel_leaves_no_output(tmp_path) -> None:
    """Cancelling mid-encode: prompt stop, no partial clip left (D)."""
    import time

    from clipforge.core.face_tracking import CropPath, FaceBox

    video = _make_video(tmp_path, duration=30.0)
    out = str(tmp_path / "cancelled_clip.mp4")
    # A ready-made centred crop path skips face tracking so the test
    # reaches the encode immediately (640x360 source, 9:16 crop 202x360).
    path = CropPath(
        times=[0.0, 30.0],
        boxes=[FaceBox(219, 0, 202, 360), FaceBox(219, 0, 202, 360)],
        faces=[None, None], mode="track",
    )
    started = time.monotonic()
    with pytest.raises(RenderError, match="cancelled"):
        render_clip(
            video, start=0.0, end=30.0,
            settings=RenderSettings(preset="slow", crf=15),
            index=1, crop_path=path, words=[], language="en",
            output_path=out,
            cancel=lambda: time.monotonic() - started > 0.8,
        )
    assert time.monotonic() - started < 20.0
    assert not Path(out).exists()
    assert not list(tmp_path.glob("*.part.mp4"))


def test_pipeline_tracking_cancel_is_not_swallowed(tmp_path) -> None:
    """A cancel raised inside face tracking aborts the pipeline (D).

    The tracking stage's fallback (one bad range -> centred crop) must
    not swallow a cancellation and carry on.
    """
    video = _make_video(tmp_path, duration=10.0)
    state = {"cancel": False}

    class _FakeDeepgram:
        def transcribe_chunk(self, audio_bytes: bytes, language: str = "auto") -> dict:
            return {
                "results": {
                    "channels": [{
                        "detected_language": "en",
                        "alternatives": [{"words": [
                            {"word": "hello", "start": 0.5, "end": 0.9,
                             "confidence": 0.99},
                        ]}],
                    }],
                    "utterances": [],
                }
            }

    def _cancelled_tracker(video_path, **kwargs):
        # The user pressed Cancel mid-scan; the tracker aborts and the
        # cancel flag is set by the time the pipeline catches it.
        state["cancel"] = True
        raise RuntimeError("face tracking cancelled")

    config = PipelineConfig(
        video_path=video,
        language="en",
        engine="deepgram",
        manual_ranges=[ClipRange(1.0, 4.0)],
        render_settings=RenderSettings(
            width=270, height=480, preset="ultrafast", crf=28,
        ),
    )
    with pytest.raises(PipelineError, match="Cancelled"):
        run_pipeline(
            config,
            cancel=lambda: state["cancel"],
            deepgram_client=_FakeDeepgram(),
            track_faces_fn=_cancelled_tracker,
        )
