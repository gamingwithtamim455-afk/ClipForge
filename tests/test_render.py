"""Tests: rendering end-to-end (ffprobe-verified) + TikTok labels."""

import hashlib
import subprocess
import time
from pathlib import Path

import pytest

from clipforge.core.captions import build_ass, fonts_dir
from clipforge.core.face_tracking import FaceBox, track_faces
from clipforge.core.ffmpeg import find_ffmpeg, probe
from clipforge.core.render import (
    RenderError,
    _escape_filter_arg,
    RenderJob,
    RenderSettings,
    cleanup_clipforge_temp,
    generate_clip_label,
    render_clip,
    render_clips,
    write_label_file,
)
from clipforge.core.transcription import Word


def _make_source(tmp_path, *, duration: float = 5.0) -> str:
    path = str(tmp_path / "source.mp4")
    subprocess.run(
        [
            find_ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i",
            f"testsrc2=size=1280x720:rate=30:duration={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", path,
        ],
        check=True,
    )
    return path


def _words() -> list[Word]:
    # Mixed English + Bengali words across the clip.
    return [
        Word("Hello", 0.0, 0.4), Word("world", 0.4, 0.8),
        Word("আমি", 0.8, 1.2), Word("আসছি", 1.2, 1.7),
        Word("today", 1.7, 2.1), Word("we", 2.1, 2.3),
        Word("learn", 2.3, 2.8),
    ]


def _frame_md5(video: str, frame_no: int) -> str:
    out = subprocess.run(
        [
            find_ffmpeg(), "-y", "-v", "error",
            "-i", video, "-vf", f"select=eq(n\\,{frame_no})",
            "-frames:v", "1", "-f", "md5", "-",
        ],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip().split("=")[-1]


# -- end-to-end render -------------------------------------------------
def test_render_clip_e2e_no_face_fill_crop(tmp_path) -> None:
    """Spec defaults (CRF 15, preset slow) on a faceless clip.

    Verifies: 1080x1920 output, duration within 1 frame of 3.0 s,
    centred-fill-crop fallback flag (the letterboxing blur-fit graph
    was removed in v0.2.1), upscaling flag, label file with 4 hashtags.
    """
    src = _make_source(tmp_path)
    out = str(tmp_path / "clip01.mp4")
    result = render_clip(
        src, start=1.0, end=4.0,
        settings=RenderSettings(),  # defaults: 1080x1920, CRF 15, slow
        index=1, words=_words(), language="bn",
        output_path=out,
        label=("আজকের ভিডিওতে শিখুন!", ["#ভিডিও", "#শেখা", "#বাংলা", "#ভাইরাল"]),
    )
    assert Path(result.video_path).exists()
    assert result.duration == pytest.approx(3.0, abs=1 / 30 + 0.02)
    info = probe(result.video_path)
    assert (info.width, info.height) == (1080, 1920)
    assert info.duration == pytest.approx(3.0, abs=1 / 30 + 0.02)
    assert result.encoder == "libx264"  # no GPU encoder on this VM
    assert result.upscaled  # 720p source -> 1920 high
    assert any("centred fill crop" in f for f in result.flags)
    assert not any("blur-fit" in f for f in result.flags)
    assert result.label_written
    label_path = Path(result.label_path)
    assert label_path.name == "01_label.txt"
    assert label_path.parent == Path(result.video_path).parent
    text = label_path.read_text(encoding="utf-8")
    assert "আজকের ভিডিওতে শিখুন!" in text
    assert text.count("#") == 4


def test_render_clip_captions_change_pixels(tmp_path) -> None:
    """Burned captions must actually appear in the output frames."""
    src = _make_source(tmp_path, duration=3.0)
    fast = RenderSettings(preset="veryfast", crf=20)
    with_words = render_clip(
        src, start=0.0, end=2.0, settings=fast, index=1,
        words=_words(), language="en",
        output_path=str(tmp_path / "with.mp4"),
    )
    without_words = render_clip(
        src, start=0.0, end=2.0, settings=fast, index=2,
        words=None, language="en",
        output_path=str(tmp_path / "without.mp4"),
    )
    assert with_words.duration == pytest.approx(2.0, abs=1 / 30 + 0.02)
    assert without_words.duration == pytest.approx(2.0, abs=1 / 30 + 0.02)
    # Same frame number, different pixels => captions were burned in.
    assert _frame_md5(with_words.video_path, 45) != _frame_md5(
        without_words.video_path, 45
    )


def test_render_clip_rejects_bad_range(tmp_path) -> None:
    src = _make_source(tmp_path)
    with pytest.raises(RenderError):
        render_clip(src, start=4.0, end=6.0, output_path=str(tmp_path / "x.mp4"))
    with pytest.raises(RenderError):
        render_clip(src, start=2.0, end=2.0, output_path=str(tmp_path / "x.mp4"))


def test_render_clips_batch(tmp_path) -> None:
    src = _make_source(tmp_path, duration=4.0)
    fast = RenderSettings(preset="veryfast", crf=20)
    jobs = [
        RenderJob(src, 0.0, 1.0, index=1, settings=fast,
                  output_path=str(tmp_path / "c1.mp4")),
        RenderJob(src, 2.0, 3.0, index=2, settings=fast,
                  output_path=str(tmp_path / "c2.mp4")),
    ]
    results = render_clips(jobs)
    assert len(results) == 2
    assert all(Path(r.video_path).exists() for r in results)
    assert results[0].duration == pytest.approx(1.0, abs=1 / 30 + 0.02)
    assert results[1].duration == pytest.approx(1.0, abs=1 / 30 + 0.02)


# -- tracked crop path (mocked detector) -------------------------------
def test_track_faces_with_mocked_detector(tmp_path, monkeypatch) -> None:
    """A slowly moving face: mode track, QA >= 99%, crop follows it."""
    src = _make_source(tmp_path, duration=3.0)

    def fake_detect(frames, frame_w, frame_h, **_kwargs):
        detected = []
        for i in range(len(frames)):
            x = 0.35 + 0.01 * i  # drifts right, slowly
            detected.append([
                FaceBox(x=x, y=0.2, width=0.2, height=0.3, identity=1,
                        lip_open=0.05 + 0.01 * (i % 3))
            ])
        return detected, [False] * len(frames)

    monkeypatch.setattr(
        "clipforge.core.face_tracking._detect_faces_in_frames", fake_detect
    )
    path = track_faces(src, start=0.0, end=3.0)
    assert path.mode == "track"
    assert path.qa_face_inside_fraction >= 0.99
    assert path.qa_passed
    assert not path.needs_review
    # Crop is centered on the face (source is 1280x720; face center
    # starts at 0.45 * 1280 = 576).
    first = path.boxes[0]
    face_cx = 0.45 * 1280
    crop_cx = first.x + first.width / 2
    assert abs(crop_cx - face_cx) < 60


# -- TikTok labels -----------------------------------------------------
class _FakeDeepSeek:
    def __init__(self, content: str) -> None:
        self._content = content

    def chat_completions(self, messages, *, thinking=True, **_kwargs):
        return {"choices": [{"message": {"content": self._content}}]}


class _FailingDeepSeek:
    def chat_completions(self, messages, *, thinking=True, **_kwargs):
        raise NotImplementedError("provider not wired up")


def test_generate_clip_label_parses_four_hashtags() -> None:
    client = _FakeDeepSeek(
        "This clip will surprise you!\n\n#funny #viral #podcast #shorts"
    )
    result = generate_clip_label(
        client, transcript_text="hello world this is a test", language="en"
    )
    assert result is not None
    caption, hashtags = result
    assert caption == "This clip will surprise you!"
    assert hashtags == ["#funny", "#viral", "#podcast", "#shorts"]


def test_generate_clip_label_rejects_wrong_hashtag_count() -> None:
    client = _FakeDeepSeek("Nice clip!\n\n#one #two #three")
    assert generate_clip_label(
        client, transcript_text="hello", language="en"
    ) is None
    client5 = _FakeDeepSeek("Nice clip!\n\n#a #b #c #d #e")
    assert generate_clip_label(
        client5, transcript_text="hello", language="en"
    ) is None


def test_generate_clip_label_handles_provider_failure() -> None:
    assert generate_clip_label(
        _FailingDeepSeek(), transcript_text="hello", language="en"
    ) is None
    assert generate_clip_label(
        _FakeDeepSeek("x #a #b #c #d"), transcript_text="   ", language="en"
    ) is None


def test_write_label_file(tmp_path) -> None:
    video = tmp_path / "01_clip.mp4"
    video.touch()
    target = write_label_file(
        video, 1, "My caption", ["#a", "#b", "#c", "#d"]
    )
    assert target.name == "01_label.txt"
    assert target.parent == tmp_path
    assert target.read_text(encoding="utf-8") == "My caption\n\n#a #b #c #d\n"


# -- temp cleanup ------------------------------------------------------
def test_cleanup_clipforge_temp(tmp_path, monkeypatch) -> None:
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    old = tmp_path / "clipforge-render-old"
    old.mkdir()
    (old / "x.txt").write_text("x")
    os_old = time.time() - 8 * 86400
    import os

    os.utime(old, (os_old, os_old))
    fresh = tmp_path / "clipforge-render-new"
    fresh.mkdir()
    other = tmp_path / "something-else"
    other.mkdir()
    assert cleanup_clipforge_temp(max_age_days=7.0) == 1
    assert not old.exists()
    assert fresh.exists()
    assert other.exists()


# -- caption font / volume / face-safe placement --------------------------

def test_render_with_bundled_caption_font(tmp_path) -> None:
    """A bundled viral font (Anton) renders without error."""
    from clipforge.core.captions import build_ass

    src = _make_source(tmp_path)
    out = str(tmp_path / "clip-anton.mp4")
    result = render_clip(
        src, start=1.0, end=4.0,
        settings=RenderSettings(custom_font="Anton-Regular.ttf"),
        index=1, words=_words(), language="en", output_path=out,
    )
    assert Path(result.video_path).exists()
    assert result.duration == pytest.approx(3.0, abs=1 / 30 + 0.02)
    # The ASS the renderer builds for that font names the Anton family.
    ass = build_ass(_words(), custom_font="Anton")
    assert "Anton," in ass


def test_render_muted_clip_is_silent(tmp_path) -> None:
    """Volume 0.0 renders a (near-)silent audio track."""
    src = _make_source(tmp_path)
    out = str(tmp_path / "clip-muted.mp4")
    result = render_clip(
        src, start=1.0, end=4.0,
        settings=RenderSettings(volume=0.0),
        index=1, words=_words(), language="en", output_path=out,
    )
    assert "muted" in result.flags
    probe_out = subprocess.run(
        [
            find_ffmpeg(), "-i", out, "-vn",
            "-af", "volumedetect", "-f", "null", "-",
        ],
        capture_output=True, text=True, check=False,
    )
    assert "mean_volume" in probe_out.stderr
    mean = float(
        [l for l in probe_out.stderr.splitlines() if "mean_volume" in l][0]
        .split("mean_volume:")[1].split("dB")[0]
    )
    assert mean < -60.0  # effectively silent


def test_render_captions_move_when_face_low(tmp_path) -> None:
    """Face low in the frame -> captions move to the top (flagged)."""
    from clipforge.core.face_tracking import CropPath, FaceBox

    src = _make_source(tmp_path)
    out = str(tmp_path / "clip-facesafe.mp4")
    # Full-frame crop; face low (normalized y 0.75) in the source frame.
    crop = CropPath(
        times=[0.0, 1.0, 2.0],
        boxes=[
            FaceBox(x=0, y=0, width=1280, height=720),
            FaceBox(x=0, y=0, width=1280, height=720),
            FaceBox(x=0, y=0, width=1280, height=720),
        ],
        faces=[
            FaceBox(x=0.3, y=0.75, width=0.3, height=0.2),
            FaceBox(x=0.3, y=0.75, width=0.3, height=0.2),
            FaceBox(x=0.3, y=0.75, width=0.3, height=0.2),
        ],
        mode="track",
    )
    result = render_clip(
        src, start=1.0, end=4.0,
        settings=RenderSettings(),
        index=1, crop_path=crop, words=_words(), language="en",
        output_path=out,
    )
    assert Path(result.video_path).exists()
    assert "captions moved to the top to keep the face clear" in result.flags


def test_render_with_overlay_and_edited_words(tmp_path) -> None:
    """Edited words + a custom text overlay render into the clip."""
    from clipforge.core.caption_edit import delete_word
    from clipforge.core.captions import CaptionOverlay

    src = _make_source(tmp_path)
    out = str(tmp_path / "clip-edited.mp4")
    words = delete_word(_words(), 0)  # drop "Hello"
    result = render_clip(
        src, start=1.0, end=4.0,
        settings=RenderSettings(),
        index=1, words=words, language="en", output_path=out,
        overlays=[CaptionOverlay(text="MY TEXT", start=1.0, end=2.0)],
    )
    assert Path(result.video_path).exists()
    assert result.duration == pytest.approx(3.0, abs=1 / 30 + 0.02)


def test_render_keyword_pop_preset(tmp_path) -> None:
    """Keyword Pop renders: keyword word big + magenta, rest small."""
    src = _make_source(tmp_path)
    out = str(tmp_path / "clip-keyword-pop.mp4")
    result = render_clip(
        src, start=1.0, end=4.0,
        settings=RenderSettings(caption_preset="Keyword Pop"),
        index=1, words=_words(), language="en", output_path=out,
    )
    assert Path(result.video_path).exists()
    assert result.duration == pytest.approx(3.0, abs=1 / 30 + 0.02)
    # Frame at clip t=2.0 (line "আসছি today we", keyword "today"):
    # the big magenta keyword and the small white words must both be
    # visible in the pixels — not just a colour change.
    raw = subprocess.run(
        [find_ffmpeg(), "-y", "-v", "error", "-ss", "2.0",
         "-i", out, "-frames:v", "1", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    assert len(raw) == 1080 * 1920 * 3
    magenta = sum(
        1
        for i in range(0, len(raw), 3)
        if raw[i] > 200 and raw[i + 1] < 120 and raw[i + 2] > 150
    )
    white = sum(
        1
        for i in range(0, len(raw), 3)
        if raw[i] > 200 and raw[i + 1] > 200 and raw[i + 2] > 200
    )
    assert magenta > 1000  # big keyword glyphs in #FF2DC6
    assert white > 200  # small white words present


# -- box presets: the yellow box must actually draw ---------------------
# Regression: the Highlight Box preset rendered INVISIBLE text — the
# BorderStyle-3 box did not draw at all with Outline=0, and with
# Outline=1 it drew BLACK (this libass build draws the box in the
# OutlineColour, not BackColour). Both old variants give 0 yellow
# pixels, so this test fails on either of them.


def _render_ass_frame(tmp_path, ass_text: str, *, at: float) -> bytes:
    """One 1080x1920 RGB frame of ``ass_text`` on black, at ``at`` seconds."""
    ass_path = tmp_path / "box.ass"
    ass_path.write_text(ass_text, encoding="utf-8")
    raw = subprocess.run(
        [
            find_ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=black:s=1080x1920:r=30:d=2",
            "-vf",
            f"ass={_escape_filter_arg(str(ass_path))}"
            f":fontsdir={_escape_filter_arg(str(fonts_dir()))}",
            "-ss", str(at), "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True, check=True,
    ).stdout
    assert len(raw) == 1080 * 1920 * 3
    return raw


def _count_yellow(raw: bytes) -> int:
    return sum(
        1
        for i in range(0, len(raw), 3)
        if raw[i] > 200 and raw[i + 1] > 150 and raw[i + 2] < 100
    )


def _count_magenta(raw: bytes) -> int:
    return sum(
        1
        for i in range(0, len(raw), 3)
        if raw[i] > 200 and raw[i + 1] < 100 and raw[i + 2] > 150
    )


def test_highlight_box_renders_yellow_box(tmp_path) -> None:
    words = [
        Word(t, i * 0.4, (i + 1) * 0.4)
        for i, t in enumerate(["a", "quick", "brown", "fox"])
    ]
    ass = build_ass(words, preset_name="Highlight Box")
    # t=0.5: the word "quick" in its yellow box, black text on #FEC923.
    raw = _render_ass_frame(tmp_path, ass, at=0.5)
    assert _count_yellow(raw) > 1000


def test_keyword_boxes_render_yellow_and_magenta(tmp_path) -> None:
    words = [
        Word(t, i * 0.4, (i + 1) * 0.4)
        for i, t in enumerate(["a", "quick", "brown", "fox"])
    ]
    ass = build_ass(words, preset_name="Keyword Boxes")
    # t=0.5: "quick" in the yellow Box; t=1.0: "brown" in the magenta Box2.
    assert _count_yellow(_render_ass_frame(tmp_path, ass, at=0.5)) > 1000
    assert _count_magenta(_render_ass_frame(tmp_path, ass, at=1.0)) > 1000

