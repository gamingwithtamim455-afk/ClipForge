"""Tests: ffprobe wrapper + encoder selection (system ffmpeg)."""

import subprocess

import pytest

from clipforge.core.ffmpeg import (
    FFmpegError,
    encoder_is_usable,
    encoder_quality_args,
    find_ffmpeg,
    find_ffprobe,
    list_video_encoders,
    probe,
    select_video_encoder,
)

def _make_src(tmp_path) -> str:
    path = str(tmp_path / "src.mp4")
    subprocess.run(
        [
            find_ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", path,
        ],
        check=True,
    )
    return path


def test_find_binaries() -> None:
    assert find_ffmpeg()
    assert find_ffprobe()


def test_probe(tmp_path) -> None:
    path = _make_src(tmp_path)
    result = probe(path)
    assert result.width == 1280
    assert result.height == 720
    assert result.duration == pytest.approx(3.0, abs=0.1)
    assert result.fps == pytest.approx(30.0, abs=0.1)
    assert result.has_audio


def test_probe_missing_file_raises() -> None:
    # The transcription layer's probe raises RuntimeError on failure.
    with pytest.raises(RuntimeError):
        probe("/nonexistent/video.mp4")


def test_list_video_encoders_has_x264() -> None:
    encoders = list_video_encoders()
    assert "libx264" in encoders


def test_select_video_encoder_auto() -> None:
    name, is_gpu = select_video_encoder("Auto")
    assert name in list_video_encoders()
    assert isinstance(is_gpu, bool)


def test_select_video_encoder_skips_unusable_gpu() -> None:
    # This VM's ffmpeg lists h264_nvenc but has no NVIDIA driver, so
    # Auto must fall back to libx264 rather than a broken GPU encoder.
    assert not encoder_is_usable("h264_nvenc")
    name, is_gpu = select_video_encoder("Auto")
    assert name == "libx264"
    assert is_gpu is False


def test_select_video_encoder_cpu() -> None:
    name, is_gpu = select_video_encoder("CPU")
    assert name == "libx264"
    assert is_gpu is False


def test_encoder_quality_args() -> None:
    assert encoder_quality_args("libx264", 15) == ["-crf", "15"]
    assert encoder_quality_args("h264_nvenc", 15) == [
        "-rc", "vbr", "-cq", "15", "-preset", "slow",
    ]
    assert encoder_quality_args("h264_qsv", 16) == [
        "-global_quality", "16", "-preset", "slow",
    ]
