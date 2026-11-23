"""Tests for the faster-whisper first-use model download progress.

Covers the progress math (percent + MB), the tqdm adapter that feeds
it, the cached-model skip (no download, no bar), and the wiring into
``WhisperLocalClient.transcribe`` / ``_transcribe_with_whisper``.
No network is used: the Hugging Face calls are monkeypatched.
"""

from __future__ import annotations

import pytest

from clipforge.providers.whisper_local import (
    DownloadProgress,
    WhisperLocalClient,
    _ReportingTqdm,
    ensure_model_downloaded,
    model_is_cached,
    whisper_repo_id,
)

MB = 1024 * 1024


def _collect(seen: list):
    """Turn a list into a (percent, message) progress callback."""
    return lambda percent, message: seen.append((percent, message))


# -- DownloadProgress math -------------------------------------------------
def test_progress_percent_and_message() -> None:
    seen: list[tuple[int, str]] = []
    state = DownloadProgress(
        model_size="small", total_bytes=200 * MB, on_progress=_collect(seen)
    )
    assert state.percent == 0
    state.add(50 * MB)
    assert state.percent == 25
    assert seen == [(25, "Downloading Whisper model (small)… 25% — 50 MB of 200 MB")]
    state.add(50 * MB)
    assert state.percent == 50
    state.add(150 * MB)  # overshoot clamps at 100
    assert state.percent == 100
    assert seen[-1][0] == 100
    assert "100 MB of 200 MB" not in seen[-1][1]  # message uses real bytes
    assert "250 MB of 200 MB" in seen[-1][1]


def test_progress_throttles_to_percent_changes() -> None:
    seen: list[tuple[int, str]] = []
    state = DownloadProgress(
        model_size="tiny", total_bytes=1000 * MB, on_progress=_collect(seen)
    )
    # 1 MB chunks: the first chunk reports 0% at once (download started),
    # then the percent ticks 1, 2... as whole percents are crossed.
    for _ in range(25):
        state.add(1 * MB)
    assert [p for p, _ in seen] == [0, 1, 2]
    assert len(seen) == 3


def test_progress_unknown_total_reports_per_mb() -> None:
    seen: list[tuple[int, str]] = []
    state = DownloadProgress(model_size="base", on_progress=_collect(seen))
    assert state.percent == 0
    state.add(512 * 1024)  # half a MB: reports the download has started
    assert seen == [(0, "Downloading Whisper model (base)… 0 MB downloaded")]
    state.add(512 * 1024)  # first whole MB
    assert seen[-1] == (0, "Downloading Whisper model (base)… 1 MB downloaded")
    state.add(2 * MB)
    assert seen[-1] == (0, "Downloading Whisper model (base)… 3 MB downloaded")


def test_progress_no_callback_is_safe() -> None:
    state = DownloadProgress(model_size="small", total_bytes=10 * MB)
    state.add(5 * MB)  # no callback configured: must not raise
    assert state.percent == 50


def test_register_file_accumulates_total() -> None:
    state = DownloadProgress(model_size="small")
    state.register_file(100 * MB)
    state.register_file(90 * MB)
    assert state.total_bytes == 190 * MB
    state.add(95 * MB)
    assert state.percent == 50


# -- tqdm adapter ------------------------------------------------------------
def test_reporting_tqdm_feeds_progress() -> None:
    seen: list[tuple[int, str]] = []
    state = DownloadProgress(model_size="small", on_progress=_collect(seen))
    bar = _ReportingTqdm(total=1000, state=state)
    assert bar.disable is True  # no terminal spam from the app
    assert state.total_bytes == 1000
    bar.update(250)
    bar.update(250)
    bar.update(500)
    assert [p for p, _ in seen] == [25, 50, 100]
    assert "100% — 0 MB of 0 MB" in seen[-1][1]  # 1000 bytes < 1 MB


def test_whisper_repo_id() -> None:
    assert whisper_repo_id("small") == "Systran/faster-whisper-small"
    assert whisper_repo_id("large-v3") == "Systran/faster-whisper-large-v3"
    # A full repo id passes through unchanged.
    assert whisper_repo_id("my-org/my-model") == "my-org/my-model"


# -- cached vs download ------------------------------------------------------
def test_model_is_cached(monkeypatch) -> None:
    import clipforge.providers.whisper_local as wl

    calls: list[dict] = []

    def fake_download(repo_id, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})
        return "/cache/models--Systran--faster-whisper-small"

    monkeypatch.setattr(wl, "snapshot_download", fake_download)
    assert model_is_cached("small") is True
    assert calls[0]["local_files_only"] is True
    assert calls[0]["repo_id"] == "Systran/faster-whisper-small"


def test_model_is_cached_false_when_missing(monkeypatch) -> None:
    import clipforge.providers.whisper_local as wl

    def fake_download(repo_id, **kwargs):
        raise RuntimeError("not cached")

    monkeypatch.setattr(wl, "snapshot_download", fake_download)
    assert model_is_cached("small") is False


def test_ensure_skips_download_when_cached(monkeypatch) -> None:
    import clipforge.providers.whisper_local as wl

    monkeypatch.setattr(wl, "model_is_cached", lambda size: True)
    seen: list[tuple[int, str]] = []
    assert ensure_model_downloaded("small", progress=_collect(seen)) is False
    assert seen == []  # no download, no bar


def test_ensure_downloads_with_progress(monkeypatch) -> None:
    import clipforge.providers.whisper_local as wl

    monkeypatch.setattr(wl, "model_is_cached", lambda size: False)
    captured: dict = {}

    def fake_download(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        # Simulate HF creating one tqdm bar per file and streaming it.
        bar = kwargs["tqdm_class"](total=10 * MB)
        for _ in range(10):
            bar.update(1 * MB)
        return "/cache/snapshot"

    monkeypatch.setattr(wl, "snapshot_download", fake_download)
    seen: list[tuple[int, str]] = []
    assert ensure_model_downloaded("small", progress=_collect(seen)) is True
    assert captured["args"][0] == "Systran/faster-whisper-small"
    assert [p for p, _ in seen] == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert seen[-1][1].startswith("Downloading Whisper model (small)… 100%")


def test_ensure_download_failure_returns_false(monkeypatch) -> None:
    import clipforge.providers.whisper_local as wl

    monkeypatch.setattr(wl, "model_is_cached", lambda size: False)
    monkeypatch.setattr(
        wl, "snapshot_download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    seen: list[tuple[int, str]] = []
    assert ensure_model_downloaded("small", progress=_collect(seen)) is False
    assert seen == []


# -- client + transcription wiring -------------------------------------------
class _FakeModel:
    def transcribe(self, audio_path, language=None, word_timestamps=True):
        return [], _FakeInfo()


class _FakeInfo:
    language = "en"


def test_client_transcribe_calls_download_progress(monkeypatch, tmp_path) -> None:
    import clipforge.providers.whisper_local as wl
    import faster_whisper

    seen: list[tuple[int, str]] = []
    monkeypatch.setattr(
        wl, "ensure_model_downloaded",
        lambda size, progress=None: (progress(42, "fake download 42%") or True),
    )
    monkeypatch.setattr(faster_whisper, "WhisperModel", lambda *a, **k: _FakeModel())
    client = WhisperLocalClient(model_size="small")
    result = client.transcribe(
        str(tmp_path / "audio.wav"), download_progress=_collect(seen)
    )
    assert seen == [(42, "fake download 42%")]
    assert result["detected_language"] == "en"


def test_transcribe_with_whisper_scales_download_to_first_30_percent(monkeypatch, tmp_path) -> None:
    """The download bar occupies 0-30% of the transcription phase."""
    from clipforge.core.transcription import _transcribe_with_whisper

    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 100)
    import clipforge.core.transcription as tr

    calls: list[tuple[int, str]] = []

    class _FakeClient:
        def __init__(self, model_size=None, device="auto"):
            pass

        def transcribe(self, path, language=None, download_progress=None,
                       progress=None, cancel=None):
            # Simulate a first-use download reporting 0-100%.
            if download_progress is not None:
                download_progress(50, "Downloading Whisper model (small)… 50% — 1 MB of 2 MB")
                download_progress(100, "Downloading Whisper model (small)… 100% — 2 MB of 2 MB")
            return {"words": [], "utterances": [], "detected_language": "en"}

    monkeypatch.setattr(tr, "WhisperLocalClient", _FakeClient)
    transcript = _transcribe_with_whisper(
        str(wav), language="en", model_size="small",
        progress=lambda p, m: calls.append((p, m)),
    )
    # Download 50% -> 15% of the phase; 100% -> 30%; then 100% finished.
    assert (15, "Downloading Whisper model (small)… 50% — 1 MB of 2 MB") in calls
    assert (30, "Downloading Whisper model (small)… 100% — 2 MB of 2 MB") in calls
    assert calls[-1] == (100, "Local transcription finished.")
    assert transcript.note == "Transcribed locally with Whisper (offline)."
