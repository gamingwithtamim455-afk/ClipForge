"""Local Whisper adapter (offline transcription via faster-whisper).

Used when the user selects Local Whisper, when Deepgram reports a language
unsupported, or when the app is offline. The status line must say so
explicitly when this adapter is in use (see ``core.transcription``).

Notes:
- The model is downloaded from Hugging Face on first use (needs internet
  once); afterwards it runs fully offline. The download goes through
  :func:`ensure_model_downloaded`, which hooks the Hugging Face progress
  (``snapshot_download(..., tqdm_class=...)``) into a ``(percent, message)``
  callback so the Transcription step status line shows a real progress
  bar (percent + MB downloaded). When the model is already cached the
  download is skipped entirely — no bar, no callback.
- ``faster-whisper`` is imported lazily inside :meth:`_build_model` so the
  app still starts (and its tests still run) on machines without the
  model files — and so tests can patch ``faster_whisper.WhisperModel``.
- Device ``auto`` tries CUDA first and falls back to CPU (int8) when CUDA
  is unavailable; the fallback is silent because CPU is a normal path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import threading
from typing import Callable

from huggingface_hub import snapshot_download

from ._http import ProviderError

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
DEFAULT_MODEL = "small"

#: Built Whisper models, keyed by (model_size, device, compute_type).
#: Loading a model takes seconds and hundreds of MB; before v0.2.1 a
#: fresh model was built for EVERY transcribe() call, so transcribing
#: ten manual ranges paid the load ten times. Models are immutable in
#: use (faster-whisper inference is stateless per call), so sharing one
#: instance across calls in the same process is safe.
_MODEL_CACHE: dict[tuple[str, str, str], object] = {}
_MODEL_CACHE_LOCK = threading.Lock()

#: Signature of a download-progress callback: (percent 0-100, message).
ProgressCallback = Callable[[int, str], None]


def whisper_repo_id(model_size: str) -> str:
    """Hugging Face repo for a faster-whisper model (or the id as given)."""
    if model_size in MODEL_SIZES:
        return f"Systran/faster-whisper-{model_size}"
    return model_size


@dataclass
class DownloadProgress:
    """Accumulates Hugging Face download bytes into percent + MB text.

    Fed by :class:`_ReportingTqdm` (one instance per downloaded file);
    each bar registers its file's total size on creation, so the percent
    is measured against the bytes that will actually be downloaded
    (files already cached are skipped by Hugging Face and never counted).
    The callback fires only when the integer percent changes (or, when
    the total size is unknown, when another whole MB has landed) so the
    UI is not spammed with a signal per network chunk.
    """

    model_size: str
    total_bytes: int = 0
    downloaded_bytes: int = 0
    on_progress: ProgressCallback | None = None
    _last_percent: int = field(default=-1, init=False, repr=False)
    _last_mb: int = field(default=-1, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def percent(self) -> int:
        if self.total_bytes <= 0:
            return 0
        return max(0, min(100, 100 * self.downloaded_bytes // self.total_bytes))

    @property
    def downloaded_mb(self) -> float:
        return self.downloaded_bytes / (1024 * 1024)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    def message(self) -> str:
        if self.total_bytes > 0:
            return (
                f"Downloading Whisper model ({self.model_size})… "
                f"{self.percent}% — {self.downloaded_mb:.0f} MB "
                f"of {self.total_mb:.0f} MB"
            )
        return (
            f"Downloading Whisper model ({self.model_size})… "
            f"{self.downloaded_mb:.0f} MB downloaded"
        )

    def register_file(self, total_bytes: int) -> None:
        """Add one file's size to the total (called by each tqdm bar)."""
        with self._lock:
            self.total_bytes += max(0, int(total_bytes or 0))

    def add(self, n_bytes: int) -> None:
        """Record ``n_bytes`` more downloaded and report if due."""
        with self._lock:
            self.downloaded_bytes += max(0, int(n_bytes))
            percent, message = self.percent, self.message()
            if self.total_bytes > 0:
                if percent == self._last_percent:
                    return
                self._last_percent = percent
            else:
                whole_mb = int(self.downloaded_mb)
                if whole_mb == self._last_mb:
                    return
                self._last_mb = whole_mb
        if self.on_progress is not None:
            self.on_progress(percent, message)


class _ReportingTqdm:
    """A tqdm-compatible progress bar that feeds a :class:`DownloadProgress`.

    Hugging Face creates one per file (``tqdm_class(total=file_size)``)
    and calls ``update(chunk_bytes)``; the visual bar is disabled (the
    app has no terminal — progress goes to the UI callback instead).
    """

    def __init__(self, *args, state: DownloadProgress | None = None, **kwargs):
        from tqdm import tqdm

        kwargs["disable"] = True
        self._tqdm = tqdm(*args, **kwargs)
        self._state = state
        if state is not None:
            state.register_file(self._tqdm.total or 0)

    def __getattr__(self, name: str):
        return getattr(self._tqdm, name)

    def update(self, n: int = 1):
        shown = self._tqdm.update(n)
        if self._state is not None:
            self._state.add(n)
        return shown


def model_is_cached(model_size: str) -> bool:
    """True when the model is already in the local HF cache."""
    try:
        snapshot_download(whisper_repo_id(model_size), local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 - not cached (or cache unreadable)
        return False


def ensure_model_downloaded(
    model_size: str,
    progress: ProgressCallback | None = None,
) -> bool:
    """Download the faster-whisper model if it is not cached yet.

    Returns True when a download ran, False when the model was already
    cached (then ``progress`` is never called — no bar) or when the
    download attempt failed (the caller still tries ``WhisperModel``,
    which retries the download itself and raises the real error).
    """
    if model_is_cached(model_size):
        return False
    repo_id = whisper_repo_id(model_size)
    # Total accumulates from the tqdm bars (only files actually
    # downloaded), so the bar reaches exactly 100% at the end.
    state = DownloadProgress(model_size=model_size, on_progress=progress)
    try:
        snapshot_download(
            repo_id, tqdm_class=partial(_ReportingTqdm, state=state)
        )
    except Exception:  # noqa: BLE001 - retried by WhisperModel below
        return False
    return True



@dataclass
class WhisperLocalClient:
    """Offline transcription with faster-whisper."""

    model_size: str = DEFAULT_MODEL
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    compute_type: str = "auto"  # "auto" | "int8" | "float16" | ...

    def _resolve_device(self) -> tuple[str, str]:
        """Return ``(device, compute_type)`` for the requested setting."""
        if self.device == "cuda":
            return "cuda", "float16" if self.compute_type == "auto" else self.compute_type
        if self.device == "cpu":
            return "cpu", "int8" if self.compute_type == "auto" else self.compute_type
        # auto: prefer CUDA, fall back to CPU at model-construction time.
        return "cuda", "float16" if self.compute_type == "auto" else self.compute_type

    def _build_model(self, download_progress: ProgressCallback | None = None):
        """Construct the faster-whisper model, with a CPU fallback.

        Downloads the model first (with progress) when it is not cached;
        when cached, no download runs and ``download_progress`` is never
        called. Built models are cached process-wide (see
        ``_MODEL_CACHE``) — repeat calls with the same size/device are
        effectively free.
        """
        from faster_whisper import WhisperModel

        ensure_model_downloaded(self.model_size, progress=download_progress)
        device, compute_type = self._resolve_device()
        key = (self.model_size, device, compute_type)
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
            actual_key = key
        except Exception:
            if device != "cpu":
                model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                actual_key = (self.model_size, "cpu", "int8")
            else:
                raise
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[actual_key] = model
        return model

    def transcribe(
        self, audio_path: str, *, language: str | None = None,
        download_progress: ProgressCallback | None = None,
        progress: ProgressCallback | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> dict:
        """Transcribe an audio file locally.

        ``language`` is a BCP-47 code (``"bn"``, ``"en"``, ...) or ``None``
        for auto-detect. ``download_progress`` is called (percent, message)
        only while the model is being downloaded on first use; when the
        model is already cached it is never called. ``progress`` is called
        per transcribed segment with the percent of the audio covered, so
        a long file reports steady progress instead of 0% then 100%
        (v0.2.1). ``cancel`` is polled per segment; when it fires,
        ``ProviderError("cancelled", ...)`` is raised within a segment's
        time instead of after the whole file (v0.2.1: Cancel used to wait
        for the entire transcription). Returns a dict with the same shape
        as ``core.transcription.parse_whisper_result`` expects::

            {"words": [{"text", "start", "end", "confidence"}],
             "utterances": [{"text", "start", "end"}],
             "detected_language": "en" | None}
        """
        model = self._build_model(download_progress=download_progress)
        segments, info = model.transcribe(
            audio_path, language=language, word_timestamps=True
        )
        total_duration = float(getattr(info, "duration", 0.0) or 0.0)
        words: list[dict] = []
        utterances: list[dict] = []
        for segment in segments:
            if cancel is not None and cancel():
                raise ProviderError("cancelled", "Transcription cancelled")
            utterances.append(
                {
                    "text": segment.text.strip(),
                    "start": float(segment.start),
                    "end": float(segment.end),
                }
            )
            for word in getattr(segment, "words", None) or []:
                words.append(
                    {
                        "text": word.word.strip(),
                        "start": float(word.start),
                        "end": float(word.end),
                        "confidence": float(getattr(word, "probability", 1.0)),
                    }
                )
            if progress is not None and total_duration > 0:
                percent = int(100 * float(segment.end) / total_duration)
                progress(
                    max(0, min(100, percent)),
                    "Transcribing locally with Whisper…",
                )
        return {
            "words": words,
            "utterances": utterances,
            "detected_language": getattr(info, "language", None),
        }
