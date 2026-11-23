"""Provider adapter tests: presets content + real behavior with mocked HTTP."""

import pytest
import requests

from clipforge.providers import load_presets
from clipforge.providers._http import ProviderError
from clipforge.providers.deepgram import DeepgramClient, one_second_silence_wav
from clipforge.providers.deepseek import DeepSeekClient
from clipforge.providers.whisper_local import WhisperLocalClient
from conftest import FakeResponse, FakeSession

MODELS_OK = {
    "object": "list",
    "data": [
        {"id": "deepseek-flash"},
        {"id": "deepseek-v4-pro"},
    ],
}


def test_presets_deepseek_models() -> None:
    deepseek = load_presets()["deepseek"]
    assert deepseek["default_model"] == "deepseek-flash"
    assert deepseek["fallback_model"] == "deepseek-v4-pro"
    # Legacy alias maps to the current model; it is never a default.
    assert deepseek["legacy_aliases"]["deepseek-v4-flash"] == "deepseek-flash"


def test_deepseek_client_urls() -> None:
    client = DeepSeekClient(base_url="https://api.deepseek.com/v1/", api_key="sk-test")
    assert client.base_url == "https://api.deepseek.com"
    assert client.models_url == "https://api.deepseek.com/models"
    assert client.chat_url == "https://api.deepseek.com/chat/completions"


def test_deepseek_validate_key_connected() -> None:
    session = FakeSession({("GET", "/models"): FakeResponse(200, MODELS_OK)})
    client = DeepSeekClient(api_key="sk-good", session=session, backoff_base=0)
    status = client.validate_key()
    assert status.connected
    assert status.message == "Connected"
    assert status.models == ["deepseek-flash", "deepseek-v4-pro"]
    call = session.calls[0]
    assert call["headers"]["Authorization"] == "Bearer sk-good"


def test_deepseek_validate_key_wrong_key_401() -> None:
    session = FakeSession({("GET", "/models"): FakeResponse(401, {"error": "bad key"})})
    client = DeepSeekClient(api_key="sk-wrong", session=session, backoff_base=0)
    status = client.validate_key()
    assert not status.connected
    assert status.message == "Invalid API key (401)"
    # The key never appears in the message.
    assert "sk-wrong" not in status.message


def test_deepseek_validate_key_no_balance_402() -> None:
    session = FakeSession({("GET", "/models"): FakeResponse(402)})
    client = DeepSeekClient(api_key="sk-broke", session=session, backoff_base=0)
    assert client.validate_key().message == "No balance — top up your account (402)"


def test_deepseek_validate_key_wrong_base_url_404() -> None:
    session = FakeSession({("GET", "/models"): FakeResponse(404)})
    client = DeepSeekClient(api_key="sk-x", session=session, backoff_base=0)
    assert client.validate_key().message == "Wrong base URL or model (404)"


def test_deepseek_chat_thinking_modes() -> None:
    """Discovery uses thinking enabled/high; captions disable thinking."""
    session = FakeSession(
        {("POST", "/chat/completions"): FakeResponse(200, {"choices": [{"message": {"content": "hi"}}]})}
    )
    client = DeepSeekClient(api_key="sk-test", session=session, backoff_base=0)
    assert client.chat_text([{"role": "user", "content": "x"}], thinking=True) == "hi"
    body = session.calls[0]["json"]
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"
    assert body["model"] == "deepseek-flash"
    assert "temperature" not in body  # thinking mode ignores temperature

    client.chat_text([{"role": "user", "content": "x"}], thinking=False)
    body = session.calls[1]["json"]
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body


def test_deepseek_chat_legacy_alias_resolves() -> None:
    session = FakeSession(
        {("POST", "/chat/completions"): FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})}
    )
    client = DeepSeekClient(api_key="sk-test", model="deepseek-v4-flash", session=session)
    client.chat_text([{"role": "user", "content": "x"}])
    assert session.calls[0]["json"]["model"] == "deepseek-flash"


def test_deepgram_validate_key_connected() -> None:
    session = FakeSession({("GET", "/v1/projects"): FakeResponse(200, {"projects": []})})
    client = DeepgramClient(api_key="dg-good", session=session, backoff_base=0)
    status = client.validate_key()
    assert status.connected
    assert status.message == "Connected"
    assert status.validated_by == "projects"
    assert session.calls[0]["headers"]["Authorization"] == "Token dg-good"


def test_deepgram_validate_key_wrong_key_401() -> None:
    session = FakeSession({("GET", "/v1/projects"): FakeResponse(401)})
    client = DeepgramClient(api_key="dg-wrong", session=session, backoff_base=0)
    status = client.validate_key()
    assert not status.connected
    assert status.message == "Invalid API key (401)"
    assert "dg-wrong" not in status.message


def test_deepgram_validate_403_falls_back_to_test_listen() -> None:
    """A listen-only key gets 403 on /v1/projects — validate by test listen."""
    session = FakeSession(
        {
            ("GET", "/v1/projects"): FakeResponse(403),
            ("POST", "/v1/listen"): FakeResponse(200, {"results": {}}),
        }
    )
    client = DeepgramClient(api_key="dg-listen-only", session=session, backoff_base=0)
    status = client.validate_key()
    assert status.connected
    assert status.validated_by == "test_transcription"
    # The fallback posts 1 s of silence to /v1/listen.
    listen_call = session.calls[1]
    assert listen_call["headers"]["Content-Type"] == "audio/wav"
    assert listen_call["data"] == one_second_silence_wav()


def test_deepgram_validate_403_then_test_listen_401_is_invalid() -> None:
    session = FakeSession(
        {
            ("GET", "/v1/projects"): FakeResponse(403),
            ("POST", "/v1/listen"): FakeResponse(401),
        }
    )
    client = DeepgramClient(api_key="dg-bad", session=session, backoff_base=0)
    status = client.validate_key()
    assert not status.connected
    assert status.message == "Invalid API key (401)"


def test_deepgram_transcribe_chunk_params() -> None:
    """Params: nova-3, language code, smart_format/punctuate/utterances.

    There is deliberately NO word_timestamps parameter — words come back
    by default (research §1.2).
    """
    payload = {
        "results": {
            "channels": [
                {"alternatives": [{"words": [{"word": "hello", "start": 0.1, "end": 0.4}]}]}
            ]
        }
    }
    session = FakeSession({("POST", "/v1/listen"): FakeResponse(200, payload)})
    client = DeepgramClient(api_key="dg-test", session=session, backoff_base=0)
    data = client.transcribe_chunk(b"RIFF....", language="bn")
    assert data is payload
    call = session.calls[0]
    assert call["headers"]["Authorization"] == "Token dg-test"
    assert call["params"]["model"] == "nova-3"
    assert call["params"]["language"] == "bn"
    assert call["params"]["smart_format"] == "true"
    assert call["params"]["utterances"] == "true"
    assert "word_timestamps" not in call["params"]


def test_deepgram_transcribe_chunk_auto_detect() -> None:
    session = FakeSession({("POST", "/v1/listen"): FakeResponse(200, {"results": {}})})
    client = DeepgramClient(api_key="dg-test", session=session, backoff_base=0)
    client.transcribe_chunk(b"RIFF....", language="auto")
    params = session.calls[0]["params"]
    assert params["detect_language"] == "true"
    assert "language" not in params


def test_deepgram_transcribe_chunk_402_no_balance() -> None:
    session = FakeSession({("POST", "/v1/listen"): FakeResponse(402)})
    client = DeepgramClient(api_key="dg-broke", session=session, backoff_base=0)
    with pytest.raises(ProviderError) as excinfo:
        client.transcribe_chunk(b"RIFF....", language="en")
    assert excinfo.value.kind == "no_balance"


def test_whisper_local_interface() -> None:
    client = WhisperLocalClient()
    assert client.model_size == "small"


def test_whisper_local_transcribe_uses_faster_whisper(monkeypatch) -> None:
    """faster-whisper is called with word timestamps; output is normalized."""
    import sys
    import types

    class FakeWord:
        def __init__(self, word, start, end, probability=0.9):
            self.word = word
            self.start = start
            self.end = end
            self.probability = probability

    class FakeSegment:
        def __init__(self):
            self.text = " hello world "
            self.start = 0.0
            self.end = 1.0
            self.words = [FakeWord("hello", 0.0, 0.5), FakeWord("world", 0.5, 1.0)]

    class FakeInfo:
        language = "en"

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def transcribe(self, path, **kwargs):
            self.transcribe_kwargs = kwargs
            return [FakeSegment()], FakeInfo()

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    client = WhisperLocalClient(model_size="tiny")
    result = client.transcribe("audio.wav", language="en")
    assert result["detected_language"] == "en"
    assert [w["text"] for w in result["words"]] == ["hello", "world"]
    assert result["words"][0]["confidence"] == 0.9
    assert result["utterances"][0]["text"] == "hello world"


def test_no_internet_is_friendly(monkeypatch) -> None:
    """A dead connection raises a friendly no-internet error, not a crash."""

    class DeadSession:
        def request(self, *args, **kwargs):
            raise requests.exceptions.ConnectionError("dns fail")

    client = DeepSeekClient(api_key="sk-test", session=DeadSession(), backoff_base=0)
    status = client.validate_key()
    assert not status.connected
    assert status.message == "No internet connection — check your network"


# ---------------------------------------------------------------------------
# Multi-provider support (DeepSeek / OpenAI / Gemini / custom endpoints)
# ---------------------------------------------------------------------------

def test_openai_base_url_keeps_v1() -> None:
    """OpenAI's /v1 must NOT be stripped (the old normalizer broke it)."""
    from clipforge.providers.deepseek import normalize_base_url

    assert normalize_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"
    assert normalize_base_url("https://api.openai.com/v1/") == "https://api.openai.com/v1"
    client = DeepSeekClient(base_url="https://api.openai.com/v1", api_key="<redacted>")
    assert client.chat_url == "https://api.openai.com/v1/chat/completions"
    assert client.models_url == "https://api.openai.com/v1/models"


def test_gemini_base_url_and_models() -> None:
    """Gemini's OpenAI-compatible endpoint: URL kept, /models listed live."""
    session = FakeSession(
        {("GET", "/models"): FakeResponse(200, {"data": [{"id": "gemini-2.5-flash"}]})}
    )
    client = DeepSeekClient(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key="<redacted>", session=session, backoff_base=0,
    )
    assert client.chat_url == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert client.list_models() == ["gemini-2.5-flash"]


def test_non_deepseek_chat_body_has_no_deepseek_thinking_fields() -> None:
    """OpenAI/Gemini must not receive DeepSeek's thinking parameters."""
    session = FakeSession(
        {("POST", "/chat/completions"): FakeResponse(200, {"choices": [{"message": {"content": "hi"}}]})}
    )
    client = DeepSeekClient(
        base_url="https://api.openai.com/v1", api_key="<redacted>",
        session=session, backoff_base=0,
    )
    assert client.chat_text([{"role": "user", "content": "x"}], thinking=True) == "hi"
    body = session.calls[0]["json"]
    assert "thinking" not in body
    assert "reasoning_effort" not in body


def test_deepgram_model_choice_reaches_listen_params() -> None:
    """Settings > Transcription model (Nova-2) must reach the API call."""
    from clipforge.providers.deepgram import build_listen_params, model_id_for_label

    assert model_id_for_label("Nova-3") == "nova-3"
    assert model_id_for_label("Nova-2") == "nova-2"
    assert model_id_for_label("Enhanced") == "enhanced"
    assert build_listen_params("en", model="nova-2")["model"] == "nova-2"
    assert build_listen_params("en")["model"] == "nova-3"  # preset default

    session = FakeSession({("POST", "/v1/listen"): FakeResponse(200, {"results": {}})})
    client = DeepgramClient(api_key="<redacted>", model="nova-2",
                            session=session, backoff_base=0)
    client.transcribe_chunk(b"wav-bytes", language="en")
    assert session.calls[0]["params"]["model"] == "nova-2"


# ---------------------------------------------------------------------------
# v0.2.1 C/D/E: Whisper model cache, per-segment progress, cancel
# ---------------------------------------------------------------------------

def _install_counting_whisper(monkeypatch, *, n_segments=3, duration=3.0):
    """Fake faster-whisper whose model counts its own constructions."""
    import sys
    import types

    from clipforge.providers import whisper_local

    whisper_local._MODEL_CACHE.clear()
    monkeypatch.setattr(
        whisper_local, "ensure_model_downloaded", lambda *a, **k: False
    )
    state = {"constructions": 0, "consumed": 0}

    class FakeWord:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end
            self.probability = 0.9

    class FakeSegment:
        def __init__(self, i):
            self.text = f" segment {i} "
            self.start = float(i)
            self.end = float(i + 1)
            self.words = [FakeWord(f"w{i}", self.start, self.end)]

    class FakeInfo:
        language = "en"

        def __init__(self):
            self.duration = duration

    class FakeModel:
        def __init__(self, *args, **kwargs):
            state["constructions"] += 1

        def transcribe(self, path, **kwargs):
            def _gen():
                for i in range(n_segments):
                    state["consumed"] += 1
                    yield FakeSegment(i)

            return _gen(), FakeInfo()

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    return state


def test_whisper_model_is_built_once_across_calls(monkeypatch) -> None:
    """C: transcribing many ranges must not reload the model per range."""
    state = _install_counting_whisper(monkeypatch)
    WhisperLocalClient(model_size="base", device="cpu").transcribe("a.wav")
    WhisperLocalClient(model_size="base", device="cpu").transcribe("b.wav")
    WhisperLocalClient(model_size="base", device="cpu").transcribe("c.wav")
    assert state["constructions"] == 1


def test_whisper_progress_reports_per_segment(monkeypatch) -> None:
    """E: progress climbs per segment instead of jumping 0 -> 100."""
    _install_counting_whisper(monkeypatch, n_segments=3, duration=3.0)
    seen: list[int] = []
    WhisperLocalClient(model_size="base", device="cpu").transcribe(
        "a.wav", progress=lambda p, _m: seen.append(p)
    )
    assert seen == [33, 66, 100]


def test_whisper_cancel_stops_between_segments(monkeypatch) -> None:
    """D: cancel fires mid-transcription, not after the whole file."""
    state = _install_counting_whisper(monkeypatch, n_segments=50, duration=50.0)
    with pytest.raises(ProviderError) as excinfo:
        WhisperLocalClient(model_size="base", device="cpu").transcribe(
            "a.wav", cancel=lambda: state["consumed"] >= 2
        )
    assert excinfo.value.kind == "cancelled"
    assert state["consumed"] <= 3  # stopped almost immediately
