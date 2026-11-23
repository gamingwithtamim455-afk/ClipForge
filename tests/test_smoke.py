"""Smoke tests: package imports, provider presets and pure helpers."""

from clipforge import __version__
from clipforge.providers import load_presets
from clipforge.providers.deepgram import LANGUAGE_CODES, build_listen_params
from clipforge.providers.deepseek import (
    DEFAULT_MODEL,
    LEGACY_ALIASES,
    normalize_base_url,
    resolve_model_id,
)


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_presets_load() -> None:
    presets = load_presets()
    assert presets["version"] == 1
    assert presets["deepseek"]["default_model"] == "deepseek-flash"
    assert presets["deepseek"]["fallback_model"] == "deepseek-v4-pro"
    assert presets["deepgram"]["default_model"] == "nova-3"


def test_deepseek_default_is_current_model() -> None:
    # deepseek-flash is the current model; deepseek-v4-flash is a retired
    # legacy alias and must never be the default.
    assert DEFAULT_MODEL == "deepseek-flash"
    assert resolve_model_id("deepseek-v4-flash") == "deepseek-flash"
    assert "deepseek-v4-flash" in LEGACY_ALIASES


def test_normalize_base_url() -> None:
    assert normalize_base_url("https://api.deepseek.com") == "https://api.deepseek.com"
    assert normalize_base_url("https://api.deepseek.com/") == "https://api.deepseek.com"
    assert normalize_base_url("https://api.deepseek.com/v1") == "https://api.deepseek.com"
    assert normalize_base_url(" https://api.deepseek.com/v1/ ") == "https://api.deepseek.com"


def test_deepgram_language_coverage() -> None:
    # nova-3 covers all 12 app languages; no per-language model switch.
    expected = {"auto", "bn", "en", "ar", "ur", "hi", "es", "fr", "tr", "id", "pt", "ru"}
    assert set(LANGUAGE_CODES) == expected


def test_deepgram_listen_params() -> None:
    params = build_listen_params("bn")
    assert params["model"] == "nova-3"
    assert params["language"] == "bn"
    assert params["smart_format"] == "true"
    assert "word_timestamps" not in params  # words come back by default
    auto = build_listen_params("auto")
    assert auto["detect_language"] == "true"
    assert "language" not in auto
