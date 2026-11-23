"""Settings-dialog wiring tests: debounce, background validation, status line.

Validation functions and key storage are mocked — no network, no keyring.
"""

import pytest
from PySide6.QtCore import QEventLoop, QTimer

from clipforge.app import create_app
from clipforge.providers.deepgram import DeepgramStatus
from clipforge.providers.deepseek import DeepSeekStatus
from clipforge.ui.settings_dialog import SettingsDialog


@pytest.fixture(scope="module")
def window(app_window):
    yield app_window


def _wait_until(predicate, timeout_ms: int = 5000) -> bool:
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    checker = QTimer()
    checker.setInterval(10)
    checker.timeout.connect(lambda: loop.quit() if predicate() else None)
    checker.start()
    loop.exec()
    checker.stop()
    return predicate()


@pytest.fixture
def mocked_keys(monkeypatch):
    """Replace key storage + validation with in-memory fakes."""
    saved: dict[str, str] = {}
    calls: dict[str, list] = {"deepgram": [], "deepseek": []}

    def fake_validate_deepgram(key, session=None):
        calls["deepgram"].append(key)
        if key == "dg-wrong":
            return DeepgramStatus(False, "Invalid API key (401)")
        return DeepgramStatus(True, "Connected")

    def fake_validate_deepseek(base_url, key, session=None):
        calls["deepseek"].append((base_url, key))
        if key == "sk-wrong":
            return DeepSeekStatus(False, "Invalid API key (401)")
        return DeepSeekStatus(
            True, "Connected",
            models=["deepseek-flash", "deepseek-v4-pro", "deepseek-next"],
        )

    monkeypatch.setattr(
        "clipforge.ui.settings_dialog.validate_deepgram", fake_validate_deepgram
    )
    monkeypatch.setattr(
        "clipforge.ui.settings_dialog.validate_deepseek", fake_validate_deepseek
    )
    monkeypatch.setattr("clipforge.ui.settings_dialog.has_key", lambda p: p in saved)
    monkeypatch.setattr("clipforge.ui.settings_dialog.get_key", saved.get)
    monkeypatch.setattr(
        "clipforge.ui.settings_dialog.save_key",
        lambda provider, key: saved.__setitem__(provider, key),
    )
    return {"saved": saved, "calls": calls}


def _open_dialog(window):
    dialog = SettingsDialog(window._settings, window)
    return dialog


def test_typing_deepgram_key_debounces_then_connects(window, mocked_keys) -> None:
    dialog = _open_dialog(window)
    dialog._dg_key.setText("dg-good")
    assert dialog._dg_status.text() == "Checking…"
    # Debounce: nothing validated yet, 1 s has not passed.
    assert mocked_keys["calls"]["deepgram"] == []
    dialog._on_dg_timer()  # simulate the 1 s timer firing
    assert _wait_until(lambda: dialog._dg_status.text() == "Connected")
    assert mocked_keys["calls"]["deepgram"] == ["dg-good"]
    # The key is saved encrypted and never shown again.
    assert mocked_keys["saved"]["deepgram"] == "dg-good"
    assert dialog._dg_key.text() == ""
    assert dialog._dg_key.placeholderText() == "Key saved — type to replace"
    dialog.close()


def test_wrong_deepgram_key_shows_exact_reason(window, mocked_keys) -> None:
    dialog = _open_dialog(window)
    dialog._dg_key.setText("dg-wrong")
    dialog._on_dg_timer()
    assert _wait_until(
        lambda: dialog._dg_status.text() == "Invalid API key (401)"
    )
    assert "dg-wrong" not in mocked_keys["saved"]
    assert dialog._dg_key.text() == "dg-wrong"  # kept so the user can fix it
    dialog.close()


def test_deepseek_key_fills_model_dropdown(window, mocked_keys) -> None:
    dialog = _open_dialog(window)
    dialog._ds_key.setText("sk-good")
    dialog._on_ds_timer()
    assert _wait_until(lambda: dialog._ds_status.text() == "Connected")
    items = [dialog._ds_model.itemText(i) for i in range(dialog._ds_model.count())]
    assert items == ["deepseek-flash", "deepseek-v4-pro", "deepseek-next"]
    assert mocked_keys["saved"]["deepseek"] == "sk-good"
    assert dialog._ds_key.text() == ""
    dialog.close()


def test_wrong_deepseek_key_shows_401(window, mocked_keys) -> None:
    dialog = _open_dialog(window)
    dialog._ds_key.setText("sk-wrong")
    dialog._on_ds_timer()
    assert _wait_until(
        lambda: dialog._ds_status.text() == "Invalid API key (401)"
    )
    dialog.close()


def test_paste_restarts_debounce(window, mocked_keys) -> None:
    """Typing again before the timer fires restarts the 1 s wait."""
    dialog = _open_dialog(window)
    dialog._dg_key.setText("dg-partial")
    assert dialog._dg_timer.isActive()
    dialog._dg_key.setText("dg-good")  # paste a full key over it
    assert dialog._dg_timer.isActive()
    assert mocked_keys["calls"]["deepgram"] == []
    dialog.close()


# -- caption settings (schema v2) ------------------------------------------

def test_caption_settings_schema_v3() -> None:
    """v3 defaults exist and round-trip (preset, colours, keyword size,
    layout + gradient JSON)."""
    from clipforge.settings import DEFAULTS, SCHEMA_VERSION, Settings

    assert SCHEMA_VERSION == 3
    for key in (
        "caption_preset", "caption_font", "custom_font",
        "caption_text_color", "caption_keyword_color",
        "caption_highlight_color", "caption_outline_color",
        "caption_box_color", "caption_color_scheme",
        "caption_keyword_scale", "caption_layout_json",
        "caption_gradient_json",
    ):
        assert key in DEFAULTS, f"missing caption default: {key}"
    settings = Settings()
    settings.set("caption_preset", "Gradient")
    settings.set("caption_keyword_color", "#00FF00")
    settings.set("caption_keyword_scale", 2.2)
    assert settings.get("caption_preset") == "Gradient"
    assert settings.caption_colors().keyword == "#00FF00"
    assert settings.get("caption_keyword_scale") == pytest.approx(2.2)
    assert settings.caption_colors().text == "#FFFFFF"
    # Layout + gradient round-trip through JSON; bad JSON falls back.
    from clipforge.core.captions import CaptionLayout, GradientConfig

    settings.set_caption_layout(CaptionLayout(words_per_line=2, align="left"))
    assert settings.caption_layout().words_per_line == 2
    assert settings.caption_layout().align == "left"
    settings.set_caption_gradient(
        GradientConfig(mode="solid", stops=((0.0, "#123456"),))
    )
    assert settings.caption_gradient().mode == "solid"
    settings.set("caption_layout_json", "not json{")
    assert settings.caption_layout() == CaptionLayout()
    settings.set("caption_gradient_json", "not json{")
    assert settings.caption_gradient() == GradientConfig()
    # Restore the defaults so later tests see them.
    settings.set("caption_preset", "Hormozi")
    settings.set("caption_keyword_color", "#FF2DC6")
    settings.set("caption_keyword_scale", 0.0)
    settings.set("caption_layout_json", "{}")
    settings.set("caption_gradient_json", "{}")
