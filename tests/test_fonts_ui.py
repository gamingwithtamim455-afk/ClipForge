"""UI tests (offscreen): Fonts tab — style preview cards, gradient
editor, layout controls. Each test restores the settings it changes."""

import pytest
from PySide6.QtGui import QPixmap

from clipforge.core.captions import (
    PRESETS,
    CaptionLayout,
    GradientConfig,
    ensure_preview_card,
    preview_card_path,
    previews_dir,
)
from clipforge.ui.settings_dialog import SettingsDialog


@pytest.fixture()
def dialog(app_window):
    dlg = SettingsDialog(app_window._settings, app_window)
    yield dlg
    # Restore the defaults this dialog may have changed.
    settings = app_window._settings
    settings.set("caption_preset", "Hormozi")
    settings.set_caption_layout(CaptionLayout())
    settings.set_caption_gradient(GradientConfig())
    dlg.close()


def test_preview_cards_all_presets_with_images(dialog) -> None:
    cards = dialog._preview_group.buttons()
    assert len(cards) == len(PRESETS) == 40
    names = {b.property("preset_name") for b in cards}
    assert names == set(PRESETS)
    # Every card shows a real rendered preview image. A preset whose
    # PNG is not bundled gets its card generated on demand by the app
    # itself — the dialog runs ensure_preview_card in a background job
    # and sets each icon as it lands; drive that same ensure path here
    # (synchronously), then refresh the buttons the same way.
    for name in PRESETS:
        path = ensure_preview_card(name)
        assert path.exists(), f"missing preview image: {name}"
    for button in cards:
        name = button.property("preset_name")
        assert dialog._set_card_icon(button, name), name
        assert not button.icon().isNull(), name
        card_path = preview_card_path(name)
        assert card_path is not None
        assert not QPixmap(str(card_path)).isNull(), name
    # A bundled card is used as-is, never regenerated into app data.
    assert preview_card_path("Hormozi") == previews_dir() / "hormozi.png"
    # The current preset's card is checked.
    checked = [b for b in cards if b.isChecked()]
    assert len(checked) == 1
    assert checked[0].property("preset_name") == "Hormozi"


def test_preview_card_click_selects_preset(dialog) -> None:
    for button in dialog._preview_group.buttons():
        if button.property("preset_name") == "Gradient":
            button.click()
            break
    assert dialog._settings.get("caption_preset") == "Gradient"
    assert dialog._caption_preset.currentData() == "Gradient"
    checked = [b for b in dialog._preview_group.buttons() if b.isChecked()]
    assert [b.property("preset_name") for b in checked] == ["Gradient"]


def test_layout_controls_persist(dialog) -> None:
    dialog._layout_words.setValue(2)
    assert dialog._settings.caption_layout().words_per_line == 2
    dialog._layout_align.setCurrentIndex(
        dialog._layout_align.findData("left")
    )
    assert dialog._settings.caption_layout().align == "left"
    dialog._layout_h_pct.setValue(25)
    dialog._layout_v_pct.setValue(30)
    layout = dialog._settings.caption_layout()
    assert layout.h_pct == 25.0
    assert layout.v_pct == 30.0
    dialog._layout_font_size.setValue(5.0)
    assert dialog._settings.caption_layout().font_size_pct == 5.0
    dialog._layout_lines.setCurrentIndex(
        dialog._layout_lines.findData(2)
    )
    assert dialog._settings.caption_layout().lines == 2
    dialog._layout_max_chars.setValue(10)
    assert dialog._settings.caption_layout().max_chars == 10


def test_gradient_editor_add_remove_and_solid(dialog) -> None:
    assert len(dialog._gradient_rows) == 2
    dialog._on_gradient_add_stop()
    assert len(dialog._settings.caption_gradient().stops) == 3
    assert len(dialog._gradient_rows) == 3
    dialog._on_gradient_remove_stop()
    assert len(dialog._settings.caption_gradient().stops) == 2
    # Solid mode collapses to one stop; back to Gradient restores two.
    dialog._gradient_mode.setCurrentIndex(
        dialog._gradient_mode.findData("solid")
    )
    assert dialog._settings.caption_gradient().mode == "solid"
    assert len(dialog._settings.caption_gradient().stops) == 1
    assert len(dialog._gradient_rows) == 1
    dialog._gradient_mode.setCurrentIndex(
        dialog._gradient_mode.findData("gradient")
    )
    assert dialog._settings.caption_gradient().mode == "gradient"
    assert len(dialog._settings.caption_gradient().stops) == 2
    # Adding stops caps at 5.
    for _ in range(5):
        dialog._on_gradient_add_stop()
    assert len(dialog._settings.caption_gradient().stops) == 5
