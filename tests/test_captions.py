"""Tests: ASS caption generation (fonts, presets, karaoke, RTL)."""

import pytest

from clipforge.core.captions import (
    PRESETS,
    CaptionLayout,
    GradientConfig,
    build_ass,
    font_family_for_language,
    font_path_for_language,
    fonts_dir,
    is_rtl_language,
)
from clipforge.core.transcription import Word


def _words() -> list[Word]:
    return [
        Word("Hello", 0.0, 0.4),
        Word("world", 0.4, 0.8),
        Word("this", 0.8, 1.0),
        Word("is", 1.0, 1.2),
        Word("a", 1.2, 1.3),
        Word("test", 1.3, 1.8),
        Word("clip", 1.8, 2.2),
    ]


def test_font_files_exist() -> None:
    for lang in ("bn", "ar", "ur", "en"):
        path = font_path_for_language(lang)
        assert path.exists(), f"missing bundled font for {lang}: {path}"
        assert path.stat().st_size > 100_000


def test_font_family_for_language() -> None:
    assert font_family_for_language("bn") == "Noto Sans Bengali"
    assert font_family_for_language("ar") == "Noto Naskh Arabic"
    assert font_family_for_language("ur") == "Noto Nastaliq Urdu"
    assert font_family_for_language("en") == "Noto Sans"
    assert font_family_for_language("xx") == "Noto Sans"  # unknown -> Latin


def test_rtl_languages() -> None:
    assert is_rtl_language("ar")
    assert is_rtl_language("ur")
    assert not is_rtl_language("bn")
    assert not is_rtl_language("en")


def test_ass_header_and_style() -> None:
    ass = build_ass(_words(), language="en", preset_name="Hormozi")
    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass
    assert "Noto Sans," in ass
    # Hormozi: yellow highlight as the karaoke (secondary) colour.
    assert "&H0023C9FE" in ass
    # Bottom-center alignment, TikTok safe-zone margin.
    assert ",2,120,120,268,1" in ass


def test_ass_karaoke_word_highlight() -> None:
    ass = build_ass(_words(), language="en")
    # Every word gets a {\k<centiseconds>} tag.
    assert ass.count("{\\k") == 7
    assert "{\\k40}Hello" in ass
    assert "{\\k40}world" in ass


def test_ass_groups_words_into_lines() -> None:
    # 7 words -> 2 lines (max 5 words per line).
    ass = build_ass(_words(), language="en")
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) == 2
    assert dialogues[0].startswith("Dialogue: 0,0:00:00.00,0:00:01.30,")
    assert dialogues[1].startswith("Dialogue: 0,0:00:01.30,0:00:02.20,")


def test_ass_bengali_uses_bengali_font() -> None:
    words = [Word("আমি", 0.0, 0.4), Word("আসছি", 0.4, 0.9)]
    ass = build_ass(words, language="bn")
    assert "Noto Sans Bengali," in ass
    assert "আমি" in ass and "আসছি" in ass


def test_ass_arabic_and_urdu_fonts() -> None:
    ar = build_ass([Word("مرحبا", 0.0, 0.5)], language="ar")
    assert "Noto Naskh Arabic," in ar
    assert "مرحبا" in ar
    ur = build_ass([Word("سلام", 0.0, 0.5)], language="ur")
    assert "Noto Nastaliq Urdu," in ur
    assert "سلام" in ur


def test_ass_presets_differ() -> None:
    hormozi = build_ass(_words(), preset_name="Hormozi")
    beast = build_ass(_words(), preset_name="Beast")
    minimal = build_ass(_words(), preset_name="Minimal")
    assert "Hormozi" not in hormozi  # preset name is not in the document
    # Font sizes differ per preset (64 / 76 / 56 at 1920 high).
    assert ",64," in hormozi
    assert ",76," in beast
    assert ",56," in minimal
    # The original three are unchanged; animation presets were added.
    assert {"Hormozi", "Beast", "Minimal"} <= set(PRESETS)
    assert set(PRESETS) == {
        "Hormozi", "Beast", "Minimal",
        "Karaoke", "Typewriter", "Slide Up", "Bounce", "Fade", "Highlight Box",
        "Keyword Pop", "Keyword Italic", "Impact", "Gary Vee", "Glow",
        "Gradient", "Lower Third", "Outline", "Clean Pop",
        "Headline Script", "Emerald Glow", "Stacked", "Keyword Boxes",
        # FuseClip preset names (v0.2.2, Boss's parity request).
        "Mumbai", "Numaya", "Josh", "Aks", "Rang", "Noor", "Rawani",
        "Lehar", "Orbit", "Bayan", "Headline", "Lakeer", "Uchhaal",
        "Ali Abdaal", "Karachi", "Lafz", "Shamani", "Saaf",
    }


def _dialogues(ass: str) -> list[str]:
    return [line for line in ass.splitlines() if line.startswith("Dialogue:")]


def test_animation_pop_is_original_karaoke() -> None:
    ass = build_ass(_words(), preset_name="Hormozi")
    assert ass.count("{\\k") == 7
    assert "{\\kf" not in ass
    assert "\\move(" not in ass and "\\fad(" not in ass and "\\t(" not in ass
    assert len(_dialogues(ass)) == 2  # grouped lines, as before


def test_animation_karaoke_fill() -> None:
    ass = build_ass(_words(), preset_name="Karaoke")
    assert "{\\kf40}Hello" in ass
    assert "{\\k40}Hello" not in ass


def test_animation_typewriter_accumulates() -> None:
    ass = build_ass(_words(), preset_name="Typewriter")
    assert "{\\k" not in ass
    dialogues = _dialogues(ass)
    assert len(dialogues) == 7  # one event per word
    assert dialogues[0].endswith(",{\\pos(540,1652)}Hello")
    assert dialogues[1].endswith(",{\\pos(540,1652)}Hello world")
    # Each prefix ends when the next word starts (only one visible).
    assert dialogues[0].startswith("Dialogue: 0,0:00:00.00,0:00:00.40,")
    assert dialogues[1].startswith("Dialogue: 0,0:00:00.40,0:00:00.80,")


def test_animation_slide_up() -> None:
    ass = build_ass(_words(), preset_name="Slide Up")
    assert "\\move(540,1832,540,1652,0,300)" in ass  # 1080x1920, margin 268
    assert "{\\k40}Hello" in ass  # karaoke highlight still applies


def test_animation_bounce() -> None:
    ass = build_ass(_words(), preset_name="Bounce")
    assert "\\t(0,150,\\fscx80\\fscy80)" in ass
    assert "\\t(150,300,\\fscx108\\fscy108)" in ass
    assert "\\t(300,450,\\fscx100\\fscy100)" in ass
    assert "{\\k40}Hello" in ass


def test_animation_fade() -> None:
    ass = build_ass(_words(), preset_name="Fade")
    assert "{\\fad(250,0)}" in ass
    assert "{\\k40}Hello" in ass


def test_animation_highlight_box() -> None:
    ass = build_ass(_words(), preset_name="Highlight Box")
    assert "Style: Box," in ass
    assert "&H0023C9FE" in ass  # yellow box (OutlineColour AND BackColour)
    # BorderStyle 3 = opaque box, Outline 1 (this libass build only
    # draws the box when Outline > 0, and draws it in the outline
    # colour — see the Highlight Box regression test in test_render.py).
    assert ",3,1,0,2," in ass
    dialogues = _dialogues(ass)
    assert len(dialogues) == 7  # one event per word
    assert all(",Box,,0,0,0,," in d for d in dialogues)
    assert "{\\k" not in ass
    assert dialogues[0].endswith(",,Hello")


def test_ass_timestamps_stay_clip_relative() -> None:
    words = [Word("hello", 30.5, 31.0), Word("world", 35.0, 35.5)]
    for preset in ("Hormozi", "Slide Up", "Highlight Box"):
        ass = build_ass(words, preset_name=preset)
        assert "0:00:30.50" in ass
        assert "0:00:35.50" in ass


def test_ass_custom_font_applies_to_all_presets() -> None:
    for preset in PRESETS:
        ass = build_ass(_words(), preset_name=preset, custom_font="My Font")
        assert "My Font," in ass
        assert "Noto Sans," not in ass
    # Without a custom font the language font is used, as before.
    assert "Noto Sans," in build_ass(_words(), language="en")


def test_ass_scales_with_resolution() -> None:
    ass = build_ass(_words(), language="en", width=1440, height=2560)
    assert "PlayResX: 1440" in ass
    assert "PlayResY: 2560" in ass
    # Font size scales with height: 64 * 2560/1920 = 85.
    assert ",85," in ass


def test_ass_escapes_special_chars() -> None:
    words = [Word("a{b}c\\d", 0.0, 0.5)]
    ass = build_ass(words, language="en")
    assert r"a\{b\}c\\d" in ass


# -- face-safe placement -------------------------------------------------

def test_placement_bottom_when_face_high() -> None:
    from clipforge.core.captions import (
        CaptionPlacement,
        caption_zone_clear,
        choose_caption_placement,
    )

    # Face in the top half: bottom zone is clear.
    placement, safe = choose_caption_placement(
        [(300.0, 200.0, 400.0, 500.0)],
        width=1080, height=1920, font_size=64, margin_v=268,
    )
    assert safe
    assert placement.alignment == 2  # bottom-center
    assert caption_zone_clear(
        [(300.0, 200.0, 400.0, 500.0)], placement, width=1080, height=1920
    )


def test_placement_moves_up_when_face_low() -> None:
    from clipforge.core.captions import choose_caption_placement

    # Face extends into the bottom caption zone -> captions move to top.
    placement, safe = choose_caption_placement(
        [(300.0, 1200.0, 400.0, 500.0)],
        width=1080, height=1920, font_size=64, margin_v=268,
    )
    assert safe
    assert placement.alignment == 8  # top-center


def test_placement_no_faces_stays_bottom() -> None:
    from clipforge.core.captions import choose_caption_placement

    placement, safe = choose_caption_placement(
        [], width=1080, height=1920, font_size=64, margin_v=268
    )
    assert safe and placement.alignment == 2


def test_build_ass_moves_captions_when_face_low() -> None:
    # Style alignment flips to 8 (top) when the face covers the bottom.
    low_face = build_ass(
        _words(), preset_name="Hormozi",
        face_rects=[(300.0, 1200.0, 400.0, 500.0)],
    )
    assert ",8,120,120,268,1" in low_face
    high_face = build_ass(
        _words(), preset_name="Hormozi",
        face_rects=[(300.0, 200.0, 400.0, 500.0)],
    )
    assert ",2,120,120,268,1" in high_face


# -- colours --------------------------------------------------------------

def test_hex_to_ass_color() -> None:
    from clipforge.core.captions import CaptionColors, hex_to_ass_color

    assert hex_to_ass_color("#FEC923") == "&H0023C9FE"
    assert hex_to_ass_color("#FFFFFF") == "&H00FFFFFF"
    assert hex_to_ass_color("#000") == "&H00000000"
    assert hex_to_ass_color("#123456") == "&H00563412"
    assert CaptionColors().to_ass() == (
        "&H00FFFFFF", "&H0023C9FE", "&H00000000", "&H0023C9FE", "&H00C62DFF",
    )


def test_custom_colors_appear_in_ass() -> None:
    from clipforge.core.captions import CaptionColors

    colors = CaptionColors(
        text="#111111", highlight="#2563EB",
        outline="#FFFFFF", box="#000000",
    )
    ass = build_ass(_words(), preset_name="Hormozi", colors=colors)
    # Style: Default,...,Primary,Secondary,Outline,Back,...
    assert "Style: Default,Noto Sans,64,&H00111111,&H00EB6325,&H00FFFFFF," in ass
    box = build_ass(_words(), preset_name="Highlight Box", colors=colors)
    assert "Style: Box,Noto Sans,64,&H00000000,&H00000000,&H00000000,&H00000000," in box


def test_color_schemes_include_light_dark_black() -> None:
    from clipforge.core.captions import COLOR_SCHEMES

    assert {"Classic", "Light", "Dark", "Black"} <= set(COLOR_SCHEMES)


# -- overlays ---------------------------------------------------------------

def test_custom_text_overlay_event() -> None:
    from clipforge.core.captions import CaptionOverlay

    ass = build_ass(
        _words(), preset_name="Hormozi",
        overlays=[CaptionOverlay(text="MY TEXT", start=1.0, end=2.0)],
    )
    assert "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,MY TEXT" in ass


# -- custom fonts -----------------------------------------------------------

def test_inspect_bundled_font_family(tmp_path) -> None:
    import shutil

    from clipforge.core.captions import FontError, inspect_font_file

    src = fonts_dir() / "NotoSans.ttf"
    copy = tmp_path / "NotoSans.ttf"
    shutil.copyfile(src, copy)
    info = inspect_font_file(copy)
    assert info.family == "Noto Sans"
    assert info.filename == "NotoSans.ttf"

    bad = tmp_path / "not-a-font.ttf"
    bad.write_text("hello, this is not a font", encoding="utf-8")
    with pytest.raises(FontError):
        inspect_font_file(bad)
    empty = tmp_path / "empty.ttf"
    empty.write_bytes(b"")
    with pytest.raises(FontError):
        inspect_font_file(empty)
    missing = tmp_path / "nope.ttf"
    with pytest.raises(FontError):
        inspect_font_file(missing)


def test_install_and_resolve_custom_font(tmp_path) -> None:
    import shutil

    from clipforge.core.captions import (
        install_custom_font,
        list_installed_fonts,
        resolve_custom_font,
    )

    src = fonts_dir() / "NotoSans.ttf"
    renamed = tmp_path / "MyFont.ttf"
    shutil.copyfile(src, renamed)
    dest = tmp_path / "user-fonts"
    installed = install_custom_font(renamed, dest_dir=dest)
    assert installed.family == "Noto Sans"
    assert installed.filename == "MyFont.ttf"
    assert (dest / "MyFont.ttf").exists()
    # Bundled fonts were copied in alongside, so the folder is a fontsdir.
    assert (dest / "NotoSansBengali.ttf").exists()
    assert list_installed_fonts(dest_dir=dest)[0].filename == "MyFont.ttf"
    family, fontsdir, error = resolve_custom_font(
        "MyFont.ttf", base_dir=dest
    )
    assert error is None
    assert family == "Noto Sans"
    assert fontsdir == dest
    _fam, _dir, error = resolve_custom_font("Missing.ttf", base_dir=dest)
    assert error is not None


def test_bundled_caption_fonts_viral_first() -> None:
    from clipforge.core.captions import (
        BUNDLED_CAPTION_FONTS,
        bundled_font_family,
        list_caption_fonts,
    )

    fonts = list_caption_fonts()
    assert fonts == list(BUNDLED_CAPTION_FONTS)
    viral = [f for f in fonts if f.viral]
    assert [f.display for f in viral[:3]] == [
        "Anton", "Bebas Neue", "Montserrat Black",
    ]
    assert all(not f.viral for f in fonts[len(viral):])
    for font in fonts:
        path = fonts_dir() / font.filename
        assert path.exists(), f"missing bundled font: {font.filename}"
        assert bundled_font_family(font.filename) == font.family
    assert bundled_font_family("Nope.ttf") is None


def test_bundled_font_licenses_present() -> None:
    licenses = fonts_dir() / "licenses"
    assert (licenses / "anton-OFL.txt").read_text().startswith(
        "SIL OPEN FONT LICENSE"
    ) or "SIL OPEN FONT LICENSE" in (licenses / "anton-OFL.txt").read_text()
    assert len(list(licenses.glob("*-OFL.txt"))) >= 7


# -- keyword emphasis (Boss's reference screenshots) ----------------------

def _sample_words(texts: list[str]) -> list[Word]:
    return [
        Word(text, i * 0.4, (i + 1) * 0.4) for i, text in enumerate(texts)
    ]


def test_pick_keyword_pyaar_sample() -> None:
    from clipforge.core.captions import pick_keyword

    line = _sample_words(["Pehli", "baat", "toh", "pyaar", "woh"])
    assert pick_keyword(line) == {3}  # "pyaar", not "Pehli"


def test_pick_keyword_stress_sample() -> None:
    from clipforge.core.captions import pick_keyword

    line = _sample_words(["Mai", "thodi", "is", "Stress", "Me", "jiyuga"])
    assert pick_keyword(line) == {3}  # "Stress", not "jiyuga"


def test_pick_keyword_modes() -> None:
    from clipforge.core.captions import pick_keyword

    line = _sample_words(["how", "rich", "is", "gary", "vee"])
    assert pick_keyword(line, mode="last_two") == {3, 4}
    assert pick_keyword(line, mode="last") == {4}
    assert pick_keyword(line, mode="first_half") == {0, 1, 2}
    assert pick_keyword([]) == set()
    # All stopwords: longest overall wins.
    assert pick_keyword(_sample_words(["is", "the", "a"])) == {1}


def test_keyword_pop_ass() -> None:
    ass = build_ass(_words(), preset_name="Keyword Pop")
    # Keyword "world" huge in magenta, small words white at 0.65x.
    assert "{\\fs128\\c&H00C62DFF&}world" in ass
    assert "{\\fs42\\c&H00FFFFFF&}Hello" in ass
    # Preset font, not the language default.
    assert "Style: Default,Poppins ExtraBold,64," in ass
    # Short lines (max 3 words) so the huge word fits.
    assert ass.count("Dialogue:") == 3


def test_keyword_italic_ass() -> None:
    ass = build_ass(_words(), preset_name="Keyword Italic")
    assert "\\i1" in ass
    assert "{\\fs128\\c&H0023C9FE&\\i1}world" in ass  # yellow keyword, italic


def test_impact_ass() -> None:
    ass = build_ass(
        _sample_words(["never", "edit", "in", "capcut"]), preset_name="Impact"
    )
    assert "NEVER" in ass and "CAPCUT" in ass  # upper-case
    assert "never" not in ass
    # First line red, second line white (sample 3's two-tone look).
    assert "{\\fs110\\c&H002D2DFF&}NEVER" in ass  # red, huge
    assert "{\\fs110\\c&H002D2DFF&}IN" in ass
    assert "{\\fs110\\c&H00FFFFFF&}CAPCUT" in ass  # white second line
    assert "Style: Default,Anton,110," in ass


def test_gary_vee_ass() -> None:
    ass = build_ass(
        _sample_words(["how", "rich", "is", "gary", "vee"]),
        preset_name="Gary Vee",
    )
    assert "{\\fs122\\c&H002D2DFF&}GARY" in ass  # last two huge + red
    assert "{\\fs122\\c&H002D2DFF&}VEE" in ass
    assert "{\\fs51\\c&H00FFFFFF&}HOW" in ass  # rest smaller + white
    assert "Style: Default,Anton,64," in ass


def test_glow_ass() -> None:
    ass = build_ass(_words(), preset_name="Glow")
    assert "\\blur6" in ass  # blurred halo copy
    assert "\\alpha&H80&" in ass
    # 3 lines x (glow layer 0 + sharp layer 1).
    assert ass.count("Dialogue: 0,") == 3
    assert ass.count("Dialogue: 1,") == 3
    assert "{\\fs112\\c&H00FFFFFF&}world" in ass  # huge white keyword


def test_gradient_ass() -> None:
    ass = build_ass(_words(), preset_name="Gradient")
    # 3 lines x 4 clip bands, each band explicitly positioned (\pos)
    # so libass does not stack the simultaneous events.
    assert ass.count("\\clip(") == 12
    assert "\\pos(540,1652)" in ass
    assert "\\c&H00FEEFE6&" in ass  # top band: near-white (12.5% blue)
    assert "\\c&H00F79254&" in ass  # bottom band: near keyword blue
    assert "Style: Default,Poppins ExtraBold,80," in ass


def test_typewriter_events_positioned() -> None:
    # Overlapping prefix events carry \pos so they do not stack.
    ass = build_ass(_words(), preset_name="Typewriter")
    assert "\\pos(540,1652)" in ass


def test_lower_third_ass() -> None:
    ass = build_ass(_words(), preset_name="Lower Third")
    # Left-aligned (1), BorderStyle 3, semi-transparent dark box
    # (outline colour with alpha 40 = 75% opaque; Outline 1.0 because
    # libass only draws the box when Outline > 0).
    assert ",3,1.0,0.0,1,120,120,268,1" in ass
    assert "&H40000000" in ass
    assert "Style: Default,Montserrat,44," in ass


def test_outline_ass() -> None:
    ass = build_ass(_words(), preset_name="Outline")
    # Transparent fill, thick outline in the keyword (yellow) colour.
    assert "Style: Default,Archivo Black,76,&HFF000000,&H0023C9FE,&H0023C9FE," in ass
    assert ",1,4.0,0.0,2,120,120,268,1" in ass


def test_clean_pop_ass() -> None:
    ass = build_ass(_words(), preset_name="Clean Pop")
    # Active word keeps the text colour (no yellow highlight).
    assert "Style: Default,Montserrat Black,64,&H00FFFFFF,&H00FFFFFF," in ass
    assert "\\fscx112" in ass  # gentle scale pulse


def test_keyword_color_changeable() -> None:
    from clipforge.core.captions import CaptionColors

    colors = CaptionColors(keyword="#00FF00")
    ass = build_ass(_words(), preset_name="Keyword Pop", colors=colors)
    assert "{\\fs128\\c&H0000FF00&}world" in ass  # green keyword


def test_keyword_scale_override() -> None:
    ass = build_ass(_words(), preset_name="Keyword Pop", keyword_scale=2.5)
    assert "{\\fs160\\c&H00C62DFF&}world" in ass


def test_preset_font_ignored_for_bengali() -> None:
    ass = build_ass(_words(), language="bn", preset_name="Impact")
    assert "Noto Sans Bengali," in ass
    assert "Anton," not in ass


def test_user_font_overrides_preset_font() -> None:
    ass = build_ass(_words(), preset_name="Impact", font="Anton")
    assert "Style: Default,Anton,110," in ass
    ass2 = build_ass(_words(), preset_name="Impact", font="Bebas Neue")
    assert "Style: Default,Bebas Neue,110," in ass2
    # A custom (installed) font wins over both.
    ass3 = build_ass(
        _words(), preset_name="Impact", font="Bebas Neue",
        custom_font="My Font",
    )
    assert "Style: Default,My Font,110," in ass3


def test_all_presets_build() -> None:
    assert len(PRESETS) >= 18  # 9 original + 9 new (6 samples + 3 extra)
    for name in PRESETS:
        ass = build_ass(_words(), preset_name=name)
        assert "Dialogue:" in ass, f"no events for preset {name}"
        assert PRESETS[name].animation in (
            "pop", "karaoke", "typewriter", "slide", "bounce", "fade",
            "box", "keyword", "impact", "glow", "garyvee", "gradient",
            "lowerthird", "outline", "cleanpop", "dualfont", "stacked",
            "keywordbox",
        )


def test_ass_to_hex_and_lerp() -> None:
    from clipforge.core.captions import _ass_to_hex, _lerp_hex

    assert _ass_to_hex("&H00C62DFF") == "#FF2DC6"
    assert _ass_to_hex("&H0023C9FE") == "#FEC923"
    assert _lerp_hex("#000000", "#FFFFFF", 0.5) == "#808080"
    assert _lerp_hex("#FFFFFF", "#3B82F6", 0.0) == "#FFFFFF"
    assert _lerp_hex("#FFFFFF", "#3B82F6", 1.0) == "#3B82F6"


# ---------------------------------------------------------------------------
# New presets (FuseClip-beating): Headline Script, Emerald Glow, Stacked,
# Keyword Boxes — plus multi-stop gradient and layout controls.
# ---------------------------------------------------------------------------


def _fox_words() -> list[Word]:
    return _sample_words(["a", "quick", "brown", "fox"])


def test_headline_script_ass() -> None:
    ass = build_ass(_fox_words(), preset_name="Headline Script")
    # Dual-font: keyword word in the Kaushan Script font, the rest in Anton.
    assert "\\fnKaushan Script" in ass
    assert "\\fnAnton" in ass
    # The script word keeps its natural case (not uppercased).
    assert "quick" in ass
    assert "QUICK" not in ass
    # Main words are uppercased by the preset.
    assert "A " in ass or "A\\" in ass or "}A " in ass


def test_emerald_glow_ass() -> None:
    ass = build_ass(_fox_words(), preset_name="Emerald Glow")
    # Glow animation (two layers) with the green glow colour.
    assert ass.count("Dialogue:") >= 2
    assert "&H003ED8A0&" in ass  # _GREEN_GLOW = #A0D83E


def test_stacked_ass() -> None:
    ass = build_ass(_fox_words(), preset_name="Stacked")
    # One event per word, each with its own \pos (stacked vertically).
    dialogues = _dialogues(ass)
    assert len(dialogues) == 4
    assert all("\\pos(" in d for d in dialogues)
    # Every word uppercased, one per event.
    assert dialogues[0].rstrip().endswith("A")
    assert dialogues[1].rstrip().endswith("QUICK")
    assert dialogues[2].rstrip().endswith("BROWN")
    assert dialogues[3].rstrip().endswith("FOX")


def test_keyword_boxes_ass() -> None:
    ass = build_ass(_fox_words(), preset_name="Keyword Boxes")
    # Both box styles exist; keyword words alternate Box / Box2.
    assert "Style: Box," in ass
    assert "Style: Box2," in ass
    dialogues = _dialogues(ass)
    styles = [d.split(",")[3] for d in dialogues]
    assert "Box" in styles
    assert "Box2" in styles
    # Each word is its own positioned event.
    assert all("\\pos(" in d for d in dialogues)


def test_gradient_multi_stop() -> None:
    # 3 words fit one line (1.2s < 1.5s line cap) -> 8 bands, 8 events.
    words = _sample_words(["a", "quick", "fox"])
    gradient = GradientConfig(
        mode="gradient",
        stops=(
            (0.0, "#FF0000"),
            (0.25, "#FFFF00"),
            (0.5, "#00FF00"),
            (0.75, "#0000FF"),
            (1.0, "#FF00FF"),
        ),
    )
    ass = build_ass(words, preset_name="Gradient", gradient=gradient)
    dialogues = _dialogues(ass)
    # More than 2 stops -> 8 bands.
    assert len(dialogues) == 8
    assert ass.count("\\clip(") == 8
    # First band samples the red->yellow leg at 1/16, last the
    # blue->magenta leg at 15/16.
    assert "&H000040FF&" in dialogues[0]  # #FF4000
    assert "&H00FF00BF&" in dialogues[-1]  # #BF00FF


def test_gradient_solid_mode() -> None:
    # 3 words fit one line -> a single solid-colour event.
    words = _sample_words(["a", "quick", "fox"])
    gradient = GradientConfig(mode="solid", stops=((0.0, "#123456"),))
    ass = build_ass(words, preset_name="Gradient", gradient=gradient)
    dialogues = _dialogues(ass)
    assert len(dialogues) == 1
    assert "&H00563412&" in dialogues[0]  # #123456 as ASS colour


def test_gradient_config_validated() -> None:
    cfg = GradientConfig(
        mode="bogus",
        stops=((0.8, "#FFFFFF"), (0.2, "#000000"), (2.0, "#FF0000")),
    ).validated()
    assert cfg.mode == "gradient"
    assert cfg.stops[0][0] == 0.2  # sorted
    assert cfg.stops[-1][0] == 1.0  # clamped
    assert GradientConfig(stops=()).validated().stops == (
        (0.0, "#FFFFFF"),
        (1.0, "#3B82F6"),
    )
    many = GradientConfig(
        stops=tuple((i / 9, "#FFFFFF") for i in range(10))
    ).validated()
    assert len(many.stops) == 5  # capped at 5


def test_layout_words_per_line() -> None:
    words = _sample_words(["one", "two", "three", "four", "five", "six"])
    ass = build_ass(
        words, preset_name="Hormozi",
        layout=CaptionLayout(words_per_line=2),
    )
    # 6 words, 2 per line -> 3 lines -> 3 events.
    assert len(_dialogues(ass)) == 3


def test_layout_max_chars() -> None:
    words = _sample_words(["abcdefghij", "klmnop", "qrst"])
    ass = build_ass(
        words, preset_name="Hormozi", layout=CaptionLayout(max_chars=10),
    )
    # "abcdefghij" (10 chars) fills a line on its own; "klmnop qrst"
    # would be 11 chars, so it splits too -> 3 lines.
    assert len(_dialogues(ass)) == 3


def test_layout_lines_merge() -> None:
    words = _sample_words(["one", "two", "three", "four", "five", "six"])
    ass = build_ass(
        words, preset_name="Hormozi",
        layout=CaptionLayout(words_per_line=2, lines=2),
    )
    # 3 lines merged 2-per-block -> 2 events; only the first block
    # holds 2 lines, so only it has a \N break.
    dialogues = _dialogues(ass)
    assert len(dialogues) == 2
    assert "\\N" in dialogues[0]
    assert "\\N" not in dialogues[1]


def test_layout_align_left_right() -> None:
    left = build_ass(_fox_words(), layout=CaptionLayout(align="left"))
    right = build_ass(_fox_words(), layout=CaptionLayout(align="right"))
    style_left = [l for l in left.splitlines() if l.startswith("Style: Default")][0]
    style_right = [l for l in right.splitlines() if l.startswith("Style: Default")][0]
    assert ",1," in style_left  # alignment 1 = bottom-left
    assert ",3," in style_right  # alignment 3 = bottom-right
    # Left align puts the text at the side margin (120px at 1080 wide).
    assert "\\pos(120," in _dialogues(left)[0]


def test_layout_position() -> None:
    ass = build_ass(
        _fox_words(),
        layout=CaptionLayout(h_pct=25.0, v_pct=30.0),
    )
    # Horizontal 25% -> \pos x = 270; vertical 30% is the TOP zone
    # (v_pct < 50) with MarginV 576 = 30% of 1920.
    assert "\\pos(270,576)" in _dialogues(ass)[0]
    style = [l for l in ass.splitlines() if l.startswith("Style: Default")][0]
    assert ",576,1" in style


def test_layout_font_size_pct() -> None:
    # font_size_pct is a % of the frame height: 5% of 1920 = 96.
    ass = build_ass(
        _fox_words(), layout=CaptionLayout(font_size_pct=5.0),
    )
    assert ",96," in ass


def test_layout_validated() -> None:
    layout = CaptionLayout(
        lines=9, words_per_line=0, h_pct=150.0, v_pct=-5.0,
        font_size_pct=10.0, align="bogus",
    ).validated()
    assert layout.lines == 3
    assert layout.words_per_line == 1
    assert layout.h_pct == 100.0
    assert layout.v_pct == 0.0
    # font_size_pct is % of frame height, clamped to 1–10.
    assert layout.font_size_pct == 10.0
    assert layout.align == "auto"
    assert CaptionLayout(font_size_pct=150.0).validated().font_size_pct == 10.0
    assert CaptionLayout(font_size_pct=0.5).validated().font_size_pct == 1.0


def test_layout_defaults_change_nothing() -> None:
    # A default layout produces exactly the same document as no layout.
    assert build_ass(_words(), layout=CaptionLayout()) == build_ass(_words())


# ---------------------------------------------------------------------------
# v0.2.2: FuseClip preset parity, per-preset anchors, position sanity
# ---------------------------------------------------------------------------

#: The FuseClip style names Boss asked to find in the picker.
FUSECLIP_PRESET_NAMES = {
    "Mumbai", "Numaya", "Josh", "Aks", "Rang", "Noor", "Rawani",
    "Lehar", "Orbit", "Bayan", "Headline", "Lakeer", "Uchhaal",
    "Ali Abdaal", "Karachi", "Lafz", "Shamani", "Saaf",
}


def test_fuseclip_preset_names_all_exist() -> None:
    """Every FuseClip preset name exists (Boss's parity request)."""
    assert FUSECLIP_PRESET_NAMES <= set(PRESETS)
    for name in FUSECLIP_PRESET_NAMES:
        preset = PRESETS[name]
        # A plausible look: a real bundled font (or the language
        # default) and an animation the renderer implements.
        assert preset.animation in (
            "pop", "karaoke", "typewriter", "slide", "bounce", "fade",
            "box", "keyword", "impact", "glow", "garyvee", "gradient",
            "lowerthird", "outline", "cleanpop", "dualfont", "stacked",
            "keywordbox",
        )


def test_fuseclip_signature_looks() -> None:
    """The frame-analysed FuseClip looks match their descriptions."""
    # Mumbai: Anton caps + Kaushan Script keyword in yellow.
    ass = build_ass(_fox_words(), preset_name="Mumbai")
    assert "\\fnKaushan Script" in ass and "\\fnAnton" in ass
    # Numaya: the emphasised word glows FuseClip green (#A0D83E).
    ass = build_ass(_fox_words(), preset_name="Numaya")
    assert "&H003ED8A0&" in ass
    # Josh: magenta keyword.
    ass = build_ass(_fox_words(), preset_name="Josh")
    assert "&H00C62DFF&" in ass
    # Rawani: teal keyword (#14B8A6).
    ass = build_ass(_fox_words(), preset_name="Rawani")
    assert "&H00A6B814&" in ass
    # Rang: per-keyword boxes in both box styles.
    ass = build_ass(_fox_words(), preset_name="Rang")
    assert "Style: Box," in ass and "Style: Box2," in ass
    # Headline: stacked one-word-per-line events.
    dialogues = _dialogues(build_ass(_fox_words(), preset_name="Headline"))
    assert len(dialogues) == 4
    assert all("\\pos(" in d for d in dialogues)


def _style_margin_v(ass: str) -> int:
    style = [
        line for line in ass.splitlines() if line.startswith("Style: Default")
    ][0]
    return int(style.rsplit(",", 2)[1])


def test_preset_anchor_keyword_family_lower_middle() -> None:
    """Keyword-family presets anchor at v=70% (the reference frames'
    lower-middle block), not the 86% safe-zone default."""
    for name in ("Keyword Pop", "Impact", "Stacked", "Mumbai", "Headline"):
        ass = build_ass(_words(), preset_name=name, width=1080, height=1920)
        # Bottom margin = (100 - 70)% of 1920.
        assert _style_margin_v(ass) == int(1920 * 0.30), name
    # Plain styles keep the historical safe-zone anchor.
    ass = build_ass(_words(), preset_name="Hormozi", width=1080, height=1920)
    assert _style_margin_v(ass) == int(1920 * 0.14)


def test_preset_anchor_user_position_wins() -> None:
    """A user-chosen Fonts-tab position overrides the preset anchor."""
    ass = build_ass(
        _words(), preset_name="Keyword Pop", width=1080, height=1920,
        layout=CaptionLayout(v_pct=90.0),
    )
    assert _style_margin_v(ass) == int(1920 * 0.10)


def test_every_preset_positions_inside_frame() -> None:
    """Position sanity for every preset at every output ratio.

    Parses the generated ASS at 9:16, 1:1 and 16:9 sizes: the style
    margin must be a legal in-frame value and every explicit \\pos
    coordinate must land inside the frame.
    """
    import re

    pos_re = re.compile(r"\\pos\((-?\d+),(-?\d+)\)")
    sizes = [(1080, 1920), (1080, 1080), (1920, 1080), (360, 640)]
    for name in PRESETS:
        for width, height in sizes:
            ass = build_ass(
                _words(), preset_name=name, width=width, height=height
            )
            margin_v = _style_margin_v(ass)
            assert 0 <= margin_v < height, (name, width, height)
            for match in pos_re.finditer(ass):
                x, y = int(match.group(1)), int(match.group(2))
                assert 0 <= x <= width, (name, width, height, x)
                assert 0 <= y <= height, (name, width, height, y)


def test_stacked_block_shrinks_to_fit_short_frame() -> None:
    """A tall stack on a short (16:9) frame shrinks instead of
    running off the top/bottom edge."""
    import re

    # Eight fast words fit ONE caption line (1.2 s < the 1.5 s cap),
    # so the stack is 8 rows; a 10%-of-height font makes the natural
    # block (8 x 1.25 x 108 = 1080 px) taller than the space between
    # the v=70 anchor and the top of the 1080-tall frame.
    words = [
        Word(text, i * 0.15, (i + 1) * 0.15)
        for i, text in enumerate(
            ["one", "two", "three", "four", "five", "six", "seven", "eight"]
        )
    ]
    ass = build_ass(
        words, preset_name="Stacked", width=1920, height=1080,
        layout=CaptionLayout(words_per_line=8, font_size_pct=10.0),
    )
    ys = [
        int(m.group(1))
        for m in re.finditer(r"\\pos\(\d+,(\d+)\)", ass)
    ]
    assert len(ys) == 8, "one event per stacked word"
    assert all(0 <= y <= 1080 for y in ys)
    # Rows shrunk to fit: the block's last row sits at the anchor
    # (bottom edge = 70% of 1080 = 756) and no row uses the natural
    # 108 px font.
    assert max(ys) == 756
    assert "\\fs108" not in ass


def test_block_height_covers_stacked_and_keyword() -> None:
    """The face-avoidance zone uses the real block height."""
    from clipforge.core.captions import caption_block_height

    layout = CaptionLayout()
    stacked = caption_block_height(PRESETS["Stacked"], layout, 96)
    assert stacked == pytest.approx(4 * 96 * 1.25)  # 4 words x 1.25 rows
    keyword = caption_block_height(PRESETS["Keyword Pop"], layout, 64)
    assert keyword >= 2.0 * 64  # the 2x keyword word sets the height
    plain = caption_block_height(PRESETS["Hormozi"], layout, 64)
    assert plain == pytest.approx(2.8 * 64)


def test_every_preset_renders_a_frame_with_captions_inside(tmp_path) -> None:
    """Every preset burns onto a real frame; caption pixels exist and
    the frame is exactly the requested size (libass draws in-frame by
    construction — this proves no preset errors or renders blank)."""
    import subprocess

    from clipforge.core.ffmpeg import ass_filter, find_ffmpeg, probe

    ffmpeg = find_ffmpeg()
    assert ffmpeg
    for name in PRESETS:
        ass = build_ass(
            _fox_words(), preset_name=name, width=360, height=640
        )
        ass_path = tmp_path / "c.ass"
        ass_path.write_text(ass, encoding="utf-8")
        out = tmp_path / "f.png"
        subprocess.run(
            [
                ffmpeg, "-y", "-v", "error",
                "-f", "lavfi", "-i", "color=c=black:s=360x640:r=30:d=2",
                "-vf", ass_filter(ass_path, fonts_dir()),
                "-ss", "0.6", "-frames:v", "1", str(out),
            ],
            check=True,
        )
        info = probe(str(out))
        assert (info.width, info.height) == (360, 640), name
        # Caption pixels: the black background gained non-black pixels.
        import cv2
        import numpy as np

        frame = cv2.imread(str(out))
        assert frame is not None and np.max(frame) > 40, name
