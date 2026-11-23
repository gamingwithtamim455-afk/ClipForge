"""Burned captions: ASS subtitle generation with word highlight.

Rendered by FFmpeg's ``ass`` filter (libass + HarfBuzz), so Bengali
shaping and Arabic/Urdu right-to-left text come out correct. Bundled
fonts (SIL OFL, see ``clipforge/assets/fonts``):

- Bengali -> Noto Sans Bengali
- Arabic  -> Noto Naskh Arabic
- Urdu    -> Noto Nastaliq Urdu
- anything else (Latin etc.) -> Noto Sans

A custom font can be installed by the user (Settings > Fonts); it is
copied into the app-data fonts folder, which then serves as the
``fontsdir`` for the render so libass finds it, and its family name is
used in the ASS style for every preset.

Captions sit in the TikTok safe zone: bottom-center, with a bottom margin
that keeps them clear of the TikTok caption bar and the right-hand action
rail. Word highlight uses ASS karaoke tags (``{\\k}``): the swept word is
drawn in the preset's highlight colour, the rest in the base colour.

Caption animations: each preset carries an ``animation`` mode. ``pop``
is the original Hormozi word-by-word karaoke (kept byte-identical).
The others use real ASS animation tags: ``\\kf`` fill sweep,
``\\move`` slide-in, chained ``\\t`` scale bounce, ``\\fad`` fade-in,
accumulating typewriter events, and per-word yellow-box events.
Keyword-emphasis presets (Boss's reference screenshots) render the
keyword word much bigger in a different colour via inline ``\\fs`` /
``\\c`` overrides; the gradient preset fakes a vertical gradient fill
with ``\\clip`` bands; the glow preset layers a blurred copy under the
sharp text. Every colour in every preset is user-changeable
(``CaptionColors``: text, keyword, highlight, outline, box).
"""

from __future__ import annotations

import shutil
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .paths import app_data_dir
from .transcription import Word

__all__ = [
    "BundledFont",
    "BUNDLED_CAPTION_FONTS",
    "CaptionColors",
    "CaptionLayout",
    "CaptionOverlay",
    "CaptionPlacement",
    "CaptionPreset",
    "COLOR_SCHEMES",
    "DEFAULT_GRADIENT",
    "FontError",
    "FontInfo",
    "GradientConfig",
    "PRESETS",
    "build_ass",
    "bundled_font_family",
    "caption_zone_clear",
    "choose_caption_placement",
    "font_family_for_language",
    "font_family_from_file",
    "font_path_for_language",
    "fonts_dir",
    "group_words",
    "hex_to_ass_color",
    "inspect_font_file",
    "install_custom_font",
    "is_rtl_language",
    "layout_align",
    "layout_font_size",
    "layout_margin_v",
    "list_caption_fonts",
    "list_installed_fonts",
    "pick_keyword",
    "resolve_custom_font",
    "user_fonts_dir",
    "write_ass_file",
]

#: (file name, fontconfig family name) per language code.
_FONTS: dict[str, tuple[str, str]] = {
    "bn": ("NotoSansBengali.ttf", "Noto Sans Bengali"),
    "ar": ("NotoNaskhArabic.ttf", "Noto Naskh Arabic"),
    "ur": ("NotoNastaliqUrdu.ttf", "Noto Nastaliq Urdu"),
}
_DEFAULT_FONT = ("NotoSans.ttf", "Noto Sans")


@dataclass(frozen=True)
class BundledFont:
    """One bundled caption font for the font picker."""

    filename: str
    family: str  # fontconfig family name, as used in the ASS style
    display: str  # picker label, e.g. "Montserrat Black"
    viral: bool  # viral podcast fonts are listed first


#: Bundled caption fonts, viral podcast fonts FIRST, then language fonts.
#: All SIL OFL (see assets/fonts/licenses/).
BUNDLED_CAPTION_FONTS: tuple[BundledFont, ...] = (
    BundledFont("Anton-Regular.ttf", "Anton", "Anton", True),
    BundledFont("BebasNeue-Regular.ttf", "Bebas Neue", "Bebas Neue", True),
    BundledFont("Montserrat-Black.ttf", "Montserrat Black", "Montserrat Black", True),
    BundledFont("Montserrat-Bold.ttf", "Montserrat", "Montserrat Bold", True),
    BundledFont("Oswald[wght].ttf", "Oswald", "Oswald", True),
    BundledFont("Poppins-Bold.ttf", "Poppins", "Poppins Bold", True),
    BundledFont(
        "Poppins-ExtraBold.ttf", "Poppins ExtraBold", "Poppins ExtraBold", True
    ),
    BundledFont("ArchivoBlack-Regular.ttf", "Archivo Black", "Archivo Black", True),
    BundledFont(
        "BarlowCondensed-Bold.ttf", "Barlow Condensed", "Barlow Condensed Bold", True
    ),
    # Script font for the dual-font headline preset (Headline Script:
    # Anton caps + Kaushan Script keyword). SIL OFL, see licenses/.
    BundledFont(
        "KaushanScript-Regular.ttf", "Kaushan Script", "Kaushan Script", False
    ),
    BundledFont("NotoSans.ttf", "Noto Sans", "Noto Sans", False),
    BundledFont("NotoSansBengali.ttf", "Noto Sans Bengali", "Noto Sans Bengali", False),
    BundledFont("NotoNaskhArabic.ttf", "Noto Naskh Arabic", "Noto Naskh Arabic", False),
    BundledFont(
        "NotoNastaliqUrdu.ttf", "Noto Nastaliq Urdu", "Noto Nastaliq Urdu", False
    ),
)


def list_caption_fonts() -> list[BundledFont]:
    """Bundled fonts for the picker (viral podcast fonts first)."""
    return list(BUNDLED_CAPTION_FONTS)


def bundled_font_family(filename: str) -> str | None:
    """Family name for a bundled font file, or None if not bundled."""
    name = Path(filename).name
    for font in BUNDLED_CAPTION_FONTS:
        if font.filename == name:
            return font.family
    return None

#: Languages rendered right-to-left by libass/HarfBuzz.
RTL_LANGUAGES = {"ar", "ur"}

#: Bottom margin as a fraction of the output height (TikTok safe zone).
SAFE_ZONE_MARGIN_FRACTION = 0.14
#: Side margins in px at 1080 wide (keeps text clear of the action rail).
SIDE_MARGIN = 120


def fonts_dir() -> Path:
    """Directory holding the bundled caption fonts."""
    return Path(__file__).resolve().parent.parent / "assets" / "fonts"


def preview_slug(name: str) -> str:
    """File slug for a preset's preview image (``"Slide Up"`` -> ``"slide-up"``)."""
    import re

    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def previews_dir() -> Path:
    """Directory holding the bundled style-preview images."""
    return Path(__file__).resolve().parent.parent / "assets" / "previews"


def font_path_for_language(language: str) -> Path:
    """Bundled font file for a language (falls back to Noto Sans)."""
    name = _FONTS.get(language, _DEFAULT_FONT)[0]
    return fonts_dir() / name


def font_family_for_language(language: str) -> str:
    """Font family name (as fontconfig sees it) for a language."""
    return _FONTS.get(language, _DEFAULT_FONT)[1]


def is_rtl_language(language: str) -> bool:
    return language in RTL_LANGUAGES


@dataclass(frozen=True)
class CaptionLayout:
    """User-chosen caption layout (Fonts tab), honoured by ``build_ass``.

    Every field defaults to "no change" (``None`` / ``"auto"`` / 50 / 86
    reproduce the historical rendering exactly), so existing presets and
    tests keep working when no layout is given:

    - ``lines``: max lines per caption block (``None`` = Auto, one line
      per event, as before; 1–3 merges consecutive lines into one event
      with ``\\N`` breaks).
    - ``words_per_line`` / ``max_chars``: line breaking (``None`` = the
      preset's own ``max_words`` / no char limit).
    - ``align``: ``"auto"`` = the preset's default (left for Lower Third,
      centre otherwise), else ``"left"`` / ``"center"`` / ``"right"``.
    - ``h_pct``: horizontal anchor as % of frame width (50 = centre).
      For centre alignment it is the text centre; for left/right it is
      the text edge.
    - ``v_pct``: vertical anchor as % of frame height (86 = the TikTok
      safe zone, as before). >= 50 puts captions at the bottom,
      < 50 at the top.
    - ``font_size_pct``: font size as % of frame height (``None`` =
      the preset's size; FuseClip's default is 2.5%).
    """

    lines: int | None = None
    words_per_line: int | None = None
    max_chars: int | None = None
    align: str = "auto"
    h_pct: float = 50.0
    v_pct: float = 86.0
    font_size_pct: float | None = None

    def validated(self) -> "CaptionLayout":
        """A copy with every value clamped to its legal range."""
        lines = self.lines
        if lines is not None:
            lines = max(1, min(3, int(lines)))
        words = self.words_per_line
        if words is not None:
            words = max(1, min(12, int(words)))
        chars = self.max_chars
        if chars is not None:
            chars = max(1, min(120, int(chars)))
        align = self.align if self.align in ("auto", "left", "center", "right") else "auto"
        size = self.font_size_pct
        if size is not None:
            size = max(1.0, min(10.0, float(size)))
        return CaptionLayout(
            lines=lines,
            words_per_line=words,
            max_chars=chars,
            align=align,
            h_pct=max(0.0, min(100.0, float(self.h_pct))),
            v_pct=max(0.0, min(100.0, float(self.v_pct))),
            font_size_pct=size,
        )

    def to_dict(self) -> dict:
        """JSON-safe dict for Settings persistence."""
        return {
            "lines": self.lines,
            "words_per_line": self.words_per_line,
            "max_chars": self.max_chars,
            "align": self.align,
            "h_pct": self.h_pct,
            "v_pct": self.v_pct,
            "font_size_pct": self.font_size_pct,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "CaptionLayout":
        """Rebuild from :meth:`to_dict` output (tolerates bad input)."""
        if not isinstance(data, dict):
            return cls()
        try:
            return cls(
                lines=data.get("lines"),
                words_per_line=data.get("words_per_line"),
                max_chars=data.get("max_chars"),
                align=str(data.get("align") or "auto"),
                h_pct=float(data.get("h_pct", 50.0)),
                v_pct=float(data.get("v_pct", 86.0)),
                font_size_pct=data.get("font_size_pct"),
            ).validated()
        except (TypeError, ValueError):
            return cls()


@dataclass(frozen=True)
class GradientConfig:
    """Multi-stop caption gradient (Fonts tab gradient editor).

    ``stops`` are (position 0.0–1.0, ``#RRGGBB``) pairs, up to 5, in
    ascending position order. ``mode`` ``"solid"`` renders one flat
    colour (the first stop); ``"gradient"`` blends across the stops.
    ``angle`` is 0 (top -> bottom, the sample look) or 90 (left ->
    right); other values snap to the nearest of the two, because
    libass clip bands are axis-aligned.
    """

    mode: str = "gradient"  # "gradient" | "solid"
    angle: float = 0.0
    stops: tuple[tuple[float, str], ...] = ((0.0, "#FFFFFF"), (1.0, "#3B82F6"))

    def validated(self) -> "GradientConfig":
        """A copy with sorted, clamped stops (1–5) and a legal mode/angle."""
        mode = self.mode if self.mode in ("gradient", "solid") else "gradient"
        angle = 0.0 if abs(float(self.angle)) < 45 else 90.0
        stops: list[tuple[float, str]] = []
        for pos, color in self.stops:
            try:
                hex_to_ass_color(color)
            except ValueError:
                continue
            stops.append((max(0.0, min(1.0, float(pos))), color))
        stops.sort(key=lambda s: s[0])
        if not stops:
            stops = [(0.0, "#FFFFFF"), (1.0, "#3B82F6")]
        # A single stop stays a single stop (solid mode) — the band
        # sampler handles it (every ratio returns that colour).
        return GradientConfig(mode=mode, angle=angle, stops=tuple(stops[:5]))

    def to_dict(self) -> dict:
        """JSON-safe dict for Settings persistence."""
        return {
            "mode": self.mode,
            "angle": self.angle,
            "stops": [[pos, color] for pos, color in self.stops],
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "GradientConfig":
        """Rebuild from :meth:`to_dict` output (tolerates bad input)."""
        if not isinstance(data, dict):
            return cls()
        try:
            stops = tuple(
                (float(pos), str(color))
                for pos, color in data.get("stops") or []
            )
            return cls(
                mode=str(data.get("mode") or "gradient"),
                angle=float(data.get("angle", 0.0)),
                stops=stops,
            ).validated()
        except (TypeError, ValueError):
            return cls()


#: The gradient editor's default: white on top fading to blue
#: (the "trendy effect" sample), matching the Gradient preset's look.
DEFAULT_GRADIENT = GradientConfig()


@dataclass(frozen=True)
class CaptionPreset:
    """One Fonts-tab caption preset."""

    name: str
    font_size: int  # at 1080x1920; scaled with the output height
    bold: bool
    primary: str  # base text colour, ASS &HBBGGRR
    highlight: str  # karaoke (word-highlight) colour, ASS &HBBGGRR
    outline: float
    shadow: float
    animation: str = "pop"
    # UI label, e.g. "Pop — পপ" (English name + Bengali name).
    display_name: str = ""
    # Font family for Latin scripts (ignored for Bengali/Arabic/Urdu,
    # where the language font wins, and overridden by a custom font).
    # "" = the language default font.
    font: str = ""
    # Keyword-emphasis presets: the keyword word(s) render at
    # ``font_size * keyword_scale`` in the keyword colour, the rest at
    # ~0.65x. 1.0 = no size emphasis.
    keyword_scale: float = 1.0
    # Keyword word(s) rendered italic (sample 2, "Stress").
    keyword_italic: bool = False
    # Render every word upper-case (samples 3 and 5).
    uppercase: bool = False
    # Which word(s) are the keyword: "longest" (longest content word,
    # ties -> latest), "last", "last_two", "first_half".
    keyword_mode: str = "longest"
    # Default keyword colour, ASS &HBBGGRR. "" = the highlight colour.
    keyword: str = ""
    # Max words per caption line for this preset (keyword/gradient looks
    # need short lines so the huge word still fits the frame).
    max_words: int = 5
    # Dual-font presets (Headline Script): the keyword word(s) render in
    # this script font family (e.g. "Kaushan Script") instead of the
    # preset's font, at the keyword size.
    script_font: str = ""


# Colours are ASS &HAABBGGRR (alpha 00 = opaque).
_WHITE = "&H00FFFFFF"
_YELLOW = "&H0023C9FE"  # #FEC923
_BLACK = "&H00000000"
_SHADOW = "&H80000000"

#: Animation modes (``CaptionPreset.animation``).
ANIMATIONS = (
    "pop", "karaoke", "typewriter", "slide", "bounce", "fade", "box",
    # Keyword-emphasis / style-reference presets (Boss's 6 screenshots).
    "keyword", "impact", "glow", "garyvee", "gradient",
    # Extra professional presets.
    "lowerthird", "outline", "cleanpop",
    # Even-more-professional presets (Boss's FuseClip recording).
    "dualfont", "stacked", "keywordbox",
)

#: Keyword colour defaults (ASS &HBBGGRR).
_MAGENTA = "&H00C62DFF"  # #FF2DC6 — sample 1, "pyaar"
_RED = "&H002D2DFF"  # #FF2D2D — samples 3 and 5
_BLUE = "&H00F6823B"  # #3B82F6 — sample 6 gradient bottom
_GREEN_GLOW = "&H003ED8A0"  # #A0D83E — FuseClip Numaya emphasize green

PRESETS: dict[str, CaptionPreset] = {
    # Viral podcast styles first: Hormozi pop and karaoke word highlight
    # are the reference looks for podcast/TikTok clips.
    # Hormozi-style: bold white words, yellow highlight, strong outline.
    # animation "pop" = the original word-by-word karaoke, kept identical.
    "Hormozi": CaptionPreset(
        "Hormozi", 64, True, _WHITE, _YELLOW, 2.5, 1.0, "pop", "Pop — পপ"
    ),
    # Karaoke fill: whole line shown, active word fills yellow left-to-right.
    "Karaoke": CaptionPreset(
        "Karaoke", 64, True, _WHITE, _YELLOW, 2.5, 1.0,
        "karaoke", "Karaoke fill — কারাওকে ফিল",
    ),
    # --- Keyword emphasis (Boss's reference screenshots 1, 2, 5) --------
    # Sample 1 "pyaar": small white words, ONE keyword huge in magenta.
    "Keyword Pop": CaptionPreset(
        "Keyword Pop", 64, True, _WHITE, _YELLOW, 2.5, 1.0,
        "keyword", "Keyword Pop — কীওয়ার্ড পপ",
        font="Poppins ExtraBold", keyword_scale=2.0, keyword=_MAGENTA,
        max_words=3,
    ),
    # Sample 2 "Stress": small white italic words, keyword huge in yellow.
    "Keyword Italic": CaptionPreset(
        "Keyword Italic", 64, True, _WHITE, _YELLOW, 2.5, 1.0,
        "keyword", "Keyword Italic — কীওয়ার্ড ইটালিক",
        font="Poppins ExtraBold", keyword_scale=2.0, keyword_italic=True,
        keyword=_YELLOW, max_words=3,
    ),
    # Sample 3 "NEVER EDIT / IN CAPCUT": huge bold condensed all-caps,
    # lines alternating red and white.
    "Impact": CaptionPreset(
        "Impact", 110, True, _WHITE, _YELLOW, 3.0, 2.0,
        "impact", "Impact — ইমপ্যাক্ট",
        font="Anton", uppercase=True, keyword=_RED, max_words=4,
    ),
    # Sample 5 "GARY VEE": small yellow + white words, last two words
    # huge in red, heavy condensed all-caps.
    "Gary Vee": CaptionPreset(
        "Gary Vee", 64, True, _WHITE, _YELLOW, 3.0, 2.0,
        "garyvee", "Gary Vee — গ্যারি ভি",
        font="Anton", uppercase=True, keyword_mode="last_two",
        keyword_scale=1.9, keyword=_RED, max_words=3,
    ),
    # Sample 4 "like this": thin small white words + huge bold white
    # keyword with a soft glow.
    "Glow": CaptionPreset(
        "Glow", 56, True, _WHITE, _YELLOW, 1.5, 0.0,
        "glow", "Glow — গ্লো",
        font="Poppins ExtraBold", keyword_scale=2.0, keyword=_WHITE,
        max_words=3,
    ),
    # Sample 6 "trendy effect": huge heavy text with a vertical gradient
    # fill (white on top fading to blue at the bottom).
    "Gradient": CaptionPreset(
        "Gradient", 80, True, _WHITE, _YELLOW, 2.0, 2.0,
        "gradient", "Gradient — গ্রেডিয়েন্ট",
        font="Poppins ExtraBold", keyword=_BLUE, max_words=3,
    ),
    # Beast-style: bigger, heavier, same palette.
    "Beast": CaptionPreset(
        "Beast", 76, True, _WHITE, _YELLOW, 3.0, 1.5, "pop", "Pop big — পপ বড়"
    ),
    # Minimal: plain white, thin outline, no shadow.
    "Minimal": CaptionPreset(
        "Minimal", 56, False, _WHITE, _YELLOW, 1.0, 0.0, "pop", "Minimal — মিনিমাল"
    ),
    # Typewriter: words appear one by one, no highlight.
    "Typewriter": CaptionPreset(
        "Typewriter", 64, True, _WHITE, _YELLOW, 2.5, 1.0,
        "typewriter", "Typewriter — টাইপরাইটার",
    ),
    # Slide up: each line slides up from below into place.
    "Slide Up": CaptionPreset(
        "Slide Up", 64, True, _WHITE, _YELLOW, 2.5, 1.0,
        "slide", "Slide up — স্লাইড আপ",
    ),
    # Bounce: the line bounces in (scale 80% -> 108% -> 100%).
    "Bounce": CaptionPreset(
        "Bounce", 64, True, _WHITE, _YELLOW, 2.5, 1.0,
        "bounce", "Bounce — বাউন্স",
    ),
    # Fade: each line fades in.
    "Fade": CaptionPreset(
        "Fade", 64, True, _WHITE, _YELLOW, 2.5, 1.0,
        "fade", "Fade — ফেড",
    ),
    # Highlight box: each word alone in a solid yellow box, black text.
    "Highlight Box": CaptionPreset(
        "Highlight Box", 64, True, _BLACK, _YELLOW, 0.0, 0.0,
        "box", "Highlight box — হাইলাইট বক্স",
    ),
    # --- Extra professional presets --------------------------------------
    # Clean minimal lower-third: small text, left-aligned, on a dark box.
    "Lower Third": CaptionPreset(
        "Lower Third", 44, False, _WHITE, _YELLOW, 0.0, 0.0,
        "lowerthird", "Lower Third — লোয়ার থার্ড",
        font="Montserrat",
    ),
    # Bold outline-only text (transparent fill); the active karaoke word
    # fills with the highlight colour.
    "Outline": CaptionPreset(
        "Outline", 76, True, _WHITE, _YELLOW, 4.0, 0.0,
        "outline", "Outline — আউটলাইন",
        font="Archivo Black", keyword=_YELLOW,
    ),
    # Clean pop: gentle scale pulse, active word keeps the text colour
    # (no colour change) — the quiet professional look.
    "Clean Pop": CaptionPreset(
        "Clean Pop", 64, True, _WHITE, _WHITE, 1.5, 0.0,
        "cleanpop", "Clean Pop — ক্লিন পপ",
        font="Montserrat Black",
    ),
    # --- Even more professional (Boss's FuseClip recording) --------------
    # Dual-font headline+script: Anton caps for normal words, the keyword
    # word in Kaushan Script (yellow) — the Mumbai look, ClipForge's own.
    "Headline Script": CaptionPreset(
        "Headline Script", 72, True, _WHITE, _YELLOW, 2.5, 1.0,
        "dualfont", "Headline Script — হেডলাইন স্ক্রিপ্ট",
        font="Anton", script_font="Kaushan Script", uppercase=True,
        keyword_scale=1.6, keyword=_YELLOW, max_words=3,
    ),
    # Green-glow emphasize: white text, the keyword glowing green — the
    # Numaya look, ClipForge's own.
    "Emerald Glow": CaptionPreset(
        "Emerald Glow", 56, True, _WHITE, _YELLOW, 1.5, 0.0,
        "glow", "Emerald Glow — এমারাল্ড গ্লো",
        font="Poppins ExtraBold", keyword_scale=2.0, keyword=_GREEN_GLOW,
        max_words=3,
    ),
    # Stacked headline: each word on its own line, huge bold all-caps,
    # centred — the Headline look, ClipForge's own.
    "Stacked": CaptionPreset(
        "Stacked", 96, True, _WHITE, _YELLOW, 3.0, 2.0,
        "stacked", "Stacked — স্ট্যাকড",
        font="Anton", uppercase=True, keyword=_WHITE, max_words=4,
    ),
    # Per-keyword highlight boxes: every content word in its own box,
    # alternating yellow / magenta — the Rang look, ClipForge's own.
    "Keyword Boxes": CaptionPreset(
        "Keyword Boxes", 64, True, _WHITE, _YELLOW, 0.0, 0.0,
        "keywordbox", "Keyword Boxes — কীওয়ার্ড বক্স",
        font="Poppins ExtraBold", uppercase=True, keyword_mode="content",
        keyword=_MAGENTA, max_words=4,
    ),
}


# ---------------------------------------------------------------------------
# Caption colours
# ---------------------------------------------------------------------------

def hex_to_ass_color(value: str) -> str:
    """``#RRGGBB`` (or ``#RGB``) -> ASS ``&H00BBGGRR``."""
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"not a #RRGGBB colour: {value!r}")
    r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    return f"&H00{b:02X}{g:02X}{r:02X}"


@dataclass(frozen=True)
class CaptionOverlay:
    """A custom text line the user added at a chosen time (Edit step)."""

    text: str
    start: float  # clip-relative seconds
    end: float  # clip-relative seconds


@dataclass(frozen=True)
class CaptionColors:
    """User-chosen caption colours (Settings > Fonts).

    Every colour in every preset is changeable: text, keyword
    (the huge emphasised word in the keyword presets, and the bottom
    of the gradient fill), highlight (active karaoke word), outline,
    and box (background / highlight-box colour).
    """

    text: str = "#FFFFFF"  # base text colour
    highlight: str = "#FEC923"  # active-word colour
    outline: str = "#000000"  # outline colour
    box: str = "#FEC923"  # background / highlight-box colour
    keyword: str = "#FF2DC6"  # keyword-emphasis colour (magenta, sample 1)

    def to_ass(self) -> tuple[str, str, str, str, str]:
        """(text, highlight, outline, box, keyword) as ASS colours."""
        return (
            hex_to_ass_color(self.text),
            hex_to_ass_color(self.highlight),
            hex_to_ass_color(self.outline),
            hex_to_ass_color(self.box),
            hex_to_ass_color(self.keyword),
        )


#: Named colour schemes (Settings > Fonts). Light / dark / black included.
COLOR_SCHEMES: dict[str, CaptionColors] = {
    "Classic": CaptionColors("#FFFFFF", "#FEC923", "#000000", "#FEC923", "#FF2DC6"),
    "Light": CaptionColors("#111111", "#2563EB", "#FFFFFF", "#FFFFFF", "#D63384"),
    "Dark": CaptionColors("#FFFFFF", "#FEC923", "#000000", "#111111", "#FF2DC6"),
    "Black": CaptionColors("#FFFFFF", "#FEC923", "#000000", "#000000", "#FF2DC6"),
}

#: Max words per caption line and max line span in seconds.
_MAX_WORDS_PER_LINE = 5
_MAX_LINE_SECONDS = 1.5


def _escape_ass(text: str) -> str:
    """Escape characters that are special in ASS dialogue text."""
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _group_words(
    words: list[Word],
    *,
    max_words: int = _MAX_WORDS_PER_LINE,
    max_chars: int | None = None,
) -> list[list[Word]]:
    """Group words into caption lines (<=``max_words``, <=1.5 s each).

    ``max_chars`` (the Fonts-tab layout control) also breaks a line
    when adding a word would push it past that many characters; a
    single long word never splits.
    """
    lines: list[list[Word]] = []
    current: list[Word] = []
    current_chars = 0
    for word in words:
        extra = len(word.text) + (1 if current else 0)
        if current and (
            len(current) >= max_words
            or word.end - current[0].start > _MAX_LINE_SECONDS
            or (max_chars is not None and current_chars + extra > max_chars)
        ):
            lines.append(current)
            current = []
            current_chars = 0
        current.append(word)
        current_chars += len(word.text) + (1 if len(current) > 1 else 0)
    if current:
        lines.append(current)
    return lines


#: Public alias (caption editing reuses the same line grouping).
group_words = _group_words


# ---------------------------------------------------------------------------
# Keyword emphasis (Boss's reference screenshots 1, 2, 5)
# ---------------------------------------------------------------------------
# The viral podcast look: small normal words + ONE keyword word rendered
# MUCH bigger and in a different colour. The keyword is the longest
# content word in the line (ties go to the word closest to the centre —
# in the samples the emotional word sits mid-line, e.g. "pyaar", "Stress").

#: Function words that are never the keyword.
_STOPWORDS = frozenset(
    "a an the is are was were be been to in on of and or but it its this that "
    "my me he she we they you i am do does did so as at by for with not no "
    "yeh woh toh hai ka ki ke me ne se mai main jo bhi abhi kabhi thoda thodi "
    "bahut bahot aur ya phir jab tab".split()
)


def _word_letters(word: Word) -> str:
    """Letters of a word (punctuation stripped) for keyword scoring."""
    return "".join(ch for ch in word.text if ch.isalpha())


def pick_keyword(
    line: list[Word], *, mode: str = "longest"
) -> set[int]:
    """Indices of the keyword word(s) in ``line``.

    Modes: ``"longest"`` (longest content word, ties -> closest to the
    centre of the line — in the samples the emotional word sits
    mid-line, e.g. "pyaar", "Stress"), ``"last"`` (last word),
    ``"last_two"`` (last two words), ``"first_half"`` (first half of
    the line, sample 3's red line), ``"content"`` (every content word
    — the per-keyword highlight-boxes look).
    """
    if not line:
        return set()
    if mode == "last":
        return {len(line) - 1}
    if mode == "last_two":
        return set(range(max(0, len(line) - 2), len(line)))
    if mode == "first_half":
        return set(range((len(line) + 1) // 2))
    if mode == "content":
        idx = {
            i
            for i, word in enumerate(line)
            if _word_letters(word).lower() not in _STOPWORDS
        }
        return idx or {0}
    centre = (len(line) - 1) / 2
    best: int | None = None
    best_len = -1
    best_dist = float("inf")
    for i, word in enumerate(line):
        letters = _word_letters(word)
        if letters.lower() in _STOPWORDS:
            continue
        dist = abs(i - centre)
        if len(letters) > best_len or (
            len(letters) == best_len and dist < best_dist
        ):
            best, best_len, best_dist = i, len(letters), dist
    if best is None:  # every word was a stopword: longest overall wins
        best = max(range(len(line)), key=lambda i: len(_word_letters(line[i])))
    return {best}


def _ass_to_hex(ass_color: str) -> str:
    """ASS ``&H00BBGGRR`` -> ``#RRGGBB`` (inverse of hex_to_ass_color)."""
    text = ass_color.strip().lstrip("&").lstrip("H")
    if len(text) == 8:  # AABBGGRR
        text = text[2:]
    if len(text) != 6:
        raise ValueError(f"not an ASS colour: {ass_color!r}")
    b, g, r = text[0:2], text[2:4], text[4:6]
    return f"#{r}{g}{b}"


def _lerp_hex(first: str, second: str, ratio: float) -> str:
    """Blend two ``#RRGGBB`` colours (0.0 = first, 1.0 = second)."""
    a = tuple(int(first[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(second[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(round(x + (y - x) * ratio) for x, y in zip(a, b))
    return "#%02X%02X%02X" % mixed


def _keyword_line_text(
    line: list[Word],
    preset: CaptionPreset,
    *,
    font_size: int,
    keyword_size: int,
    keyword_ass: str,
    text_ass: str,
    small_factor: float = 0.65,
    keyword_idx: set[int] | None = None,
    keyword_font: str | None = None,
    main_font: str | None = None,
) -> str:
    """One line with inline size/colour overrides per word.

    Keyword word(s) render at ``keyword_size`` in the keyword colour
    (italic when the preset says so); the rest render small in the text
    colour. Upper-case when the preset says so. libass draws mixed
    sizes on one line on a shared baseline, which is the sample look
    (small words left/right, one huge word in the middle).
    ``keyword_idx`` overrides the preset's keyword detection (the
    Impact preset passes the whole line on even lines, none on odd).
    ``keyword_font`` / ``main_font`` switch fonts per word (the
    dual-font preset: keyword words in a script font, the rest in the
    preset's font); the keyword word keeps its original case so the
    script stays readable.
    """
    if keyword_idx is None:
        keyword_idx = pick_keyword(line, mode=preset.keyword_mode)
    small_size = max(16, round(font_size * small_factor))
    italic_on = "\\i1" if preset.keyword_italic else ""
    italic_off = "\\i0" if preset.keyword_italic else ""
    parts: list[str] = []
    for i, word in enumerate(line):
        if i in keyword_idx:
            # Keyword word: uppercase when the preset says so — except
            # in a script font (dual-font look), where the natural case
            # is the look (a script font in all-caps is unreadable).
            if keyword_font is not None:
                text = _escape_ass(word.text)
            else:
                text = _escape_ass(
                    word.text.upper() if preset.uppercase else word.text
                )
            font_tag = f"\\fn{keyword_font}" if keyword_font else ""
            parts.append(
                f"{{\\fs{keyword_size}\\c{keyword_ass}&{font_tag}{italic_on}}}"
                f"{text} "
            )
        else:
            text = _escape_ass(
                word.text.upper() if preset.uppercase else word.text
            )
            font_tag = f"\\fn{main_font}" if main_font and keyword_font else ""
            parts.append(
                f"{{\\fs{small_size}\\c{text_ass}&{font_tag}{italic_off}}}"
                f"{text} "
            )
    return "".join(parts).rstrip()


# ---------------------------------------------------------------------------
# Face-safe caption placement
# ---------------------------------------------------------------------------
# Rule (absolute): caption text must NEVER sit on the speaker's face or
# mouth. The caption zone is the bottom safe zone by default; when a
# tracked face box intersects it, the zone moves to the top safe zone
# (and shrinks if even the top is covered).

@dataclass(frozen=True)
class CaptionPlacement:
    """Where captions sit for one clip."""

    alignment: int  # ASS alignment: 2 = bottom-center, 8 = top-center
    margin_v: int  # px from the screen edge (bottom or top)
    font_size: int  # effective font size (may be shrunk to fit)


def _zone_rect(
    placement: CaptionPlacement, *, width: int, height: int
) -> tuple[float, float, float, float]:
    """Caption zone as (x, y, w, h) in output pixels.

    Up to two lines of text, so the zone is ~2.8x the font size tall.
    """
    zone_h = placement.font_size * 2.8
    margin_lr = int(SIDE_MARGIN * width / 1080.0)
    if placement.alignment == 8:  # top-center
        return (margin_lr, placement.margin_v, width - 2 * margin_lr, zone_h)
    return (
        margin_lr,
        height - placement.margin_v - zone_h,
        width - 2 * margin_lr,
        zone_h,
    )


def _rects_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def caption_zone_clear(
    face_rects: list[tuple[float, float, float, float]],
    placement: CaptionPlacement,
    *,
    width: int,
    height: int,
) -> bool:
    """True when no face rect intersects the caption zone."""
    zone = _zone_rect(placement, width=width, height=height)
    return not any(_rects_overlap(zone, rect) for rect in face_rects)


def _align_code(align: str, *, top: bool) -> int:
    """ASS alignment code for a horizontal align + top/bottom zone."""
    base = {"left": 1, "center": 2, "right": 3}.get(align, 2)
    return base + 6 if top else base


def choose_caption_placement(
    face_rects: list[tuple[float, float, float, float]],
    *,
    width: int,
    height: int,
    font_size: int,
    margin_v: int,
    align: str = "center",
    prefer_top: bool = False,
) -> tuple[CaptionPlacement, bool]:
    """Pick a caption zone that never overlaps a face.

    ``face_rects`` are (x, y, w, h) in output pixels, one per sampled
    frame. Tries the preferred zone first (bottom by default, top when
    ``prefer_top`` — the layout's vertical % is under 50), then the
    other zone, then the top zone with a smaller font. ``align`` is
    ``"left"`` / ``"center"`` / ``"right"``. Returns ``(placement,
    safe)`` — ``safe`` is False only when no placement clears every
    face (best effort then).
    """
    first = "top" if prefer_top else "bottom"
    for zone in (first, "top" if first == "bottom" else "bottom"):
        placement = CaptionPlacement(
            alignment=_align_code(align, top=zone == "top"),
            margin_v=margin_v,
            font_size=font_size,
        )
        if caption_zone_clear(face_rects, placement, width=width, height=height):
            return placement, True
    shrunk = CaptionPlacement(
        alignment=_align_code(align, top=True),
        margin_v=margin_v,
        font_size=max(16, int(font_size * 0.75)),
    )
    return shrunk, caption_zone_clear(
        face_rects, shrunk, width=width, height=height
    )


def layout_align(preset: "CaptionPreset", layout: CaptionLayout) -> str:
    """Effective horizontal align: the layout's, else the preset default.

    Lower Third is left-aligned by design; every other preset is
    centred unless the user picks otherwise.
    """
    if layout.align in ("left", "center", "right"):
        return layout.align
    return "left" if preset.animation == "lowerthird" else "center"


def layout_font_size(
    preset: "CaptionPreset", height: int, layout: CaptionLayout
) -> int:
    """Effective font size: the layout's % of frame height, else preset."""
    if layout.font_size_pct is not None:
        return max(16, round(height * layout.font_size_pct / 100.0))
    return max(16, round(preset.font_size * height / 1920.0))


def layout_margin_v(height: int, layout: CaptionLayout) -> int:
    """Bottom/top margin in px from the layout's vertical %.

    v_pct 86 reproduces the historical TikTok safe zone (14% of the
    frame height). Uses ``int()`` truncation, matching the old
    ``SAFE_ZONE_MARGIN_FRACTION`` math exactly.
    """
    if layout.v_pct >= 50:
        return int(height * (100.0 - layout.v_pct) / 100.0)
    return int(height * layout.v_pct / 100.0)


def _format_timestamp(seconds: float) -> str:
    """ASS timestamp ``H:MM:SS.cc``."""
    total_cs = max(0, int(round(seconds * 100)))
    hours, rem = divmod(total_cs, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _karaoke_text(line: list[Word], *, fill: bool = False) -> str:
    """One line as karaoke-tagged text (``\\k`` sweep, ``\\kf`` fill)."""
    tag = "kf" if fill else "k"
    return "".join(
        "{\\" + tag + str(max(1, int(round((w.end - w.start) * 100)))) + "}"
        + _escape_ass(w.text) + " "
        for w in line
    ).rstrip()


def _pos_y(placement: CaptionPlacement, height: int) -> int:
    """Absolute y for a ``\\pos`` tag: the alignment point of the text.

    Bottom alignment (2/1): the bottom of the text sits ``margin_v``
    above the bottom edge. Top alignment (8/7): the top of the text
    sits ``margin_v`` below the top edge.
    """
    if placement.alignment in (8, 7, 9):
        return placement.margin_v
    return height - placement.margin_v


def _gradient_band_color(
    stops: tuple[tuple[float, str], ...], ratio: float
) -> str:
    """Hex colour sampled from gradient ``stops`` at ``ratio`` (0–1)."""
    if ratio <= stops[0][0]:
        return stops[0][1]
    if ratio >= stops[-1][0]:
        return stops[-1][1]
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if p0 <= ratio <= p1:
            span = p1 - p0
            t = 0.0 if span <= 0 else (ratio - p0) / span
            return _lerp_hex(c0, c1, t)
    return stops[-1][1]


def _anchor_x(width: int, layout: CaptionLayout, align: str, margin_lr: int) -> int:
    """Absolute x anchor for ``\\pos`` events.

    h_pct 50 means "default": the centre for centre align, the side
    margin for left/right align. Otherwise the anchor is the h_pct
    position (the text centre for centre align, the text edge for
    left/right align).
    """
    if layout.h_pct == 50.0:
        if align == "left":
            return margin_lr
        if align == "right":
            return width - margin_lr
        return width // 2
    x = int(width * layout.h_pct / 100.0)
    if align == "left":
        return max(0, min(width - 2 * margin_lr, x))
    if align == "right":
        return max(2 * margin_lr, min(width, x))
    return x


def _style_margins(
    width: int, margin_lr: int, align: str, anchor_x: int
) -> tuple[int, int]:
    """(MarginL, MarginR) placing the style's text area on the anchor."""
    if align == "left":
        return anchor_x, margin_lr
    if align == "right":
        return margin_lr, width - anchor_x
    if anchor_x * 2 >= width:
        return max(0, 2 * anchor_x - width + margin_lr), margin_lr
    return margin_lr, max(0, margin_lr + width - 2 * anchor_x)


@lru_cache(maxsize=64)
def _pil_font(family: str, font_size: int):
    """PIL font for a bundled family, or None (falls back to estimates)."""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for font in BUNDLED_CAPTION_FONTS:
        if font.family == family:
            path = fonts_dir() / font.filename
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), font_size)
                except OSError:
                    return None
    return None


def _measure_word_widths(
    words: list[str], *, font_size: int, family: str
) -> list[float]:
    """Width in px of each word at ``font_size``.

    Measured with PIL on the bundled font file; when the font or PIL
    is unavailable, 0.6 * font_size per character is used. Used by the
    per-keyword highlight-boxes preset to place each boxed word
    exactly where it sits in the line.
    """
    pil = _pil_font(family, font_size)
    if pil is None:
        return [0.6 * font_size * max(1, len(w)) for w in words]
    return [float(pil.getlength(w)) for w in words]


def _line_events(
    blocks: list[list[list[Word]]],
    preset: CaptionPreset,
    placement: CaptionPlacement,
    *,
    width: int,
    height: int,
    colors: CaptionColors | None = None,
    keyword_scale: float = 0.0,
    text_ass: str = _WHITE,
    keyword_ass: str = _YELLOW,
    layout: CaptionLayout | None = None,
    gradient: GradientConfig | None = None,
    family: str = "Noto Sans",
) -> list[str]:
    """Dialogue events for one animation mode.

    ``blocks`` are groups of lines shown together (one line per block
    unless the layout merges lines). Every event's timestamps are
    exactly the (clip-relative) word times given by the caller — no
    re-timing happens here. ``colors`` is the effective (user-overridden)
    colour set; ``keyword_scale`` overrides the preset's keyword size
    (0.0 = preset default). ``layout`` positions the text (horizontal
    % / vertical % / align); ``gradient`` is the multi-stop gradient
    config for the gradient animation (``None`` = the classic
    text-colour -> keyword-colour blend).
    """
    animation = preset.animation if preset.animation in ANIMATIONS else "pop"
    layout = (layout or CaptionLayout()).validated()
    font_size = placement.font_size
    scale = keyword_scale if keyword_scale > 0 else preset.keyword_scale
    keyword_size = max(16, round(font_size * scale))
    if colors is None:
        text_hex = _ass_to_hex(preset.primary)
        keyword_hex = _ass_to_hex(preset.keyword or preset.highlight)
    else:
        text_hex, keyword_hex = colors.text, colors.keyword
    eff_align = layout_align(preset, layout)
    margin_lr = int(SIDE_MARGIN * width / 1080.0)
    pos_x = _anchor_x(width, layout, eff_align, margin_lr)
    pos_y = _pos_y(placement, height)
    # Events that already carry \pos use the anchor; the rest get an
    # explicit \pos only when the user moved the text off centre.
    use_pos = layout.h_pct != 50.0 or eff_align != "center"
    pos_tag = f"{{\\pos({pos_x},{pos_y})}}" if use_pos else ""
    events: list[str] = []
    line_no = 0  # running line index (Impact alternates colours per line)
    for block in blocks:
        lines = block
        start = _format_timestamp(lines[0][0].start)
        end = _format_timestamp(lines[-1][-1].end)
        line = lines[0]
        if animation == "karaoke":
            text = "\\N".join(_karaoke_text(l, fill=True) for l in lines)
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos_tag}{text}")
        elif animation == "typewriter":
            # One event per word, each showing the line so far: words
            # appear one by one with no highlight. Each event ends when
            # the next word starts, so exactly one prefix is visible at
            # a time. The events carry an explicit \pos — without it
            # libass stacks simultaneous same-style events vertically.
            # (Merging is ignored: typewriter is per line by nature.)
            for line in lines:
                line_end = _format_timestamp(line[-1].end)
                for i, word in enumerate(line):
                    prefix = " ".join(_escape_ass(w.text) for w in line[: i + 1])
                    event_end = (
                        _format_timestamp(line[i + 1].start)
                        if i + 1 < len(line)
                        else line_end
                    )
                    events.append(
                        "Dialogue: 0,"
                        f"{_format_timestamp(word.start)},{event_end},Default,,0,0,0,,"
                        f"{{\\pos({pos_x},{pos_y})}}{prefix}"
                    )
        elif animation == "slide":
            # Line slides into place in 300 ms: up from below at the
            # bottom, down from above at the top.
            if placement.alignment in (7, 8, 9):
                y0 = placement.margin_v - 180
                y1 = placement.margin_v
            else:
                y0 = height - placement.margin_v + 180
                y1 = height - placement.margin_v
            text = (
                f"{{\\move({pos_x},{y0},{pos_x},{y1},0,300)}}"
                + "\\N".join(_karaoke_text(l) for l in lines)
            )
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos_tag}{text}")
        elif animation == "bounce":
            # Line bounces in: 80% -> 108% -> 100% scale, then karaoke.
            text = (
                "{\\t(0,150,\\fscx80\\fscy80)"
                "\\t(150,300,\\fscx108\\fscy108)"
                "\\t(300,450,\\fscx100\\fscy100)}"
                + "\\N".join(_karaoke_text(l) for l in lines)
            )
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos_tag}{text}")
        elif animation == "fade":
            text = "{\\fad(250,0)}" + "\\N".join(_karaoke_text(l) for l in lines)
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos_tag}{text}")
        elif animation == "box":
            # Each word alone, centred, in the yellow-box style.
            # (Merging is ignored: each word is its own event anyway.)
            for line in lines:
                for word in line:
                    events.append(
                        "Dialogue: 0,"
                        f"{_format_timestamp(word.start)},"
                        f"{_format_timestamp(word.end)},Box,,0,0,0,,"
                        f"{pos_tag}{_escape_ass(word.text)}"
                    )
        elif animation in ("keyword", "garyvee"):
            # Viral keyword look (samples 1, 2, 5): small normal words,
            # ONE keyword word huge in the keyword colour. Whole line
            # visible for the line's duration, fades in.
            small = 0.8 if animation == "garyvee" else 0.65
            text = "{\\fad(120,0)}" + "\\N".join(
                _keyword_line_text(
                    l, preset, font_size=font_size,
                    keyword_size=keyword_size, keyword_ass=keyword_ass,
                    text_ass=text_ass, small_factor=small,
                )
                for l in lines
            )
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos_tag}{text}")
        elif animation == "dualfont":
            # Dual-font headline+script: normal words in the preset's
            # font (Anton caps), the keyword word(s) in the script font
            # (Kaushan Script) at the keyword size, in the keyword
            # colour. The script word keeps its original case.
            text = "{\\fad(120,0)}" + "\\N".join(
                _keyword_line_text(
                    l, preset, font_size=font_size,
                    keyword_size=keyword_size, keyword_ass=keyword_ass,
                    text_ass=text_ass, small_factor=0.65,
                    keyword_font=preset.script_font or None,
                    main_font=family,
                )
                for l in lines
            )
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos_tag}{text}")
        elif animation == "impact":
            # Sample 3: huge condensed all-caps, lines alternating
            # keyword colour (red) and text colour (white).
            parts = []
            for li, l in enumerate(lines):
                keyword_idx = (
                    set(range(len(l)))
                    if (line_no + li) % 2 == 0
                    else set()
                )
                parts.append(
                    _keyword_line_text(
                        l, preset, font_size=font_size,
                        keyword_size=font_size, keyword_ass=keyword_ass,
                        text_ass=text_ass, small_factor=1.0,
                        keyword_idx=keyword_idx,
                    )
                )
            text = "{\\t(0,120,\\fscx92\\fscy92)}" + "\\N".join(parts)
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos_tag}{text}")
        elif animation == "glow":
            # Sample 4: thin small words + huge bold white keyword with
            # a soft glow — a blurred, semi-transparent copy of the
            # same line on a lower layer, sharp text on top.
            mixed = "\\N".join(
                _keyword_line_text(
                    l, preset, font_size=font_size,
                    keyword_size=keyword_size, keyword_ass=keyword_ass,
                    text_ass=text_ass, small_factor=0.65,
                )
                for l in lines
            )
            events.append(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,"
                f"{pos_tag}{{\\blur6\\alpha&H80&}}{mixed}"
            )
            events.append(
                f"Dialogue: 1,{start},{end},Default,,0,0,0,,"
                f"{pos_tag}{{\\fad(120,0)}}{mixed}"
            )
        elif animation == "gradient":
            # Sample 6: huge heavy text with a vertical gradient fill.
            # libass has no gradient fill, so the text is drawn once
            # per band, each time clipped to one band with a colour
            # sampled from the gradient stops — a close visual match.
            # The events are simultaneous, so each carries an explicit
            # \pos: without it libass stacks same-style events
            # vertically and the bands no longer line up with the text.
            # Without a GradientConfig the classic blend is used (text
            # colour on top -> keyword colour at the bottom, 4 bands),
            # so the historical look is unchanged.
            cfg = (gradient or GradientConfig()).validated()
            if gradient is None:
                stops: tuple[tuple[float, str], ...] = (
                    (0.0, text_hex),
                    (1.0, keyword_hex),
                )
                band_count = 4
                horizontal = False
            else:
                stops = cfg.stops
                band_count = 1 if cfg.mode == "solid" else (8 if len(stops) > 2 else 4)
                horizontal = cfg.angle >= 45
            plain = "\\N".join(
                " ".join(
                    _escape_ass(w.text.upper() if preset.uppercase else w.text)
                    for w in l
                )
                for l in lines
            )
            line_h = font_size * 1.4
            block_h = line_h * len(lines)
            if placement.alignment in (7, 8, 9):
                top = float(placement.margin_v)
            else:
                top = height - placement.margin_v - block_h
            for band in range(band_count):
                ratio = (band + 0.5) / band_count
                band_hex = _gradient_band_color(stops, ratio)
                band_ass = hex_to_ass_color(band_hex)
                if horizontal:
                    x1 = band * width / band_count
                    x2 = (band + 1) * width / band_count
                    clip = f"\\clip({x1:.0f},{top:.0f},{x2:.0f},{top + block_h:.0f})"
                else:
                    y1 = top + band * block_h / band_count
                    y2 = top + (band + 1) * block_h / band_count
                    clip = f"\\clip(0,{y1:.0f},{width},{y2:.0f})"
                events.append(
                    f"Dialogue: 0,{start},{end},Default,,0,0,0,,"
                    f"{{\\pos({pos_x},{pos_y})"
                    f"{clip}"
                    f"\\c{band_ass}&}}{plain}"
                )
        elif animation == "stacked":
            # Stacked headline: every word on its own line, huge bold
            # all-caps, centred — the whole block visible for the
            # block's duration, the keyword word in the keyword colour.
            flat = [w for l in lines for w in l]
            row_h = font_size * 1.25
            block_h = row_h * len(flat)
            if placement.alignment in (7, 8, 9):
                top = float(placement.margin_v)
            else:
                top = height - placement.margin_v - block_h
            keyword_idx = pick_keyword(flat, mode=preset.keyword_mode)
            for i, word in enumerate(flat):
                y_i = top + (i + 1) * row_h
                color = keyword_ass if i in keyword_idx else text_ass
                text = _escape_ass(
                    word.text.upper() if preset.uppercase else word.text
                )
                events.append(
                    "Dialogue: 0,"
                    f"{start},{end},Default,,0,0,0,,"
                    f"{{\\pos({pos_x},{y_i:.0f})"
                    f"\\fs{font_size}\\c{color}&}}"
                    f"{text}"
                )
        elif animation == "keywordbox":
            # Per-keyword highlight boxes: every content word in its
            # own box, alternating yellow / magenta, each placed
            # exactly where it sits in the line (widths measured on
            # the preset's font). Other words render plain at their
            # own positions. Words appear only while spoken.
            space_w = _measure_word_widths([" "], font_size=font_size, family=family)[0]
            for l in lines:
                texts = [
                    w.text.upper() if preset.uppercase else w.text for w in l
                ]
                widths = _measure_word_widths(
                    texts, font_size=font_size, family=family
                )
                total = sum(widths) + space_w * max(0, len(l) - 1)
                x = pos_x - total / 2
                keyword_idx = pick_keyword(l, mode=preset.keyword_mode)
                box_no = 0
                for i, word in enumerate(l):
                    cx = x + widths[i] / 2
                    x += widths[i] + space_w
                    if i in keyword_idx:
                        style = "Box" if box_no % 2 == 0 else "Box2"
                        box_no += 1
                    else:
                        style = "Default"
                    events.append(
                        "Dialogue: 0,"
                        f"{_format_timestamp(word.start)},"
                        f"{_format_timestamp(word.end)},{style},,0,0,0,,"
                        f"{{\\pos({cx:.0f},{pos_y})}}"
                        f"{_escape_ass(texts[i])}"
                    )
        elif animation == "lowerthird":
            # Clean minimal lower-third: the whole line, static, on the
            # dark box drawn by the style (alignment handled in build_ass).
            plain = "\\N".join(
                " ".join(_escape_ass(w.text) for w in l) for l in lines
            )
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos_tag}{plain}")
        elif animation == "outline":
            # Outline-only text; the karaoke word fills with highlight.
            text = "\\N".join(_karaoke_text(l) for l in lines)
            events.append(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos_tag}{text}"
            )
        elif animation == "cleanpop":
            # Gentle scale pulse, active word keeps the text colour.
            text = (
                "{\\t(0,150,\\fscx100\\fscy100)"
                "\\t(150,300,\\fscx112\\fscy112)"
                "\\t(300,450,\\fscx100\\fscy100)}"
                + "\\N".join(_karaoke_text(l) for l in lines)
            )
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos_tag}{text}")
        else:  # "pop" — the original Hormozi karaoke, byte-identical.
            text = "\\N".join(_karaoke_text(l) for l in lines)
            events.append(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos_tag}{text}"
            )
        line_no += len(lines)
    return events


def build_ass(
    words: list[Word],
    *,
    language: str = "en",
    preset_name: str = "Hormozi",
    width: int = 1080,
    height: int = 1920,
    custom_font: str | None = None,
    font: str | None = None,
    colors: CaptionColors | None = None,
    keyword_scale: float = 0.0,
    face_rects: list[tuple[float, float, float, float]] | None = None,
    overlays: list[CaptionOverlay] | None = None,
    layout: CaptionLayout | None = None,
    gradient: GradientConfig | None = None,
) -> str:
    """Build an ASS subtitle document for ``words``.

    ``words`` must be time-ordered with start/end in seconds (already
    clip-relative). The document's PlayRes matches the output resolution
    so positions and sizes are exact; libass does the shaping/bidi at
    render time. Font precedence: ``custom_font`` (installed custom
    font family) wins, then ``font`` (the user's picked caption font),
    then the preset's own font, then the language font. Preset/picked
    fonts apply to Latin scripts only — Bengali/Arabic/Urdu always use
    their language font so the glyphs shape correctly. ``colors``
    overrides the preset's text/highlight/outline/box/keyword colours.
    ``keyword_scale`` overrides the preset's keyword size (0.0 =
    preset default). ``face_rects`` are (x, y, w, h) face boxes in
    output pixels — when given, the caption zone is moved (bottom ->
    top, then shrunk) until no caption text overlaps any face.
    ``overlays`` are custom text lines the user added (Edit step),
    rendered as plain caption events at their own times. ``layout``
    is the Fonts-tab layout (lines, words per line, max chars, align,
    horizontal/vertical position, font size %); every field defaults
    to "no change". ``gradient`` is the multi-stop gradient config
    for the gradient animation (``None`` = the classic text-colour ->
    keyword-colour blend).
    """
    preset = PRESETS.get(preset_name, PRESETS["Hormozi"])
    layout = (layout or CaptionLayout()).validated()
    font_size = layout_font_size(preset, height, layout)
    margin_v = layout_margin_v(height, layout)
    margin_lr = int(SIDE_MARGIN * width / 1080.0)
    eff_align = layout_align(preset, layout)
    anchor_x = _anchor_x(width, layout, eff_align, margin_lr)
    margin_l, margin_r = _style_margins(width, margin_lr, eff_align, anchor_x)
    if colors is None:
        colors = CaptionColors(
            text=_ass_to_hex(preset.primary),
            highlight=_ass_to_hex(preset.highlight),
            outline="#000000",
            # The box colour defaults to the highlight (yellow), NOT the
            # keyword: for Keyword Boxes the keyword is magenta and the
            # second box style (Box2) already uses it — using the keyword
            # here too would make both box styles magenta.
            box=_ass_to_hex(preset.highlight),
            keyword=_ass_to_hex(preset.keyword or preset.highlight),
        )
    primary, highlight, outline, box, keyword = colors.to_ass()
    if language in _FONTS:
        family = custom_font or font_family_for_language(language)
    else:
        family = custom_font or font or preset.font or "Noto Sans"
    if preset.animation == "cleanpop":
        highlight = primary  # active word keeps the text colour
    if face_rects is not None:
        placement, _safe = choose_caption_placement(
            face_rects, width=width, height=height,
            font_size=font_size, margin_v=margin_v,
            align=eff_align, prefer_top=layout.v_pct < 50,
        )
    else:
        placement = CaptionPlacement(
            alignment=_align_code(eff_align, top=layout.v_pct < 50),
            margin_v=margin_v, font_size=font_size,
        )
    font_size = placement.font_size

    if preset.animation == "outline":
        # Transparent fill, thick outline in the keyword colour; the
        # karaoke word fills with the highlight colour.
        primary = "&HFF000000"
        outline = keyword
        outline_width = max(preset.outline, 4.0)
        border_style = 1
        back = _SHADOW
    elif preset.animation == "lowerthird":
        # Dark semi-transparent box behind the text. The box uses the
        # outline colour (black in Classic/Dark/Black, white in Light)
        # so it stays readable in every colour scheme; the box colour
        # picker still controls the Highlight Box preset. ASS alpha
        # 0x40 = 75% opaque (0x00 is opaque, 0xFF transparent).
        # NOTE: this libass build only draws the BorderStyle-3 box when
        # Outline > 0, so the outline is 1px (invisible on the box).
        outline_width = 1.0
        border_style = 3
        back = "&H40" + outline[4:]
    else:
        outline_width = preset.outline
        border_style = 1
        back = _SHADOW

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{family},{font_size},{primary},"
        f"{highlight},{outline},{back},"
        f"{-1 if preset.bold else 0},0,0,0,100,100,0,0,{border_style},"
        f"{outline_width},{preset.shadow},{placement.alignment},"
        f"{margin_l},{margin_r},{placement.margin_v},1\n"
    )
    if preset.animation in ("box", "keywordbox"):
        # Opaque-box style: black text on a solid box (colour from settings).
        # NOTE: Outline must be 1, not 0 — this libass build only draws
        # the BorderStyle-3 box when Outline > 0 (same quirk the
        # lower-third box works around). And the box is drawn in the
        # OUTLINE colour, not BackColour, in libass 0.17.1 — verified by
        # a swap experiment (Outline=yellow, Back=blue rendered a yellow
        # box). So the outline colour IS the box colour; BackColour is
        # kept the same for other renderers. The 1px outline in the box
        # colour around black text on the same-colour box is invisible.
        header += (
            f"Style: Box,{family},{font_size},{_BLACK},"
            f"{_BLACK},{box},{box},"
            f"{-1 if preset.bold else 0},0,0,0,100,100,0,0,3,"
            f"1,0,{placement.alignment},{margin_l},{margin_r},{placement.margin_v},1\n"
        )
    if preset.animation == "keywordbox":
        # Second box colour (the keyword colour, magenta by default)
        # so per-keyword boxes alternate yellow / magenta.
        header += (
            f"Style: Box2,{family},{font_size},{_BLACK},"
            f"{_BLACK},{keyword},{keyword},"
            f"{-1 if preset.bold else 0},0,0,0,100,100,0,0,3,"
            f"1,0,{placement.alignment},{margin_l},{margin_r},{placement.margin_v},1\n"
        )
    header += (
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    lines = _group_words(
        words,
        max_words=layout.words_per_line or preset.max_words,
        max_chars=layout.max_chars,
    )
    if layout.lines and layout.lines > 1:
        blocks = [
            lines[i : i + layout.lines]
            for i in range(0, len(lines), layout.lines)
        ]
    else:
        blocks = [[line] for line in lines]
    events = _line_events(
        blocks, preset, placement, width=width, height=height,
        colors=colors, keyword_scale=keyword_scale,
        text_ass=primary if preset.animation != "outline" else _WHITE,
        keyword_ass=keyword,
        layout=layout, gradient=gradient, family=family,
    )
    for overlay in overlays or []:
        style = "Box" if preset.animation == "box" else "Default"
        events.append(
            "Dialogue: 0,"
            f"{_format_timestamp(overlay.start)},"
            f"{_format_timestamp(overlay.end)},{style},,0,0,0,,"
            f"{_escape_ass(overlay.text)}"
        )
    return header + "\n".join(events) + "\n"


def write_ass_file(ass_text: str, path: str | Path) -> Path:
    """Write an ASS document to ``path`` and return it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(ass_text, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Custom caption fonts (Settings > Fonts)
# ---------------------------------------------------------------------------

class FontError(ValueError):
    """A font file was rejected (not a font, unreadable, too big)."""


@dataclass(frozen=True)
class FontInfo:
    """An installed custom font."""

    filename: str  # file name inside the user fonts folder
    family: str  # family name, as used in the ASS style
    path: str  # full path of the installed copy


#: Magic bytes -> container kind. WOFF2 has no stdlib Brotli decoder, so
#: its name table is not parsed (family falls back to the file stem).
_FONT_MAGICS: dict[bytes, str] = {
    b"\x00\x01\x00\x00": "ttf",
    b"true": "ttf",
    b"OTTO": "otf",
    b"wOFF": "woff",
    b"wOF2": "woff2",
}

#: Reject absurd files early (a font is rarely over a few MB).
_MAX_FONT_BYTES = 50 * 1024 * 1024


def user_fonts_dir() -> Path:
    """User fonts folder (app data dir); user files never go to assets."""
    return app_data_dir() / "fonts"


def _parse_name_table(data: bytes) -> str | None:
    """Family name (nameID 1, else 16) from raw ``name`` table bytes."""
    if len(data) < 6:
        return None
    _fmt, count, string_offset = struct.unpack(">HHH", data[:6])
    best: tuple[int, str] | None = None
    for i in range(count):
        off = 6 + i * 12
        if off + 12 > len(data):
            break
        platform, _enc, _lang, name_id, length, str_off = struct.unpack(
            ">HHHHHH", data[off : off + 12]
        )
        if name_id not in (1, 16):
            continue
        start = string_offset + str_off
        raw = data[start : start + length]
        try:
            if platform in (0, 3):
                text = raw.decode("utf-16-be")
            elif platform == 1:
                text = raw.decode("mac_roman")
            else:
                text = raw.decode("utf-16-be", errors="ignore")
        except (UnicodeDecodeError, LookupError):
            continue
        text = text.strip()
        if not text:
            continue
        score = (
            0 if (name_id == 1 and platform == 3)
            else 1 if name_id == 1
            else 2
        )
        if best is None or score < best[0]:
            best = (score, text)
    return best[1] if best else None


def _find_sfnt_table(data: bytes, tag: bytes) -> bytes | None:
    """Raw table bytes from a TrueType/OpenType offset table."""
    if len(data) < 12:
        return None
    num_tables = struct.unpack(">H", data[4:6])[0]
    for i in range(num_tables):
        off = 12 + i * 16
        if off + 16 > len(data):
            break
        found, _chk, table_off, table_len = struct.unpack(
            ">4sIII", data[off : off + 16]
        )
        if found == tag:
            return data[table_off : table_off + table_len]
    return None


def _woff_name_table(data: bytes) -> bytes | None:
    """Raw ``name`` table from a WOFF file (tables are zlib-compressed)."""
    if len(data) < 44:
        return None
    num_tables = struct.unpack(">H", data[12:14])[0]
    for i in range(num_tables):
        off = 44 + i * 20
        if off + 20 > len(data):
            break
        tag, table_off, comp_len, _orig_len, _chk = struct.unpack(
            ">4sIIII", data[off : off + 20]
        )
        if tag != b"name":
            continue
        raw = data[table_off : table_off + comp_len]
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return raw
    return None


def font_family_from_file(path: str | Path) -> str | None:
    """Family name from a font file's name table, or None if unreadable."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    kind = _FONT_MAGICS.get(data[:4])
    if kind in ("ttf", "otf"):
        table = _find_sfnt_table(data, b"name")
    elif kind == "woff":
        table = _woff_name_table(data)
    else:
        return None  # woff2 / unknown: caller falls back to the stem
    return _parse_name_table(table) if table else None


def inspect_font_file(path: str | Path) -> FontInfo:
    """Validate a font file and read its family name.

    Accepts .ttf/.otf (name table parsed), .woff (name table parsed) and
    .woff2 (accepted by magic bytes; family falls back to the file stem).
    Raises :class:`FontError` with a friendly message for anything else.
    """
    source = Path(path)
    if not source.is_file():
        raise FontError(f"Font file not found: {source.name}")
    size = source.stat().st_size
    if size < 100:
        raise FontError(f"'{source.name}' is too small to be a font file.")
    if size > _MAX_FONT_BYTES:
        raise FontError(f"'{source.name}' is too large for a font file.")
    try:
        head = source.read_bytes()[:4]
    except OSError as exc:
        raise FontError(f"Could not read '{source.name}': {exc}") from exc
    if head not in _FONT_MAGICS:
        raise FontError(
            f"'{source.name}' is not a font file "
            "(expected .ttf, .otf, .woff or .woff2)."
        )
    family = font_family_from_file(source) or source.stem
    return FontInfo(filename=source.name, family=family, path=str(source))


def install_custom_font(
    source: str | Path, *, dest_dir: str | Path | None = None
) -> FontInfo:
    """Install a user font into the app-data fonts folder.

    Validates the file, copies it in, and copies the bundled fonts in
    alongside it so the folder is a complete ``fontsdir`` for libass
    (the render then needs only this one directory). Returns the
    installed :class:`FontInfo`.
    """
    info = inspect_font_file(source)
    dest = Path(dest_dir) if dest_dir is not None else user_fonts_dir()
    dest.mkdir(parents=True, exist_ok=True)
    for bundled in fonts_dir().glob("*.ttf"):
        target = dest / bundled.name
        if not target.exists():
            shutil.copyfile(bundled, target)
    safe_name = Path(info.filename).name or "custom.ttf"
    installed = dest / safe_name
    shutil.copyfile(source, installed)
    return FontInfo(filename=safe_name, family=info.family, path=str(installed))


def list_installed_fonts(
    *, dest_dir: str | Path | None = None
) -> list[FontInfo]:
    """Custom fonts installed in the user fonts folder (bundled excluded)."""
    dest = Path(dest_dir) if dest_dir is not None else user_fonts_dir()
    bundled = {p.name for p in fonts_dir().glob("*.ttf")}
    found: list[FontInfo] = []
    if not dest.is_dir():
        return found
    for path in sorted(dest.glob("*")):
        if not path.is_file() or path.name in bundled:
            continue
        if path.suffix.lower() not in (".ttf", ".otf", ".woff", ".woff2"):
            continue
        family = font_family_from_file(path) or path.stem
        found.append(FontInfo(filename=path.name, family=family, path=str(path)))
    return found


def resolve_custom_font(
    filename: str, *, base_dir: str | Path | None = None
) -> tuple[str | None, Path | None, str | None]:
    """Resolve a caption font file for rendering.

    ``filename`` may be a bundled font (viral/language list) or an
    installed custom font in the user fonts folder. Returns
    ``(family, fontsdir, error)`` — ``error`` is None on success.
    The fontsdir is the user fonts folder for custom fonts (bundled
    fonts were copied in at install time, so libass finds every font
    there) and the bundled folder for bundled fonts.
    """
    safe = Path(filename).name
    if not safe:
        return None, None, "No caption font selected."
    bundled = bundled_font_family(safe)
    if bundled is not None:
        return bundled, fonts_dir(), None
    base = Path(base_dir) if base_dir is not None else user_fonts_dir()
    installed = base / safe
    if not installed.is_file():
        return None, None, f"Custom font '{safe}' is missing — using the built-in font."
    family = font_family_from_file(installed) or installed.stem
    return family, base, None
