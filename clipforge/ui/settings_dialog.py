"""Settings dialog — 7 tabs, matching the reference recording.

Tabs (in order): General, Transcription, AI Provider, Processing,
Viral Clips, Fonts, Updates. The recording's "Experimental" tab is
REMOVED per the build spec (multi-person split screen is automatic).

Changes apply immediately (no Save/Cancel). API keys validate
automatically 1 second after typing or pasting; the inline status line
shows "Connected" or the exact reason (401 invalid key, 402 no balance,
404 wrong base URL or model, 429 rate limit, no internet). A saved key is
never shown again — the field stays empty with a "Key saved" placeholder.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..core.captions import (
    COLOR_SCHEMES,
    PRESETS,
    CaptionLayout,
    FontError,
    GradientConfig,
    install_custom_font,
    list_caption_fonts,
    list_installed_fonts,
    preview_slug,
    previews_dir,
)
from ..core.jobs import Job, run_job
from ..core.keys import (
    KeyringUnavailable,
    get_key,
    has_key,
    save_key,
    validate_deepgram,
    validate_deepseek,
)
from ..core.paths import app_data_dir
from ..core.updates import (
    UpdateCheck,
    UpdateResult,
    apply_update,
    check_for_updates,
    export_diagnostics,
)
from ..i18n import SUPPORTED_LANGUAGES, t
from ..settings import DEFAULT_DISCOVERY_PROMPT, DEFAULT_REVIEW_PROMPT, Settings
from .widgets import SegmentedControl, ToggleSwitch

TABS = [
    "settings.general",
    "settings.transcription",
    "settings.ai_provider",
    "settings.processing",
    "settings.viral_clips",
    "settings.fonts",
    "settings.updates",
]


class SettingsDialog(QDialog):
    """Modal settings dialog."""

    appearance_changed = Signal(str)
    language_changed = Signal(str)

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self.setWindowTitle(t("settings.title"))
        self.setModal(True)
        self.resize(680, 480)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._nav = QListWidget(self)
        self._nav.setFixedWidth(150)
        for key in TABS:
            QListWidgetItem(t(key), self._nav)
        self._nav.setCurrentRow(0)
        self._nav.currentRowChanged.connect(self._on_tab_changed)
        nav_wrap = QVBoxLayout()
        nav_wrap.setContentsMargins(10, 12, 6, 12)
        nav_title = QLabel(t("settings.title"), self)
        nav_title.setStyleSheet("font-weight: 600; padding-left: 8px;")
        nav_wrap.addWidget(nav_title)
        nav_wrap.addWidget(self._nav, 1)
        root.addLayout(nav_wrap)

        self._stack = QStackedWidget(self)
        self._pages = [
            self._general_page(),
            self._transcription_page(),
            self._ai_provider_page(),
            self._processing_page(),
            self._viral_clips_page(),
            self._fonts_page(),
            self._updates_page(),
        ]
        for page in self._pages:
            self._stack.addWidget(page)
        content = QVBoxLayout()
        content.setContentsMargins(12, 12, 12, 12)
        content.addWidget(self._stack, 1)
        root.addLayout(content, 1)

        # Key-validation state: debounce timers + running jobs.
        self._suppress_key_signals = False
        self._dg_seq = 0
        self._ds_seq = 0
        self._dg_pending_key = ""
        self._ds_pending_key = ""
        self._jobs: list[Job] = []
        self._revalidate_saved_keys()

    # -- helpers ---------------------------------------------------------
    def _page(self, title_key: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel(t(title_key), self)
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(title)
        card = QFrame(self)
        card.setProperty("panel", True)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(14)
        # Pages scroll: the Fonts tab now holds the preview grid, the
        # gradient editor and the layout controls as well.
        scroll = QScrollArea(self)
        scroll.setWidget(card)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(scroll, 1)
        return page, card_layout

    @staticmethod
    def _row(label_text: str, hint: str, control) -> QHBoxLayout:
        row = QHBoxLayout()
        texts = QVBoxLayout()
        texts.setSpacing(1)
        texts.addWidget(QLabel(label_text))
        hint_label = QLabel(hint)
        hint_label.setProperty("muted", True)
        hint_label.setStyleSheet("font-size: 12px;")
        hint_label.setWordWrap(True)
        texts.addWidget(hint_label)
        row.addLayout(texts, 1)
        if isinstance(control, QHBoxLayout):
            row.addLayout(control, 0)
        else:
            row.addWidget(control, 0)
        return row

    def _on_tab_changed(self, row: int) -> None:
        self._stack.setCurrentIndex(row)

    # -- pages -----------------------------------------------------------
    def _general_page(self) -> QWidget:
        page, card = self._page("settings.general")

        appearance = SegmentedControl(
            ["System", "Light", "Dark"],
            current=self._settings.get("appearance").capitalize(),
        )
        appearance.valueChanged.connect(
            lambda v: self._on_appearance(v.lower())
        )
        card.addLayout(self._row(
            t("settings.appearance"), t("settings.appearance_hint"), appearance,
        ))

        language = QComboBox(self)
        for code, name in SUPPORTED_LANGUAGES.items():
            language.addItem(name, code)
        language.setCurrentText(SUPPORTED_LANGUAGES[str(self._settings.get("app_language"))])
        language.currentIndexChanged.connect(
            lambda i: self.language_changed.emit(language.itemData(i))
        )
        card.addLayout(self._row(
            t("settings.app_language"), "", language,
        ))

        always = ToggleSwitch(bool(self._settings.get("always_approve")), self)
        always.toggled.connect(lambda v: self._settings.set("always_approve", v))
        card.addLayout(self._row(
            t("settings.always_approve"), t("settings.always_approve_hint"), always,
        ))

        export_btn = QPushButton(f"📁 {self._settings.export_dir.split('/')[-1]}", self)
        export_btn.clicked.connect(self._on_pick_export_dir)
        card.addLayout(self._row(
            t("settings.export_destination"), t("settings.export_hint"), export_btn,
        ))
        return page

    def _transcription_page(self) -> QWidget:
        page, card = self._page("settings.transcription")

        engine = QComboBox(self)
        engine.addItems(["Deepgram (Cloud API)", "Local Whisper (Offline)"])
        engine.setCurrentIndex(
            0 if self._settings.get("transcription_engine") == "deepgram" else 1
        )
        engine.currentIndexChanged.connect(
            lambda i: self._settings.set(
                "transcription_engine", "deepgram" if i == 0 else "whisper_local"
            )
        )
        card.addLayout(self._row(
            "Transcription engine", "Use local Whisper or Deepgram Cloud.", engine,
        ))

        key_row = QHBoxLayout()
        self._dg_key = QLineEdit(self)
        self._dg_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._dg_key.setPlaceholderText(
            "Key saved — type to replace" if has_key("deepgram") else "Deepgram API key"
        )
        self._dg_key.textChanged.connect(self._on_dg_key_edited)
        self._dg_timer = QTimer(self)
        self._dg_timer.setSingleShot(True)
        self._dg_timer.setInterval(1000)  # validate 1 s after typing/pasting
        self._dg_timer.timeout.connect(self._on_dg_timer)
        key_row.addWidget(self._dg_key, 1)
        eye = QToolButton(self)
        eye.setText("👁")
        eye.setCheckable(True)
        eye.toggled.connect(
            lambda c: self._dg_key.setEchoMode(
                QLineEdit.EchoMode.Normal if c else QLineEdit.EchoMode.Password
            )
        )
        key_row.addWidget(eye)
        card.addLayout(self._row(
            "API Key", "Required for cloud speech recognition. Get API key ↗", key_row,
        ))

        model = QComboBox(self)
        model.addItems(["Enhanced"])
        card.addLayout(self._row(
            "Model", "Speech recognition model for fast word-by-word timestamps", model,
        ))

        self._dg_status = QLabel("", self)
        self._dg_status.setProperty("muted", True)
        self._dg_status.setWordWrap(True)
        card.addWidget(self._dg_status)
        return page

    def _ai_provider_page(self) -> QWidget:
        page, card = self._page("settings.ai_provider")

        provider = QComboBox(self)
        provider.addItems(["OpenAI-compatible API"])
        card.addLayout(self._row("Provider", "Choose the AI Provider.", provider))

        base_url = QLineEdit(self)
        base_url.setText(str(self._settings.get("ai_base_url")))
        base_url.editingFinished.connect(
            lambda: self._on_base_url_changed(base_url.text())
        )
        card.addLayout(self._row(
            "Base URL", "API root ending in /v1; chat completions are used.", base_url,
        ))

        key_row = QHBoxLayout()
        self._ds_key = QLineEdit(self)
        self._ds_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._ds_key.setPlaceholderText(
            "Key saved — type to replace" if has_key("deepseek") else "DeepSeek API key"
        )
        self._ds_key.textChanged.connect(self._on_ds_key_edited)
        self._ds_timer = QTimer(self)
        self._ds_timer.setSingleShot(True)
        self._ds_timer.setInterval(1000)  # validate 1 s after typing/pasting
        self._ds_timer.timeout.connect(self._on_ds_timer)
        key_row.addWidget(self._ds_key, 1)
        eye = QToolButton(self)
        eye.setText("👁")
        eye.setCheckable(True)
        eye.toggled.connect(
            lambda c: self._ds_key.setEchoMode(
                QLineEdit.EchoMode.Normal if c else QLineEdit.EchoMode.Password
            )
        )
        key_row.addWidget(eye)
        card.addLayout(self._row(
            "API Key", "Optional for local servers; sent only to your endpoint.", key_row,
        ))

        self._ds_model = QComboBox(self)
        self._ds_model.setEditable(True)
        self._ds_model.addItems(["deepseek-flash", "deepseek-v4-pro"])
        self._ds_model.setCurrentText(str(self._settings.get("ai_model")))
        self._ds_model.currentTextChanged.connect(
            lambda text: self._settings.set("ai_model", text)
        )
        card.addLayout(self._row(
            "Model ID", "Exact model identifier accepted by your endpoint.", self._ds_model,
        ))

        self._ds_status = QLabel("", self)
        self._ds_status.setProperty("muted", True)
        self._ds_status.setWordWrap(True)
        card.addWidget(self._ds_status)

        note = QLabel(
            "These settings are stored on this device. Requests and any API key "
            "are sent to your configured endpoint.",
            self,
        )
        note.setProperty("muted", True)
        note.setWordWrap(True)
        card.addWidget(note)
        return page

    def _processing_page(self) -> QWidget:
        page, card = self._page("settings.processing")
        # The recording titles this tab's content "Acceleration".
        title = QLabel("Acceleration", self)
        title.setStyleSheet("font-weight: 600;")
        card.addWidget(title)

        transcription = SegmentedControl(
            ["Auto", "GPU", "CPU"],
            current=str(self._settings.get("accel_transcription")),
        )
        transcription.valueChanged.connect(
            lambda v: self._settings.set("accel_transcription", v)
        )
        card.addLayout(self._row(
            "Transcription", "Cloud transcription uses Deepgram.", transcription,
        ))

        rendering = SegmentedControl(
            ["Auto", "GPU", "CPU"],
            current=str(self._settings.get("accel_rendering")),
        )
        rendering.valueChanged.connect(
            lambda v: self._settings.set("accel_rendering", v)
        )
        card.addLayout(self._row(
            "Video rendering", "Auto uses GPU, with CPU fallback.", rendering,
        ))
        return page

    def _viral_clips_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel(t("settings.viral_clips"), self)
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(title)

        from PySide6.QtWidgets import QTabWidget

        tabs = QTabWidget(self)
        discovery = QPlainTextEdit(self)
        discovery.setPlainText(self._settings.discovery_prompt or DEFAULT_DISCOVERY_PROMPT)
        discovery.textChanged.connect(
            lambda: self._settings.set("discovery_prompt", discovery.toPlainText())
        )
        tabs.addTab(discovery, "Discovery")
        review = QPlainTextEdit(self)
        review.setPlainText(self._settings.review_prompt or DEFAULT_REVIEW_PROMPT)
        review.textChanged.connect(
            lambda: self._settings.set("review_prompt", review.toPlainText())
        )
        tabs.addTab(review, "Review")
        layout.addWidget(tabs, 1)

        variables = QToolButton(self)
        variables.setText("▸ Template variables")
        variables.setCheckable(True)
        variables.setStyleSheet("QToolButton { border: none; text-align: left; }")
        variables_label = QLabel(
            "{{transcript}} {{clipLength}} {{targetClips}} {{requestedClips}} "
            "{{searchBudget}} {{targetCount}} {{minDuration}} {{maxDuration}} "
            "{{targetDuration}} {{excludedRangesInstruction}}",
            self,
        )
        variables_label.setProperty("muted", True)
        variables_label.setWordWrap(True)
        variables_label.setVisible(False)
        variables.toggled.connect(variables_label.setVisible)
        layout.addWidget(variables)
        layout.addWidget(variables_label)
        return page

    def _fonts_page(self) -> QWidget:
        page, card = self._page("settings.fonts")
        # The recording titles this page "Custom Fonts".
        drop = QFrame(self)
        drop.setProperty("panel", True)
        drop.setStyleSheet("QFrame[panel=\"true\"] { border-style: dashed; }")
        drop_layout = QVBoxLayout(drop)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        aa = QLabel("Aa", self)
        aa.setStyleSheet("font-size: 20px; font-weight: 600;")
        aa.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(aa)
        drop_layout.addWidget(
            QLabel("Click to select or drop font files here", self),
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        supports = QLabel("Supports .ttf, .otf, .woff, and .woff2", self)
        supports.setProperty("muted", True)
        drop_layout.addWidget(supports, alignment=Qt.AlignmentFlag.AlignCenter)
        drop.setMinimumHeight(110)
        card.addWidget(drop)

        # Caption animation — viral podcast styles first.
        self._caption_preset = QComboBox(self)
        for name, preset in PRESETS.items():
            self._caption_preset.addItem(preset.display_name or name, name)
        self._caption_preset.setCurrentIndex(
            self._caption_preset.findData(
                str(self._settings.get("caption_preset"))
            )
        )
        self._caption_preset.currentIndexChanged.connect(
            self._on_caption_preset
        )
        card.addLayout(
            self._row(
                "Caption animation",
                "Viral podcast styles first — Hormozi pop and karaoke word "
                "highlight are the reference looks.",
                self._caption_preset,
            )
        )

        # Caption font picker — viral podcast fonts first, then the rest.
        self._caption_font = QComboBox(self)
        self._caption_font.currentIndexChanged.connect(self._on_caption_font)
        card.addLayout(
            self._row(
                "Caption font",
                "Viral podcast fonts first. A picked font replaces the "
                "preset's font (Latin scripts only).",
                self._caption_font,
            )
        )

        # Custom caption font — the user's own file, installed into the
        # app's fonts folder and used for every preset.
        custom_row = QHBoxLayout()
        choose = QPushButton("Choose font file…", self)
        choose.clicked.connect(self._on_pick_custom_font)
        custom_row.addWidget(choose)
        self._custom_font_label = QLabel(self)
        self._custom_font_label.setProperty("muted", True)
        custom_row.addWidget(self._custom_font_label, 1)
        clear = QPushButton("Clear", self)
        clear.clicked.connect(self._on_clear_custom_font)
        custom_row.addWidget(clear)
        card.addLayout(
            self._row(
                "Custom caption font",
                "Your own .ttf/.otf — installed into the app's fonts "
                "folder and used for every preset.",
                custom_row,
            )
        )
        self._custom_font_error = QLabel(self)
        self._custom_font_error.setStyleSheet("color: #ff6b6b; font-size: 12px;")
        self._custom_font_error.setWordWrap(True)
        card.addWidget(self._custom_font_error)

        # Colour scheme + the five changeable colours.
        self._color_scheme = QComboBox(self)
        self._color_scheme.addItem("Custom", "Custom")
        for name in COLOR_SCHEMES:
            self._color_scheme.addItem(name, name)
        self._color_scheme.setCurrentIndex(
            self._color_scheme.findData(
                str(self._settings.get("caption_color_scheme"))
            )
        )
        self._color_scheme.currentIndexChanged.connect(self._on_color_scheme)
        card.addLayout(
            self._row(
                "Colour scheme",
                "Classic, Light, Dark or Black — or pick each colour "
                "yourself below.",
                self._color_scheme,
            )
        )
        colors_row = QHBoxLayout()
        self._color_swatches: dict[str, QPushButton] = {}
        for key, label in (
            ("caption_text_color", "Text"),
            ("caption_keyword_color", "Keyword"),
            ("caption_highlight_color", "Highlight"),
            ("caption_outline_color", "Outline"),
            ("caption_box_color", "Box"),
        ):
            col = QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(QLabel(label, self))
            swatch = QPushButton(self)
            swatch.setFixedSize(34, 24)
            swatch.setToolTip(f"Change the {label.lower()} colour")
            swatch.clicked.connect(
                lambda _checked=False, k=key: self._on_pick_color(k)
            )
            col.addWidget(swatch)
            self._color_swatches[key] = swatch
            colors_row.addLayout(col)
        colors_row.addStretch(1)
        card.addLayout(
            self._row(
                "Caption colours",
                "Every colour in every preset is changeable — text, "
                "keyword, highlight, outline and box.",
                colors_row,
            )
        )

        # Keyword size — how much bigger the keyword word renders.
        self._keyword_scale = QDoubleSpinBox(self)
        self._keyword_scale.setRange(0.0, 2.5)
        self._keyword_scale.setSingleStep(0.1)
        self._keyword_scale.setSpecialValueText("Auto (preset)")
        self._keyword_scale.setValue(
            float(self._settings.get("caption_keyword_scale"))
        )
        self._keyword_scale.valueChanged.connect(self._on_keyword_scale)
        card.addLayout(
            self._row(
                "Keyword size",
                "How much bigger the keyword word is (the samples use "
                "about 2x). Auto uses the preset's size.",
                self._keyword_scale,
            )
        )

        # Style previews — every preset rendered with "a quick brown
        # fox", so the style is picked visually. Clicking a card picks
        # that preset (same as the dropdown above).
        previews_title = QLabel("Style previews", self)
        previews_title.setStyleSheet("font-weight: 600;")
        card.addWidget(previews_title)
        previews_hint = QLabel(
            "Every style rendered with “a quick brown fox” — click one "
            "to use it. The previews are real renders of each preset.",
            self,
        )
        previews_hint.setProperty("muted", True)
        previews_hint.setWordWrap(True)
        card.addWidget(previews_hint)
        self._preview_group = QButtonGroup(self)
        self._preview_group.setExclusive(True)
        self._preview_group.buttonClicked.connect(self._on_preview_picked)
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, (name, preset) in enumerate(PRESETS.items()):
            button = QToolButton(self)
            button.setCheckable(True)
            button.setToolButtonStyle(
                Qt.ToolButtonStyle.ToolButtonTextUnderIcon
            )
            button.setText(preset.display_name or name)
            button.setProperty("preset_name", name)
            pixmap_path = previews_dir() / f"{preview_slug(name)}.png"
            if pixmap_path.exists():
                pixmap = QPixmap(str(pixmap_path))
                button.setIcon(QIcon(pixmap))
                button.setIconSize(
                    pixmap.size().scaled(
                        96, 171, Qt.AspectRatioMode.KeepAspectRatio
                    )
                )
            button.setMinimumSize(104, 196)
            self._preview_group.addButton(button)
            grid.addWidget(button, i // 4, i % 4)
        card.addLayout(grid)
        self._sync_preview_checked()

        # Gradient editor — up to 5 stops, each colour + position
        # changeable, Solid/Gradient toggle + direction. Applies to
        # the Gradient preset's fill (other presets ignore it).
        gradient_title = QLabel("Gradient", self)
        gradient_title.setStyleSheet("font-weight: 600;")
        card.addWidget(gradient_title)
        self._gradient_mode = QComboBox(self)
        self._gradient_mode.addItem("Gradient", "gradient")
        self._gradient_mode.addItem("Solid", "solid")
        self._gradient_mode.currentIndexChanged.connect(
            self._on_gradient_mode
        )
        self._gradient_angle = QComboBox(self)
        self._gradient_angle.addItem("Vertical (top → bottom)", 0.0)
        self._gradient_angle.addItem("Horizontal (left → right)", 90.0)
        self._gradient_angle.currentIndexChanged.connect(
            self._on_gradient_angle
        )
        gradient_top = QHBoxLayout()
        gradient_top.addWidget(QLabel("Fill", self))
        gradient_top.addWidget(self._gradient_mode)
        gradient_top.addWidget(QLabel("Direction", self))
        gradient_top.addWidget(self._gradient_angle)
        gradient_top.addStretch(1)
        card.addLayout(gradient_top)
        self._gradient_rows_layout = QVBoxLayout()
        card.addLayout(self._gradient_rows_layout)
        gradient_buttons = QHBoxLayout()
        add_stop = QPushButton("Add stop", self)
        add_stop.clicked.connect(self._on_gradient_add_stop)
        gradient_buttons.addWidget(add_stop)
        remove_stop = QPushButton("Remove stop", self)
        remove_stop.clicked.connect(self._on_gradient_remove_stop)
        gradient_buttons.addWidget(remove_stop)
        gradient_buttons.addStretch(1)
        card.addLayout(gradient_buttons)
        self._gradient_rows: list[tuple[QPushButton, QDoubleSpinBox]] = []
        self._refresh_gradient_editor()

        # Layout — lines, words per line, max chars, align, position,
        # font size %. Every field defaults to the preset's own layout
        # ("Auto"); a set value overrides it for every preset.
        layout_title = QLabel("Layout", self)
        layout_title.setStyleSheet("font-weight: 600;")
        card.addWidget(layout_title)
        layout = self._settings.caption_layout()
        self._layout_lines = QComboBox(self)
        self._layout_lines.addItem("Auto", None)
        for n in (1, 2, 3):
            self._layout_lines.addItem(str(n), n)
        self._layout_lines.setCurrentIndex(
            self._layout_lines.findData(layout.lines)
        )
        self._layout_lines.currentIndexChanged.connect(self._on_layout_changed)
        card.addLayout(
            self._row(
                "Lines shown together",
                "Auto shows one line at a time; 2–3 stack lines in one "
                "caption event.",
                self._layout_lines,
            )
        )
        self._layout_words = QSpinBox(self)
        self._layout_words.setRange(0, 12)
        self._layout_words.setSpecialValueText("Auto (preset)")
        self._layout_words.setValue(layout.words_per_line or 0)
        self._layout_words.valueChanged.connect(self._on_layout_changed)
        card.addLayout(
            self._row(
                "Words per line",
                "How many words each caption line holds (Auto uses the "
                "preset's grouping).",
                self._layout_words,
            )
        )
        self._layout_max_chars = QSpinBox(self)
        self._layout_max_chars.setRange(0, 120)
        self._layout_max_chars.setSpecialValueText("Auto (preset)")
        self._layout_max_chars.setValue(layout.max_chars or 0)
        self._layout_max_chars.valueChanged.connect(self._on_layout_changed)
        card.addLayout(
            self._row(
                "Max characters per line",
                "A line breaks early when it would pass this many "
                "characters (Auto = no limit).",
                self._layout_max_chars,
            )
        )
        self._layout_align = QComboBox(self)
        for label, value in (
            ("Auto", "auto"),
            ("Left", "left"),
            ("Center", "center"),
            ("Right", "right"),
        ):
            self._layout_align.addItem(label, value)
        self._layout_align.setCurrentIndex(
            self._layout_align.findData(layout.align)
        )
        self._layout_align.currentIndexChanged.connect(
            self._on_layout_changed
        )
        card.addLayout(
            self._row(
                "Align",
                "Left, centre or right — Auto keeps the preset's "
                "alignment (centred, except Lower Third).",
                self._layout_align,
            )
        )
        self._layout_h_pct = QSpinBox(self)
        self._layout_h_pct.setRange(0, 100)
        self._layout_h_pct.setSuffix("%")
        self._layout_h_pct.setValue(int(layout.h_pct))
        self._layout_h_pct.valueChanged.connect(self._on_layout_changed)
        card.addLayout(
            self._row(
                "Horizontal position",
                "Where the caption sits across the frame (50% = centre).",
                self._layout_h_pct,
            )
        )
        self._layout_v_pct = QSpinBox(self)
        self._layout_v_pct.setRange(0, 100)
        self._layout_v_pct.setSuffix("%")
        self._layout_v_pct.setValue(int(layout.v_pct))
        self._layout_v_pct.valueChanged.connect(self._on_layout_changed)
        card.addLayout(
            self._row(
                "Vertical position",
                "Where the caption sits down the frame (86% = the usual "
                "TikTok safe zone at the bottom).",
                self._layout_v_pct,
            )
        )
        self._layout_font_size = QDoubleSpinBox(self)
        self._layout_font_size.setRange(0.0, 10.0)
        self._layout_font_size.setSingleStep(0.5)
        self._layout_font_size.setSuffix("%")
        self._layout_font_size.setSpecialValueText("Auto (preset)")
        self._layout_font_size.setValue(layout.font_size_pct or 0.0)
        self._layout_font_size.valueChanged.connect(self._on_layout_changed)
        card.addLayout(
            self._row(
                "Font size",
                "% of the frame height (Auto uses the preset's size; "
                "about 3.3% matches Hormozi at 1080p).",
                self._layout_font_size,
            )
        )

        installed = QFrame(self)
        installed.setProperty("panel", True)
        installed_layout = QVBoxLayout(installed)
        installed_title = QLabel("Installed fonts", self)
        installed_title.setStyleSheet("font-weight: 600;")
        installed_layout.addWidget(installed_title)
        self._installed_label = QLabel(self)
        self._installed_label.setProperty("muted", True)
        self._installed_label.setWordWrap(True)
        installed_layout.addWidget(self._installed_label)
        note = QLabel(
            "Installed fonts will automatically appear in the caption styling "
            "and presets panels for rendering in video clips.",
            self,
        )
        note.setProperty("muted", True)
        note.setWordWrap(True)
        installed_layout.addWidget(note)
        card.addWidget(installed)
        self._refresh_fonts_page()
        return page

    # -- Fonts tab ---------------------------------------------------------
    def _refresh_fonts_page(self) -> None:
        """Rebuild the font picker + installed list from current state."""
        custom = str(self._settings.get("custom_font"))
        picked = str(self._settings.get("caption_font"))
        self._caption_font.blockSignals(True)
        self._caption_font.clear()
        self._caption_font.addItem("Preset default", "")
        for font in list_caption_fonts():
            self._caption_font.addItem(font.display, font.filename)
        installed = list_installed_fonts()
        for font in installed:
            self._caption_font.addItem(f"{font.family} (custom)", font.filename)
        current = custom or picked
        index = self._caption_font.findData(current) if current else 0
        self._caption_font.setCurrentIndex(max(0, index))
        self._caption_font.blockSignals(False)
        if custom:
            self._custom_font_label.setText(f"Using: {custom}")
        else:
            self._custom_font_label.setText("No custom font — preset default")
        if installed:
            self._installed_label.setText(
                "Installed: "
                + ", ".join(f.family for f in installed)
            )
        else:
            self._installed_label.setText("No custom fonts installed yet")
        self._refresh_color_swatches()

    def _refresh_color_swatches(self) -> None:
        for key, swatch in self._color_swatches.items():
            color = str(self._settings.get(key))
            swatch.setStyleSheet(
                f"background-color: {color}; "
                "border: 1px solid #888; border-radius: 3px;"
            )

    def _on_caption_preset(self, _index: int) -> None:
        name = self._caption_preset.currentData()
        if name:
            self._settings.set("caption_preset", str(name))
            self._sync_preview_checked()

    def _on_caption_font(self, _index: int) -> None:
        filename = str(self._caption_font.currentData() or "")
        installed = {f.filename for f in list_installed_fonts()}
        if not filename:
            self._settings.set("caption_font", "")
            self._settings.set("custom_font", "")
        elif filename in installed:
            self._settings.set("caption_font", filename)
            self._settings.set("custom_font", filename)
        else:
            self._settings.set("caption_font", filename)
            self._settings.set("custom_font", "")
        self._refresh_fonts_page()

    def _on_pick_custom_font(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Choose a caption font",
            "",
            "Font files (*.ttf *.otf *.woff *.woff2)",
        )
        if not path:
            return
        try:
            info = install_custom_font(path)
        except FontError as exc:
            self._custom_font_error.setText(str(exc))
            return
        self._custom_font_error.setText("")
        self._settings.set("custom_font", info.filename)
        self._settings.set("caption_font", "")
        self._refresh_fonts_page()

    def _on_clear_custom_font(self) -> None:
        self._custom_font_error.setText("")
        self._settings.set("custom_font", "")
        self._refresh_fonts_page()

    def _on_color_scheme(self, _index: int) -> None:
        name = str(self._color_scheme.currentData() or "")
        if name == "Custom" or name not in COLOR_SCHEMES:
            return
        scheme = COLOR_SCHEMES[name]
        self._settings.set("caption_color_scheme", name)
        self._settings.set("caption_text_color", scheme.text)
        self._settings.set("caption_keyword_color", scheme.keyword)
        self._settings.set("caption_highlight_color", scheme.highlight)
        self._settings.set("caption_outline_color", scheme.outline)
        self._settings.set("caption_box_color", scheme.box)
        self._refresh_color_swatches()

    def _on_pick_color(self, key: str) -> None:
        current = QColor(str(self._settings.get(key)))
        chosen = QColorDialog.getColor(current, self, "Choose colour")
        if not chosen.isValid():
            return
        self._settings.set(key, chosen.name().upper())
        self._color_scheme.blockSignals(True)
        self._color_scheme.setCurrentIndex(0)  # Custom
        self._color_scheme.blockSignals(False)
        self._refresh_color_swatches()

    def _on_keyword_scale(self, value: float) -> None:
        self._settings.set("caption_keyword_scale", float(value))

    # -- Fonts tab: style previews ---------------------------------------
    def _sync_preview_checked(self) -> None:
        """Check the preview card matching the current preset."""
        group = getattr(self, "_preview_group", None)
        if group is None:
            return
        current = str(self._settings.get("caption_preset"))
        for button in group.buttons():
            button.setChecked(button.property("preset_name") == current)

    def _on_preview_picked(self, button) -> None:
        """A preview card was clicked — pick that preset."""
        name = button.property("preset_name")
        if not name:
            return
        index = self._caption_preset.findData(name)
        if index >= 0:
            # Fires _on_caption_preset, which saves + syncs the cards.
            self._caption_preset.setCurrentIndex(index)

    # -- Fonts tab: gradient editor --------------------------------------
    def _refresh_gradient_editor(self) -> None:
        """Rebuild the stop rows from the saved gradient config."""
        gradient = self._settings.caption_gradient()
        while self._gradient_rows_layout.count():
            item = self._gradient_rows_layout.takeAt(0)
            child_layout = item.layout()
            if child_layout is not None:
                while child_layout.count():
                    sub = child_layout.takeAt(0)
                    if sub.widget() is not None:
                        sub.widget().deleteLater()
            elif item.widget() is not None:
                item.widget().deleteLater()
        self._gradient_rows = []
        self._gradient_colors = []
        self._gradient_mode.blockSignals(True)
        self._gradient_mode.setCurrentIndex(
            self._gradient_mode.findData(gradient.mode)
        )
        self._gradient_mode.blockSignals(False)
        self._gradient_angle.blockSignals(True)
        self._gradient_angle.setCurrentIndex(
            self._gradient_angle.findData(gradient.angle)
        )
        self._gradient_angle.blockSignals(False)
        # Solid mode shows a single stop (the solid colour).
        stops = gradient.stops[:1] if gradient.mode == "solid" else gradient.stops
        for pos, color in stops:
            row = QHBoxLayout()
            swatch = QPushButton(self)
            swatch.setFixedSize(34, 24)
            swatch.setToolTip("Change this stop's colour")
            swatch.setStyleSheet(
                f"background-color: {color}; "
                "border: 1px solid #888; border-radius: 3px;"
            )
            swatch.clicked.connect(
                lambda _checked=False, idx=len(self._gradient_rows):
                    self._on_gradient_stop_color(idx)
            )
            row.addWidget(swatch)
            spin = QDoubleSpinBox(self)
            spin.setRange(0.0, 100.0)
            spin.setSingleStep(5.0)
            spin.setSuffix("%")
            spin.setValue(pos * 100.0)
            spin.valueChanged.connect(self._on_gradient_stop_position)
            row.addWidget(spin)
            row.addStretch(1)
            self._gradient_rows_layout.addLayout(row)
            self._gradient_rows.append((swatch, spin))
            self._gradient_colors.append(color)

    def _collect_gradient(self) -> GradientConfig:
        """Gradient config from the editor's current widgets."""
        mode = str(self._gradient_mode.currentData() or "gradient")
        angle = float(self._gradient_angle.currentData() or 0.0)
        stops = tuple(
            (spin.value() / 100.0, self._gradient_colors[i])
            for i, (_swatch, spin) in enumerate(self._gradient_rows)
        )
        return GradientConfig(mode=mode, angle=angle, stops=stops).validated()

    def _save_gradient(self) -> None:
        self._settings.set_caption_gradient(self._collect_gradient())

    def _on_gradient_mode(self, _index: int) -> None:
        # Switching modes adjusts the stops first: Solid keeps only
        # the first stop (the solid colour); back to Gradient restores
        # a second stop. The refresh then rebuilds the rows to match.
        mode = str(self._gradient_mode.currentData() or "gradient")
        gradient = self._collect_gradient()
        if mode == "solid":
            stops = gradient.stops[:1]
        elif len(gradient.stops) < 2:
            stops = gradient.stops + ((1.0, gradient.stops[0][1]),)
        else:
            stops = gradient.stops
        self._settings.set_caption_gradient(
            GradientConfig(
                mode=mode, angle=gradient.angle, stops=stops,
            ).validated()
        )
        self._refresh_gradient_editor()

    def _on_gradient_angle(self, _index: int) -> None:
        self._save_gradient()

    def _on_gradient_stop_position(self, _value: float) -> None:
        # Save only (no refresh) so dragging the position keeps focus.
        self._save_gradient()

    def _on_gradient_stop_color(self, index: int) -> None:
        if index >= len(self._gradient_colors):
            return
        current = QColor(self._gradient_colors[index])
        chosen = QColorDialog.getColor(current, self, "Choose stop colour")
        if not chosen.isValid():
            return
        color = chosen.name().upper()
        self._gradient_colors[index] = color
        swatch = self._gradient_rows[index][0]
        swatch.setStyleSheet(
            f"background-color: {color}; "
            "border: 1px solid #888; border-radius: 3px;"
        )
        self._save_gradient()

    def _on_gradient_add_stop(self) -> None:
        gradient = self._collect_gradient()
        if len(gradient.stops) >= 5:
            return  # 5 stops max
        stops = list(gradient.stops)
        # New stop halfway between the last two (or at 50%).
        if len(stops) >= 2:
            pos = (stops[-2][0] + stops[-1][0]) / 2.0
            color = stops[-1][1]
        else:
            pos, color = 0.5, stops[-1][1]
        stops.append((pos, color))
        self._settings.set_caption_gradient(
            GradientConfig(
                mode=gradient.mode, angle=gradient.angle,
                stops=tuple(stops),
            ).validated()
        )
        self._refresh_gradient_editor()

    def _on_gradient_remove_stop(self) -> None:
        gradient = self._collect_gradient()
        minimum = 1 if gradient.mode == "solid" else 2
        if len(gradient.stops) <= minimum:
            return
        stops = gradient.stops[:-1]
        self._settings.set_caption_gradient(
            GradientConfig(
                mode=gradient.mode, angle=gradient.angle, stops=stops,
            ).validated()
        )
        self._refresh_gradient_editor()

    # -- Fonts tab: layout controls --------------------------------------
    def _on_layout_changed(self, _value=None) -> None:
        """Any layout control changed — persist the layout."""
        layout = CaptionLayout(
            lines=self._layout_lines.currentData(),
            words_per_line=self._layout_words.value() or None,
            max_chars=self._layout_max_chars.value() or None,
            align=str(self._layout_align.currentData() or "auto"),
            h_pct=float(self._layout_h_pct.value()),
            v_pct=float(self._layout_v_pct.value()),
            font_size_pct=self._layout_font_size.value() or None,
        ).validated()
        self._settings.set_caption_layout(layout)

    def _updates_page(self) -> QWidget:
        page, card = self._page("settings.updates")
        version = QLabel(f"ClipForge {__version__}", self)
        version.setStyleSheet("font-weight: 600;")
        card.addWidget(version)

        check = QCheckBox("Check for updates on launch", self)
        check.setChecked(bool(self._settings.get("check_updates_on_launch")))
        check.toggled.connect(
            lambda v: self._settings.set("check_updates_on_launch", v)
        )
        card.addWidget(check)

        self._update_status = QLabel("", self)
        self._update_status.setProperty("muted", True)
        self._update_status.setWordWrap(True)
        card.addWidget(self._update_status)

        row = QHBoxLayout()
        check_btn = QPushButton(t("settings.check_updates"), self)
        check_btn.clicked.connect(self._on_check_updates)
        row.addWidget(check_btn)
        diag_btn = QPushButton(t("settings.export_diagnostics"), self)
        diag_btn.clicked.connect(self._on_export_diagnostics)
        row.addWidget(diag_btn)
        row.addStretch(1)
        card.addLayout(row)
        return page

    # -- updates ---------------------------------------------------------
    def _on_check_updates(self) -> None:
        """Check the GitHub Releases feed in a background job."""
        self._update_status.setText("Checking for updates…")

        def _work(ctx):
            return check_for_updates()

        run_job(
            _work,
            on_finished=self._on_update_checked,
            on_failed=lambda err: self._update_status.setText(
                f"Could not check for updates ({err})"
            ),
            parent=self,
        )

    def _on_update_checked(self, check: object) -> None:
        if not isinstance(check, UpdateCheck):
            return
        self._update_status.setText(check.message)
        if check.available and check.info is not None and check.info.installer_url:
            from PySide6.QtWidgets import QMessageBox  # noqa: PLC0415 - local use

            answer = QMessageBox.question(
                self,
                t("settings.updates"),
                f"{check.message}\n\nDownload and install it now?",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._download_and_install(check.info)

    def _download_and_install(self, info) -> None:
        """One click: download, verify, back up, install (auto-rollback).

        The whole flow is :func:`core.updates.apply_update` — the dialog
        only reports the outcome, so the UI flow and the tested core flow
        can never drift apart.
        """
        from ..settings import DEFAULTS  # noqa: PLC0415 - local, avoids a cycle

        self._update_status.setText(f"Downloading ClipForge {info.version}…")

        def _work(ctx):
            snapshot = {key: self._settings.get(key) for key in DEFAULTS}
            return apply_update(info, settings_snapshot=snapshot)

        def _done(result: object) -> None:
            if not isinstance(result, UpdateResult):
                return
            if result.outcome == "installer_launched":
                self._update_status.setText("Installer launched — ClipForge will close.")
                from PySide6.QtWidgets import QApplication  # noqa: PLC0415 - local use

                app = QApplication.instance()
                if app is not None:
                    app.quit()
            else:
                self._update_status.setText(
                    f"Installer downloaded: {result.installer_path}"
                )

        def _failed(error: str) -> None:
            self._update_status.setText(f"Update failed: {error}")

        run_job(_work, on_finished=_done, on_failed=_failed, parent=self)

    def _on_export_diagnostics(self) -> None:
        """Zip the rotating logs (scrubbed) and save them where asked."""
        from pathlib import Path  # noqa: PLC0415 - local use

        default = str(app_data_dir() / "clipforge-diagnostics.zip")
        path, _ = QFileDialog.getSaveFileName(
            self, t("settings.export_diagnostics"), default,
            "ZIP archives (*.zip)",
        )
        if not path:
            return
        try:
            out = export_diagnostics(
                app_data_dir() / "logs", Path(path)
            )
        except OSError as exc:
            self._update_status.setText(f"Diagnostics export failed: {exc}")
            return
        self._update_status.setText(f"Diagnostics saved to {out}")
        self._show_info(f"Diagnostics saved to:\n{out}")

    # -- actions ---------------------------------------------------------
    def _on_appearance(self, value: str) -> None:
        self._settings.set("appearance", value)
        self.appearance_changed.emit(value)

    def _on_pick_export_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, t("settings.export_destination"), self._settings.export_dir
        )
        if directory:
            self._settings.set("export_destination", directory)

    # -- API key validation (1 s debounce, background job, inline status) --
    def _revalidate_saved_keys(self) -> None:
        """Silently re-check keys saved from an earlier session on open."""
        stored_dg = get_key("deepgram")
        if stored_dg:
            self._dg_status.setText("Checking saved key…")
            self._start_dg_validation(stored_dg, from_field=False)
        stored_ds = get_key("deepseek")
        if stored_ds:
            self._ds_status.setText("Checking saved key…")
            self._start_ds_validation(stored_ds, from_field=False)

    def _on_dg_key_edited(self, text: str) -> None:
        if self._suppress_key_signals:
            return
        if not text.strip():
            self._dg_status.setText(
                "Connected" if has_key("deepgram") else ""
            )
            return
        self._dg_status.setText("Checking…")
        self._dg_timer.start()  # restarts on every keystroke/paste

    def _on_dg_timer(self) -> None:
        key = self._dg_key.text().strip()
        if key:
            self._start_dg_validation(key, from_field=True)

    def _start_dg_validation(self, key: str, *, from_field: bool) -> None:
        self._dg_seq += 1
        seq = self._dg_seq
        if from_field:
            self._dg_pending_key = key

        def _run(ctx) -> object:
            return validate_deepgram(key)

        job = Job(_run, self)
        self._jobs.append(job)
        job.finished.connect(
            lambda status: self._on_dg_validated(seq, status, from_field)
        )
        job.failed.connect(
            lambda err: self._on_dg_validated(
                seq, type("S", (), {"connected": False, "message": err})(), from_field
            )
        )
        job.start()

    def _on_dg_validated(self, seq: int, status, from_field: bool) -> None:
        if seq != self._dg_seq:
            return  # a newer validation superseded this one
        if status.connected:
            if from_field:
                try:
                    save_key("deepgram", self._dg_pending_key)
                except KeyringUnavailable:
                    self._dg_status.setText(
                        "Connected — but the key could not be saved "
                        "(secure storage unavailable)"
                    )
                    return
                # Never show the key again: clear the field, keep a placeholder.
                self._suppress_key_signals = True
                self._dg_key.clear()
                self._dg_key.setPlaceholderText("Key saved — type to replace")
                self._suppress_key_signals = False
            self._dg_status.setText("Connected")
        else:
            self._dg_status.setText(status.message)

    def _on_ds_key_edited(self, text: str) -> None:
        if self._suppress_key_signals:
            return
        if not text.strip():
            self._ds_status.setText(
                "Connected" if has_key("deepseek") else ""
            )
            return
        self._ds_status.setText("Checking…")
        self._ds_timer.start()  # restarts on every keystroke/paste

    def _on_ds_timer(self) -> None:
        key = self._ds_key.text().strip()
        if key:
            self._start_ds_validation(key, from_field=True)

    def _on_base_url_changed(self, text: str) -> None:
        self._settings.set("ai_base_url", text)
        stored = get_key("deepseek")
        if stored:
            self._ds_status.setText("Checking…")
            self._start_ds_validation(stored, from_field=False)

    def _start_ds_validation(self, key: str, *, from_field: bool) -> None:
        self._ds_seq += 1
        seq = self._ds_seq
        base_url = str(self._settings.get("ai_base_url"))
        if from_field:
            self._ds_pending_key = key

        def _run(ctx) -> object:
            return validate_deepseek(base_url, key)

        job = Job(_run, self)
        self._jobs.append(job)
        job.finished.connect(
            lambda status: self._on_ds_validated(seq, status, from_field)
        )
        job.failed.connect(
            lambda err: self._on_ds_validated(
                seq, type("S", (), {"connected": False, "message": err, "models": []})(),
                from_field,
            )
        )
        job.start()

    def _on_ds_validated(self, seq: int, status, from_field: bool) -> None:
        if seq != self._ds_seq:
            return  # a newer validation superseded this one
        if status.connected:
            if getattr(status, "models", None):
                self._fill_model_dropdown(status.models)
            if from_field:
                try:
                    save_key("deepseek", self._ds_pending_key)
                except KeyringUnavailable:
                    self._ds_status.setText(
                        "Connected — but the key could not be saved "
                        "(secure storage unavailable)"
                    )
                    return
                # Never show the key again: clear the field, keep a placeholder.
                self._suppress_key_signals = True
                self._ds_key.clear()
                self._ds_key.setPlaceholderText("Key saved — type to replace")
                self._suppress_key_signals = False
            self._ds_status.setText("Connected")
        else:
            self._ds_status.setText(status.message)

    def _fill_model_dropdown(self, models: list[str]) -> None:
        """Fill the Model ID dropdown from GET /models (editable fallback)."""
        current = self._ds_model.currentText().strip()
        saved = str(self._settings.get("ai_model") or "").strip()
        self._ds_model.blockSignals(True)
        self._ds_model.clear()
        self._ds_model.addItems(models)
        for preferred in (saved, current, "deepseek-flash"):
            if preferred and preferred in models:
                self._ds_model.setCurrentText(preferred)
                break
        else:
            if models:
                self._ds_model.setCurrentIndex(0)
        self._ds_model.blockSignals(False)
        if self._ds_model.currentText():
            self._settings.set("ai_model", self._ds_model.currentText())

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt naming
        for job in self._jobs:
            job.shutdown()
        super().closeEvent(event)

    @staticmethod
    def _show_info(message: str) -> None:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.information(None, t("settings.updates"), message)
