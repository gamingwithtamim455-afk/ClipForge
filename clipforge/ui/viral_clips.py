"""Viral Clips screen (wizard step 2).

Matches the reference recording:
- Left panel: Model (Enhanced), Language (placeholder "Select language...",
  "Auto (Recommended)" first), Captions (Burn Captions / Cutout Only),
  Clip length (sec, default 30), Clips (Auto / number), info paragraph.
- Center: Player panel (9:16 preview, "Face tracking at render" badge,
  transport row) with the Timeline panel below it.
- Right panel "Clip & Framing": Aspect ratio 9:16, Subject Face Tracking,
  Quality 1080p.

"Start clipping" is enabled only once a language is chosen; the page emits
``readiness_changed`` so the wizard footer can enable/disable the button.
Transcription / discovery itself is a later milestone.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..i18n import t
from .timeline import TimelineWidget
from .widgets import Badge, VideoPlayerWidget

#: (label, app language code) — Auto first, as in the spec.
LANGUAGES: list[tuple[str, str]] = [
    ("Auto (Recommended)", "auto"),
    ("Bengali", "bn"),
    ("English", "en"),
    ("Arabic", "ar"),
    ("Urdu", "ur"),
    ("Hindi", "hi"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Turkish", "tr"),
    ("Indonesian", "id"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
]

INFO_TEXT = (
    "Transcription automatically finds viral clips. Review the results "
    "unless automatic approval is enabled. Caption correction and "
    "transliteration run only for approved clips when Burn Captions is "
    "selected. Word emphasis is found after the caption text is ready and "
    "appears in presets that support it."
)


class ViralClipsPage(QWidget):
    """The Viral Clips step."""

    readiness_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ready = False

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        root.addWidget(self._build_left_panel(), 0)

        center = QVBoxLayout()
        center.setContentsMargins(0, 0, 0, 0)
        center.setSpacing(10)
        center.addWidget(self._build_player_panel(), 1)
        self.timeline = TimelineWidget(self)
        self.timeline.manual_mode_changed.connect(self._on_manual_mode_changed)
        center.addWidget(self.timeline, 0)
        root.addLayout(center, 1)

        root.addWidget(self._build_right_panel(), 0)

    # -- panels ----------------------------------------------------------
    def _row(self, label_text: str, control: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel(label_text, self)
        label.setProperty("muted", True)
        row.addWidget(label, 1)
        row.addWidget(control, 2)  # controls get the wider share, as in the recording
        return row

    def _build_left_panel(self) -> QFrame:
        panel = QFrame(self)
        panel.setProperty("panel", True)
        panel.setFixedWidth(250)  # wide enough for "Select language..."
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel(t("viral.clips"), self)
        title.setProperty("title", True)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(Badge("Deepgram Cloud", parent=self))
        layout.addLayout(header)

        self._model = QComboBox(self)
        self._model.addItems([t("viral.model_enhanced")])
        layout.addLayout(self._row(t("viral.model"), self._model))

        self._language = QComboBox(self)
        for label, _code in LANGUAGES:
            self._language.addItem(label)
        self._language.setCurrentIndex(-1)
        self._language.setPlaceholderText("Select language...")
        self._language.currentIndexChanged.connect(self._on_language_changed)
        layout.addLayout(self._row(t("viral.language"), self._language))

        self._captions = QComboBox(self)
        self._captions.addItems([t("viral.captions_burn"), t("viral.captions_cutout")])
        layout.addLayout(self._row(t("viral.captions"), self._captions))

        self._clip_length = QSpinBox(self)
        self._clip_length.setRange(5, 180)
        self._clip_length.setValue(30)
        self._clip_length.setSuffix(" sec")
        layout.addLayout(self._row(t("viral.clip_length"), self._clip_length))

        self._clips = QComboBox(self)
        self._clips.setEditable(True)
        self._clips.addItem(t("viral.clips_auto"))
        self._clips.addItems([str(n) for n in range(1, 21)])
        layout.addLayout(self._row(t("viral.clips"), self._clips))

        info = QLabel(INFO_TEXT, self)
        info.setProperty("muted", True)
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 12px;")
        layout.addWidget(info)
        layout.addStretch(1)

        self._status = QLabel("", self)
        self._status.setProperty("muted", True)
        layout.addWidget(self._status)
        return panel

    def _build_player_panel(self) -> QFrame:
        panel = QFrame(self)
        panel.setProperty("panel", True)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel(t("player.title"), self)
        title.setProperty("title", True)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(Badge(t("player.face_tracking_badge"), parent=self))
        layout.addLayout(header)

        # Real player (QMediaPlayer): the source video actually plays
        # here, with working play/pause + seek. The previous panel was a
        # static pixmap whose transport buttons were never connected.
        self._player_widget = VideoPlayerWidget(self)
        layout.addWidget(self._player_widget, 1)
        return panel

    def _build_right_panel(self) -> QFrame:
        panel = QFrame(self)
        panel.setProperty("panel", True)
        panel.setFixedWidth(205)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        title = QLabel(t("framing.title"), self)
        title.setProperty("title", True)
        layout.addWidget(title)

        self._aspect = QComboBox(self)
        self._aspect.addItems(["9:16", "1:1", "16:9"])
        layout.addLayout(self._row(t("framing.aspect_ratio"), self._aspect))

        self._subject = QComboBox(self)
        self._subject.addItems([t("framing.subject_face")])
        layout.addLayout(self._row(t("framing.subject"), self._subject))

        self._quality = QComboBox(self)
        self._quality.addItems(["1080p", "1440p", "4K"])
        layout.addLayout(self._row(t("framing.quality"), self._quality))

        layout.addStretch(1)
        return panel

    # -- state -----------------------------------------------------------
    def _on_language_changed(self, index: int) -> None:
        ready = index >= 0
        if ready != self._ready:
            self._ready = ready
            self.readiness_changed.emit(ready)

    @property
    def is_ready(self) -> bool:
        """True once a language has been chosen."""
        return self._ready

    @property
    def language_code(self) -> str:
        """App language code for the selected language ("" if none)."""
        index = self._language.currentIndex()
        return LANGUAGES[index][1] if index >= 0 else ""

    @property
    def clip_length(self) -> int:
        return self._clip_length.value()

    @property
    def clips_requested(self) -> str:
        return self._clips.currentText()

    @property
    def clips_requested_count(self) -> int | None:
        """Requested clip count, or None when Clips is on Auto."""
        text = self._clips.currentText().strip()
        return int(text) if text.isdigit() else None

    @property
    def captions_mode(self) -> str:
        """``"burn"`` for Burn Captions, ``"none"`` for Cutout Only."""
        return "burn" if self._captions.currentIndex() == 0 else "none"

    @property
    def quality(self) -> str:
        return self._quality.currentText()

    @property
    def aspect(self) -> tuple[int, int]:
        """Selected aspect ratio as (width, height) parts, e.g. (9, 16)."""
        text = self._aspect.currentText()
        try:
            w, h = text.split(":")
            return int(w), int(h)
        except ValueError:
            return (9, 16)

    def set_thumbnail(self, path: str) -> None:
        """Show a real video frame as the player's poster image."""
        self._player_widget.set_poster(path)

    def set_source(self, path: str) -> None:
        """Load the source video into the player."""
        self._player_widget.set_source(path)

    @property
    def manual_ranges(self) -> list:
        """Manual timeline ranges (empty when AI discovery is in use)."""
        return self.timeline.manual_ranges()

    @property
    def has_manual_ranges(self) -> bool:
        return self.timeline.has_manual_ranges()

    def _on_manual_mode_changed(self, manual: bool) -> None:
        """Manual ranges take over: Clip length / Clips are disabled and
        Start clipping uses exactly those ranges (AI discovery skipped)."""
        self._clip_length.setEnabled(not manual)
        self._clips.setEnabled(not manual)
        if manual:
            count = len(self.timeline.manual_ranges())
            self.set_status(
                f"Manual ranges: {count} — AI discovery skipped, "
                "Start clipping uses exactly these ranges."
            )
        else:
            self.set_status("")

    def set_video(self, duration: float, time_label: str) -> None:
        """Attach a loaded video (duration in seconds)."""
        self.timeline.set_duration(duration)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def reset(self) -> None:
        """Clear all per-project state: no language, no video, no ranges."""
        self._language.setCurrentIndex(-1)  # -> _ready False, readiness_changed
        self.timeline.clear_ranges()
        self.timeline.set_clips([])
        self.timeline.set_duration(0.0)
        self._player_widget.clear()
        self._clip_length.setEnabled(True)
        self._clips.setEnabled(True)
        self.set_status("")
