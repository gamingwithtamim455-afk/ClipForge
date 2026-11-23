"""Shared widgets: segmented control, toggle switch, project card, thumbnails,
and the real video player (QMediaPlayer-backed)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)


class SegmentedControl(QWidget):
    """A row of mutually exclusive options, like the recording's
    System/Light/Dark and Auto/GPU/CPU controls."""

    valueChanged = Signal(str)

    def __init__(self, options: list[str], current: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: dict[str, QPushButton] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for option in options:
            btn = QPushButton(option, self)
            btn.setCheckable(True)
            btn.setFlat(True)
            btn.clicked.connect(lambda _checked, o=option: self.set_value(o))
            layout.addWidget(btn)
            self._buttons[option] = btn
        self._value = current or (options[0] if options else "")
        self._refresh()

    def value(self) -> str:
        return self._value

    def set_value(self, value: str) -> None:
        if value == self._value or value not in self._buttons:
            return
        self._value = value
        self._refresh()
        self.valueChanged.emit(value)

    def _refresh(self) -> None:
        for option, btn in self._buttons.items():
            selected = option == self._value
            btn.setChecked(selected)
            # Selected segment: lighter pill; others plain text.
            btn.setStyleSheet(
                "QPushButton { border-radius: 5px; padding: 4px 10px; }"
                + ("QPushButton:checked { background: palette(button); font-weight: 600; }"
                   if selected else "")
            )


class ToggleSwitch(QCheckBox):
    """A checkbox rendered as an on/off switch."""

    def __init__(self, checked: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setChecked(checked)
        self.setStyleSheet(
            "QCheckBox::indicator { width: 34px; height: 20px; border-radius: 10px; "
            "background: #3a3a3a; border: 1px solid #4a4a4a; }"
            "QCheckBox::indicator:checked { background: #2f7de1; border-color: #2f7de1; }"
            "QCheckBox::indicator:unchecked:hover { background: #444444; }"
        )


def thumbnail_pixmap(name: str, width: int = 320, height: int = 180) -> QPixmap:
    """Deterministic gradient thumbnail for a project (placeholder art)."""
    digest = hashlib.md5(name.encode("utf-8")).digest()
    hue1 = (digest[0] / 255.0) * 360
    hue2 = (hue1 + 40 + digest[1] % 60) % 360
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.black)
    painter = QPainter(pixmap)
    gradient = QLinearGradient(0, 0, width, height)
    gradient.setColorAt(0, QColor.fromHsv(int(hue1), 120, 70))
    gradient.setColorAt(1, QColor.fromHsv(int(hue2), 150, 45))
    painter.fillRect(0, 0, width, height, gradient)
    # Soft light blobs for texture.
    painter.setPen(Qt.PenStyle.NoPen)
    for i in range(3):
        x = (digest[2 + i * 2] / 255.0) * width
        y = (digest[3 + i * 2] / 255.0) * height
        r = 40 + digest[8 + i] % 60
        color = QColor(255, 255, 255, 14)
        painter.setBrush(color)
        painter.drawEllipse(int(x - r), int(y - r), r * 2, r * 2)
    # Center play glyph.
    cx, cy = width // 2, height // 2
    painter.setBrush(QColor(255, 255, 255, 60))
    painter.drawEllipse(cx - 22, cy - 22, 44, 44)
    painter.setBrush(QColor(255, 255, 255, 220))
    from PySide6.QtCore import QPoint

    painter.drawPolygon([
        QPoint(cx - 7, cy - 11), QPoint(cx - 7, cy + 11), QPoint(cx + 11, cy),
    ])
    painter.end()
    return pixmap


def _format_ms(ms: int) -> str:
    """Milliseconds -> "m:ss" for the player time label."""
    total = max(0, int(ms) // 1000)
    return f"{total // 60}:{total % 60:02d}"


class VideoPlayerWidget(QWidget):
    """A real video player: QMediaPlayer + QVideoWidget + transport row.

    This replaces the original static-pixmap "player" mock, whose play
    button and seek slider were never connected to anything (the root
    cause of "video won't play / buttons don't work"). The video area is
    a fixed-size 9:16-style frame; a poster pixmap (video thumbnail) is
    shown until a source is loaded.

    ``set_play_range(start, end)`` optionally confines playback to a
    window (used by the Edit step to preview one clip inside the source).
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        video_width: int = 270,
        video_height: int = 480,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Video area: poster (placeholder art / thumbnail) under the
        # real video surface; the surface takes over once a source loads.
        self._stack = QStackedLayout()
        self._poster = QLabel(self)
        self._poster.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._poster.setFixedSize(video_width, video_height)
        self._poster.setPixmap(thumbnail_pixmap("player", video_width, video_height))
        self._stack.addWidget(self._poster)
        self._video_widget = QVideoWidget(self)
        self._video_widget.setFixedSize(video_width, video_height)
        self._video_widget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self._stack.addWidget(self._video_widget)
        self._stack.setCurrentWidget(self._poster)
        video_row = QHBoxLayout()
        video_row.addStretch(1)
        video_row.addLayout(self._stack)
        video_row.addStretch(1)
        layout.addLayout(video_row)

        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video_widget)

        transport = QHBoxLayout()
        self._play_btn = QPushButton("▶", self)
        self._play_btn.setFlat(True)
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self.toggle)
        transport.addWidget(self._play_btn)
        self._seek = QSlider(Qt.Orientation.Horizontal, self)
        self._seek.setEnabled(False)
        self._seek.setRange(0, 0)
        transport.addWidget(self._seek, 1)
        self._time_label = QLabel("0:00 / 0:00", self)
        self._time_label.setProperty("muted", True)
        transport.addWidget(self._time_label)
        layout.addLayout(transport)

        self._range: tuple[float, float] | None = None  # seconds
        self._seeking = False
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state)
        self._seek.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
        self._seek.sliderReleased.connect(self._on_seek_released)
        self._seek.sliderMoved.connect(self._on_seek_moved)

    # -- source ----------------------------------------------------------
    def set_source(self, path: str) -> None:
        """Load a video file and show the real video surface."""
        self._range = None
        self._player.setSource(QUrl.fromLocalFile(path))
        self._stack.setCurrentWidget(self._video_widget)
        self._play_btn.setEnabled(True)
        self._seek.setEnabled(True)

    def set_poster(self, path: str) -> None:
        """Show a still frame as the poster (before/without playback)."""
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self._poster.setPixmap(
                pixmap.scaled(
                    self._poster.width(), self._poster.height(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def clear(self) -> None:
        """Unload the media and return to the poster."""
        self._player.stop()
        self._player.setSource(QUrl())
        self._range = None
        self._stack.setCurrentWidget(self._poster)
        self._play_btn.setEnabled(False)
        self._seek.setEnabled(False)
        self._seek.setRange(0, 0)
        self._time_label.setText("0:00 / 0:00")

    # -- transport ---------------------------------------------------------
    def play(self) -> None:
        self._player.play()

    def pause(self) -> None:
        self._player.pause()

    def toggle(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def seek_seconds(self, seconds: float) -> None:
        self._player.setPosition(int(max(0.0, seconds) * 1000))

    def set_play_range(self, start: float, end: float) -> None:
        """Confine playback to ``[start, end]`` seconds (clip preview)."""
        self._range = (start, end)
        self.seek_seconds(start)

    @property
    def is_playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    @property
    def duration_seconds(self) -> float:
        return self._player.duration() / 1000.0

    @property
    def position_seconds(self) -> float:
        return self._player.position() / 1000.0

    # -- player signal handlers --------------------------------------------
    def _on_duration(self, duration_ms: int) -> None:
        self._seek.setRange(0, max(0, duration_ms))
        self._update_time_label()

    def _on_position(self, position_ms: int) -> None:
        if self._range is not None:
            _start, end = self._range
            if position_ms >= int(end * 1000):
                self._player.pause()
                self._player.setPosition(int(end * 1000))
        if not self._seeking:
            self._seek.setValue(position_ms)
        self._update_time_label()

    def _on_state(self, state) -> None:  # noqa: ANN001 - Qt enum
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_btn.setText("⏸" if playing else "▶")

    def _on_seek_moved(self, value: int) -> None:
        self._time_label.setText(
            f"{_format_ms(value)} / {_format_ms(self._player.duration())}"
        )

    def _on_seek_released(self) -> None:
        self._seeking = False
        self._player.setPosition(self._seek.value())

    def _update_time_label(self) -> None:
        self._time_label.setText(
            f"{_format_ms(self._player.position())} / "
            f"{_format_ms(self._player.duration())}"
        )


class Badge(QLabel):
    """Small pill badge used on thumbnails and panels."""

    def __init__(self, text: str, *, accent: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        bg = "#2f7de1" if accent else "rgba(0, 0, 0, 184)"  # Qt QSS alpha is 0-255
        self.setStyleSheet(
            f"QLabel {{ background: {bg}; color: white; border-radius: 4px; "
            "padding: 1px 5px; font-size: 11px; font-weight: 600; }"
        )


class ProjectCard(QFrame):
    """One project card: 16:9 thumbnail with badges, name below.

    Matches the recording: "9:16" badge top-left, optional blue "Cutout"
    badge, duration bottom-right, name centered below the thumbnail.
    """

    clicked = Signal(str)  # project id

    def __init__(self, project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Thumbnail with overlaid badges: a real video frame when the
        # project has one, otherwise the generated placeholder art.
        self._thumb = QLabel(self)
        thumb_path = getattr(project, "thumbnail", "") or ""
        pixmap = None
        if thumb_path and Path(thumb_path).exists():
            pixmap = QPixmap(thumb_path)
        self._thumb.setPixmap(pixmap if pixmap and not pixmap.isNull() else thumbnail_pixmap(project.name))
        self._thumb.setScaledContents(True)
        self._thumb.setStyleSheet("border-radius: 8px;")
        self._thumb.setFixedHeight(132)
        # Ignore the pixmap's 320px sizeHint so 5 columns fit the window.
        self._thumb.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        layout.addWidget(self._thumb)

        badge_916 = Badge(project.badge, parent=self._thumb)
        badge_916.move(6, 6)
        badge_916.show()
        if project.cutout:
            cutout = Badge("Cutout", accent=True, parent=self._thumb)
            cutout.move(6 + badge_916.sizeHint().width() + 6, 6)
            cutout.show()
        if project.duration:
            duration = Badge(project.duration, parent=self._thumb)
            duration.move(
                self._thumb.width() - duration.sizeHint().width() - 6,
                self._thumb.height() - duration.sizeHint().height() - 6,
            )
            duration.show()
        self._duration_badge = duration if project.duration else None

        name = QLabel(project.name, self)
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name.setProperty("muted", True)
        # Ignored horizontal policy: a long project name must not stretch
        # its grid column (that is what squeezed neighbouring cards into
        # narrow strips and pushed the grid into a horizontal scrollbar).
        name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        name.setToolTip(project.name)
        layout.addWidget(name)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        # Keep the duration badge pinned to the thumbnail's bottom-right.
        if self._duration_badge is not None:
            badge = self._duration_badge
            badge.move(
                self._thumb.width() - badge.sizeHint().width() - 6,
                self._thumb.height() - badge.sizeHint().height() - 6,
            )

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.clicked.emit(self.project.id)
        super().mousePressEvent(event)
