"""Animated intro splash + sidebar branding.

Splash (shown once at launch): the user's portrait assembles
shard-by-shard from below in blue crystal tones — the same motion
language as his portfolio website intro (darkness, pieces fly in and
lock together, title fades in). Pure Qt (QPropertyAnimation over a
QGraphicsScene); no video files, no extra dependencies, no launch
delay (the photo is one small JPEG). Click or any key dismisses it.

Sidebar branding: the same photo, small, in the Home sidebar with a
gentle pulsing blue glow so the branding stays visible while the app
runs.

The photo is swappable: replace
``clipforge/assets/branding/user-photo.png`` (or set Settings >
``branding_photo``) and the next launch uses the new one.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPauseAnimation,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QSequentialAnimationGroup,
    Qt,
    QTimer,
    QVariantAnimation,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QGraphicsEllipseItem,
    QGraphicsObject,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..core.branding import load_branding_photo

__all__ = ["SidebarBranding", "SplashScreen"]

#: Splash window size.
SPLASH_WIDTH = 640
SPLASH_HEIGHT = 420
#: Photo area inside the splash (16:9).
PHOTO_RECT = QRectF(80, 36, 480, 270)
#: Tile grid for the shard-assemble effect.
TILE_COLS = 6
TILE_ROWS = 4
#: Whole intro runs ~2.1 s, then fades out by itself.
AUTO_CLOSE_MS = 2100
FADE_MS = 300

#: Blue-crystal palette (matches the portrait + portfolio intro).
_CRYSTAL = QColor(86, 156, 255)
_CRYSTAL_DEEP = QColor(28, 68, 160)
_BG = QColor(8, 12, 26)


class _TileItem(QGraphicsObject):
    """One shard of the portrait (animatable pos + opacity)."""

    def __init__(self, pixmap: QPixmap, parent=None) -> None:
        super().__init__(parent)
        self._pixmap = pixmap
        self.setOpacity(0.0)

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt override
        return QRectF(0, 0, self._pixmap.width(), self._pixmap.height())

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802 - Qt override
        painter.drawPixmap(0, 0, self._pixmap)


class SplashScreen(QWidget):
    """Frameless launch splash with the shard-assemble intro."""

    def __init__(self, parent: QWidget | None = None, photo_path=None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.SplashScreen,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedSize(SPLASH_WIDTH, SPLASH_HEIGHT)
        self.setWindowTitle("ClipForge")
        self._closed = False
        self._photo_path = photo_path

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._view = QGraphicsView(self)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(self._view)

        self._scene = QGraphicsScene(0, 0, SPLASH_WIDTH, SPLASH_HEIGHT, self)
        self._scene.setBackgroundBrush(QBrush(_BG))
        self._view.setScene(self._scene)

        self._animations = QParallelAnimationGroup(self)
        self._build_scene()
        self._animations.start()
        QTimer.singleShot(AUTO_CLOSE_MS, self._fade_out)

    # -- animation helper ------------------------------------------------
    def _value_anim(self, duration: int, start, end, setter, *,
                    delay: int = 0,
                    easing: QEasingCurve.Type = QEasingCurve.Type.Linear):
        """Animate any value onto a setter (works for plain QGraphicsItems,
        which QPropertyAnimation cannot target — they are not QObjects)."""
        seq = QSequentialAnimationGroup(self)
        if delay:
            seq.addAnimation(QPauseAnimation(delay, self))
        anim = QVariantAnimation(self)
        anim.setDuration(duration)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setEasingCurve(easing)
        anim.valueChanged.connect(setter)
        seq.addAnimation(anim)
        self._animations.addAnimation(seq)
        return seq

    def _load_photo(self):
        """The portrait (explicit path first, else the bundled/env default)."""
        if self._photo_path:
            pixmap = QPixmap(str(self._photo_path))
            if not pixmap.isNull():
                return pixmap
        return load_branding_photo()

    # -- scene -----------------------------------------------------------
    def _build_scene(self) -> None:
        # Soft blue glow behind the photo.
        glow = QRadialGradient(
            SPLASH_WIDTH / 2, PHOTO_RECT.center().y(), 260
        )
        glow.setColorAt(0.0, QColor(64, 120, 255, 90))
        glow.setColorAt(1.0, QColor(64, 120, 255, 0))
        glow_item = QGraphicsEllipseItem(
            SPLASH_WIDTH / 2 - 260, PHOTO_RECT.center().y() - 260, 520, 520
        )
        glow_item.setBrush(QBrush(glow))
        glow_item.setPen(Qt.PenStyle.NoPen)
        glow_item.setOpacity(0.0)
        self._scene.addItem(glow_item)
        self._value_anim(900, 0.0, 1.0, glow_item.setOpacity)

        pixmap = self._load_photo()
        if pixmap is not None and not pixmap.isNull():
            self._add_photo_tiles(pixmap)
        else:
            # No photo: a blue gradient panel keeps the splash presentable.
            panel = QGraphicsRectItem(PHOTO_RECT)
            panel.setBrush(QBrush(_CRYSTAL_DEEP))
            panel.setPen(Qt.PenStyle.NoPen)
            self._scene.addItem(panel)

        # Title + tagline fade in after the shards land.
        title = QGraphicsTextItem("ClipForge")
        title.setFont(QFont("Segoe UI", 30, QFont.Weight.Bold))
        title.setDefaultTextColor(QColor(255, 255, 255))
        title.setPos(
            (SPLASH_WIDTH - title.boundingRect().width()) / 2,
            PHOTO_RECT.bottom() + 14,
        )
        title.setOpacity(0.0)
        self._scene.addItem(title)
        self._value_anim(500, 0.0, 1.0, title.setOpacity, delay=950)

        # Accent line grows from the center under the title.
        line = QGraphicsRectItem(-60, 0, 120, 3)
        line.setBrush(QBrush(_CRYSTAL))
        line.setPen(Qt.PenStyle.NoPen)
        line.setPos(SPLASH_WIDTH / 2, PHOTO_RECT.bottom() + 58)
        line.setTransformOriginPoint(0, 1.5)
        line.setScale(0.0)
        self._scene.addItem(line)
        self._value_anim(450, 0.0, 1.0, line.setScale, delay=1050,
                         easing=QEasingCurve.Type.OutCubic)

        tagline = QGraphicsTextItem("Turn long videos into viral clips")
        tagline.setFont(QFont("Segoe UI", 12))
        tagline.setDefaultTextColor(QColor(180, 195, 220))
        tagline.setPos(
            (SPLASH_WIDTH - tagline.boundingRect().width()) / 2,
            PHOTO_RECT.bottom() + 66,
        )
        tagline.setOpacity(0.0)
        self._scene.addItem(tagline)
        self._value_anim(500, 0.0, 1.0, tagline.setOpacity, delay=1150)

    def _add_photo_tiles(self, pixmap: QPixmap) -> None:
        """Split the photo into shards; each flies in from below, staggered
        bottom-row first (the portfolio intro assembles bottom-to-top)."""
        scaled = pixmap.scaled(
            int(PHOTO_RECT.width()), int(PHOTO_RECT.height()),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        tile_w = int(PHOTO_RECT.width()) // TILE_COLS
        tile_h = int(PHOTO_RECT.height()) // TILE_ROWS
        for row in range(TILE_ROWS):
            for col in range(TILE_COLS):
                x = col * tile_w
                y = row * tile_h
                w = tile_w if col < TILE_COLS - 1 else scaled.width() - x
                h = tile_h if row < TILE_ROWS - 1 else scaled.height() - y
                tile = _TileItem(scaled.copy(x, y, w, h))
                final = QPointF(PHOTO_RECT.x() + x, PHOTO_RECT.y() + y)
                # Deterministic sideways drift + a drop from below.
                drift = ((col * 37 + row * 17) % 60) - 30
                start = QPointF(final.x() + drift, final.y() + 130 + row * 24)
                tile.setPos(start)
                self._scene.addItem(tile)

                # Bottom rows land first (delay grows towards the top).
                delay = (TILE_ROWS - 1 - row) * 120 + col * 40
                seq = QSequentialAnimationGroup(self)
                seq.addAnimation(QPauseAnimation(delay, self))
                parallel = QParallelAnimationGroup(self)
                pos_anim = QVariantAnimation(self)
                pos_anim.setDuration(450)
                pos_anim.setStartValue(start)
                pos_anim.setEndValue(final)
                pos_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
                pos_anim.valueChanged.connect(tile.setPos)
                parallel.addAnimation(pos_anim)
                fade_anim = QVariantAnimation(self)
                fade_anim.setDuration(400)
                fade_anim.setStartValue(0.0)
                fade_anim.setEndValue(1.0)
                fade_anim.valueChanged.connect(tile.setOpacity)
                parallel.addAnimation(fade_anim)
                seq.addAnimation(parallel)
                self._animations.addAnimation(seq)

    # -- lifecycle -------------------------------------------------------
    def _fade_out(self) -> None:
        if self._closed:
            return
        self._closed = True
        fade = QPropertyAnimation(self, b"windowOpacity", self)
        fade.setDuration(FADE_MS)
        fade.setStartValue(1.0)
        fade.setEndValue(0.0)
        fade.finished.connect(self.close)
        fade.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def finish(self) -> None:
        """Close now (called when the main window appears)."""
        self._fade_out()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.close()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._closed = True
        super().closeEvent(event)


class SidebarBranding(QWidget):
    """Small portrait in the Home sidebar with a gentle pulsing glow."""

    def __init__(self, parent: QWidget | None = None, photo_path=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        pixmap = None
        if photo_path:
            pixmap = QPixmap(str(photo_path))
            if pixmap.isNull():
                pixmap = None
        if pixmap is None:
            pixmap = load_branding_photo()
        if pixmap is None or pixmap.isNull():
            self.setVisible(False)
            return
        photo = QLabel(self)
        photo.setPixmap(
            pixmap.scaled(
                126, 71,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        photo.setFixedSize(126, 71)
        photo.setStyleSheet("border-radius: 8px;")
        photo.setScaledContents(True)
        # Pulsing blue glow (slow, subtle — branding, not a distraction).
        glow = QGraphicsDropShadowEffect(self)
        glow.setColor(QColor(86, 156, 255, 140))
        glow.setBlurRadius(8)
        glow.setOffset(0, 0)
        photo.setGraphicsEffect(glow)
        self._glow_anim = QPropertyAnimation(glow, b"blurRadius", self)
        self._glow_anim.setDuration(2600)
        self._glow_anim.setStartValue(6)
        self._glow_anim.setEndValue(18)
        self._glow_anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._glow_anim.setLoopCount(-1)
        self._glow_anim.start()
        layout.addWidget(photo, alignment=Qt.AlignmentFlag.AlignCenter)

        caption = QLabel("ClipForge", self)
        caption.setProperty("muted", True)
        caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        caption.setStyleSheet("font-size: 11px; font-weight: 600;")
        layout.addWidget(caption)
