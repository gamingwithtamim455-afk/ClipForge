"""Timeline panel (bottom of the Viral Clips / Edit screens).

Matches the recording: "Timeline" header with zoom slider and + / delete
buttons, a time ruler, and the empty state "+ Add a clip or start viral
detection". Clip ranges are drawn as blocks on the track.

Manual ranges (build spec §1): the "+" button adds a clip range as an
inline row with Start / End fields (``core.timeparse`` accepts ``90``,
``1:30``, ``01:30``, ``1:02:03`` and decimals) plus draggable handles on
the timeline bar. Validation: end > start, inside the video length, no
overlaps, warn above 60 s, up to 20 ranges. When at least one manual
range exists, ``manual_mode_changed`` fires and Start clipping must use
exactly those ranges (AI discovery is skipped); the Viral Clips page
disables its Clip length / Clips controls in that case. Ranges can be
edited again in the Edit step (same widget, ``set_ranges``).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..core.ranges import MAX_RANGES, ClipRange, RangeValidation, validate_ranges
from ..core.timeparse import TimecodeError, format_timecode, parse_timecode
from ..i18n import t

#: Pixels around a range edge that count as a drag handle.
HANDLE_TOLERANCE_PX = 8
#: Minimum range length while dragging (seconds).
MIN_RANGE_S = 0.1


def _format_time(seconds: float) -> str:
    """Ruler label, e.g. 90 s -> "01:30.0" (as in the recording)."""
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:04.1f}"


class _Ruler(QWidget):
    """Time ruler + clip track area (with draggable range handles)."""

    def __init__(self, timeline: "TimelineWidget") -> None:
        super().__init__(timeline)
        self._timeline = timeline
        self.duration = 0.0
        self.pixels_per_second = 1.0
        self.setMinimumHeight(96)
        self._drag: tuple[int, str] | None = None  # (range index, "start"|"end")

    # -- geometry ------------------------------------------------------
    def _time_at(self, x: int) -> float:
        if self.pixels_per_second <= 0:
            return 0.0
        return max(0.0, min(self.duration, x / self.pixels_per_second))

    def _handle_at(self, x: int) -> tuple[int, str] | None:
        """Return (range index, edge) if x is near a manual range handle."""
        for i, rng in enumerate(self._timeline.manual_ranges()):
            for edge, edge_x in (
                ("start", rng.start * self.pixels_per_second),
                ("end", rng.end * self.pixels_per_second),
            ):
                if abs(x - edge_x) <= HANDLE_TOLERANCE_PX:
                    return i, edge
        return None

    def _block_at(self, x: int) -> int | None:
        for i, rng in enumerate(self._timeline.manual_ranges()):
            if rng.start * self.pixels_per_second <= x <= rng.end * self.pixels_per_second:
                return i
        return None

    # -- mouse ---------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        handle = self._handle_at(int(event.position().x()))
        if handle is not None:
            self._drag = handle
            self._timeline.select_range(handle[0])
            return
        block = self._block_at(int(event.position().x()))
        if block is not None:
            self._timeline.select_range(block)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._drag is None:
            return
        index, edge = self._drag
        ranges = self._timeline.manual_ranges()
        if index >= len(ranges):
            return
        new_time = self._timeline.snap_time(self._time_at(int(event.position().x())))
        rng = ranges[index]
        if edge == "start":
            rng.start = min(max(new_time, 0.0), rng.end - MIN_RANGE_S)
        else:
            rng.end = max(min(new_time, self.duration), rng.start + MIN_RANGE_S)
        self._timeline.refresh_ranges(emit=False)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._drag is not None:
            self._drag = None
            self._timeline.refresh_ranges(emit=True)

    # -- painting ------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setPen(QPen(self.palette().color(self.palette().ColorRole.WindowText)))
        # Ruler ticks: aim for a tick every ~120 px.
        if self.duration > 0 and self.pixels_per_second > 0:
            step = max(1.0, round(120 / self.pixels_per_second))
            for nice in (1, 5, 15, 30, 60, 300, 900, 1800):
                if nice >= step:
                    step = float(nice)
                    break
            muted = self.palette().color(self.palette().ColorRole.WindowText)
            muted.setAlpha(140)
            painter.setPen(QPen(muted))
            tick = 0.0
            while tick <= self.duration:
                x = int(tick * self.pixels_per_second)
                painter.drawLine(x, 0, x, 8)
                painter.drawText(x + 3, 20, _format_time(tick))
                tick += step
        highlight = self.palette().highlight().color()
        selected = self._timeline.selected_index
        manual = self._timeline.manual_ranges()
        if manual:
            for i, rng in enumerate(manual):
                x = int(rng.start * self.pixels_per_second)
                w = max(4, int((rng.end - rng.start) * self.pixels_per_second))
                color = highlight.lighter(120) if i == selected else highlight
                painter.fillRect(x, 28, w, 44, color)
                # Handles: darker edge bars.
                painter.fillRect(x, 28, 4, 44, highlight.darker(130))
                painter.fillRect(x + w - 4, 28, 4, 44, highlight.darker(130))
        else:
            for start, end in self._timeline.ai_clips:
                x = int(start * self.pixels_per_second)
                w = max(4, int((end - start) * self.pixels_per_second))
                painter.fillRect(x, 28, w, 44, highlight)
        painter.end()


class _RangeRow(QWidget):
    """One inline Start / End editor row for a manual range."""

    def __init__(
        self, index: int, rng: ClipRange, timeline: "TimelineWidget"
    ) -> None:
        super().__init__(timeline)
        self._index = index
        self._timeline = timeline
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        name = QLabel(f"Clip {index + 1}", self)
        name.setFixedWidth(48)
        layout.addWidget(name)

        self._start_edit = QLineEdit(format_timecode(rng.start), self)
        self._start_edit.setFixedWidth(84)
        self._start_edit.setPlaceholderText("Start")
        self._start_edit.editingFinished.connect(self._on_start_edited)
        layout.addWidget(self._start_edit)

        layout.addWidget(QLabel("–", self))

        self._end_edit = QLineEdit(format_timecode(rng.end), self)
        self._end_edit.setFixedWidth(84)
        self._end_edit.setPlaceholderText("End")
        self._end_edit.editingFinished.connect(self._on_end_edited)
        layout.addWidget(self._end_edit)

        self._duration_label = QLabel("", self)
        self._duration_label.setProperty("muted", True)
        layout.addWidget(self._duration_label)

        layout.addStretch(1)
        delete_btn = QPushButton("✕", self)
        delete_btn.setFlat(True)
        delete_btn.setFixedWidth(26)
        delete_btn.setToolTip("Delete this range")
        delete_btn.clicked.connect(lambda: timeline.delete_range(index))
        layout.addWidget(delete_btn)
        self.refresh(rng)

    def refresh(self, rng: ClipRange) -> None:
        """Update the fields from the model without firing edit signals."""
        for edit, value in ((self._start_edit, rng.start), (self._end_edit, rng.end)):
            edit.blockSignals(True)
            edit.setText(format_timecode(value))
            edit.blockSignals(False)
        self._duration_label.setText(f"{rng.duration:.1f}s")

    def mark_invalid(self, invalid: bool) -> None:
        style = "border: 1px solid #c0392b;" if invalid else ""
        self._start_edit.setStyleSheet(style)
        self._end_edit.setStyleSheet(style)

    def _apply(self, which: str, edit: QLineEdit) -> None:
        try:
            value = parse_timecode(edit.text())
        except TimecodeError as exc:
            self._timeline.set_error(str(exc))
            self.refresh(self._timeline.manual_ranges()[self._index])
            return
        ranges = self._timeline.manual_ranges()
        if which == "start":
            ranges[self._index].start = value
        else:
            ranges[self._index].end = value
        self._timeline.refresh_ranges(emit=True)

    def _on_start_edited(self) -> None:
        self._apply("start", self._start_edit)

    def _on_end_edited(self) -> None:
        self._apply("end", self._end_edit)


class TimelineWidget(QFrame):
    """Timeline panel with manual clip ranges."""

    add_clip_requested = Signal()
    delete_clip_requested = Signal()
    ranges_changed = Signal(list)  # current manual ranges
    manual_mode_changed = Signal(bool)  # True when >= 1 manual range exists

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("panel", True)
        self._duration = 0.0
        self._fps = 30.0
        self._ranges: list[ClipRange] = []
        self._ai_clips: list[tuple[float, float]] = []
        self._selected: int | None = None
        self._manual_mode = False
        self._validation = RangeValidation()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel(t("timeline.title"), self)
        title.setProperty("title", True)
        header.addWidget(title)
        header.addStretch(1)
        zoom_out = QPushButton("−", self)
        zoom_out.setFlat(True)
        zoom_out.setFixedWidth(26)
        zoom_out.setStyleSheet("font-size: 14px; font-weight: 600;")
        zoom_out.clicked.connect(self._zoom_out)
        header.addWidget(zoom_out)
        self._zoom = QSlider(Qt.Orientation.Horizontal, self)
        self._zoom.setRange(1, 100)
        self._zoom.setValue(20)
        self._zoom.setFixedWidth(90)
        self._zoom.valueChanged.connect(self._apply_zoom)
        header.addWidget(self._zoom)
        zoom_in = QPushButton("+", self)
        zoom_in.setFlat(True)
        zoom_in.setFixedWidth(26)
        zoom_in.setStyleSheet("font-size: 14px; font-weight: 600;")
        zoom_in.clicked.connect(self._zoom_in)
        header.addWidget(zoom_in)
        self._add_btn = QPushButton("+", self)
        self._add_btn.setFlat(True)
        self._add_btn.setFixedWidth(26)
        self._add_btn.setStyleSheet("font-size: 14px; font-weight: 600;")
        self._add_btn.setToolTip("Add a clip range")
        self._add_btn.clicked.connect(self._on_add_clicked)
        header.addWidget(self._add_btn)
        delete_btn = QPushButton("✕", self)
        delete_btn.setFlat(True)
        delete_btn.setFixedWidth(26)
        delete_btn.setStyleSheet("font-size: 13px; font-weight: 600;")
        delete_btn.setToolTip("Delete selected clip")
        delete_btn.clicked.connect(self._on_delete_clicked)
        header.addWidget(delete_btn)
        layout.addLayout(header)

        self._ruler = _Ruler(self)
        layout.addWidget(self._ruler)

        self._rows_container = QVBoxLayout()
        self._rows_container.setContentsMargins(0, 0, 0, 0)
        self._rows_container.setSpacing(4)
        layout.addLayout(self._rows_container)
        self._rows: list[_RangeRow] = []

        self._empty = QLabel(f"+ {t('timeline.empty')}", self)
        self._empty.setProperty("muted", True)
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty)

        self._error_label = QLabel("", self)
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #e74c3c;")
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)
        self._warning_label = QLabel("", self)
        self._warning_label.setWordWrap(True)
        self._warning_label.setProperty("muted", True)
        self._warning_label.setVisible(False)
        layout.addWidget(self._warning_label)

    # -- API -------------------------------------------------------------
    @property
    def duration(self) -> float:
        return self._duration

    @property
    def selected_index(self) -> int | None:
        return self._selected

    @property
    def ai_clips(self) -> list[tuple[float, float]]:
        return list(self._ai_clips)

    def set_duration(self, seconds: float, fps: float = 30.0) -> None:
        """Set the source video length (seconds); redraws the ruler."""
        self._duration = max(0.0, seconds)
        self._fps = fps if fps > 0 else 30.0
        self._apply_zoom()
        self.refresh_ranges(emit=False)

    def set_clips(self, clips: list[tuple[float, float]]) -> None:
        """Draw AI-discovered clip ranges (start, end) in seconds."""
        self._ai_clips = list(clips)
        self._empty.setVisible(not clips and not self._ranges)
        self._ruler.update()

    def manual_ranges(self) -> list[ClipRange]:
        """Current manual ranges (the model objects — edit in place)."""
        return self._ranges

    def has_manual_ranges(self) -> bool:
        return len(self._ranges) > 0

    @property
    def validation(self) -> RangeValidation:
        return self._validation

    def set_ranges(self, ranges: list[ClipRange]) -> None:
        """Replace the manual ranges (used by the Edit step)."""
        self._ranges = [
            ClipRange(start=r.start, end=r.end, source=r.source, label=r.label)
            for r in ranges
        ]
        self._selected = None
        self._rebuild_rows()
        self.refresh_ranges(emit=True)

    def clear_ranges(self) -> None:
        self.set_ranges([])

    def snap_time(self, seconds: float) -> float:
        """Snap a time to the frame grid (1/fps), 2-decimal precision."""
        step = 1.0 / self._fps
        return round(round(seconds / step) * step, 2)

    # -- range editing ---------------------------------------------------
    def add_range(self, start: float | None = None, end: float | None = None) -> bool:
        """Add a manual range; returns False when the limit is reached."""
        if len(self._ranges) >= MAX_RANGES:
            self.set_error(f"Up to {MAX_RANGES} ranges are allowed.")
            return False
        if self._duration <= 0:
            self.set_error("Load a video first — the timeline has no length yet.")
            return False
        if start is None or end is None:
            last_end = max((r.end for r in self._ranges), default=0.0)
            start = last_end
            end = min(self._duration, start + 30.0)
            if end - start < 1.0:  # no room at the end: start from 0
                start, end = 0.0, min(30.0, self._duration)
        self._ranges.append(ClipRange(start=start, end=end, label=f"Clip {len(self._ranges) + 1}"))
        self._selected = len(self._ranges) - 1
        self._rebuild_rows()
        self.refresh_ranges(emit=True)
        return True

    def delete_range(self, index: int) -> None:
        if 0 <= index < len(self._ranges):
            del self._ranges[index]
            self._selected = None
            self._rebuild_rows()
            self.refresh_ranges(emit=True)

    def select_range(self, index: int | None) -> None:
        self._selected = index if index is not None and 0 <= index < len(self._ranges) else None
        self._ruler.update()

    def refresh_ranges(self, *, emit: bool) -> None:
        """Re-validate, redraw the ruler/rows, and notify listeners."""
        self._validation = validate_ranges(self._ranges, self._duration)
        self._error_label.setVisible(bool(self._validation.errors))
        self._error_label.setText("\n".join(self._validation.errors))
        self._warning_label.setVisible(bool(self._validation.warnings))
        self._warning_label.setText("\n".join(self._validation.warnings))
        # Mark invalid rows (overlap / out of bounds).
        invalid_indices: set[int] = set()
        for message in self._validation.errors:
            for i, rng in enumerate(self._ranges):
                if (rng.label or f"Clip {i + 1}") in message:
                    invalid_indices.add(i)
        for i, row in enumerate(self._rows):
            row.mark_invalid(i in invalid_indices)
            if i < len(self._ranges):
                row.refresh(self._ranges[i])
        self._empty.setVisible(not self._ranges and not self._ai_clips)
        self._ruler.update()
        manual = len(self._ranges) > 0
        if manual != self._manual_mode:
            self._manual_mode = manual
            self.manual_mode_changed.emit(manual)
        if emit:
            self.ranges_changed.emit(list(self._ranges))

    def set_error(self, message: str) -> None:
        self._error_label.setVisible(bool(message))
        self._error_label.setText(message)

    # -- internal --------------------------------------------------------
    def _rebuild_rows(self) -> None:
        for row in self._rows:
            self._rows_container.removeWidget(row)
            row.deleteLater()
        self._rows = [
            _RangeRow(i, rng, self) for i, rng in enumerate(self._ranges)
        ]
        for row in self._rows:
            self._rows_container.addWidget(row)

    def _on_add_clicked(self) -> None:
        self.add_clip_requested.emit()
        self.add_range()

    def _on_delete_clicked(self) -> None:
        self.delete_clip_requested.emit()
        if self._selected is not None:
            self.delete_range(self._selected)
        elif self._ranges:
            self.delete_range(len(self._ranges) - 1)

    # -- zoom ------------------------------------------------------------
    def _apply_zoom(self) -> None:
        if self._duration > 0:
            base = self._ruler.width() / self._duration if self._ruler.width() else 1.0
            self._ruler.pixels_per_second = base * (self._zoom.value() / 20.0)
        self._ruler.update()

    def _zoom_in(self) -> None:
        self._zoom.setValue(min(100, self._zoom.value() + 10))

    def _zoom_out(self) -> None:
        self._zoom.setValue(max(1, self._zoom.value() - 10))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._apply_zoom()
