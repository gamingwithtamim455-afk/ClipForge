"""Edit step (wizard step 3) — basic editing.

Left: clip list (check = keep in render, uncheck = drop). Center: 9:16
preview. Right: the selected clip's editor — trim start/end, split,
move up/down, delete, volume (mute + level), captions on/off, caption
line editor (edit text, delete a word or a whole line, remove all),
and add a custom text line at a chosen time. Bottom: timeline showing
the updated ranges.

Every edit flows into the render jobs via :meth:`EditPage.kept_clips`
and :meth:`EditPage.edited_overrides` (read by ``app.py``), so what the
user edits here is exactly what gets rendered.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..core.caption_edit import line_spans, line_texts
from ..core.captions import PRESETS, CaptionOverlay
from ..core.edit_ops import (
    EditableClip,
    add_overlay,
    clip_words_relative,
    delete_clip,
    delete_clip_line,
    delete_clip_word,
    edit_clip_line,
    move_clip,
    split_clip,
    trim_clip,
)
from ..core.ranges import ClipRange
from ..core.timeparse import format_timecode, parse_timecode
from ..core.transcription import Word
from ..i18n import t
from .timeline import TimelineWidget
from .widgets import VideoPlayerWidget


class EditPage(QWidget):
    """The Edit step.

    Clips are listed (checkable — uncheck to drop a clip from rendering),
    ranges are editable again in the timeline below, and face-QA failures
    are flagged on the clip ("review framing"). Selecting a clip opens
    its editor on the right.
    """

    clips_changed = Signal()  # keep/drop toggled or ranges edited

    @property
    def caption_preset(self) -> str:
        """Selected caption preset name ("Hormozi" if unset)."""
        data = self._preset.currentData()
        return str(data) if data else "Hormozi"

    def set_caption_preset(self, name: str) -> None:
        """Select a preset by name (used to apply the Settings default)."""
        index = self._preset.findData(name)
        if index >= 0:
            self._preset.setCurrentIndex(index)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._video_source: str = ""
        self._clips: list[EditableClip] = []
        self._video_duration: float = 0.0
        self._selected: int | None = None
        self._syncing_timeline = False
        self._last_volume: float = 1.0  # restored when un-muting
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)

        # Left: clip list.
        left = QFrame(self)
        left.setProperty("panel", True)
        left.setFixedWidth(215)
        left_layout = QVBoxLayout(left)
        title = QLabel("Clips", self)
        title.setProperty("title", True)
        left_layout.addWidget(title)
        self._clip_list = QListWidget(self)
        self._clip_list.addItem("No clips yet — run Start clipping first.")
        self._clip_list.setEnabled(False)
        self._clip_list.itemChanged.connect(self._on_clip_toggled)
        self._clip_list.currentRowChanged.connect(self._on_clip_selected)
        left_layout.addWidget(self._clip_list)
        body.addWidget(left, 0)

        # Center: preview — the real player over the source video;
        # selecting a clip seeks to its range and plays just that range.
        center = QFrame(self)
        center.setProperty("panel", True)
        center_layout = QVBoxLayout(center)
        preview_title = QLabel(t("player.title"), self)
        preview_title.setProperty("title", True)
        center_layout.addWidget(preview_title)
        self._player_widget = VideoPlayerWidget(self)
        center_layout.addWidget(self._player_widget, 1)
        body.addWidget(center, 1)

        # Right: caption preset + clip editor (scrollable).
        right = QFrame(self)
        right.setProperty("panel", True)
        right.setFixedWidth(280)
        right_layout = QVBoxLayout(right)
        caption_title = QLabel("Captions", self)
        caption_title.setProperty("title", True)
        right_layout.addWidget(caption_title)
        self._preset = QComboBox(self)
        for name, preset in PRESETS.items():
            self._preset.addItem(preset.display_name or name, name)
        self._preset.setEnabled(False)
        right_layout.addWidget(QLabel("Preset", self))
        right_layout.addWidget(self._preset)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._editor = QWidget(self)
        self._editor_layout = QVBoxLayout(self._editor)
        self._editor_layout.setContentsMargins(0, 4, 0, 4)
        self._build_editor()
        scroll.setWidget(self._editor)
        right_layout.addWidget(scroll, 1)
        body.addWidget(right, 0)

        root.addLayout(body, 1)
        self.timeline = TimelineWidget(self)
        self.timeline.ranges_changed.connect(self._on_ranges_changed)
        root.addWidget(self.timeline, 0)

    # -- editor controls -------------------------------------------------

    def _build_editor(self) -> None:
        layout = self._editor_layout
        self._editor_title = QLabel("Select a clip to edit", self)
        self._editor_title.setProperty("title", True)
        self._editor_title.setWordWrap(True)
        layout.addWidget(self._editor_title)

        # Trim: start/end in video time, same validation as the timeline.
        layout.addWidget(QLabel("Trim (video time)", self))
        trim_row = QHBoxLayout()
        self._trim_start = QLineEdit(self)
        self._trim_start.setPlaceholderText("Start (0:00)")
        self._trim_end = QLineEdit(self)
        self._trim_end.setPlaceholderText("End (0:40)")
        trim_row.addWidget(self._trim_start)
        trim_row.addWidget(self._trim_end)
        layout.addLayout(trim_row)
        self._trim_btn = QPushButton("Apply trim", self)
        self._trim_btn.clicked.connect(self._on_apply_trim)
        layout.addWidget(self._trim_btn)

        # Split at a chosen time.
        layout.addWidget(QLabel("Split", self))
        split_row = QHBoxLayout()
        self._split_at = QLineEdit(self)
        self._split_at.setPlaceholderText("Split at (0:20)")
        self._split_btn = QPushButton("Split here", self)
        self._split_btn.clicked.connect(self._on_split)
        split_row.addWidget(self._split_at)
        split_row.addWidget(self._split_btn)
        layout.addLayout(split_row)

        # Reorder / delete.
        order_row = QHBoxLayout()
        self._up_btn = QPushButton("↑ Move up", self)
        self._up_btn.clicked.connect(lambda: self._on_move(-1))
        self._down_btn = QPushButton("↓ Move down", self)
        self._down_btn.clicked.connect(lambda: self._on_move(1))
        self._delete_btn = QPushButton("Delete clip", self)
        self._delete_btn.clicked.connect(self._on_delete)
        order_row.addWidget(self._up_btn)
        order_row.addWidget(self._down_btn)
        order_row.addWidget(self._delete_btn)
        layout.addLayout(order_row)

        # Volume: slider + mute.
        layout.addWidget(QLabel("Volume", self))
        vol_row = QHBoxLayout()
        self._mute = QCheckBox("Mute", self)
        self._mute.stateChanged.connect(self._on_mute_changed)
        self._volume = QSlider(Qt.Orientation.Horizontal, self)
        self._volume.setRange(0, 100)
        self._volume.setValue(100)
        self._volume.valueChanged.connect(self._on_volume_changed)
        vol_row.addWidget(self._mute)
        vol_row.addWidget(self._volume, 1)
        layout.addLayout(vol_row)

        # Captions on/off.
        self._captions_on = QCheckBox("Captions on for this clip", self)
        self._captions_on.setChecked(True)
        self._captions_on.stateChanged.connect(self._on_captions_toggled)
        layout.addWidget(self._captions_on)

        # Caption lines: timestamped, editable, with a delete button
        # per line (FuseClip's corrected-captions list). Timestamps are
        # clip-relative, so they match what the render burns in.
        layout.addWidget(
            QLabel("Caption lines (click a line to edit, × to delete)", self)
        )
        self._lines = QListWidget(self)
        self._lines.setMaximumHeight(132)
        self._lines.itemChanged.connect(self._on_line_edited)
        self._lines.currentRowChanged.connect(self._on_line_selected)
        layout.addWidget(self._lines)
        line_row = QHBoxLayout()
        self._delete_line_btn = QPushButton("Delete line", self)
        self._delete_line_btn.clicked.connect(self._on_delete_line)
        self._word_combo = QComboBox(self)
        self._delete_word_btn = QPushButton("Delete word", self)
        self._delete_word_btn.clicked.connect(self._on_delete_word)
        line_row.addWidget(self._delete_line_btn)
        line_row.addWidget(self._word_combo, 1)
        line_row.addWidget(self._delete_word_btn)
        layout.addLayout(line_row)
        self._remove_all_btn = QPushButton("Remove all captions", self)
        self._remove_all_btn.clicked.connect(self._on_remove_all_captions)
        layout.addWidget(self._remove_all_btn)

        # Add a custom text line at a chosen (video) time.
        layout.addWidget(QLabel("Add custom text", self))
        self._overlay_text = QLineEdit(self)
        self._overlay_text.setPlaceholderText("Text to show")
        layout.addWidget(self._overlay_text)
        overlay_row = QHBoxLayout()
        self._overlay_start = QLineEdit(self)
        self._overlay_start.setPlaceholderText("Start (0:00)")
        self._overlay_end = QLineEdit(self)
        self._overlay_end.setPlaceholderText("End (0:02)")
        overlay_row.addWidget(self._overlay_start)
        overlay_row.addWidget(self._overlay_end)
        layout.addLayout(overlay_row)
        self._overlay_btn = QPushButton("Add text", self)
        self._overlay_btn.clicked.connect(self._on_add_overlay)
        layout.addWidget(self._overlay_btn)
        self._overlay_list = QLabel("", self)
        self._overlay_list.setProperty("muted", True)
        self._overlay_list.setWordWrap(True)
        layout.addWidget(self._overlay_list)

        # Validation / status line.
        self._error = QLabel("", self)
        self._error.setWordWrap(True)
        self._error.setStyleSheet("color: #e5484d;")
        layout.addWidget(self._error)
        layout.addStretch(1)
        self._set_editor_enabled(False)

    def _set_editor_enabled(self, enabled: bool) -> None:
        for widget in (
            self._trim_start, self._trim_end, self._trim_btn,
            self._split_at, self._split_btn,
            self._up_btn, self._down_btn, self._delete_btn,
            self._mute, self._volume, self._captions_on,
            self._lines, self._delete_line_btn, self._word_combo,
            self._delete_word_btn, self._remove_all_btn,
            self._overlay_text, self._overlay_start, self._overlay_end,
            self._overlay_btn,
        ):
            widget.setEnabled(enabled)

    # -- model ------------------------------------------------------------

    def set_clips(
        self,
        clips: list,
        *,
        qa_flags: dict | None = None,
        words: list[Word] | None = None,
        video_duration: float = 0.0,
    ) -> None:
        """Load clips into the Edit step.

        ``clips`` is a list of ``ClipRange`` objects; ``qa_flags`` maps a
        clip index to True when its face-tracking QA failed (framing
        needs review). ``words`` are the transcript words in source-video
        time — every clip keeps the full transcript so a trim can grow
        back into a gap and still find its words. Ranges are editable
        again in the timeline below.
        """
        flags = qa_flags or {}
        self._video_duration = video_duration
        if video_duration > 0:
            # The timeline validates ranges against the video length — without
            # this it reports every range as "past the video length (0.00s)".
            self.timeline.set_duration(video_duration)
        self._clips = []
        for c in clips:
            rng = (
                c if isinstance(c, ClipRange)
                else ClipRange(start=c[0], end=c[1])
            )
            self._clips.append(
                EditableClip(
                    start=rng.start, end=rng.end,
                    source=rng.source, label=rng.label,
                    words=list(words) if words else None,
                )
            )
        self._qa_flags = flags
        self._selected = 0 if self._clips else None
        self._refresh_list()
        self._refresh_editor()
        self._sync_timeline()

    def kept_clips(self) -> list[EditableClip]:
        """Clips still checked (approved) in the list, in render order."""
        return [c for c in self._clips if c.kept]

    def kept_ranges(self) -> list[ClipRange]:
        """Ranges whose clip is still checked (approved) in the list."""
        return [c.to_range() for c in self.kept_clips()]

    def edited_overrides(self) -> tuple[dict, dict, dict, dict]:
        """Render overrides for the kept clips, keyed by kept index.

        Returns ``(words_by_index, overlays_by_index, volumes,
        captions_enabled)`` — words and overlays re-timed to clip time,
        matching :func:`clipforge.core.pipeline.build_render_jobs`.
        """
        kept = self.kept_clips()
        words_by_index = {
            i: clip_words_relative(c) for i, c in enumerate(kept)
        }
        overlays_by_index = {
            i: [
                CaptionOverlay(
                    text=o.text,
                    start=o.start - c.start,
                    end=o.end - c.start,
                )
                for o in c.overlays
            ]
            for i, c in enumerate(kept)
        }
        volumes = {i: c.volume for i, c in enumerate(kept)}
        captions_enabled = {i: c.captions_enabled for i, c in enumerate(kept)}
        return words_by_index, overlays_by_index, volumes, captions_enabled

    # -- list / editor refresh --------------------------------------------

    def _refresh_list(self) -> None:
        self._clip_list.blockSignals(True)
        self._clip_list.clear()
        if not self._clips:
            self._clip_list.addItem("No clips yet — run Start clipping first.")
            self._clip_list.setEnabled(False)
            self._preset.setEnabled(False)
            self._clip_list.blockSignals(False)
            return
        self._clip_list.setEnabled(True)
        self._preset.setEnabled(True)
        for i, clip in enumerate(self._clips):
            label = clip.label or f"Clip {i + 1}"
            text = f"{label}  {clip.start:.1f}s – {clip.end:.1f}s"
            if getattr(self, "_qa_flags", {}).get(i):
                text += "  ⚠ review framing"
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if clip.kept
                else Qt.CheckState.Unchecked
            )
            if getattr(self, "_qa_flags", {}).get(i):
                item.setToolTip("Face tracking QA failed — check the framing.")
            self._clip_list.addItem(item)
        if self._selected is not None:
            self._clip_list.setCurrentRow(self._selected)
        self._clip_list.blockSignals(False)

    def _refresh_editor(self) -> None:
        clip = self._selected_clip()
        if clip is None:
            self._editor_title.setText("Select a clip to edit")
            self._set_editor_enabled(False)
            self._lines.blockSignals(True)
            self._lines.clear()
            self._lines.blockSignals(False)
            self._word_combo.clear()
            self._overlay_list.setText("")
            return
        self._set_editor_enabled(True)
        label = clip.label or f"Clip {(self._selected or 0) + 1}"
        self._editor_title.setText(
            f"{label} — {clip.start:.1f}s to {clip.end:.1f}s"
            f" ({clip.duration:.1f}s)"
        )
        self._trim_start.setText(format_timecode(clip.start))
        self._trim_end.setText(format_timecode(clip.end))
        mid = (clip.start + clip.end) / 2
        self._split_at.setText(format_timecode(mid))
        self._volume.blockSignals(True)
        self._volume.setValue(int(round(clip.volume * 100)))
        self._volume.blockSignals(False)
        self._mute.blockSignals(True)
        self._mute.setChecked(clip.volume == 0.0)
        self._mute.blockSignals(False)
        self._captions_on.blockSignals(True)
        self._captions_on.setChecked(clip.captions_enabled)
        self._captions_on.blockSignals(False)
        self._overlay_start.setText(format_timecode(clip.start))
        self._overlay_end.setText(
            format_timecode(min(clip.start + 2.0, clip.end))
        )
        self._refresh_lines()
        self._overlay_list.setText(
            "\n".join(
                f"• {o.text}  {o.start:.1f}s – {o.end:.1f}s"
                for o in clip.overlays
            )
        )

    def _refresh_lines(self) -> None:
        clip = self._selected_clip()
        self._lines.blockSignals(True)
        self._lines.clear()
        if clip is not None:
            for row, (start, end, text) in enumerate(
                line_spans(clip_words_relative(clip))
            ):
                item = QListWidgetItem(text)
                item.setFlags(
                    item.flags() | Qt.ItemFlag.ItemIsEditable
                )
                self._lines.addItem(item)
                self._lines.setItemWidget(
                    item, self._line_row_widget(row, start, end, text)
                )
        self._lines.blockSignals(False)
        self._refresh_word_combo()

    def _line_row_widget(
        self, row: int, start: float, end: float, text: str
    ) -> QWidget:
        """One corrected-caption row: timestamp + editable text + ×."""
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        stamp = QLabel(
            f"{format_timecode(start)}–{format_timecode(end)}", widget
        )
        stamp.setProperty("muted", True)
        layout.addWidget(stamp)
        edit = QLineEdit(text, widget)
        edit.setProperty("line_row", row)
        edit.installEventFilter(self)
        edit.editingFinished.connect(
            lambda _r=row, e=edit: self._on_line_text_edited(_r, e.text())
        )
        layout.addWidget(edit, 1)
        delete = QPushButton("×", widget)
        delete.setToolTip("Delete this line")
        delete.setFixedWidth(24)
        delete.clicked.connect(
            lambda _checked=False, r=row: self._on_delete_line_row(r)
        )
        layout.addWidget(delete)
        return widget

    def eventFilter(self, watched, event) -> bool:
        # Focusing a line's text field selects that row, so the word
        # combo (delete word) follows the line being edited.
        if event.type() == QEvent.Type.FocusIn:
            row = watched.property("line_row")
            if row is not None:
                self._lines.setCurrentRow(int(row))
        return super().eventFilter(watched, event)

    def _refresh_word_combo(self) -> None:
        """Words of the selected caption line, for word delete."""
        clip = self._selected_clip()
        self._word_combo.blockSignals(True)
        self._word_combo.clear()
        if clip is not None:
            words = clip_words_relative(clip)
            # Flat word index across the clip's lines.
            line_index = self._lines.currentRow()
            if line_index >= 0:
                lines = line_texts(words)
                offset = sum(len(line.split()) for line in lines[:line_index])
                line_words = lines[line_index].split()
                for j, word in enumerate(line_words):
                    self._word_combo.addItem(word, offset + j)
        self._word_combo.blockSignals(False)

    def _selected_clip(self) -> EditableClip | None:
        if self._selected is None or not 0 <= self._selected < len(self._clips):
            return None
        return self._clips[self._selected]

    def _replace_selected(self, clip: EditableClip) -> None:
        if self._selected is not None:
            self._clips[self._selected] = clip

    def _sync_timeline(self) -> None:
        self._syncing_timeline = True
        try:
            self.timeline.set_ranges(
                [c.to_range() for c in self._clips]
            )
        finally:
            self._syncing_timeline = False

    def _after_edit(self) -> None:
        """Model changed — refresh the views and rebuild render jobs."""
        self._refresh_list()
        self._refresh_editor()
        self._sync_timeline()
        self.clips_changed.emit()

    def _show_error(self, message: str) -> None:
        self._error.setText(message)

    def _parse(self, field: QLineEdit, name: str) -> float | None:
        try:
            return parse_timecode(field.text())
        except ValueError:
            self._show_error(f"Could not read {name} time “{field.text()}”.")
            return None

    # -- slots --------------------------------------------------------------

    def _on_ranges_changed(self, ranges: list) -> None:
        """Timeline ranges were edited by dragging — keep the model in sync."""
        if self._syncing_timeline:
            return
        if len(ranges) != len(self._clips):
            return
        for clip, rng in zip(self._clips, ranges):
            clip.start, clip.end = rng.start, rng.end
            if clip.words is not None:
                clip.words = [
                    w for w in clip.words
                    if rng.start <= w.start < rng.end
                ]
            clip.overlays = [
                o for o in clip.overlays
                if rng.start <= o.start < rng.end
            ]
        self._refresh_list()
        self._refresh_editor()
        self.clips_changed.emit()

    def _on_clip_toggled(self, item: QListWidgetItem) -> None:
        """A keep/drop checkbox changed — the render list is rebuilt."""
        if self._clip_list.signalsBlocked():
            return
        row = self._clip_list.row(item)
        if 0 <= row < len(self._clips):
            self._clips[row].kept = (
                item.checkState() == Qt.CheckState.Checked
            )
            self.clips_changed.emit()

    def set_video_source(self, path: str) -> None:
        """Load the source video into the preview player."""
        self._video_source = path
        if path:
            self._player_widget.set_source(path)
        else:
            self._player_widget.clear()

    def _on_clip_selected(self, row: int) -> None:
        if 0 <= row < len(self._clips):
            self._selected = row
            self._show_error("")
            self._refresh_editor()
            clip = self._clips[row]
            if getattr(self, "_video_source", ""):
                # Preview exactly this clip's range of the source.
                self._player_widget.set_play_range(clip.start, clip.end)

    def _on_apply_trim(self) -> None:
        clip = self._selected_clip()
        if clip is None:
            return
        start = self._parse(self._trim_start, "start")
        end = self._parse(self._trim_end, "end")
        if start is None or end is None:
            return
        try:
            trimmed = trim_clip(
                clip, start, end,
                video_duration=self._video_duration,
                siblings=self._clips,
            )
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self._show_error("")
        self._replace_selected(trimmed)
        self._after_edit()

    def _on_split(self) -> None:
        clip = self._selected_clip()
        if clip is None:
            return
        at = self._parse(self._split_at, "split")
        if at is None:
            return
        try:
            first, second = split_clip(clip, at)
        except ValueError as exc:
            self._show_error(str(exc))
            return
        if clip.label:
            second.label = f"{clip.label} (2)"
        self._clips[self._selected : self._selected + 1] = [first, second]
        self._show_error("")
        self._after_edit()

    def _on_move(self, direction: int) -> None:
        if self._selected is None:
            return
        target = self._selected + direction
        if not 0 <= target < len(self._clips):
            return
        self._clips = move_clip(self._clips, self._selected, target)
        self._selected = target
        self._after_edit()

    def _on_delete(self) -> None:
        if self._selected is None:
            return
        self._clips = delete_clip(self._clips, self._selected)
        self._selected = (
            min(self._selected, len(self._clips) - 1)
            if self._clips else None
        )
        self._after_edit()

    def _on_mute_changed(self, state: int) -> None:
        clip = self._selected_clip()
        if clip is None:
            return
        if state == Qt.CheckState.Checked.value:
            self._last_volume = clip.volume if clip.volume > 0 else 1.0
            self._set_volume(0.0)
        else:
            self._set_volume(self._last_volume)

    def _on_volume_changed(self, value: int) -> None:
        clip = self._selected_clip()
        if clip is None:
            return
        self._set_volume(value / 100.0)

    def _set_volume(self, volume: float) -> None:
        clip = self._selected_clip()
        if clip is None:
            return
        self._clips[self._selected] = EditableClip(
            start=clip.start, end=clip.end, source=clip.source,
            label=clip.label, words=clip.words,
            captions_enabled=clip.captions_enabled,
            overlays=list(clip.overlays), volume=volume, kept=clip.kept,
        )
        self._volume.blockSignals(True)
        self._volume.setValue(int(round(volume * 100)))
        self._volume.blockSignals(False)
        self._mute.blockSignals(True)
        self._mute.setChecked(volume == 0.0)
        self._mute.blockSignals(False)
        self.clips_changed.emit()

    def _on_captions_toggled(self, state: int) -> None:
        clip = self._selected_clip()
        if clip is None:
            return
        clip.captions_enabled = state == Qt.CheckState.Checked.value
        self._refresh_lines()
        self.clips_changed.emit()

    def _on_line_edited(self, item: QListWidgetItem) -> None:
        if self._lines.signalsBlocked():
            return
        self._on_line_text_edited(self._lines.row(item), item.text())

    def _on_line_text_edited(self, row: int, text: str) -> None:
        clip = self._selected_clip()
        if clip is None:
            return
        try:
            edited = edit_clip_line(clip, row, text)
        except (ValueError, IndexError):
            self._refresh_lines()
            return
        self._replace_selected(edited)
        self._refresh_lines()
        self.clips_changed.emit()

    def _on_line_selected(self, row: int) -> None:
        self._refresh_word_combo()

    def _on_delete_line(self) -> None:
        row = self._lines.currentRow()
        if row >= 0:
            self._on_delete_line_row(row)

    def _on_delete_line_row(self, row: int) -> None:
        clip = self._selected_clip()
        if clip is None:
            return
        try:
            edited = delete_clip_line(clip, row)
        except IndexError:
            return
        self._replace_selected(edited)
        self._refresh_lines()
        self.clips_changed.emit()

    def _on_delete_word(self) -> None:
        clip = self._selected_clip()
        if clip is None:
            return
        flat_index = self._word_combo.currentData()
        if flat_index is None:
            return
        try:
            edited = delete_clip_word(clip, int(flat_index))
        except (ValueError, IndexError):
            return
        self._replace_selected(edited)
        self._refresh_lines()
        self.clips_changed.emit()

    def _on_remove_all_captions(self) -> None:
        clip = self._selected_clip()
        if clip is None:
            return
        self._replace_selected(
            EditableClip(
                start=clip.start, end=clip.end, source=clip.source,
                label=clip.label, words=[],
                captions_enabled=clip.captions_enabled,
                overlays=list(clip.overlays), volume=clip.volume,
                kept=clip.kept,
            )
        )
        self._refresh_lines()
        self.clips_changed.emit()

    def _on_add_overlay(self) -> None:
        clip = self._selected_clip()
        if clip is None:
            return
        start = self._parse(self._overlay_start, "custom text start")
        end = self._parse(self._overlay_end, "custom text end")
        if start is None or end is None:
            return
        try:
            edited = add_overlay(
                clip, self._overlay_text.text(), start, end
            )
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self._show_error("")
        self._overlay_text.clear()
        self._replace_selected(edited)
        self._refresh_editor()
        self.clips_changed.emit()
