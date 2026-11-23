"""Wizard: Select Video → Viral Clips → Edit → Render.

Matches the reference recording's header: "‹ Back" left, the four steps
centered (✓ Select Video, ① Viral Clips, Edit, Render), project name +
"Change" button right. Footer (bottom-left): status line, Back and
Start clipping buttons — Start clipping stays disabled until a video is
selected AND a language is chosen.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.ffmpeg import probe
from ..core.jobs import run_job
from ..core.paths import app_data_dir, default_video_dir
from ..core.projects import Project
from ..core.thumbnails import generate_thumbnail
from ..core.timeparse import format_timecode
from ..i18n import t
from .edit import EditPage
from .render import RenderPage
from .viral_clips import ViralClipsPage

STEP_NAMES = ["wizard.select_video", "wizard.viral_clips", "wizard.edit", "wizard.render"]


class _SelectVideoPage(QWidget):
    """Step 1: pick a source video (drop zone + Browse)."""

    video_chosen = Signal(str)  # file path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        drop = QFrame(self)
        drop.setProperty("panel", True)
        drop.setFixedSize(420, 220)
        drop_layout = QVBoxLayout(drop)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel(t("wizard.select_video"), self)
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        drop_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        hint = QLabel("Drop a video here or browse — MP4, MOV, MKV, WebM", self)
        hint.setProperty("muted", True)
        drop_layout.addWidget(hint, alignment=Qt.AlignmentFlag.AlignCenter)
        self._browse = QPushButton("Browse…", self)
        self._browse.setProperty("primary", True)
        self._browse.clicked.connect(self._on_browse)
        drop_layout.addWidget(self._browse, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(drop, alignment=Qt.AlignmentFlag.AlignCenter)

        self._info = QLabel("", self)
        self._info.setProperty("muted", True)
        layout.addWidget(self._info, alignment=Qt.AlignmentFlag.AlignCenter)

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            t("wizard.select_video"),
            default_video_dir(),
            "Video files (*.mp4 *.mov *.mkv *.webm *.avi);;All files (*)",
        )
        if path:
            self.set_video(path)

    def set_video(self, path: str) -> None:
        self._info.setText(path)
        self.video_chosen.emit(path)


class WizardPage(QWidget):
    """The 4-step wizard."""

    back_to_home = Signal()
    start_clipping_requested = Signal()
    cancel_requested = Signal()
    video_probed = Signal(str, float, str)  # path, duration, thumbnail path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None
        self._video_path: str = ""
        self._video_duration: float = 0.0
        self._video_fps: float = 30.0
        self._step = 0
        self._running = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addLayout(self._build_header())

        self._stack = QStackedWidget(self)
        self._select_video = _SelectVideoPage(self)
        self._select_video.video_chosen.connect(self._on_video_chosen)
        self._viral = ViralClipsPage(self)
        self._viral.readiness_changed.connect(self._update_start_button)
        self._edit = EditPage(self)
        self._render = RenderPage(self)
        for page in (self._select_video, self._viral, self._edit, self._render):
            self._stack.addWidget(page)
        root.addWidget(self._stack, 1)
        root.addLayout(self._build_footer())
        self._update_steps()

    # -- header ----------------------------------------------------------
    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setContentsMargins(8, 8, 12, 8)
        back = QPushButton(f"‹ {t('wizard.back')}", self)
        back.setFlat(True)
        back.clicked.connect(self._on_back)
        header.addWidget(back)
        header.addStretch(1)

        self._step_labels: list[QLabel] = []
        steps_row = QHBoxLayout()
        steps_row.setSpacing(10)
        for index, key in enumerate(STEP_NAMES):
            label = QLabel(t(key), self)
            label.setProperty("muted", True)
            self._step_labels.append(label)
            steps_row.addWidget(label)
            if index < len(STEP_NAMES) - 1:
                sep = QLabel("—", self)
                sep.setProperty("muted", True)
                steps_row.addWidget(sep)
        header.addLayout(steps_row)
        header.addStretch(1)

        self._project_label = QLabel("", self)
        self._project_label.setStyleSheet("font-weight: 600;")
        header.addWidget(self._project_label)
        change = QPushButton(t("wizard.change"), self)
        change.setFlat(True)
        change.clicked.connect(self._on_change_video)
        header.addWidget(change)
        return header

    # -- footer ----------------------------------------------------------
    # Two left-aligned rows, as in the recording: status line on top,
    # Back + Start clipping buttons below it.
    def _build_footer(self) -> QVBoxLayout:
        footer = QVBoxLayout()
        footer.setContentsMargins(12, 8, 12, 10)
        footer.setSpacing(6)
        self._status = QLabel("", self)
        footer.addWidget(self._status)
        buttons = QHBoxLayout()
        back = QPushButton(f"‹ {t('wizard.back')}", self)
        back.clicked.connect(self._on_back)
        buttons.addWidget(back)
        self._start_btn = QPushButton(t("wizard.start_clipping"), self)
        self._start_btn.setProperty("primary", True)
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start_clipping)
        buttons.addWidget(self._start_btn)
        self._cancel_btn = QPushButton("Cancel", self)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_requested.emit)
        buttons.addWidget(self._cancel_btn)
        buttons.addStretch(1)
        footer.addLayout(buttons)
        return footer

    # -- navigation ------------------------------------------------------
    def reset(self) -> None:
        """Clear all per-project wizard state so opening a project starts fresh.

        Without this, the video + language chosen in one project leaked
        into the next (and into other tests sharing the window), so Start
        clipping could be enabled with nothing chosen for the new project.
        """
        self._video_path = ""
        self._video_duration = 0.0
        self._video_fps = 30.0
        self._select_video._info.setText("")
        self._viral.reset()
        self._edit.set_clips([])
        self._edit.set_video_source("")
        self._render.set_pending_clips([])
        self._render.set_results([])
        self._status.setText("")
        self.go_to_step(0)

    def set_project(self, project: Project) -> None:
        self.reset()
        self._project = project
        self._project_label.setText(project.name)
        # Opening a project must open the project: load its source video
        # (probe + preview + Viral Clips step) and restore its saved clips
        # on the timeline. Before this, set_project never loaded the
        # source, so every project click landed back on Select Video and
        # the user had to re-pick the file in a file browser.
        source = project.source_path or ""
        if source and Path(source).exists():
            self._select_video.set_video(source)
            if project.clips:
                self._viral.timeline.set_clips(
                    [
                        (float(c["start"]), float(c["end"]))
                        for c in project.clips
                        if "start" in c and "end" in c
                    ]
                )
        else:
            self.go_to_step(0)

    def go_to_step(self, index: int) -> None:
        self._step = max(0, min(len(STEP_NAMES) - 1, index))
        self._stack.setCurrentIndex(self._step)
        self._update_steps()
        self._update_start_button()

    def _update_steps(self) -> None:
        for index, label in enumerate(self._step_labels):
            if index < self._step:
                label.setText(f"✓ {t(STEP_NAMES[index])}")
                label.setProperty("muted", True)
            elif index == self._step:
                label.setText(f"{index + 1}  {t(STEP_NAMES[index])}")
                label.setProperty("muted", False)
                label.setStyleSheet("font-weight: 600;")
            else:
                label.setText(t(STEP_NAMES[index]))
                label.setProperty("muted", True)
                label.setStyleSheet("")
            label.style().unpolish(label)
            label.style().polish(label)

    def _on_back(self) -> None:
        if self._step > 0:
            self.go_to_step(self._step - 1)
        else:
            self.back_to_home.emit()

    def _on_change_video(self) -> None:
        self.go_to_step(0)

    def _on_video_chosen(self, path: str) -> None:
        self._video_path = path
        try:
            info = probe(path)
        except Exception as exc:
            self._status.setText(f"Could not read this video file: {exc}")
            return
        self._video_duration = info.duration
        self._video_fps = info.fps or 30.0
        label = format_timecode(info.duration)
        self._viral.set_video(duration=info.duration, time_label=label)
        self._viral.set_source(path)
        self._edit.set_video_source(path)
        self._viral.timeline.set_duration(info.duration, self._video_fps)
        self.video_probed.emit(path, info.duration, "")
        self.go_to_step(1)
        # Real frame in the 9:16 preview, grabbed in a background job —
        # an ffmpeg frame grab can take a second or two and must not
        # freeze the UI on every video pick (best effort: the placeholder
        # stays when the grab fails).
        thumb_target = (
            app_data_dir() / "thumbnails"
            / f"{self._project.id if self._project else 'preview'}.jpg"
        )

        def _grab(_ctx, _path=path, _target=thumb_target) -> str:
            return generate_thumbnail(_path, _target)

        def _thumb_done(thumb: object) -> None:
            if thumb and self._video_path == path:
                self._viral.set_thumbnail(str(thumb))
                # Re-emit so the app stores the thumbnail on the project.
                self.video_probed.emit(path, info.duration, str(thumb))

        run_job(_grab, on_finished=_thumb_done, parent=self)

    def set_running(self, running: bool) -> None:
        """While a pipeline job runs, Start becomes Cancel."""
        self._running = running
        self._cancel_btn.setEnabled(running)
        self._start_btn.setEnabled(not running and bool(self._video_path) and self._viral.is_ready)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    @property
    def video_path(self) -> str:
        return self._video_path

    @property
    def video_duration(self) -> float:
        return self._video_duration

    @property
    def project(self) -> Project | None:
        return self._project

    def _on_start_clipping(self) -> None:
        # Manual ranges (if any) travel with the signal: the pipeline uses
        # exactly those ranges and skips AI discovery for this project.
        self.start_clipping_requested.emit()

    @property
    def manual_ranges(self) -> list:
        """Manual timeline ranges from the Viral Clips step."""
        return self._viral.manual_ranges

    def _update_start_button(self, _ready: bool | None = None) -> None:
        """Enabled only with a video selected and a language chosen."""
        ready = bool(self._video_path) and self._viral.is_ready
        self._start_btn.setEnabled(ready and not self._running)
        if not self._video_path:
            self._status.setText("")
        elif not self._viral.is_ready:
            self._status.setText(f"⚠ {t('wizard.select_language_hint')} — {t('wizard.select_language_text')}")
        elif not self._running:
            self._status.setText(f"✓ {t('wizard.ready_to_clip')}")
