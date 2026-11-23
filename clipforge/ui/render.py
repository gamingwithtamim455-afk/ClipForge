"""Render step (wizard step 4).

Shows the output summary, renders the pending clips in a background job
(the UI never freezes), then lists each finished clip with its TikTok
caption + exactly 4 hashtags (``01_label.txt`` next to the video) and a
Copy button. Labels are generated with DeepSeek when a client is wired
in (``set_label_client``); otherwise the clip is flagged "label pending"
and no ``.txt`` is faked.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.jobs import run_job
from ..core.pipeline import render_pipeline_jobs
from ..core.render import (
    ClipResult,
    RenderJob,
    RenderSettings,
)
from ..i18n import t
from .widgets import VideoPlayerWidget

__all__ = ["RenderPage", "whatsapp_share_folder"]


def whatsapp_share_folder(video_path: str) -> str:
    """Folder to open for a WhatsApp share of a finished clip.

    There is no free official WhatsApp API and unofficial login
    libraries risk an account ban, so ClipForge never logs into
    WhatsApp. Sharing = open the clip's folder in the OS file manager
    so Boss can attach the video in WhatsApp himself (the caption is
    copied to the clipboard by the button handler).
    """
    from pathlib import Path

    return str(Path(video_path).expanduser().parent)


class RenderPage(QWidget):
    """The Render step."""

    render_finished = Signal(list)  # finished ClipResults

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._jobs: list[RenderJob] = []
        self._results: list[ClipResult] = []
        self._label_client = None  # DeepSeekClient, wired in by the app
        self._running = False
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        # Left: output summary.
        summary = QFrame(self)
        summary.setProperty("panel", True)
        summary.setFixedWidth(260)
        summary_layout = QVBoxLayout(summary)
        title = QLabel("Output", self)
        title.setProperty("title", True)
        summary_layout.addWidget(title)
        self._summary_rows: dict[str, QLabel] = {}
        for label, value in (
            ("Resolution", "1080 × 1920"),
            ("Codec", "H.264 · CRF 15 · preset slow"),
            ("Audio", "AAC 256k"),
            ("Captions", "Burned in · word highlight"),
        ):
            row = QHBoxLayout()
            key = QLabel(label, self)
            key.setProperty("muted", True)
            row.addWidget(key, 1)
            value_label = QLabel(value, self)
            row.addWidget(value_label, 1)
            self._summary_rows[label] = value_label
            summary_layout.addLayout(row)
        summary_layout.addStretch(1)
        root.addWidget(summary, 0)

        # Center: clip captions.
        center = QFrame(self)
        center.setProperty("panel", True)
        center_layout = QVBoxLayout(center)
        cap_title = QLabel("TikTok captions", self)
        cap_title.setProperty("title", True)
        center_layout.addWidget(cap_title)
        self._clip_list = QListWidget(self)
        self._clip_list.addItem("No clips rendered yet.")
        self._clip_list.setEnabled(False)
        self._clip_list.currentRowChanged.connect(self._on_clip_selected)
        center_layout.addWidget(self._clip_list, 1)
        self._caption_view = QTextEdit(self)
        self._caption_view.setReadOnly(True)
        self._caption_view.setPlaceholderText(
            "Caption + exactly 4 hashtags will appear here."
        )
        center_layout.addWidget(self._caption_view, 1)
        self._copy_btn = QPushButton("Copy", self)
        self._copy_btn.setEnabled(False)
        self._copy_btn.clicked.connect(self._on_copy_clicked)
        center_layout.addWidget(self._copy_btn, 0)
        # In-app preview of the finished clip (FuseClip's "Preview Clip") —
        # before this, the only way to watch a rendered clip was the OS
        # file manager.
        self._preview_btn = QPushButton("Preview clip", self)
        self._preview_btn.setEnabled(False)
        self._preview_btn.clicked.connect(self._on_preview_clicked)
        center_layout.addWidget(self._preview_btn, 0)
        root.addWidget(center, 1)

        # Right: render action.
        actions = QFrame(self)
        actions.setProperty("panel", True)
        actions.setFixedWidth(205)
        actions_layout = QVBoxLayout(actions)
        self._render_btn = QPushButton(t("render.start"), self)
        self._render_btn.setProperty("primary", True)
        self._render_btn.setEnabled(False)
        self._render_btn.clicked.connect(self._on_render_clicked)
        actions_layout.addWidget(self._render_btn)
        # WhatsApp share: no API/login — opens the clip folder so the
        # video can be attached in WhatsApp by hand (caption copied too).
        self._share_btn = QPushButton("Share via WhatsApp", self)
        self._share_btn.setEnabled(False)
        self._share_btn.setToolTip(
            "WhatsApp has no free official API, so ClipForge does not log "
            "into it. This opens the finished clip's folder — attach the "
            "video in WhatsApp yourself. The caption is copied to the "
            "clipboard."
        )
        self._share_btn.clicked.connect(self._on_share_clicked)
        actions_layout.addWidget(self._share_btn)
        self._progress = QLabel("", self)
        self._progress.setProperty("muted", True)
        self._progress.setWordWrap(True)
        actions_layout.addWidget(self._progress)
        actions_layout.addStretch(1)
        root.addWidget(actions, 0)

    # -- inputs ----------------------------------------------------------
    def set_label_client(self, client) -> None:
        """DeepSeek client used for per-clip caption generation (optional)."""
        self._label_client = client

    def set_pending_clips(self, jobs: list[RenderJob]) -> None:
        """Clips waiting to be rendered; enables the Render button."""
        self._jobs = list(jobs)
        self._render_btn.setEnabled(bool(jobs) and not self._running)
        if jobs:
            settings = jobs[0].settings or RenderSettings()
            self._summary_rows["Resolution"].setText(
                f"{settings.width} × {settings.height}"
            )
            self._summary_rows["Codec"].setText(
                f"H.264 · CRF {settings.crf} · preset {settings.preset}"
            )
            self._progress.setText(f"{len(jobs)} clip(s) ready to render.")

    def set_results(self, results: list[ClipResult]) -> None:
        """Show finished clips with their captions + hashtags."""
        self._results = list(results)
        self._clip_list.clear()
        if not results:
            self._clip_list.addItem("No clips rendered yet.")
            self._clip_list.setEnabled(False)
            self._copy_btn.setEnabled(False)
            self._preview_btn.setEnabled(False)
            return
        self._clip_list.setEnabled(True)
        for result in results:
            item = QListWidgetItem(f"Clip {result.index:02d} — {result.duration:.1f}s")
            if result.flags:
                item.setText(item.text() + "  ⚠")
                item.setToolTip("\n".join(result.flags))
            self._clip_list.addItem(item)
        self._clip_list.setCurrentRow(0)
        self._share_btn.setEnabled(True)

    # -- rendering -------------------------------------------------------
    def _on_render_clicked(self) -> None:
        if not self._jobs or self._running:
            return
        self._running = True
        self._render_btn.setEnabled(False)
        self._progress.setText("Rendering… 0%")
        run_job(
            self._work,
            on_finished=self._on_render_finished,
            on_failed=self._on_render_failed,
            on_progress=self._on_render_progress,
            parent=self,
        )

    def _work(self, ctx) -> list[ClipResult]:
        """Render + label the pending clips (runs off the UI thread)."""
        jobs = list(self._jobs)
        language = jobs[0].language if jobs else "en"
        return render_pipeline_jobs(
            jobs,
            label_client=self._label_client,
            language=language,
            progress=lambda p, m: ctx.report(p, m),
            cancel=lambda: ctx.cancelled,
        )

    def _on_render_progress(self, percent: int, message: str) -> None:
        self._progress.setText(f"Rendering… {percent}% — {message}")

    def _on_render_finished(self, results: object) -> None:
        self._running = False
        self._render_btn.setEnabled(bool(self._jobs))
        if isinstance(results, list):
            self.set_results(results)
            done = sum(1 for r in results if isinstance(r, ClipResult))
            self._progress.setText(f"Done — {done} clip(s) rendered.")
            self.render_finished.emit(results)
        else:
            self._progress.setText("Render finished.")

    def _on_render_failed(self, error: str) -> None:
        self._running = False
        self._render_btn.setEnabled(bool(self._jobs))
        self._progress.setText(f"Render failed: {error}")
        QMessageBox.warning(self, t("render.title"), f"Rendering failed:\n{error}")

    # -- captions --------------------------------------------------------
    def _on_clip_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._results):
            self._caption_view.clear()
            self._copy_btn.setEnabled(False)
            self._preview_btn.setEnabled(False)
            return
        self._preview_btn.setEnabled(True)
        result = self._results[row]
        if result.label_written:
            text = result.caption + "\n\n" + " ".join(result.hashtags)
        elif result.flags:
            text = "Label pending.\n\n" + "\n".join(result.flags)
        else:
            text = "No caption for this clip."
        self._caption_view.setPlainText(text)
        self._copy_btn.setEnabled(result.label_written)

    def _on_preview_clicked(self) -> None:
        """Play the selected rendered clip inside the app."""
        row = self._clip_list.currentRow()
        if row < 0 or row >= len(self._results):
            return
        result = self._results[row]
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Clip {result.index:02d} — preview")
        layout = QVBoxLayout(dialog)
        player = VideoPlayerWidget(dialog, video_width=324, video_height=576)
        layout.addWidget(player)
        player.set_source(result.video_path)
        player.play()
        dialog.exec()
        player.pause()

    def _on_copy_clicked(self) -> None:
        text = self._caption_view.toPlainText()
        if not text:
            return
        QGuiApplication.clipboard().setText(text)
        self._progress.setText("Copied to clipboard.")

    def _on_share_clicked(self) -> None:
        """Share via WhatsApp (folder-open only — no WhatsApp login/API)."""
        if not self._results:
            return
        row = self._clip_list.currentRow()
        result = self._results[row if 0 <= row < len(self._results) else 0]
        folder = whatsapp_share_folder(result.video_path)
        QGuiApplication.clipboard().setText(self._caption_view.toPlainText())
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
        self._progress.setText(
            "Opened the clip folder — attach the video in WhatsApp; "
            "caption copied to the clipboard."
        )
