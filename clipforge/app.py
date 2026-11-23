"""Application shell: MainWindow with Home <-> Wizard navigation.

Owns the project store, settings, theme application and the settings
dialog. Wires Start clipping to the full pipeline (probe -> transcribe ->
ranges -> face tracking -> render jobs) in a background job, the Edit
step's keep/drop + range edits back into the Render step, and the
weekly backup + rotating logs on startup.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QWidget,
)

from . import __app_name__
from .core.app_logging import setup_logging
from .core.backups import maybe_weekly_backup
from .core.branding import branding_photo_path
from .core.jobs import Job, run_job
from .core.paths import app_data_dir, default_video_dir
from .core.pipeline import (
    PipelineConfig,
    PipelineResult,
    build_render_jobs,
    deepseek_client_from_settings,
    run_pipeline,
)
from .core.projects import Project, ProjectStore
from .core.render import cleanup_clipforge_temp
from .i18n import layout_direction, set_language, t
from .settings import DEFAULTS, Settings
from .ui.home import HomePage
from .ui.settings_dialog import SettingsDialog
from .ui.theme import apply_theme
from .ui.wizard import WizardPage

log = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    """Display duration, e.g. 3254 -> "54:14", 5400 -> "1:30:00"."""
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class MainWindow(QMainWindow):
    """ClipForge main window."""

    def __init__(self, settings: Settings, store: ProjectStore) -> None:
        super().__init__()
        self._settings = settings
        self._store = store
        self._pipeline_result: PipelineResult | None = None
        self._pipeline_job: Job | None = None
        self.setWindowTitle(__app_name__)
        # Initial size: 1280x800 at most, but never larger than the
        # screen's available area — on a 1080p monitor with 150% Windows
        # scaling the logical screen is only ~1280x720, and a fixed
        # 1280x800 window overflowed it (right/bottom edges cut off).
        width, height = 1280, 800
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, max(640, int(available.width() * 0.95)))
            height = min(height, max(480, int(available.height() * 0.95)))
        self.resize(width, height)

        self._stack = QStackedWidget(self)
        self.setCentralWidget(self._stack)

        self._home = HomePage(store, self,
                              branding_photo=branding_photo_path(settings))
        self._home.new_project_requested.connect(self._on_new_project)
        self._home.project_opened.connect(self._open_project)
        self._home.open_folder_requested.connect(self._on_open_folder)
        self._home.settings_requested.connect(self._open_settings)
        self._stack.addWidget(self._home)

        self._wizard = WizardPage(self)
        self._wizard.back_to_home.connect(self._show_home)
        self._wizard.start_clipping_requested.connect(self._on_start_clipping)
        self._wizard.cancel_requested.connect(self._on_cancel_clipping)
        self._wizard.video_probed.connect(self._on_video_probed)
        self._wizard._edit.clips_changed.connect(self._on_edit_clips_changed)
        self._wizard._render.render_finished.connect(self._on_render_finished)
        self._stack.addWidget(self._wizard)

        self._home.refresh()
        self._stack.setCurrentWidget(self._home)

    # -- navigation ------------------------------------------------------
    def _show_home(self) -> None:
        self._home.refresh()
        self._stack.setCurrentWidget(self._home)

    def _open_project(self, project: Project) -> None:
        self._wizard.set_project(project)
        self._wizard._render.set_label_client(
            deepseek_client_from_settings(self._settings)
        )
        self._stack.setCurrentWidget(self._wizard)

    def _on_new_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            t("wizard.select_video"),
            default_video_dir(),
            "Video files (*.mp4 *.mov *.mkv *.webm *.avi);;All files (*)",
        )
        if not path:
            return
        name = Path(path).stem
        project = Project(name=name, source_path=path)
        self._store.add(project)
        self._wizard.set_project(project)
        self._wizard._select_video.set_video(path)
        self._stack.setCurrentWidget(self._wizard)

    def _on_open_folder(self) -> None:
        """Open the export/projects folder in the OS file manager."""
        export_dir = Path(self._settings.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(export_dir)))

    def _on_video_probed(self, path: str, duration: float, thumbnail: str) -> None:
        """The wizard probed a video — store its metadata on the project."""
        project = self._wizard.project
        if project is None:
            return
        project.source_duration = duration
        project.duration = _format_duration(duration)
        if thumbnail:
            project.thumbnail = thumbnail
        self._store.save()
        log.info("Video probed: %s (%.1fs)", Path(path).name, duration)

    # -- pipeline --------------------------------------------------------
    def _build_pipeline_config(self) -> PipelineConfig:
        wizard = self._wizard
        viral = wizard._viral
        from .providers.deepgram import model_id_for_label

        accel = str(self._settings.get("accel_transcription"))
        whisper_device = {"GPU": "cuda", "CPU": "cpu"}.get(accel, "auto")
        return PipelineConfig(
            video_path=wizard.video_path,
            language=viral.language_code or "auto",
            engine=str(self._settings.get("transcription_engine")),
            deepgram_model=model_id_for_label(
                str(self._settings.get("transcription_model"))
            ),
            whisper_device=whisper_device,
            clip_length=float(viral.clip_length),
            clips_requested=viral.clips_requested_count,
            manual_ranges=list(wizard.manual_ranges),
            captions=viral.captions_mode,
            split_screen=bool(self._settings.get("split_screen")),
            quality=viral.quality,
            aspect=viral.aspect,
            accel=str(self._settings.get("accel_rendering")),
            caption_preset=wizard._edit.caption_preset,
            custom_font=str(self._settings.get("custom_font")),
            caption_font=str(self._settings.get("caption_font")),
            keyword_scale=float(self._settings.get("caption_keyword_scale")),
            caption_colors=self._settings.caption_colors(),
            caption_layout=self._settings.caption_layout(),
            caption_gradient=self._settings.caption_gradient(),
            export_dir=self._settings.export_dir,
            project_name=(wizard.project.name if wizard.project else "ClipForge"),
            discovery_template=self._settings.discovery_prompt,
            review_template=self._settings.review_prompt,
        )

    def _on_start_clipping(self) -> None:
        if self._pipeline_job is not None:
            return  # already running
        wizard = self._wizard
        if not wizard.video_path:
            return
        # Manual ranges must be valid before any work starts.
        validation = wizard._viral.timeline.validation
        if wizard.manual_ranges and not validation.is_valid:
            QMessageBox.warning(
                self,
                t("wizard.start_clipping"),
                "Manual ranges are not valid:\n" + "\n".join(validation.errors),
            )
            return
        config = self._build_pipeline_config()
        deepseek_client = deepseek_client_from_settings(self._settings)
        wizard.set_running(True)
        wizard.set_status("Starting…")

        def _work(ctx) -> PipelineResult:
            return run_pipeline(
                config,
                progress=ctx.report,
                cancel=lambda: ctx.cancelled,
                deepseek_client=deepseek_client,
            )

        self._pipeline_job = run_job(
            _work,
            on_finished=self._on_pipeline_finished,
            on_failed=self._on_pipeline_failed,
            on_progress=lambda p, m: wizard.set_status(f"{p}% — {m}"),
            parent=self,
        )
        log.info("Pipeline started for %s", config.video_path)

    def _on_cancel_clipping(self) -> None:
        if self._pipeline_job is not None:
            self._pipeline_job.request_cancel()
            self._wizard.set_status("Cancelling…")

    def _on_pipeline_finished(self, result: object) -> None:
        self._pipeline_job = None
        self._wizard.set_running(False)
        if not isinstance(result, PipelineResult):
            self._wizard.set_status("Pipeline finished with an unexpected result.")
            return
        self._pipeline_result = result
        wizard = self._wizard
        # Show the clips on the timeline + Edit step, queue the Render step.
        wizard._viral.timeline.set_clips(
            [(r.start, r.end) for r in result.ranges]
        )
        wizard._edit.set_clips(
            result.ranges, qa_flags=result.qa_flags,
            words=result.transcript.words,
            video_duration=wizard.video_duration,
        )
        # The Edit preview shows each clip's reframed crop window, not
        # the raw letterboxed source (v0.2.1 B).
        wizard._edit.set_crop_paths(
            result.crop_paths, result.width, result.height,
        )
        # Default caption preset from Settings > Fonts (user can change
        # it per project in the Edit step before rendering).
        wizard._edit.set_caption_preset(str(self._settings.get("caption_preset")))
        wizard._render.set_label_client(
            deepseek_client_from_settings(self._settings)
        )
        wizard._render.set_pending_clips(result.render_jobs)
        # Persist the clips on the project.
        project = wizard.project
        if project is not None:
            project.clips = [
                {
                    "start": r.start,
                    "end": r.end,
                    "source": r.source,
                    "label": r.label,
                }
                for r in result.ranges
            ]
            project.status = "clips_ready"
            project.export_dir = str(result.output_dir)
            self._store.save()
        wizard.set_status(
            f"{len(result.ranges)} clip(s) ready — review them in the Edit step, then Render."
        )
        wizard.go_to_step(2)
        log.info("Pipeline finished: %d clip(s)", len(result.ranges))

    def _on_pipeline_failed(self, error: str) -> None:
        self._pipeline_job = None
        self._wizard.set_running(False)
        if error.strip() == "Cancelled.":
            # The user pressed Cancel: clean idle state, no error dialog.
            self._wizard.set_status("Cancelled")
            log.info("Pipeline cancelled by user")
            return
        # run_job stringifies the exception; PipelineError messages are
        # already friendly, so show them as-is.
        self._wizard.set_status(f"Failed: {error}")
        QMessageBox.warning(self, t("wizard.start_clipping"), error)
        log.warning("Pipeline failed: %s", error)

    def _on_edit_clips_changed(self) -> None:
        """Edit step changed (range edit / keep-drop) — rebuild render jobs."""
        result = self._pipeline_result
        if result is None:
            return
        edit = self._wizard._edit
        kept = edit.kept_clips()
        if not kept:
            self._wizard._render.set_pending_clips([])
            return
        ranges = [c.to_range() for c in kept]
        # Reuse the tracked crop path for ranges that did not change
        # (matched by start/end); changed ranges re-track at render time.
        crop_paths: dict[int, object] = {}
        for new_index, rng in enumerate(ranges):
            for old_index, old in enumerate(result.ranges):
                if abs(old.start - rng.start) < 0.01 and abs(old.end - rng.end) < 0.01:
                    crop_paths[new_index] = result.crop_paths.get(old_index)
                    break
        settings = result.render_jobs[0].settings if result.render_jobs else None
        words_by_index, overlays_by_index, volumes, captions_enabled = (
            edit.edited_overrides()
        )
        jobs = build_render_jobs(
            video_path=result.video_path,
            ranges=ranges,
            words=result.transcript.words,
            crop_paths=crop_paths,  # type: ignore[arg-type]
            language=result.transcript.language,
            settings=settings,
            output_dir=result.output_dir,
            words_by_index=words_by_index,
            overlays_by_index=overlays_by_index,
            volumes=volumes,
            captions_enabled=captions_enabled,
        )
        self._wizard._render.set_pending_clips(jobs)
        project = self._wizard.project
        if project is not None:
            project.clips = [
                {"start": r.start, "end": r.end, "source": r.source, "label": r.label}
                for r in ranges
            ]
            self._store.save()

    def _on_render_finished(self, results: list) -> None:
        """Rendering done — mark the project rendered and save."""
        project = self._wizard.project
        if project is not None and results:
            project.status = "rendered"
            self._store.save()
            log.info("Render finished: %d clip(s) for %s", len(results), project.name)

    # -- settings --------------------------------------------------------
    def _open_settings(self) -> None:
        dialog = SettingsDialog(self._settings, self)
        dialog.appearance_changed.connect(self._on_appearance_changed)
        dialog.language_changed.connect(self._on_language_changed)
        dialog.exec()

    def _on_appearance_changed(self, value: str) -> None:
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, value)

    def _on_language_changed(self, code: str) -> None:
        self._settings.set("app_language", code)
        set_language(code)
        app = QApplication.instance()
        if app is not None:
            app.setLayoutDirection(layout_direction(code))
        self._home.retranslate()
        self.setWindowTitle(t("app.name"))


def create_app(argv: list[str] | None = None) -> tuple[QApplication, MainWindow]:
    """Build the QApplication + MainWindow (also used by tests).

    Reuses an existing QApplication when one is already running (Qt allows
    exactly one per process) instead of raising.
    """
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(__app_name__)
    app.setOrganizationName(__app_name__)

    settings = Settings()
    set_language(str(settings.get("app_language")))
    app.setLayoutDirection(layout_direction())
    apply_theme(app, str(settings.get("appearance")))

    # Logging, weekly backups, temp cleanup — once per process start.
    data_dir = app_data_dir()
    setup_logging(data_dir)
    store = ProjectStore(data_dir / "projects.json")
    snapshot = {key: settings.get(key) for key in DEFAULTS}
    try:
        maybe_weekly_backup(store.path, snapshot, data_dir / "backups")
    except Exception:  # noqa: BLE001 - backups must never block startup
        log.exception("Weekly backup failed")
    try:
        cleanup_clipforge_temp()
    except Exception:  # noqa: BLE001 - temp cleanup must never block startup
        pass

    window = MainWindow(settings, store)
    return app, window
