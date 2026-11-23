"""Home screen: left sidebar + projects grid.

Matches the reference recording (FaceClip home):
- Sidebar: New Project button, Projects / Open Folder / Settings nav.
  (The recording's bottom "Join Amfaiz" user card is removed — ClipForge
  has no accounts.)
- Main: "Projects" + count badge, search box and "+ New Project" top-right,
  grid of video cards (9:16 badge top-left, duration bottom-right, name
  below).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core.jobs import run_job
from ..core.paths import app_data_dir
from ..core.projects import Project, ProjectStore, delete_project_files
from ..core.thumbnails import generate_thumbnail
from ..i18n import t
from .splash import SidebarBranding
from .widgets import ProjectCard

#: Minimum card width the grid aims for; the column count follows the
#: window width instead of being hard-coded.
CARD_MIN_WIDTH = 176
GRID_SPACING = 12


class HomePage(QWidget):
    """The Home screen."""

    new_project_requested = Signal()
    project_opened = Signal(Project)
    open_folder_requested = Signal()
    settings_requested = Signal()

    def __init__(self, store: ProjectStore, parent: QWidget | None = None,
                 branding_photo=None) -> None:
        super().__init__(parent)
        self._store = store
        self._branding_photo = branding_photo
        self._cards: list[ProjectCard] = []
        self._projects: list[Project] = []
        self._columns = 0
        self._thumb_job_running = False

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        main = QVBoxLayout()
        main.setContentsMargins(16, 12, 16, 12)
        main.setSpacing(12)
        main.addLayout(self._build_header())
        main.addWidget(self._build_grid())
        root.addLayout(main, 1)

    # -- layout builders -------------------------------------------------
    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame(self)
        sidebar.setProperty("sidebar", True)
        sidebar.setFixedWidth(150)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        new_btn = QPushButton(f"+ {t('home.new_project')}", sidebar)
        new_btn.setProperty("primary", True)
        new_btn.clicked.connect(self.new_project_requested.emit)
        layout.addWidget(new_btn)

        layout.addSpacing(8)
        self._nav_projects = self._nav_button(t("sidebar.projects"), sidebar, checked=True)
        layout.addWidget(self._nav_projects)
        open_folder = self._nav_button(t("sidebar.open_folder"), sidebar)
        open_folder.clicked.connect(self.open_folder_requested.emit)
        layout.addWidget(open_folder)
        settings_btn = self._nav_button(t("sidebar.settings"), sidebar)
        settings_btn.clicked.connect(self.settings_requested.emit)
        layout.addWidget(settings_btn)

        layout.addStretch(1)
        # Branding: Boss's portrait with a gentle pulsing glow.
        self._branding = SidebarBranding(sidebar, photo_path=self._branding_photo)
        layout.addWidget(self._branding)
        muse_btn = QPushButton("Open Muse", sidebar)
        muse_btn.setFlat(True)
        muse_btn.setToolTip("Open Muse (muse.ai) in the browser")
        muse_btn.clicked.connect(self._on_open_muse)
        layout.addWidget(muse_btn)
        return sidebar

    @staticmethod
    def _on_open_muse() -> None:
        """Open Muse in the browser (no API — just a link)."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl("https://muse.ai"))

    @staticmethod
    def _nav_button(text: str, parent: QWidget, checked: bool = False) -> QPushButton:
        btn = QPushButton(text, parent)
        btn.setFlat(True)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 7px 10px; border-radius: 6px; }"
            "QPushButton:checked { background: palette(button); font-weight: 600; }"
        )
        return btn

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        title = QLabel(t("home.projects"), self)
        title.setStyleSheet("font-size: 15px; font-weight: 600;")
        header.addWidget(title)
        self._count_badge = QLabel(self)
        self._count_badge.setStyleSheet(
            "QLabel { background: palette(button); border-radius: 8px; "
            "padding: 1px 7px; font-size: 11px; }"
        )
        header.addWidget(self._count_badge)
        header.addStretch(1)

        self._search = QLineEdit(self)
        self._search.setPlaceholderText(t("home.search_placeholder"))
        self._search.setFixedWidth(220)
        self._search.textChanged.connect(self.refresh)
        header.addWidget(self._search)

        new_btn = QPushButton(f"+ {t('home.new_project')}", self)
        new_btn.setProperty("primary", True)
        new_btn.clicked.connect(self.new_project_requested.emit)
        header.addWidget(new_btn)
        return header

    def _build_grid(self) -> QScrollArea:
        self._grid_widget = QWidget(self)
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setHorizontalSpacing(GRID_SPACING)
        self._grid_layout.setVerticalSpacing(8)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll = QScrollArea(self)
        scroll.setWidget(self._grid_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll = scroll
        # The viewport reaches its real width only after the page's own
        # resizeEvent, so reflow is driven from the viewport's resize.
        scroll.viewport().installEventFilter(self)
        return scroll

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if (
            watched is self._scroll.viewport()
            and event.type() == QEvent.Type.Resize
            and self._cards
            and self._column_count() != self._columns
        ):
            self._relayout()
        return super().eventFilter(watched, event)

    # -- data ------------------------------------------------------------
    def _column_count(self) -> int:
        """Columns that fit the current viewport width (at least 1)."""
        width = 0
        if hasattr(self, "_scroll"):
            width = self._scroll.viewport().width()
        if width <= 0:
            width = max(self.width(), CARD_MIN_WIDTH)
        return max(1, (width + GRID_SPACING) // (CARD_MIN_WIDTH + GRID_SPACING))

    def _relayout(self) -> None:
        """Place the current cards into a grid with the right column count.

        The old layout hard-coded 5 columns with no stretch, so cards were
        clipped at the right edge, squeezed into strips, and the area grew
        a horizontal scrollbar. Columns now follow the window width and
        share the available space equally.
        """
        columns = self._column_count()
        self._columns = columns
        for card in self._cards:
            self._grid_layout.removeWidget(card)
        for index, card in enumerate(self._cards):
            self._grid_layout.addWidget(card, index // columns, index % columns)
        for col in range(columns):
            self._grid_layout.setColumnStretch(col, 1)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if self._cards and self._column_count() != self._columns:
            self._relayout()

    def refresh(self) -> None:
        """Rebuild the grid from the store, honouring the search box."""
        query = self._search.text() if hasattr(self, "_search") else ""
        projects = self._store.search(query)
        self._projects = projects
        for card in self._cards:
            card.deleteLater()
        self._cards = []
        for project in projects:
            card = ProjectCard(project, self._grid_widget)
            card.clicked.connect(lambda _pid, p=project: self.project_opened.emit(p))
            card.delete_requested.connect(self._on_delete_project)
            self._cards.append(card)
        self._relayout()
        self._count_badge.setText(str(len(self._store.projects)))
        self._queue_thumbnail_repairs(projects)

    def _on_delete_project(self, project_id: str) -> None:
        """Confirm, then delete a project + its own generated files.

        The confirmation states plainly that the original video file is
        NOT deleted; :func:`delete_project_files` enforces that.
        """
        project = next(
            (p for p in self._store.projects if p.id == project_id), None
        )
        if project is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete project",
            f"Delete project “{project.name}”?\n\n"
            "This removes the project and its rendered clips and "
            "thumbnail from ClipForge.\n"
            "Your original video file will not be deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        delete_project_files(project)
        self._store.remove(project.id)
        self.refresh()

    def _queue_thumbnail_repairs(self, projects: list[Project]) -> None:
        """Generate missing thumbnails in the background (Home B1 fix).

        A card only shows a real frame when ``project.thumbnail`` exists;
        projects whose wizard probe never produced one kept the placeholder
        forever. Repair them here, off the UI thread, and refresh once.
        """
        if self._thumb_job_running:
            return
        missing = [
            p for p in projects
            if p.source_path
            and Path(p.source_path).exists()
            and not (p.thumbnail and Path(p.thumbnail).exists())
        ]
        if not missing:
            return
        self._thumb_job_running = True

        def _work(_ctx, _missing=missing) -> dict:
            made: dict[str, str] = {}
            for project in _missing:
                target = app_data_dir() / "thumbnails" / f"{project.id}.jpg"
                got = generate_thumbnail(project.source_path, target)
                if got:
                    made[project.id] = got
            return made

        def _done(made: object) -> None:
            self._thumb_job_running = False
            if not made:
                return
            for project in self._store.projects:
                if project.id in made:  # type: ignore[operator]
                    project.thumbnail = made[project.id]  # type: ignore[index]
            self._store.save()
            self.refresh()

        def _failed(_error: str) -> None:
            self._thumb_job_running = False

        run_job(_work, on_finished=_done, on_failed=_failed, parent=self)

    def retranslate(self) -> None:
        """Re-apply translated strings (called after an app-language change)."""
        self.refresh()
