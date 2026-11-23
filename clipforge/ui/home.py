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

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core.projects import Project, ProjectStore
from ..i18n import t
from .splash import SidebarBranding
from .widgets import ProjectCard


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
        self._grid_layout.setHorizontalSpacing(12)
        self._grid_layout.setVerticalSpacing(8)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll = QScrollArea(self)
        scroll.setWidget(self._grid_widget)
        scroll.setWidgetResizable(True)
        return scroll

    # -- data ------------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the grid from the store, honouring the search box."""
        query = self._search.text() if hasattr(self, "_search") else ""
        projects = self._store.search(query)
        for card in self._cards:
            card.deleteLater()
        self._cards = []
        for index, project in enumerate(projects):
            card = ProjectCard(project, self._grid_widget)
            card.clicked.connect(lambda _pid, p=project: self.project_opened.emit(p))
            self._grid_layout.addWidget(card, index // 5, index % 5)
            self._cards.append(card)
        self._count_badge.setText(str(len(self._store.projects)))

    def retranslate(self) -> None:
        """Re-apply translated strings (called after an app-language change)."""
        self.refresh()
