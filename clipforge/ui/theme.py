"""Themes: dark (default), light, and system-following.

Colors are sampled from the reference recording (FaceClip): near-black
backgrounds, dark panels, white primary buttons, blue "Cutout" badges.
The app uses the Fusion style with a custom palette plus a small QSS for
rounded inputs, buttons and segmented controls.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

DARK = {
    "window": "#101010",
    "window_alt": "#161616",  # sidebar
    "panel": "#1b1b1b",
    "panel_border": "#2b2b2b",
    "input": "#232323",
    "input_border": "#383838",
    "text": "#ededed",
    "muted": "#9d9d9d",
    "button": "#2a2a2a",
    "button_hover": "#343434",
    "primary_bg": "#f2f2f2",
    "primary_text": "#111111",
    "accent": "#2f7de1",
    "accent_green": "#3fb96a",
    "accent_red": "#e5484d",
    "selection": "#2f5cb8",
}

LIGHT = {
    "window": "#f0f0f0",
    "window_alt": "#e6e6e6",
    "panel": "#ffffff",
    "panel_border": "#d4d4d4",
    "input": "#ffffff",
    "input_border": "#c9c9c9",
    "text": "#1a1a1a",
    "muted": "#6b6b6b",
    "button": "#e8e8e8",
    "button_hover": "#dcdcdc",
    "primary_bg": "#1a1a1a",
    "primary_text": "#ffffff",
    "accent": "#2f7de1",
    "accent_green": "#2e9e57",
    "accent_red": "#d13438",
    "selection": "#2f5cb8",
}


def _palette(colors: dict[str, str]) -> QPalette:
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(colors["window"]))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(colors["text"]))
    pal.setColor(QPalette.ColorRole.Base, QColor(colors["input"]))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(colors["panel"]))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(colors["panel"]))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(colors["text"]))
    pal.setColor(QPalette.ColorRole.Text, QColor(colors["text"]))
    pal.setColor(QPalette.ColorRole.Button, QColor(colors["button"]))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(colors["text"]))
    pal.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(colors["selection"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, QColor(colors["accent"]))
    return pal


def _stylesheet(colors: dict[str, str]) -> str:
    return f"""
    QWidget {{ font-size: 13px; }}
    QMainWindow, QDialog {{ background: {colors["window"]}; }}
    QLabel {{ color: {colors["text"]}; }}
    QLabel[muted="true"] {{ color: {colors["muted"]}; }}
    QLabel[title="true"] {{ font-weight: 600; font-size: 13px; }}
    QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextEdit {{
        background: {colors["input"]};
        border: 1px solid {colors["input_border"]};
        border-radius: 6px; padding: 5px 8px;
        color: {colors["text"]};
        selection-background-color: {colors["selection"]};
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus {{
        border: 1px solid {colors["accent"]};
    }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox::down-arrow {{ image: none; border-left: 4px solid transparent;
        border-right: 4px solid transparent; border-top: 5px solid {colors["muted"]};
        margin-right: 6px; }}
    QComboBox QAbstractItemView {{
        background: {colors["panel"]}; color: {colors["text"]};
        border: 1px solid {colors["input_border"]};
        selection-background-color: {colors["selection"]};
    }}
    QPushButton {{
        background: {colors["button"]}; color: {colors["text"]};
        border: 1px solid {colors["input_border"]};
        border-radius: 6px; padding: 6px 12px;
    }}
    QPushButton:hover {{ background: {colors["button_hover"]}; }}
    QPushButton:disabled {{ color: {colors["muted"]}; }}
    QPushButton[primary="true"] {{
        background: {colors["primary_bg"]}; color: {colors["primary_text"]};
        border: 1px solid {colors["primary_bg"]}; font-weight: 600;
    }}
    QPushButton[primary="true"]:disabled {{
        background: {colors["button"]}; color: {colors["muted"]};
        border: 1px solid {colors["input_border"]};
    }}
    QPushButton[flat="true"] {{ border: none; background: transparent; }}
    QFrame[panel="true"] {{
        background: {colors["panel"]};
        border: 1px solid {colors["panel_border"]}; border-radius: 8px;
    }}
    QFrame[sidebar="true"] {{ background: {colors["window_alt"]}; border: none; }}
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; }}
    QScrollBar::handle:vertical {{
        background: {colors["input_border"]}; border-radius: 5px; min-height: 30px;
    }}
    QSlider::groove:horizontal {{ height: 4px; background: {colors["input_border"]}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: {colors["text"]}; width: 12px; height: 12px;
        margin: -4px 0; border-radius: 6px;
    }}
    QListWidget {{
        background: transparent; border: none; outline: none;
        color: {colors["text"]};
    }}
    QListWidget::item {{ border-radius: 6px; padding: 6px 8px; }}
    QListWidget::item:selected {{ background: {colors["button"]}; }}
    QTabWidget::pane {{ border: none; }}
    QTabBar::tab {{ padding: 6px 10px; color: {colors["muted"]}; }}
    QTabBar::tab:selected {{ color: {colors["text"]}; border-bottom: 2px solid {colors["text"]}; }}
    """


def _is_system_dark(app: QApplication) -> bool:
    hints = app.styleHints()
    scheme = hints.colorScheme()
    return scheme == Qt.ColorScheme.Dark


def apply_theme(app: QApplication, appearance: str) -> dict[str, str]:
    """Apply ``appearance`` ("system" | "light" | "dark"); return the colors used.

    "system" follows the OS color scheme; anything else is explicit.
    Dark is the app default.
    """
    app.setStyle("Fusion")
    if appearance == "light":
        colors = LIGHT
    elif appearance == "system":
        colors = DARK if _is_system_dark(app) else LIGHT
    else:  # "dark" (default)
        colors = DARK
    app.setPalette(_palette(colors))
    app.setStyleSheet(_stylesheet(colors))
    return colors


def colors_for(appearance: str, app: QApplication | None = None) -> dict[str, str]:
    """Resolve the color dict for an appearance without applying it."""
    if appearance == "light":
        return LIGHT
    if appearance == "system" and app is not None and not _is_system_dark(app):
        return LIGHT
    return DARK
