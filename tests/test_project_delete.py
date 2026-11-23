"""Project deletion (v0.2.1 F): the Projects home can delete a project.

Deleting removes the store entry and the project's OWN generated
files (rendered clips folder + thumbnail) — never the user's original
source video — and only after a confirmation dialog that says so.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox

from clipforge.core.projects import Project, ProjectStore, delete_project_files


def _make_project(tmp_path: Path) -> tuple[Project, Path, Path, Path]:
    """A project with a real source file, thumbnail and output folder."""
    source = tmp_path / "original.mp4"
    source.write_bytes(b"source-video-bytes")
    thumb_dir = tmp_path / "thumbnails"
    thumb_dir.mkdir()
    thumb = thumb_dir / "p1.jpg"
    thumb.write_bytes(b"jpeg")
    out_dir = tmp_path / "exports" / "My_Project"
    out_dir.mkdir(parents=True)
    (out_dir / "01_clip.mp4").write_bytes(b"rendered")
    (out_dir / "01_label.txt").write_text("caption", encoding="utf-8")
    project = Project(
        name="My Project", source_path=str(source),
        thumbnail=str(thumb), export_dir=str(out_dir),
    )
    return project, source, thumb, out_dir


def test_delete_project_files_removes_outputs_keeps_source(tmp_path) -> None:
    project, source, thumb, out_dir = _make_project(tmp_path)
    removed = delete_project_files(project)
    assert not thumb.exists()
    assert not out_dir.exists()
    assert source.exists() and source.read_bytes() == b"source-video-bytes"
    assert thumb in removed and out_dir in removed


def test_delete_project_files_never_deletes_source_inside_output_dir(tmp_path) -> None:
    """Safety: if the source lives inside the output folder, keep both."""
    out_dir = tmp_path / "exports" / "Proj"
    out_dir.mkdir(parents=True)
    source = out_dir / "original.mp4"
    source.write_bytes(b"precious")
    project = Project(
        name="Proj", source_path=str(source), export_dir=str(out_dir),
    )
    delete_project_files(project)
    assert source.exists()
    assert out_dir.exists()


def test_store_remove_drops_entry(tmp_path) -> None:
    store = ProjectStore(tmp_path / "projects.json")
    project, *_ = _make_project(tmp_path)
    store.add(project)
    assert [p.id for p in store.projects] == [project.id]
    store.remove(project.id)
    assert store.projects == []
    reloaded = ProjectStore(tmp_path / "projects.json")
    assert reloaded.projects == []


@pytest.fixture(scope="module")
def window(app_window):
    yield app_window


def test_home_delete_flow_requires_confirmation(window, tmp_path, monkeypatch) -> None:
    project, source, thumb, out_dir = _make_project(tmp_path)
    window._store.add(project)
    home = window._home
    home.refresh()
    card = next(c for c in home._cards if c.project.id == project.id)
    assert card._delete_btn.text()  # a visible Delete button exists

    # Confirmation DECLINED: nothing is removed.
    monkeypatch.setattr(
        "clipforge.ui.home.QMessageBox.question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )
    card._delete_btn.click()
    assert any(p.id == project.id for p in window._store.projects)
    assert out_dir.exists() and thumb.exists() and source.exists()

    # Confirmation ACCEPTED: entry + generated files gone, source kept.
    seen = {}

    def _question(_self, _title, text, *_a, **_k):
        seen["text"] = text
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(
        "clipforge.ui.home.QMessageBox.question", staticmethod(_question)
    )
    card._delete_btn.click()
    assert "will not be deleted" in seen["text"]  # the dialog says so
    assert not any(p.id == project.id for p in window._store.projects)
    assert not out_dir.exists() and not thumb.exists()
    assert source.exists()
