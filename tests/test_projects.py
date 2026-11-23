"""Tests: projects store v2 (migration, clips, status, thumbnails)."""

from __future__ import annotations

import json
from pathlib import Path

from clipforge.core.projects import (
    SCHEMA_VERSION,
    Project,
    ProjectStore,
    migrate_store_data,
)


def _write_store(tmp_path, payload: dict) -> Path:
    path = tmp_path / "projects.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_v1_store_migrates_to_v2(tmp_path) -> None:
    path = _write_store(tmp_path, {
        "version": 1,
        "projects": [
            {"name": "old", "duration": "1:00", "badge": "9:16",
             "source_path": "/tmp/old.mp4"},
        ],
    })
    store = ProjectStore(path)
    assert len(store.projects) == 1
    project = store.projects[0]
    assert project.name == "old"
    assert project.status == "new"
    assert project.clips == []
    assert project.source_duration == 0.0
    assert project.thumbnail == ""
    assert project.export_dir == ""
    # The migrated file is persisted as v2.
    assert json.loads(path.read_text())["version"] == SCHEMA_VERSION


def test_migrate_store_data_is_idempotent() -> None:
    data = {"version": 1, "projects": [{"name": "x"}]}
    migrated = migrate_store_data(data)
    assert migrated["version"] == SCHEMA_VERSION
    assert migrate_store_data(migrated)["version"] == SCHEMA_VERSION


def test_v2_roundtrip_keeps_clips_and_status(tmp_path) -> None:
    path = tmp_path / "projects.json"
    store = ProjectStore(path)  # fresh store starts empty (no seeding)
    project = Project(
        name="clip-test",
        source_path="/tmp/clip.mp4",
        duration="0:20",
        clips=[{"start": 2.0, "end": 6.0, "source": "manual", "label": "Clip 1"}],
        status="clips_ready",
        source_duration=20.0,
        thumbnail="/tmp/thumb.jpg",
        export_dir="/tmp/exports/clip-test",
    )
    store.add(project)
    reloaded = ProjectStore(path)
    found = next(p for p in reloaded.projects if p.name == "clip-test")
    assert found.status == "clips_ready"
    assert found.clips == [
        {"start": 2.0, "end": 6.0, "source": "manual", "label": "Clip 1"}
    ]
    assert found.source_duration == 20.0
    assert found.thumbnail == "/tmp/thumb.jpg"
    assert found.export_dir == "/tmp/exports/clip-test"


def test_corrupt_store_is_backed_up_and_reset(tmp_path) -> None:
    path = _write_store(tmp_path, {"version": 2, "projects": []})
    path.write_text("{not json", encoding="utf-8")
    store = ProjectStore(path)
    assert store.projects == [], "corrupt store falls back to an empty store"
    assert path.with_suffix(".corrupt.json").exists()


def test_store_search_filters_by_name(tmp_path) -> None:
    path = tmp_path / "projects.json"
    store = ProjectStore(path)
    store.projects = [
        Project(name="podcast-ep1"),
        Project(name="sermon-friday"),
        Project(name="podcast-ep2"),
    ]
    assert [p.name for p in store.search("podcast")] == ["podcast-ep1", "podcast-ep2"]
    assert [p.name for p in store.search("SERMON")] == ["sermon-friday"]
    assert len(store.search("")) == 3
