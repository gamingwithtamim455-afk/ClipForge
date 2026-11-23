"""Projects: data model + versioned JSON store with automatic migrations.

A project is one source video plus its clips and settings. Projects live in
the app data directory (``QStandardPaths.AppDataLocation`` / ``ClipForge``)
as ``projects.json`` (schema versioned; migrations run automatically).

Schema history:
- v1: name, source_path, duration, badge, cutout, sample, created_at.
- v2: adds ``clips`` (list of {start, end, source, label} dicts),
  ``status`` ("new" | "clips_ready" | "rendered"), ``source_duration``
  (seconds, float), ``thumbnail`` (path to a JPEG frame) and
  ``export_dir`` (where this project's clips were written).

Weekly backups are handled by :mod:`clipforge.core.backups`.

A fresh store starts EMPTY. (Earlier versions seeded fake sample projects
on first run so the Home grid matched a reference recording — those cards
had no source file and no thumbnail, so most of the grid showed placeholder
art and clicking one led to a file picker instead of a project. That data
bug is why seeding was removed; ``SAMPLE_PROJECTS`` is kept only as a
record of the old demo data.)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2

# Sample projects mirroring the reference recording (names 047-059).
SAMPLE_PROJECTS: list[dict] = [
    {"name": "059", "duration": "1:38:38", "badge": "9:16"},
    {"name": "058", "duration": "54:32", "badge": "9:16"},
    {"name": "057", "duration": "16:40", "badge": "9:16"},
    {"name": "052", "duration": "38:21", "badge": "9:16"},
    {"name": "050", "duration": "32:38", "badge": "9:16"},
    {"name": "051", "duration": "33:00", "badge": "9:16"},
    {"name": "056", "duration": "1:12:11", "badge": "9:16"},
    {"name": "054", "duration": "25:57", "badge": "9:16", "cutout": True},
    {"name": "055", "duration": "1:29:10", "badge": "9:16", "cutout": True},
    {"name": "053", "duration": "1:38:38", "badge": "9:16", "cutout": True},
    {"name": "049", "duration": "38:10", "badge": "9:16", "cutout": True},
    {"name": "047", "duration": "1:12:12", "badge": "9:16", "cutout": True},
]


@dataclass
class Project:
    """One ClipForge project."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    source_path: str = ""
    duration: str = ""  # display string, e.g. "54:32"
    badge: str = "9:16"
    cutout: bool = False
    sample: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # v2 fields (all default so v1 files migrate cleanly).
    clips: list[dict] = field(default_factory=list)  # {start, end, source, label}
    status: str = "new"  # "new" | "clips_ready" | "rendered"
    source_duration: float = 0.0  # seconds
    thumbnail: str = ""  # path to a JPEG frame ("" = placeholder art)
    export_dir: str = ""  # where this project's clips were written

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


def migrate_project_dict(data: dict, from_version: int) -> dict:
    """Migrate one stored project dict to the current schema.

    v1 -> v2: the new fields simply take their defaults (``from_dict``
    fills them in); unknown keys are dropped by ``from_dict``. Kept as an
    explicit function so future migrations have one place to live.
    """
    migrated = dict(data)
    if from_version < 2:
        migrated.setdefault("clips", [])
        migrated.setdefault("status", "new")
        migrated.setdefault("source_duration", 0.0)
        migrated.setdefault("thumbnail", "")
        migrated.setdefault("export_dir", "")
    return migrated


def migrate_store_data(data: dict) -> dict:
    """Migrate a whole ``projects.json`` payload to the current schema."""
    version = int(data.get("version") or 0)
    if version >= SCHEMA_VERSION:
        return data
    return {
        "version": SCHEMA_VERSION,
        "projects": [
            migrate_project_dict(p, version) for p in data.get("projects", [])
        ],
    }


class ProjectStore:
    """Versioned JSON store for projects."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.projects: list[Project] = []
        self._load_or_seed()

    def _load_or_seed(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                stored_version = int(raw.get("version") or 0)
                data = migrate_store_data(raw)
                self.projects = [Project.from_dict(p) for p in data["projects"]]
                # Persist the migrated file so the next load is already v2.
                if stored_version < SCHEMA_VERSION:
                    self.save()
                return
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # Corrupt file: back it up, then start fresh.
                try:
                    backup = self.path.with_suffix(".corrupt.json")
                    backup.write_bytes(self.path.read_bytes())
                except OSError:
                    pass
                self.projects = []
        # Fresh or recovered store: start empty (see module docstring —
        # never seed fake projects into a real store again).
        self.projects = []
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SCHEMA_VERSION,
            "projects": [p.to_dict() for p in self.projects],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add(self, project: Project) -> None:
        self.projects.insert(0, project)
        self.save()

    def remove(self, project_id: str) -> None:
        self.projects = [p for p in self.projects if p.id != project_id]
        self.save()

    def search(self, query: str) -> list[Project]:
        q = query.strip().lower()
        if not q:
            return list(self.projects)
        return [p for p in self.projects if q in p.name.lower()]
