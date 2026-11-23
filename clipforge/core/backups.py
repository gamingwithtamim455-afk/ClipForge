"""Weekly project backups.

Projects are backed up automatically once a week into
``<app data>/backups/clipforge-backup-YYYY-MM-DD-HHMM.zip``
(projects.json + a settings snapshot); the 4 newest backups are kept.
"""

from __future__ import annotations

import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

__all__ = [
    "maybe_weekly_backup",
    "prune_backups",
]

_KEEP_BACKUPS = 4
_BACKUP_INTERVAL = timedelta(days=7)


def _backup_stamp(now: datetime) -> str:
    return now.strftime("%Y-%m-%d-%H%M")


def maybe_weekly_backup(
    projects_path: Path,
    settings_snapshot: dict,
    backups_dir: Path,
    *,
    now: datetime | None = None,
) -> Path | None:
    """Create a weekly backup zip when the last one is 7+ days old.

    Returns the new backup path, or ``None`` when a recent backup already
    exists (or there is nothing to back up yet).
    """
    import json  # noqa: PLC0415 - local, keeps import light

    now = now or datetime.now(timezone.utc)
    backups_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(backups_dir.glob("clipforge-backup-*.zip"))
    if existing:
        newest = existing[-1]
        try:
            stamp = newest.stem.replace("clipforge-backup-", "")
            last = datetime.strptime(stamp, "%Y-%m-%d-%H%M").replace(
                tzinfo=timezone.utc
            )
            if now - last < _BACKUP_INTERVAL:
                return None
        except ValueError:
            pass  # an oddly-named file: back up anyway
    if not projects_path.exists() and not settings_snapshot:
        return None
    target = backups_dir / f"clipforge-backup-{_backup_stamp(now)}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        if projects_path.exists():
            zf.write(projects_path, arcname="projects.json")
        zf.writestr("settings.json", json.dumps(settings_snapshot, indent=2))
    prune_backups(backups_dir)
    return target


def prune_backups(backups_dir: Path, keep: int = _KEEP_BACKUPS) -> int:
    """Delete all but the ``keep`` newest backups; returns the count removed."""
    backups = sorted(backups_dir.glob("clipforge-backup-*.zip"))
    removed = 0
    for stale in backups[:-keep] if keep > 0 else backups:
        try:
            stale.unlink()
            removed += 1
        except OSError:
            continue
    return removed
