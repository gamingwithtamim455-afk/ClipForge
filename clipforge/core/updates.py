"""Self-updates + diagnostics export (Settings > Updates).

Update feed: the GitHub Releases API for the repo configured in
``presets.json`` (``updates.repo``). Nothing here ever crashes when the
machine is offline — a failed check reports "could not check" and the
app carries on.

Install flow: download the installer asset -> verify its SHA-256
checksum (from the ``.sha256`` asset when the release ships one) -> back
up settings + projects -> launch the installer and quit. If any step
fails before the installer runs, the backup is restored (auto-rollback)
and the user gets the exact reason.

Diagnostics export zips the rotating logs and scrubs anything that
looks like a credential (API keys are never logged, but the scrubber is
a second line of defence — the zip must never contain a key).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from .. import __version__
from ..providers import load_presets

__all__ = [
    "UpdateCheck",
    "UpdateInfo",
    "UpdateResult",
    "apply_update",
    "backup_app_state",
    "check_for_updates",
    "download_asset",
    "export_diagnostics",
    "fetch_checksum_text",
    "install_update",
    "restore_app_state",
    "verify_sha256",
]

#: Lines matching these patterns are redacted from diagnostics.
#: Order matters: the whole ``Authorization:`` header value is redacted
#: FIRST, because redacting just the scheme word ("Bearer"/"Token")
#: would destroy the context the scheme patterns need and leave the
#: key itself in the log — that is how a Deepgram ``Token`` key leaked
#: before this fix (the ``Authorization:`` pattern ate "Token" and the
#: value survived). Scheme patterns still run for bare "Bearer x" /
#: "Token x" lines without an ``Authorization:`` prefix.
_SCRUB_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(Authorization:\s*).+", re.IGNORECASE),
    re.compile(r"(Bearer\s+)\S+", re.IGNORECASE),
    re.compile(r"(Token\s+)[A-Za-z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key[\"'\s:=]+)[\"']?[A-Za-z0-9_\-]{8,}", re.IGNORECASE),
)


def scrub_text(text: str) -> str:
    """Redact credential-like values from a log line."""
    for pattern in _SCRUB_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", text)
    return text


@dataclass
class UpdateInfo:
    """One GitHub release newer than the installed version."""

    version: str
    tag: str
    notes: str = ""
    installer_url: str = ""
    installer_name: str = ""
    checksum: str = ""  # hex SHA-256 of the installer ("" = not published)
    checksum_url: str = ""  # .sha256 asset URL ("" = none)


@dataclass
class UpdateResult:
    """Outcome of :func:`apply_update`."""

    installer_path: Path
    backup_path: Path
    outcome: str  # "installer_launched" (Windows) | "installer_downloaded"
    checksum_verified: bool


def apply_update(
    info: UpdateInfo,
    *,
    session: object | None = None,
    app_dir: Path | None = None,
    settings_snapshot: dict | None = None,
    projects_path: Path | None = None,
    install_fn=None,
) -> UpdateResult:
    """One-click update: download -> checksum -> back up -> install.

    This is the single implementation of the update flow, used by the
    Settings > Updates tab. On any failure *before* the installer
    launches, the pre-update backup is restored (auto-rollback) and a
    :class:`RuntimeError` explains what happened — a failed update never
    leaves the app in a half-updated state, and a bad checksum never
    reaches the installer. Once the installer itself launches (Windows)
    it owns the install, so rollback no longer applies.
    """
    from .paths import app_data_dir  # noqa: PLC0415 - local, avoids a cycle

    base = app_dir or app_data_dir()
    projects = projects_path or (base / "projects.json")
    tmp_dir = base / "updates"
    installer_path = tmp_dir / (info.installer_name or "ClipForge-Setup.exe")
    download_asset(info.installer_url, installer_path, session=session)
    checksum_verified = False
    if info.checksum_url:
        expected = fetch_checksum_text(info.checksum_url, session=session)
        if not verify_sha256(installer_path, expected):
            installer_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Downloaded installer failed the checksum check — "
                "aborting the update."
            )
        checksum_verified = True
    backup = backup_app_state(
        projects, settings_snapshot or {}, base / "backups"
    )
    install = install_fn or install_update
    try:
        outcome = install(installer_path)
    except Exception as exc:
        restore_app_state(backup, projects)
        raise RuntimeError(
            f"Installer failed ({exc}) — previous version restored."
        ) from exc
    return UpdateResult(
        installer_path=installer_path,
        backup_path=backup,
        outcome=outcome,
        checksum_verified=checksum_verified,
    )


@dataclass
class UpdateCheck:
    """Outcome of :func:`check_for_updates` (never raises)."""

    available: bool | None  # True = update found, False = up to date, None = could not check
    message: str
    info: UpdateInfo | None = None


def _version_tuple(version: str) -> tuple[int, ...]:
    """``"1.2.3"`` -> ``(1, 2, 3)`` (leading ``v`` stripped, non-numeric dropped)."""
    parts: list[int] = []
    for chunk in version.lstrip("vV").split("."):
        digits = re.sub(r"\D", "", chunk)
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def _is_newer(candidate: str, current: str) -> bool:
    return _version_tuple(candidate) > _version_tuple(current)


def _releases_url(repo: str) -> str:
    presets = load_presets()
    template = presets.get("updates", {}).get(
        "releases_api",
        "https://api.github.com/repos/{repo}/releases/latest",
    )
    return template.format(repo=repo)


def update_repo() -> str:
    """Configured GitHub repo for the update feed (``owner/name``)."""
    return str(load_presets().get("updates", {}).get("repo", "clipforge/clipforge"))


def check_for_updates(
    repo: str | None = None,
    current_version: str | None = None,
    session: object | None = None,
    timeout: float = 10.0,
) -> UpdateCheck:
    """Check the GitHub Releases feed for a newer version.

    Never raises: offline / bad responses come back as
    ``available=None`` with a plain-English message.
    """
    repo = repo or update_repo()
    current = current_version or __version__
    sess = session if session is not None else requests.Session()
    try:
        response = sess.get(
            _releases_url(repo),
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": f"ClipForge/{current}",
            },
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError:
        return UpdateCheck(None, "Could not check for updates (no internet connection).")
    except requests.exceptions.Timeout:
        return UpdateCheck(None, "Update check timed out — try again later.")
    except requests.exceptions.RequestException as exc:
        return UpdateCheck(None, f"Could not check for updates ({exc}).")
    if response.status_code == 404:
        return UpdateCheck(None, f"No releases found for {repo} yet.")
    if response.status_code >= 400:
        return UpdateCheck(None, f"Update check failed (HTTP {response.status_code}).")
    try:
        data = response.json()
    except ValueError:
        return UpdateCheck(None, "Update check got an unreadable response.")
    tag = str(data.get("tag_name") or "")
    version = tag.lstrip("vV")
    if not version or not _is_newer(version, current):
        return UpdateCheck(False, f"ClipForge {current} is up to date.")
    assets = data.get("assets", []) or []
    installer = next(
        (a for a in assets if str(a.get("name", "")).lower().endswith((".exe", ".msi"))),
        None,
    )
    checksum_asset = next(
        (a for a in assets if str(a.get("name", "")).lower().endswith(".sha256")),
        None,
    )
    if installer is None:
        return UpdateCheck(
            True,
            f"ClipForge {version} is available, but no Windows installer "
            "was attached to the release.",
            UpdateInfo(version=version, tag=tag, notes=str(data.get("body") or "")),
        )
    return UpdateCheck(
        True,
        f"ClipForge {version} is available.",
        UpdateInfo(
            version=version,
            tag=tag,
            notes=str(data.get("body") or ""),
            installer_url=str(installer.get("browser_download_url", "")),
            installer_name=str(installer.get("name", "")),
            checksum_url=(
                str(checksum_asset.get("browser_download_url", ""))
                if checksum_asset
                else ""
            ),
        ),
    )


def download_asset(
    url: str,
    dest: Path,
    *,
    session: object | None = None,
    timeout: float = 300.0,
    progress=None,
) -> Path:
    """Download a release asset (streamed); raises on failure."""
    sess = session if session is not None else requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sess.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with dest.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if progress is not None and total:
                        progress(int(100 * done / total), f"Downloading… {done // (1 << 20)} MB")
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError("No internet connection — download failed.") from exc
    return dest


def fetch_checksum_text(url: str, session: object | None = None, timeout: float = 30.0) -> str:
    """Download a ``.sha256`` asset and return its first token (the hex digest)."""
    sess = session if session is not None else requests.Session()
    response = sess.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text.strip().split()[0]


def verify_sha256(path: Path, expected_hex: str) -> bool:
    """True when the file's SHA-256 matches ``expected_hex`` (case-insensitive)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected_hex.strip().lower()


def backup_app_state(
    projects_path: Path,
    settings_snapshot: dict,
    backup_dir: Path,
) -> Path:
    """Zip projects.json + a settings snapshot; returns the zip path."""
    from datetime import datetime  # noqa: PLC0415 - local, keeps import light

    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    target = backup_dir / f"clipforge-state-{stamp}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        if projects_path.exists():
            zf.write(projects_path, arcname="projects.json")
        zf.writestr("settings.json", json.dumps(settings_snapshot, indent=2))
    return target


def restore_app_state(backup_zip: Path, projects_path: Path) -> dict:
    """Restore projects.json + settings from a state backup zip.

    Returns the settings dict that was stored in the backup (the caller
    re-applies it); projects.json is written back in place.
    """
    with zipfile.ZipFile(backup_zip, "r") as zf:
        names = zf.namelist()
        if "projects.json" in names:
            projects_path.parent.mkdir(parents=True, exist_ok=True)
            projects_path.write_bytes(zf.read("projects.json"))
        settings: dict = {}
        if "settings.json" in names:
            settings = json.loads(zf.read("settings.json").decode("utf-8"))
    return settings


def install_update(installer_path: Path) -> str:
    """Launch the installer and return what happened.

    Windows: the installer runs and the app is expected to quit right
    after (the installer replaces the running .exe). Other platforms:
    the installer path is returned for the user to run by hand.
    """
    if not installer_path.exists():
        raise FileNotFoundError(f"Installer not found: {installer_path}")
    if sys.platform.startswith("win"):
        subprocess.Popen([str(installer_path)], close_fds=True)  # noqa: S603 - local file
        return "installer_launched"
    return "installer_downloaded"


def export_diagnostics(log_dir: Path, output_path: Path) -> Path:
    """Zip the rotating logs (scrubbed) + a small system-info file.

    Never includes API keys: logs never contain them, and every line is
    scrubbed once more on the way into the zip.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = (
        f"ClipForge {__version__}\n"
        f"platform: {sys.platform}\n"
        f"python: {sys.version.split()[0]}\n"
    )
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("system-info.txt", info)
        if log_dir.exists():
            for log_file in sorted(log_dir.glob("*.log*")):
                try:
                    text = log_file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                zf.writestr(log_file.name, scrub_text(text))
    return output_path
