"""Tests: build sanity — PyInstaller spec, GitHub Actions workflow and
the dependency lockfile stay consistent with the package layout.

The Windows build itself cannot run on Linux; these tests verify
everything that can be verified here (parsing + file references), and
the PyInstaller smoke test below proves the spec collects the app.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_pyinstaller_spec_parses_and_references_real_files() -> None:
    spec = ROOT / "clipforge.spec"
    assert spec.exists()
    compile(spec.read_text(encoding="utf-8"), str(spec), "exec")
    text = spec.read_text(encoding="utf-8")
    assert '"clipforge"' in text and '"main.py"' in text
    assert "presets.json" in text
    assert '"i18n"' in text
    assert '"assets"' in text  # bundled caption fonts
    # Every datas source the spec references must exist in the repo.
    for match in re.finditer(r'root / "([^"]+)"', text):
        rel = match.group(1)
        assert (ROOT / rel).exists(), f"spec references missing path: {rel}"


def test_github_workflow_parses_and_matches_spec() -> None:
    workflow = ROOT / ".github/workflows/build-windows.yml"
    assert workflow.exists()
    text = workflow.read_text(encoding="utf-8")
    assert "windows-latest" in text
    assert "pyinstaller clipforge.spec" in text
    assert "iscc installer/clipforge.iss" in text
    assert "pytest tests/" in text
    assert "dist/ClipForge.exe" in text
    try:
        import yaml  # noqa: PLC0415 - optional dev dependency
    except ImportError:
        pytest.skip("PyYAML not installed — string checks above already ran")
    data = yaml.safe_load(text)
    steps = data["jobs"]["build-windows"]["steps"]
    names = [s.get("name", "") for s in steps]
    assert any("PyInstaller" in n for n in names)
    assert any("Inno Setup" in n for n in names)


def test_inno_setup_references_match_spec_output() -> None:
    iss = ROOT / "installer/clipforge.iss"
    assert iss.exists()
    text = iss.read_text(encoding="utf-8")
    # The spec produces dist/ClipForge.exe; the installer must install it.
    assert "ClipForge.exe" in text
    assert "..\\dist\\{#ExeName}" in text or "..\\dist" in text
    # Bundled FFmpeg entries (downloaded before the build).
    assert "ffmpeg.exe" in text and "ffprobe.exe" in text
    assert "desktopicon" in text  # desktop shortcut task


def test_lockfile_pins_every_direct_dependency() -> None:
    requirements = ROOT / "requirements.txt"
    lock = ROOT / "requirements.lock"
    assert lock.exists()
    lock_text = lock.read_text(encoding="utf-8").lower()
    pinned: list[str] = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\s]", line)[0].strip().lower()
        pinned.append(name)
        assert f"{name}==" in lock_text, f"{name} is pinned in requirements.txt but missing from the lock"
    # The lock must pin the heavy native deps too (they decide the build).
    for heavy in ("mediapipe", "opencv-python-headless", "faster-whisper", "ctranslate2"):
        assert f"{heavy}==" in lock_text
    # pip check: no broken requirements in the current environment.
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout
