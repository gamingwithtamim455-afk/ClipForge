# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ClipForge (Windows .exe, one-file, windowed).

Run:  pyinstaller clipforge.spec --noconfirm
Bundled data: provider presets + i18n JSON + bundled FFmpeg (if present).
"""

from pathlib import Path

block_cipher = None
root = Path(SPECPATH)

datas = [
    (str(root / "clipforge" / "providers" / "presets.json"), "clipforge/providers"),
    (str(root / "clipforge" / "i18n"), "clipforge/i18n"),
    # Bundled caption fonts (Noto Sans Bengali / Naskh Arabic / Nastaliq
    # Urdu / Noto Sans) — libass loads them from this directory at render.
    (str(root / "clipforge" / "assets"), "clipforge/assets"),
]
bin_dir = root / "clipforge" / "bin"
if bin_dir.exists():
    for binary in bin_dir.iterdir():
        if binary.is_file():
            datas.append((str(binary), "clipforge/bin"))

a = Analysis(
    [str(root / "main.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        # Heavy native deps PyInstaller's analysis can miss.
        "cv2",
        "mediapipe",
        "faster_whisper",
        "ctranslate2",
        "av",
        "numpy",
        "scenedetect",
        "tqdm",
        "huggingface_hub",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ClipForge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
