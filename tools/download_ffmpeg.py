"""Download a static FFmpeg build into ``clipforge/bin/``.

Windows (first target): gyan.dev essentials build (ffmpeg + ffprobe).
Linux: johnvansickle.com static build.

Usage:
    python tools/download_ffmpeg.py [--platform win|linux] [--dest clipforge/bin]

The binaries are NOT committed to git (see .gitignore); the Windows CI
workflow runs this script before the PyInstaller build.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import requests

WINDOWS_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
LINUX_URL = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"


def _download(url: str) -> bytes:
    print(f"Downloading {url} ...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def _install_windows(data: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if name.endswith(("ffmpeg.exe", "ffprobe.exe")):
                target = dest / Path(name).name
                target.write_bytes(zf.read(name))
                print(f"  -> {target}")


def _install_linux(data: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as tf:
        for member in tf.getmembers():
            if member.name.endswith(("/ffmpeg", "/ffprobe")) and member.isfile():
                target = dest / Path(member.name).name
                target.write_bytes(tf.extractfile(member).read())  # type: ignore[union-attr]
                target.chmod(0o755)
                print(f"  -> {target}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["win", "linux"],
                        default="win" if sys.platform.startswith("win") else "linux")
    parser.add_argument("--dest", default=str(Path(__file__).resolve().parent.parent / "clipforge" / "bin"))
    args = parser.parse_args(argv)
    dest = Path(args.dest)
    if args.platform == "win":
        _install_windows(_download(WINDOWS_URL), dest)
    else:
        _install_linux(_download(LINUX_URL), dest)
    print("FFmpeg ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
