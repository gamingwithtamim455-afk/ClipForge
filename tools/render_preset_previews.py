#!/usr/bin/env python3
"""Render one style-preview image per caption preset.

Each preview shows the sample text "a quick brown fox" rendered with
the preset, exactly as the Render step burns it (same build_ass +
FFmpeg libass path), at half resolution (540x960) to keep the bundled
images small. The Settings > Fonts tab shows these as clickable style
preview cards.

Usage (from the repo root):
    TMPDIR=$HOME/.tmp .venv/bin/python tools/render_preset_previews.py

Writes:
    clipforge/assets/previews/<slug>.png  (bundled, shown in the app)
    docs/screenshots/presets/<slug>.png   (documentation copy)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from clipforge.core.captions import (  # noqa: E402
    PRESETS,
    build_ass,
    fonts_dir,
    preview_slug,
    previews_dir,
)
from clipforge.core.ffmpeg import find_ffmpeg  # noqa: E402
from clipforge.core.transcription import Word  # noqa: E402

SAMPLE = ["a", "quick", "brown", "fox"]
PREVIEW_W, PREVIEW_H = 540, 960
FRAME_AT = 0.6  # inside "quick" (0.4-0.8s): every preset shows text


def _words() -> list[Word]:
    return [
        Word(text, i * 0.4, (i + 1) * 0.4)
        for i, text in enumerate(SAMPLE)
    ]


def render_preview(name: str, out: Path) -> None:
    """Render one preset's preview frame to ``out`` (PNG)."""
    ass_text = build_ass(
        _words(), preset_name=name, width=PREVIEW_W, height=PREVIEW_H
    )
    ass_path = out.with_suffix(".ass")
    ass_path.write_text(ass_text, encoding="utf-8")
    subprocess.run(
        [
            find_ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i",
            f"color=c=black:s={PREVIEW_W}x{PREVIEW_H}:r=30:d=2",
            "-vf", f"ass={ass_path}:fontsdir={fonts_dir()}",
            "-ss", str(FRAME_AT), "-frames:v", "1", str(out),
        ],
        check=True,
    )
    ass_path.unlink(missing_ok=True)


def main() -> int:
    dest = previews_dir()
    dest.mkdir(parents=True, exist_ok=True)
    docs = REPO / "docs" / "screenshots" / "presets"
    docs.mkdir(parents=True, exist_ok=True)
    for name in PRESETS:
        slug = preview_slug(name)
        out = dest / f"{slug}.png"
        render_preview(name, out)
        doc_copy = docs / f"{slug}.png"
        doc_copy.write_bytes(out.read_bytes())
        print(f"{name:16s} -> {out.relative_to(REPO)} ({out.stat().st_size} bytes)")
    print(f"\n{len(PRESETS)} previews written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
