#!/usr/bin/env python3
"""Render one style-preview image per caption preset.

Each preview shows the sample text "a quick brown fox" rendered with
the preset, exactly as the Render step burns it (same build_ass +
FFmpeg libass path), at half resolution (540x960) to keep the bundled
images small. The Settings > Fonts tab shows these as clickable style
preview cards.

The actual rendering lives in clipforge.core.captions
(``render_preview_card``) — the app itself uses the same function to
generate a card on demand when a preset's PNG is not bundled, so the
committed cards and the generated ones can never drift apart.

Usage (from the repo root):
    TMPDIR=$HOME/.tmp .venv/bin/python tools/render_preset_previews.py

Writes:
    clipforge/assets/previews/<slug>.png  (bundled, shown in the app)
    docs/screenshots/presets/<slug>.png   (documentation copy)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from clipforge.core.captions import (  # noqa: E402
    PRESETS,
    preview_slug,
    previews_dir,
    render_preview_card,
)


def main() -> int:
    dest = previews_dir()
    dest.mkdir(parents=True, exist_ok=True)
    docs = REPO / "docs" / "screenshots" / "presets"
    docs.mkdir(parents=True, exist_ok=True)
    for name in PRESETS:
        slug = preview_slug(name)
        out = dest / f"{slug}.png"
        render_preview_card(name, out)
        doc_copy = docs / f"{slug}.png"
        doc_copy.write_bytes(out.read_bytes())
        print(f"{name:16s} -> {out.relative_to(REPO)} ({out.stat().st_size} bytes)")
    print(f"\n{len(PRESETS)} previews written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
