"""Provider adapters: DeepSeek (chat completions), Deepgram (STT), local Whisper.

Rules are driven by ``docs/provider-research.md`` and ``presets.json``.
Shared retry + error mapping lives in ``providers._http``; keys are never
logged and are stored encrypted via ``core.keys`` (system keyring).
"""

from __future__ import annotations

import json
from pathlib import Path

_PRESETS_PATH = Path(__file__).with_name("presets.json")


def load_presets() -> dict:
    """Load the provider presets JSON (canonical model IDs, endpoints)."""
    with _PRESETS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)
