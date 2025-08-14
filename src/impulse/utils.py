from __future__ import annotations

import json
from pathlib import Path
from importlib import resources
from typing import Any, Dict


def get_mission_schema() -> Dict[str, Any]:
    """Return the impulse mission JSON schema as a dict.

    Attempts to load from packaged resources (warp_bubble_optimizer/schemas).
    Falls back to repo path `schemas/impulse.mission.v1.json` for dev checkouts.
    """
    # Try package resource
    try:
        with resources.files('warp_bubble_optimizer').joinpath('schemas/impulse.mission.v1.json').open('rb') as f:
            return json.load(f)
    except Exception:
        pass
    # Fallback to repo path
    root = Path(__file__).resolve().parents[2]
    schema_path = root / 'schemas' / 'impulse.mission.v1.json'
    return json.loads(schema_path.read_text())
