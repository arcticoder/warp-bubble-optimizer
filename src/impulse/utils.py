from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def get_mission_schema() -> Dict[str, Any]:
    """Return the impulse mission JSON schema as a dict.

    Looks for schemas/impulse.mission.v1.json relative to project root.
    Raises FileNotFoundError if missing.
    """
    root = Path(__file__).resolve().parents[2]
    schema_path = root / 'schemas' / 'impulse.mission.v1.json'
    data = json.loads(schema_path.read_text())
    return data
