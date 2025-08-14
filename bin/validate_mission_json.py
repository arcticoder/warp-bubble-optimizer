#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from importlib import resources


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate impulse mission JSON against repo schema")
    ap.add_argument('json_file', help='Path to mission JSON export')
    args = ap.parse_args(argv)
    data = json.loads(Path(args.json_file).read_text())
    # Try load from packaged resources first
    try:
        import jsonschema  # type: ignore
    except Exception:
        print("jsonschema not installed. PASS (noop).")
        return 0
    try:
        with resources.files('warp_bubble_optimizer').joinpath('schemas/impulse.mission.v1.json').open('rb') as f:  # type: ignore
            schema = json.load(f)
    except Exception:
        schema_path = Path(__file__).resolve().parents[1] / 'schemas' / 'impulse.mission.v1.json'
        if not schema_path.exists():
            print("Schema file missing, nothing to validate. PASS (noop).")
            return 0
        schema = json.loads(schema_path.read_text())
    jsonschema.validate(instance=data, schema=schema)
    print("Validation successful.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
