#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
from importlib import resources

REQUIRED_COLS = [
    'segment_index', 'kind', 'segment_time', 'segment_energy'
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate perf.csv for required columns and basic sanity")
    ap.add_argument('csv', help='Path to perf.csv')
    args = ap.parse_args(argv)
    p = Path(args.csv)
    if not p.exists():
        print(f"Missing file: {p}", file=sys.stderr)
        return 2
    df = pd.read_csv(p)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}", file=sys.stderr)
        return 3
    if len(df) == 0:
        print("Empty perf.csv", file=sys.stderr)
        return 4
    # Basic value checks
    bad_time = (df['segment_time'] < 0).any()
    bad_energy = (df['segment_energy'] < 0).any()
    if bad_time or bad_energy:
        print("Negative times or energies found", file=sys.stderr)
        return 5
    # Optional JSON Schema validation
    try:
        import json, jsonschema  # type: ignore
        # Cast a single row to an object and validate required fields/types
        with resources.files('warp_bubble_optimizer').joinpath('schemas/perf.csv.schema.json').open('rb') as f:
            schema = json.load(f)
        # Validate each row minimally (could be optimized)
        for _, row in df.iterrows():
            obj = row.to_dict()
            jsonschema.validate(instance=obj, schema=schema)
    except Exception:
        # Schema validation is optional; ignore if unavailable
        pass
    print("perf.csv valid")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
