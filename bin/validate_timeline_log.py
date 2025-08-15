#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, json, csv
from pathlib import Path
from importlib import resources

def validate(path: str) -> int:
    p = Path(path)
    if not p.exists():
        print(f"Missing file: {p}", file=sys.stderr)
        return 2
    ok = True
    errs: list[str] = []
    try:
        if p.suffix.lower() == '.jsonl':
            with p.open('r') as fh:
                schema = json.load(resources.files('warp_bubble_optimizer').joinpath('schemas/timeline.log.schema.json').open('rb'))
                import jsonschema  # type: ignore
                for i, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        jsonschema.validate(instance=obj, schema=schema)
                    except Exception as e:
                        ok = False
                        errs.append(f"Line {i}: {e}")
        else:
            # CSV mode: minimal header/field checks
            with p.open('r', newline='') as fh:
                reader = csv.DictReader(fh)
                required = ['iso_time','t_rel_s','event','segment_id','planned_value','actual_value']
                missing = [c for c in required if c not in (reader.fieldnames or [])]
                if missing:
                    ok = False
                    errs.append(f"Missing CSV columns: {missing}")
                for i, row in enumerate(reader, 2):
                    if not row.get('event'):
                        ok = False
                        errs.append(f"Row {i}: empty event")
    except Exception as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 3
    if not ok:
        print("\n".join(errs), file=sys.stderr)
        return 4
    print("timeline log valid")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    args = ap.parse_args(argv)
    return validate(args.path)

if __name__ == '__main__':
    raise SystemExit(main())
