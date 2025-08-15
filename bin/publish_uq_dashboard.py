#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Copy key UQ artifacts into public dashboard")
    ap.add_argument('--artifacts', default='artifacts')
    ap.add_argument('--public', default='public')
    args = ap.parse_args(argv)
    a = Path(args.artifacts)
    pub = Path(args.public)
    pub.mkdir(parents=True, exist_ok=True)
    # Copy a curated set
    for name in [
        '40eridani_energy.png',
        '40eridani_feasibility.png',
        '40eridani_energy_extended.png',
        '40eridani_feasibility_extended.png',
        '40eridani_energy_varied.png',
        '40eridani_feasibility_varied.png',
        '40eridani_uq_tiny.png',
        'feasibility_gates_report.json',
        'perf_summary.png',
        'perf_smoke.png'
    ]:
        src = a / name
        if src.exists():
            shutil.copy2(src, pub / name)
    print(f"Dashboard published to {pub}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
