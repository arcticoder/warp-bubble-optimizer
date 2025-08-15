#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', required=True)
    ap.add_argument('--out', default='uq_tiny.png')
    args = ap.parse_args()
    energies = []
    for line in Path(args.records).read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if 'planned_energy' in rec:
            energies.append(float(rec['planned_energy']))
    if not energies:
        raise SystemExit("No energies found in records")
    plt.figure(figsize=(6,4))
    plt.hist(energies, bins=min(15, max(3, len(energies)//3)), color='#1f77b4', alpha=0.8)
    plt.xlabel('Planned Energy (J)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
