#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def aggregate(paths: list[Path]) -> dict:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df['__source'] = p.name
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return {"files": 0}
    df = pd.concat(frames, ignore_index=True)
    trans = df[df['kind'] == 'translation'] if 'kind' in df.columns else df
    out = {
        "files": len(paths),
        "segments": int(len(df)),
        "segment_time_mean": float(df['segment_time'].mean()),
        "segment_time_std": float(df['segment_time'].std(ddof=1) if len(df) > 1 else 0.0),
        "segment_energy_mean": float(df['segment_energy'].mean()),
        "segment_energy_std": float(df['segment_energy'].std(ddof=1) if len(df) > 1 else 0.0),
        "translation_peak_velocity_mean": float(trans['peak_velocity'].mean() if 'peak_velocity' in trans.columns else float('nan')),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate one or more perf.csv files and produce summary JSON")
    ap.add_argument('csv', nargs='+', help='Paths to perf.csv files')
    ap.add_argument('--out', default='perf_aggregate.json')
    args = ap.parse_args(argv)
    paths = [Path(p) for p in args.csv]
    summary = aggregate(paths)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
