#!/usr/bin/env python3
"""
Plot a quick summary from a perf.csv emitted by mission execution.

Inputs:
  - perf CSV path (default: perf.csv)
  - output image path (default: perf_summary.png)

The CSV is expected to have at least the columns:
  segment_index,kind,segment_time,segment_energy,peak_velocity,total_distance,total_rotation_angle

Outputs:
  - A PNG image with two subplots: segment_time histogram and per-segment energy bar plot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

# Use non-interactive backend for CI
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plot perf.csv summary")
    ap.add_argument("--csv", default="perf.csv", help="Path to perf.csv")
    ap.add_argument("--out", default="perf_summary.png", help="Output PNG path")
    args = ap.parse_args(argv)

    perf_path = Path(args.csv)
    if not perf_path.exists():
        print(f"perf CSV not found: {perf_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(perf_path)
    # Ensure expected columns, but proceed best-effort
    seg_time = df.get("segment_time")
    seg_energy = df.get("segment_energy")
    seg_index = df.get("segment_index", pd.Series(range(len(df))))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if seg_time is not None:
        axes[0].hist(seg_time.dropna().values, bins=15, color="#4C78A8", edgecolor="white")
        axes[0].set_title("Segment Time Distribution")
        axes[0].set_xlabel("time (s)")
        axes[0].set_ylabel("count")
    else:
        axes[0].text(0.5, 0.5, "segment_time missing", ha="center", va="center")
        axes[0].set_axis_off()

    if seg_energy is not None:
        axes[1].bar(seg_index, seg_energy.fillna(0).values, color="#F58518")
        axes[1].set_title("Per-Segment Energy")
        axes[1].set_xlabel("segment index")
        axes[1].set_ylabel("energy (J)")
    else:
        axes[1].text(0.5, 0.5, "segment_energy missing", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote perf summary plot to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
