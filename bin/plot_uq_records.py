#!/usr/bin/env python3
"""
Utility to plot UQ JSONL records and optionally compute simple metrics.

Inputs:
- --records: Path to JSONL file where each line is a JSON record with at least
  keys 'planned_energy' (float) and 'feasible' (bool) when present.
- --energy-out: Output PNG for energy histogram.
- --feas-out: Output PNG for rolling feasible fraction plot.
- --bins: Histogram bins for energy (default: 20).
- --metrics-out: Optional path to write metrics JSON with fields:
    { energy_mean, energy_std, energy_cv, feasible_fraction }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except Exception:
            # Skip malformed lines
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--energy-out", required=True)
    ap.add_argument("--feas-out", required=True)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--metrics-out", default=None)
    args = ap.parse_args()

    recs = read_records(Path(args.records))
    energies = [float(r.get("planned_energy", 0.0)) for r in recs if "planned_energy" in r]
    feas_vals = [1.0 if bool(r.get("feasible")) else 0.0 for r in recs if "feasible" in r]

    # Energy histogram
    if energies:
        plt.figure(figsize=(6, 4))
        plt.hist(energies, bins=max(1, int(args.bins)), color="#4C78A8", edgecolor="white")
        plt.title("Planned Energy Distribution")
        plt.xlabel("Energy (J)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(args.energy_out, dpi=150)
        plt.close()

    # Rolling feasible fraction
    if feas_vals:
        n = len(feas_vals)
        win = max(1, n // 10)
        roll = np.convolve(feas_vals, np.ones(win) / win, mode="same")
        plt.figure(figsize=(6, 4))
        plt.plot(roll, color="#F58518")
        plt.ylim(0, 1)
        plt.title("Feasible Fraction (rolling)")
        plt.xlabel("Sample Index")
        plt.ylabel("Feasible Fraction")
        plt.tight_layout()
        plt.savefig(args.feas_out, dpi=150)
        plt.close()

    if args.metrics_out:
        energy_mean = float(np.mean(energies)) if energies else float("nan")
        energy_std = float(np.std(energies)) if energies else float("nan")
        energy_cv = float(energy_std / energy_mean) if energies and energy_mean else float("nan")
        feasible_fraction = float(np.mean(feas_vals)) if feas_vals else float("nan")
        Path(args.metrics_out).write_text(
            json.dumps(
                {
                    "energy_mean": energy_mean,
                    "energy_std": energy_std,
                    "energy_cv": energy_cv,
                    "feasible_fraction": feasible_fraction,
                }
            )
        )


if __name__ == "__main__":
    main()
