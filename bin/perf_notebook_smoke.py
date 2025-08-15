#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Notebook smoke: generate a single perf plot")
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='perf_smoke.png')
    args = ap.parse_args(argv)
    df = pd.read_csv(args.csv)
    plt.figure(figsize=(4,3))
    df.groupby('kind')['segment_energy'].sum().plot(kind='bar', color=['#1f77b4','#ff7f0e','#2ca02c'][:len(df['kind'].unique())])
    plt.ylabel('Total energy (J)')
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote {args.out}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
