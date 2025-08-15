from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    cwd = Path.cwd()
    print(f"Plot log helper running in: {cwd}")
    pngs = sorted([p.name for p in cwd.glob('*.png')])
    if pngs:
        print("PNG files in current directory:")
        for p in pngs:
            print(f" - {p}")
    else:
        print("No PNG files found in current directory.")
    # Also log artifacts dir if present
    artifacts = cwd / 'artifacts'
    if artifacts.exists():
        apngs = sorted([p.name for p in artifacts.glob('*.png')])
        if apngs:
            print("PNG files in artifacts/:")
            for p in apngs:
                print(f" - {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
