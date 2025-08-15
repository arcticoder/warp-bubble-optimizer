from __future__ import annotations
import csv
from pathlib import Path


def has_distance_column(path: str) -> bool:
    p = Path(path)
    with p.open('r', newline='') as f:
        r = csv.reader(f)
        header = next(r, [])
        return len(header) > 0 and header[0].strip().lower() == 'distance'
