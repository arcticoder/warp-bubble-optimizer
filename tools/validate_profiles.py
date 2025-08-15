from __future__ import annotations
import csv
from pathlib import Path


def has_distance_column(path: str) -> bool:
    p = Path(path)
    with p.open('r', newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            first = row[0].strip().lower()
            if first.startswith('#'):
                continue
            return first == 'distance'
    return False
