from __future__ import annotations
import sys
from pathlib import Path
import json


def validate_profile(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    text = p.read_text().strip()
    distances = []
    # Try JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            distances = [float(x) for x in data]
        elif isinstance(data, dict) and 'distances' in data:
            distances = [float(x) for x in data['distances']]
    except Exception:
        # Fallback CSV-like with comments and optional header
        parts = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ',' in line:
                tokens = [t.strip() for t in line.split(',') if t.strip()]
            else:
                tokens = [line]
            for tok in tokens:
                if tok.lower() in ("distance", "dist", "d"):
                    continue
                if '#' in tok:
                    tok = tok.split('#', 1)[0].strip()
                    if not tok:
                        continue
                try:
                    parts.append(float(tok))
                except ValueError:
                    # Ignore non-numeric tokens
                    pass
        distances = parts
    if not distances:
        raise ValueError("No distances parsed")
    if any(d < 0 for d in distances):
        raise ValueError("Negative distance encountered")
    total = sum(distances)
    if total == 0.0:
        raise ValueError("All distances sum to zero")
    return {
        "count": len(distances),
        "min": min(distances),
        "max": max(distances),
        "sum": total,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: validate_dist_profile.py <path>")
        sys.exit(2)
    info = validate_profile(sys.argv[1])
    print(json.dumps(info))
