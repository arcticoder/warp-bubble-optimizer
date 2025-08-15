from __future__ import annotations
from pathlib import Path

def is_png(path: str) -> bool:
    b = Path(path).read_bytes()[:8]
    return b == b'\x89PNG\r\n\x1a\n'
