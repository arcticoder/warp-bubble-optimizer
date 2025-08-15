from __future__ import annotations
import subprocess
import sys
from pathlib import Path

def validate_zip(path: str) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False
    try:
        # Use the `zip -T` test if available; otherwise fallback to python zipfile
        res = subprocess.run(["zip", "-T", str(p)], capture_output=True)
        if res.returncode == 0:
            return True
    except Exception:
        pass
    try:
        import zipfile
        with zipfile.ZipFile(p, 'r') as zf:
            bad = zf.testzip()
            return bad is None
    except Exception:
        return False

if __name__ == "__main__":
    ok = validate_zip(sys.argv[1] if len(sys.argv) > 1 else "40eridani-artifacts.zip")
    print("OK" if ok else "CORRUPT")
    sys.exit(0 if ok else 1)
