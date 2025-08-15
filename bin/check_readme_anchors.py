#!/usr/bin/env python3
from __future__ import annotations
import sys, re
from pathlib import Path

HEADINGS = [
    'Mission timeline logging (--timeline-log)',
    'Schemas',
    'Seed Reproducibility and Environment Variables'
]

def main() -> int:
    p = Path('README.md')
    if not p.exists():
        print('README.md not found', file=sys.stderr)
        return 2
    text = p.read_text(encoding='utf-8', errors='ignore')
    missing = [h for h in HEADINGS if re.search(r'^#+\s*' + re.escape(h) + r'\s*$', text, re.M) is None]
    if missing:
        print('Missing headings:', ', '.join(missing), file=sys.stderr)
        return 3
    print('README anchors OK')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
