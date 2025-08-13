#!/usr/bin/env python3
"""Traceability coverage checker.

Verifies every task title/identifier in UQ-TODO.ndjson and VnV-TODO.ndjson
appears at least once in docs/roadmap.ndjson associated_tasks lists.
Prints a summary and returns non-zero exit code if gaps exist.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def load_lines(path: Path):
    lines = []
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            lines.append(obj)
    return lines

def collect_task_strings(tasks):
    out = []
    for t in tasks:
        # Prefer 'task' or 'title' field
        if 'task' in t:
            out.append(t['task'])
        elif 'title' in t:
            out.append(t['title'])
    return out

def collect_roadmap_associated(roadmap):
    assoc = []
    for entry in roadmap:
        ats = entry.get('associated_tasks') or []
        assoc.extend(ats)
    return assoc

def main():
    uq_path = ROOT / 'UQ-TODO.ndjson'
    vnv_path = ROOT / 'VnV-TODO.ndjson'
    roadmap_path = ROOT / 'docs' / 'roadmap.ndjson'
    missing = []
    try:
        uq = load_lines(uq_path)
        vnv = load_lines(vnv_path)
        roadmap = load_lines(roadmap_path)
    except FileNotFoundError as e:
        print(f"ERROR: Missing file: {e}")
        return 2
    roadmap_tasks = collect_roadmap_associated(roadmap)
    uq_titles = collect_task_strings(uq)
    vnv_titles = collect_task_strings(vnv)
    for title in uq_titles:
        if title not in roadmap_tasks:
            missing.append(('UQ', title))
    for title in vnv_titles:
        if title not in roadmap_tasks:
            missing.append(('V&V', title))
    if not missing:
        print("✅ All UQ & V&V tasks accounted for in roadmap associated_tasks.")
        return 0
    print("❌ Missing associations:")
    for cat, title in missing:
        print(f"  [{cat}] {title}")
    print("\nSuggestion: Append missing titles to an appropriate milestone 'associated_tasks' in docs/roadmap.ndjson.")
    return 1

if __name__ == '__main__':
    sys.exit(main())
