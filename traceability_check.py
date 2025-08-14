#!/usr/bin/env python3
"""Simple roadmap ↔ test traceability checker.

Scans `docs/roadmap.ndjson` for V&V / UQ task strings and verifies that
at least one test file references each task substring.

Usage:
    python traceability_check.py [--fail-on-missing]

Exit codes:
  0 - All tasks covered
  1 - Missing coverage (when --fail-on-missing)
  2 - Roadmap not found / parse error

This is intentionally lightweight; extend later with richer JSON schema.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).parent
ROADMAP = ROOT / "docs" / "roadmap.ndjson"
TEST_GLOBS = ["test_*.py"]

TASK_PREFIXES = ["V&V:", "UQ:"]


def load_tasks():
    if not ROADMAP.exists():
        print(f"Roadmap file missing: {ROADMAP}", file=sys.stderr)
        sys.exit(2)
    task_strings = []
    for line in ROADMAP.read_text().splitlines():
        if not line.strip():
            continue
        # naive extract quoted task arrays
        for prefix in TASK_PREFIXES:
            if prefix in line:
                # Split by quotes and look for segments containing the prefix
                for match in re.findall(r'"([^"]+)"', line):
                    if prefix in match:
                        task_strings.append(match.strip())
    # Deduplicate
    uniq = sorted(set(task_strings))
    return uniq


def scan_tests(tasks):
    test_files = []
    for pattern in TEST_GLOBS:
        test_files.extend(ROOT.glob(pattern))
    coverage = {t: False for t in tasks}
    for tf in test_files:
        txt = tf.read_text(errors="ignore")
        for t in tasks:
            if t in txt:
                coverage[t] = True
    return coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-on-missing", action="store_true", help="Return code 1 if any tasks lack coverage")
    ap.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON report of coverage")
    args = ap.parse_args()
    tasks = load_tasks()
    coverage = scan_tests(tasks)
    missing = [t for t, ok in coverage.items() if not ok]
    print(f"Discovered {len(tasks)} V&V/UQ tasks in roadmap; {len(missing)} missing coverage")
    if missing:
        print("Missing coverage for tasks:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("All tasks covered ✅")
    if args.json_out:
        out = {
            "total_tasks": len(tasks),
            "covered": [t for t, ok in coverage.items() if ok],
            "missing": missing,
        }
        try:
            Path(args.json_out).write_text(json.dumps(out, indent=2))
            print(f"Wrote JSON report to {args.json_out}")
        except Exception as e:
            print(f"Failed to write JSON report: {e}")
    if missing and args.fail_on_missing:
        sys.exit(1)

if __name__ == "__main__":
    main()
