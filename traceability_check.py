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
import xml.etree.ElementTree as ET
import sys

ROOT = Path(__file__).parent
ROADMAP = ROOT / "docs" / "roadmap.ndjson"
TEST_GLOBS = ["test_*.py"]
SRC_GLOBS = ["src/**/*.py", "*.py"]

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


def load_coverage_hits(xml_path: Path):
    """Parse coverage.py XML and return a map file -> uncovered line numbers set."""
    if not xml_path or not xml_path.exists():
        return {}
    uncovered = {}
    try:
        tree = ET.fromstring(xml_path.read_text())
    except Exception:
        return {}
    for cls in tree.findall('.//class'):
        filename = cls.get('filename')
        if not filename:
            continue
        lines = cls.find('lines')
        if lines is None:
            continue
        misses = set()
        for ln in lines.findall('line'):
            hits = int(ln.get('hits', '0'))
            num = int(ln.get('number', '0'))
            if hits == 0:
                misses.add(num)
        if misses:
            uncovered[filename] = misses
    return uncovered


def scan_source_vnv_tags(root: Path):
    """Scan source files for lines containing 'V&V:' tags and return mapping."""
    tags = []
    for pattern in SRC_GLOBS:
        for path in root.glob(pattern):
            if path.is_dir():
                continue
            try:
                lines = path.read_text(errors='ignore').splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines, start=1):
                if 'V&V:' in line:
                    # Extract the tag text after 'V&V:' up to end or closing paren
                    text = line[line.find('V&V:'):].strip()
                    tags.append({'file': str(path), 'line': i, 'tag': text})
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-on-missing", action="store_true", help="Return code 1 if any tasks lack coverage")
    ap.add_argument("--coverage-xml", type=str, default=None, help="Path to coverage.xml for uncovered mapping")
    ap.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON report of coverage")
    args = ap.parse_args()
    tasks = load_tasks()
    coverage = scan_tests(tasks)
    missing = [t for t, ok in coverage.items() if not ok]
    # Optional uncovered mapping
    uncovered_map = load_coverage_hits(Path(args.coverage_xml)) if args.coverage_xml else {}
    vnv_tag_locations = scan_source_vnv_tags(ROOT)
    uncovered_vnv = []
    for tag in vnv_tag_locations:
        fname = tag['file']
        miss = uncovered_map.get(fname) or uncovered_map.get(str(Path(fname)))
        if miss and tag['line'] in miss:
            uncovered_vnv.append(tag)
    print(f"Discovered {len(tasks)} V&V/UQ tasks in roadmap; {len(missing)} missing coverage")
    if missing:
        print("Missing coverage for tasks:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("All tasks covered ✅")
    if uncovered_vnv:
        print(f"Found {len(uncovered_vnv)} source V&V tags on uncovered lines (from coverage.xml):")
        for t in uncovered_vnv[:20]:
            print(f"  - {t['file']}:{t['line']} -> {t['tag']}")
    if args.json_out:
        out = {
            "total_tasks": len(tasks),
            "covered": [t for t, ok in coverage.items() if ok],
            "missing": missing,
            "uncovered_vnv_tags": uncovered_vnv,
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
