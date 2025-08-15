#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Mapping, Sequence

def compact_diff(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None, path: str = "$") -> list[str]:
    diffs: list[str] = []
    a_keys = set(a.keys()) if isinstance(a, Mapping) else set()
    b_keys = set(b.keys()) if isinstance(b, Mapping) else set()
    for k in sorted(a_keys | b_keys):
        pa = f"{path}.{k}"
        va = a.get(k, None) if isinstance(a, Mapping) else None
        vb = b.get(k, None) if isinstance(b, Mapping) else None
        if type(va) != type(vb):
            diffs.append(f"TYPE {pa}: {type(va).__name__} -> {type(vb).__name__}")
            continue
        if isinstance(va, Mapping):
            diffs.extend(compact_diff(va, vb, pa))
        elif isinstance(va, Sequence) and not isinstance(va, (str, bytes, bytearray)):
            if not isinstance(vb, Sequence):
                diffs.append(f"TYPE {pa}: {type(va).__name__} -> {type(vb).__name__}")
                continue
            if len(va) != len(vb):
                diffs.append(f"LEN {pa}: {len(va)} -> {len(vb)}")
            else:
                # spot-check first elements
                for i, (xa, xb) in enumerate(zip(list(va)[:3], list(vb)[:3])):
                    if xa != xb:
                        diffs.append(f"ELEM {pa}[{i}]: {xa} -> {xb}")
                        break
        else:
            if va != vb:
                diffs.append(f"VAL {pa}: {va} -> {vb}")
    return diffs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compact JSON diff for mission objects")
    ap.add_argument('a')
    ap.add_argument('b')
    args = ap.parse_args(argv)
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    diffs = compact_diff(a, b)
    for d in diffs[:200]:
        print(d)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
