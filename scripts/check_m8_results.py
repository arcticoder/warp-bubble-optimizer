#!/usr/bin/env python3
"""
Quick script to check for M8 optimization results
"""
import json
import glob
import os
from pathlib import Path

print("🔍 Checking for M8 optimization results...")

# Check for any JSON files created recently
json_files = glob.glob("*M8*.json") + glob.glob("*8gaussian*.json")
print(f"Found {len(json_files)} M8-related JSON files:")
for f in json_files:
    print(f"  - {f}")

# Check for any files modified in the last hour
recent_files = []
current_time = os.path.getctime(".")
for file in Path(".").glob("*"):
    if file.is_file():
        file_time = file.stat().st_mtime
        if current_time - file_time < 3600:  # 1 hour
            recent_files.append((file.name, file_time))

recent_files.sort(key=lambda x: x[1], reverse=True)
print(f"\n📅 Files modified in the last hour:")
for fname, ftime in recent_files[:10]:
    print(f"  - {fname}")

# Check if CMA-ES output directory exists
if os.path.exists("outcmaes"):
    cma_files = list(Path("outcmaes").glob("*"))
    print(f"\n📊 CMA-ES output directory contains {len(cma_files)} files")
    
print("\n✅ Check complete")
