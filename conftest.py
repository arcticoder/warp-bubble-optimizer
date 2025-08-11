#!/usr/bin/env python3
"""
Root-level pytest configuration: prevent accidental collection outside tests/.

Notes:
- Actual fixtures and collection filters live in tests/conftest.py.
- This file adds a guard to ignore root-level test discovery even if a job
  overrides testpaths. CI should run `pytest tests/` explicitly.
"""

import os
from pathlib import Path

def pytest_ignore_collect(collection_path: Path, config):  # type: ignore[override]
	try:
		# If path is under repo root and not under tests/, ignore it.
		p = Path(str(collection_path))
		# Allow only tests/ tree
		if "tests" not in p.parts:
			# Ignore common legacy test filenames at root
			name = p.name
			if name.startswith("test_") and name.endswith(".py"):
				return True
	except Exception:
		return False
	return False
