#!/usr/bin/env python3
"""
Root-level pytest configuration shim (fixtures removed).

Notes:
- Previous shared fixtures migrated into individual test modules.
- This file is intentionally a no-op for collection control; legacy ignore
	hook retained but returns False to avoid masking new tests.
"""

import os
from pathlib import Path

def pytest_ignore_collect(collection_path: Path, config):  # type: ignore[override]
	return False
