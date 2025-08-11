#!/usr/bin/env python3
"""
Root-level pytest configuration is intentionally empty.

Notes:
- Actual fixtures and collection filters live in tests/conftest.py.
- Keeping this file minimal prevents import-time side effects from legacy imports.
"""

# Do not define hooks or fixtures here; tests/ scope owns pytest config.
