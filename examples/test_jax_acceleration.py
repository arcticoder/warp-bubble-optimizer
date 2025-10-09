#!/usr/bin/env python3
"""
Deprecated standalone JAX demo.

Pytest discovery is scoped to tests/ (see pytest.ini). The supported JAX demo test lives at:
    tests/test_jax_acceleration.py
and is automatically skipped when JAX isn't installed.

This file is retained as a no-op shim to avoid accidental execution in CI.
"""

if __name__ == "__main__":
        print("This script is deprecated. Run pytest to execute the optional JAX demo under tests/.")
