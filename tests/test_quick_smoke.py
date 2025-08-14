import sys

import pytest


@pytest.mark.quick
def test_quick_smoke_imports():
    # Minimal import sanity checks to ensure environment is correctly wired
    assert sys.version_info.major >= 3
    # Avoid heavy imports; just check module discoverability where safe
    import importlib
    assert importlib.util.find_spec('impulse') is not None
    assert importlib.util.find_spec('src.uq_validation.impulse_uq_runner') is not None
