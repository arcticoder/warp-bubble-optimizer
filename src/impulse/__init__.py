"""Impulse control package: relocated from root scripts.

Provides integrated impulse mission planning and execution APIs.
Ensures backward-compatible import when project not installed as a package
by adjusting sys.path at runtime if needed for tests.
"""

import sys as _sys
from pathlib import Path as _Path
_pkg_root = _Path(__file__).resolve().parents[2]
if str(_pkg_root) not in _sys.path:
    _sys.path.insert(0, str(_pkg_root))

from .integrated_impulse_control import (
    MissionWaypoint,
    ImpulseEngineConfig,
    IntegratedImpulseController,
)
from .mission_cli import main as mission_cli_main  # noqa: F401
from .seed_utils import set_seed  # noqa: F401

__all__ = [
    'MissionWaypoint',
    'ImpulseEngineConfig',
    'IntegratedImpulseController',
    'set_seed',
]
