"""DEPRECATED: Root shim for integrated_impulse_control_clean.

Functionality unified under src.impulse.integrated_impulse_control.
Import from `impulse` package instead. This file will be removed later.
"""

from src.impulse.integrated_impulse_control import *  # noqa: F401,F403

__all__ = [
    'MissionWaypoint', 'ImpulseEngineConfig', 'IntegratedImpulseController'
]
