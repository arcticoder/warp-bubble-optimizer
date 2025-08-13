"""DEPRECATED: Root shim for integrated_impulse_control.

Use `from src.impulse.integrated_impulse_control import ...` or
prefer `from impulse import IntegratedImpulseController`.
This shim will be removed after deprecation period.
"""

from src.impulse.integrated_impulse_control import *  # noqa: F401,F403

__all__ = [
    'MissionWaypoint', 'ImpulseEngineConfig', 'IntegratedImpulseController'
]
