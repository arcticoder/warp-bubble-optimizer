"""DEPRECATED: Root shim for integrated_impulse_control_clean.

Functionality unified under src.impulse.integrated_impulse_control.
Import from `impulse` package instead. Removal scheduled after version 0.3.0.
See docs/DEPRECATIONS.md for schedule.
"""
import warnings as _warnings
try:  # pragma: no cover
    from version import __version__ as _ver  # type: ignore
except Exception:  # pragma: no cover
    _ver = "unknown"
_warnings.warn(
    f"integrated_impulse_control_clean shim deprecated (loaded under version {_ver}); removal after 0.3.0.",
    DeprecationWarning,
    stacklevel=2
)

from src.impulse.integrated_impulse_control import *  # noqa: F401,F403

__all__ = [
    'MissionWaypoint', 'ImpulseEngineConfig', 'IntegratedImpulseController'
]
