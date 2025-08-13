"""DEPRECATED: Root shim for integrated_impulse_control.

Use `from impulse import IntegratedImpulseController` instead.
Scheduled for removal after version 0.3.0 (see docs/DEPRECATIONS.md).
"""
import warnings as _warnings
try:  # pragma: no cover - one-time import side effect
    from version import __version__ as _ver  # type: ignore
except Exception:  # pragma: no cover
    _ver = "unknown"
_warnings.warn(
    f"integrated_impulse_control shim is deprecated (loaded under version {_ver}); will be removed after 0.3.0.",
    DeprecationWarning,
    stacklevel=2
)

from src.impulse.integrated_impulse_control import *  # noqa: F401,F403

__all__ = [
    'MissionWaypoint', 'ImpulseEngineConfig', 'IntegratedImpulseController'
]
