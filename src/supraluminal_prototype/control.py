from __future__ import annotations
import numpy as np


def sync_rings(phases: np.ndarray, jitter: float) -> bool:
    """
    Return True if phases can be synchronized within jitter tolerance.
    Simple criterion: max phase spread <= jitter (radians or seconds as chosen consistently).
    """
    phases = np.asarray(phases, dtype=float)
    if phases.ndim != 1 or phases.size == 0:
        return False
    return (phases.max() - phases.min()) <= float(jitter)
