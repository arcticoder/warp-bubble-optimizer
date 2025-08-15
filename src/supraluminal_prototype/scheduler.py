from __future__ import annotations
import numpy as np


def phase_sync_schedule(phases: np.ndarray, jitter_budget: float) -> dict:
    """
    Very simple scheduler: compute mean phase, return offsets to bring each phase to mean within jitter budget.
    Returns {'target': mean_phase, 'offsets': offsets, 'ok': bool}.
    'ok' is True if max spread <= jitter_budget.
    """
    p = np.asarray(phases, dtype=float).reshape(-1)
    if p.size == 0:
        return {'target': 0.0, 'offsets': np.array([]), 'ok': False}
    target = float(np.mean(p))
    offsets = target - p
    ok = (p.max() - p.min()) <= float(jitter_budget)
    return {'target': target, 'offsets': offsets, 'ok': bool(ok)}
