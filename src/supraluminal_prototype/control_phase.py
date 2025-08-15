from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class PhaseSyncConfig:
    n_rings: int = 4
    base_phase_rad: float = 0.0
    step_rad: float = 0.1


def phase_sync_schedule(steps: int, cfg: PhaseSyncConfig | None = None) -> List[float]:
    """Produce a simple closed-loop phase schedule (stub).

    For testing: returns a monotonic ramp schedule limiting phase increments.
    """
    cfg = cfg or PhaseSyncConfig()
    out = []
    cur = cfg.base_phase_rad
    for _ in range(max(0, steps)):
        cur += cfg.step_rad
        out.append(cur)
    return out


def generate_current_profile(target: float, n: int) -> List[float]:
    """Simple first-order step response toward target current (stub).

    Deterministic and fast for quick tests.
    """
    if n <= 0:
        return []
    alpha = 0.25
    y = 0.0
    out: List[float] = []
    for _ in range(n):
        y = y + alpha * (target - y)
        out.append(y)
    return out
