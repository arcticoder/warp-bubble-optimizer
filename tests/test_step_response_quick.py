from __future__ import annotations
import math
from src.supraluminal_prototype.control_phase import generate_current_profile, phase_sync_schedule, PhaseSyncConfig


def test_step_response_quick():
    target = 1.0
    y = generate_current_profile(target, n=10)
    assert len(y) == 10
    # Monotonic increase and bounded by target
    assert all(0.0 <= y[i] <= target for i in range(len(y)))
    assert all(y[i] <= y[i+1] for i in range(len(y)-1))
    # Settling behavior (within 5% by 10 steps for alpha=0.25 → 1 - (1-0.25)^10 ≈ 0.94)
    assert math.isclose(y[-1], target, rel_tol=0.06)


def test_phase_sync_schedule_quick():
    sched = phase_sync_schedule(steps=5, cfg=PhaseSyncConfig(step_rad=0.1))
    expected = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert len(sched) == len(expected)
    for a, b in zip(sched, expected):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
