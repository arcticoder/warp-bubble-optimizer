from __future__ import annotations
import numpy as np


def generate_current_profile(I0: float, I1: float, t_ramp: float, dt: float, max_didt: float) -> np.ndarray:
    """
    Generate a current setpoint ramp from I0 to I1 over t_ramp seconds subject to |dI/dt| <= max_didt.
    Returns array of shape (N,), where N = floor(t_ramp/dt)+1.
    Guarantees the slope does not exceed the constraint and ends at or before the target.
    """
    t_ramp = float(max(t_ramp, 0.0))
    dt = float(max(dt, 1e-6))
    max_didt = float(max(max_didt, 1e-9))
    n = int(np.floor(t_ramp / dt)) + 1
    I = np.empty(n, dtype=float)
    I[0] = float(I0)
    sign = 1.0 if I1 >= I0 else -1.0
    step = min(abs(I1 - I0), max_didt * dt)
    for k in range(1, n):
        I[k] = I[k-1] + sign * step
        # clamp to target without overshoot
        if (sign > 0 and I[k] > I1) or (sign < 0 and I[k] < I1):
            I[k] = float(I1)
    return I
