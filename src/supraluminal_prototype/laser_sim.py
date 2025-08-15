from __future__ import annotations
import numpy as np


def injection_lock_phase_noise(bw_hz: float, duration_s: float, dt: float, floor_rad: float) -> np.ndarray:
    """
    Generate a simple phase noise trace: Ornstein-Uhlenbeck-like with bandwidth bw_hz and white floor.
    Returns array of length N = floor(duration_s/dt)+1 in radians.
    """
    dt = float(max(dt, 1e-6))
    n = int(np.floor(duration_s / dt)) + 1
    x = np.zeros(n)
    alpha = np.exp(-2.0 * np.pi * bw_hz * dt)
    sigma = floor_rad * np.sqrt(1.0 - alpha**2)
    for k in range(1, n):
        x[k] = alpha * x[k-1] + np.random.normal(0.0, sigma)
    return x
