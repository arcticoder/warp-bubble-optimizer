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


def phase_stability_margin_db(kp: float, plant_gain: float, noise_floor: float) -> float:
    """
    Toy stability margin estimate in dB: 20*log10((kp*plant_gain)/max(noise_floor,eps)).
    Returns +inf if noise_floor is zero.
    """
    kp = float(kp)
    plant_gain = float(plant_gain)
    nf = max(1e-12, float(noise_floor))
    return 20.0 * np.log10(abs(kp * plant_gain) / nf)
