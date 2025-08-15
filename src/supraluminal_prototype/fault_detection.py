from __future__ import annotations


def detect_overcurrent(I_meas: float, I_limit: float, dwell_ms: float, threshold_ms: float) -> bool:
    """Return True if overcurrent persists longer than threshold."""
    return (I_meas > I_limit) and (dwell_ms >= threshold_ms)


def detect_ground_fault(R_iso_meg: float, min_meg: float) -> bool:
    """Return True if insulation resistance below minimum megohms."""
    return R_iso_meg < min_meg
