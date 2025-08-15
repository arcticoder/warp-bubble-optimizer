from __future__ import annotations
import numpy as np


def coil_copper_loss(I_rms: float, R_ohm: float) -> float:
    return float((I_rms ** 2) * R_ohm)


def battery_heat(P_out: float, eta: float) -> float:
    eta = max(1e-3, min(0.999, float(eta)))
    P_in = P_out / eta
    return float(P_in - P_out)


def thermal_step(T_prev: float, P_in: float, C_th: float, G_th: float, dt: float, T_amb: float = 300.0) -> float:
    """First-order RC thermal model: C dT/dt = P_in - G (T - T_amb)."""
    T_prev = float(T_prev)
    C_th = float(max(C_th, 1e-6))
    G_th = float(max(G_th, 1e-9))
    dt = float(max(dt, 1e-6))
    dTdt = (P_in - G_th * (T_prev - T_amb)) / C_th
    return T_prev + dTdt * dt
