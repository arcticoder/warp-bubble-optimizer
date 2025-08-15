from __future__ import annotations
import numpy as np
from supraluminal_prototype.thermal import coil_copper_loss, battery_heat, thermal_step


def test_thermal_regression_smoke():
    P_loss = coil_copper_loss(10.0, 0.5)
    assert P_loss == 50.0
    P_batt = battery_heat(100.0, 0.9)
    assert 10.0 - 1e-6 <= P_batt <= 11.2
    T = 300.0
    for _ in range(10):
        T = thermal_step(T_prev=T, P_in=50.0, C_th=500.0, G_th=1.0, dt=0.1, T_amb=295.0)
    assert T > 300.0
