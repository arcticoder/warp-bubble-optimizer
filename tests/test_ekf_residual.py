from __future__ import annotations
import numpy as np
from supraluminal_prototype.estimation import EKF, simulate_sensors


def test_ekf_innovation_reduction():
    x0 = np.array([0.0, 0.0])
    P0 = np.eye(2)
    Q = 0.01 * np.eye(2)
    R = 0.1 * np.eye(1)
    ekf = EKF(x0, P0, Q, R)
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    true_state = np.array([1.0, 0.0])
    z = simulate_sensors(true_state, H, np.array([0.0]))
    x_pred, P_pred = ekf.predict(F)
    innov_before = float((z - H @ x_pred)**2)
    x_upd, P_upd = ekf.update(z, H)
    innov_after = float((z - H @ x_upd)**2)
    assert innov_after <= innov_before + 1e-9
