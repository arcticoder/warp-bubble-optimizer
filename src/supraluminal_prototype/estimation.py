from __future__ import annotations
import numpy as np


class EKF:
    """Extremely lightweight EKF skeleton with linearized update."""
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

    def predict(self, F: np.ndarray, u: np.ndarray | None = None, B: np.ndarray | None = None):
        F = np.asarray(F, float)
        if B is not None and u is not None:
            self.x = F @ self.x + B @ u
        else:
            self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x, self.P

    def update(self, z: np.ndarray, H: np.ndarray):
        z = np.asarray(z, float)
        H = np.asarray(H, float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P
        return self.x, self.P


def simulate_sensors(true_state: np.ndarray, H: np.ndarray, noise_std: np.ndarray) -> np.ndarray:
    true_state = np.asarray(true_state, float)
    H = np.asarray(H, float)
    noise_std = np.asarray(noise_std, float)
    mean = H @ true_state
    noise = np.random.normal(0.0, noise_std)
    return mean + noise
