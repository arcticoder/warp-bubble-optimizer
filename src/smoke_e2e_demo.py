from __future__ import annotations
"""Minimal end-to-end smoke demo: drivers → scheduler → estimation → fault detection.

This is a no-IO, deterministic sanity path to exercise key modules.
"""
import numpy as np
from supraluminal_prototype.control_phase import generate_current_profile
from supraluminal_prototype.estimation import EKF, simulate_sensors


def run_demo(steps: int = 10) -> dict:
    profile = generate_current_profile(target=1.0, n=steps)
    # Simple kinematic state: position, velocity
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    ekf = EKF(x0=np.zeros(2), P0=np.eye(2), Q=0.01*np.eye(2), R=0.1*np.eye(1))
    innov = []
    x, _ = ekf.predict(F)
    for u in profile:
        # Treat current as acceleration ~ control; simple integration into velocity
        x, _ = ekf.predict(F, u=np.array([u]), B=np.array([[0.0], [1.0]]))
        z = simulate_sensors(true_state=np.array([x[0], x[1]]), H=H, noise_std=np.array([0.0]))
        x, _ = ekf.update(z, H)
        innov.append(float((z - H @ x)**2))
    return {"steps": steps, "final_state": x.tolist(), "mean_innovation": float(np.mean(innov))}


if __name__ == '__main__':
    out = run_demo(steps=10)
    print(out)
