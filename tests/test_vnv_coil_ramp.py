import numpy as np
from src.supraluminal_prototype.drivers import generate_current_profile


def synthetic_plant_response(I_profile: np.ndarray, hysteresis: float = 0.0) -> np.ndarray:
    y = I_profile.copy()
    if hysteresis > 0:
        h = 0.0
        for k in range(1, len(y)):
            dh = hysteresis * np.sign(y[k]-y[k-1])
            h = 0.9*h + dh
            y[k] += h
    return y


def test_coil_ramp_linearity_vs_hysteresis():
    I = generate_current_profile(0.0, 10.0, t_ramp=1.0, dt=0.01, max_didt=20.0)
    y_lin = synthetic_plant_response(I, hysteresis=0.0)
    y_hys = synthetic_plant_response(I, hysteresis=0.1)
    # Linear should have lower deviation from ramp
    err_lin = float(np.mean(np.abs(y_lin - I)))
    err_hys = float(np.mean(np.abs(y_hys - I)))
    assert err_lin < err_hys
