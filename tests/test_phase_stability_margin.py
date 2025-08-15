import numpy as np
from src.supraluminal_prototype.control import phase_stability_margin_db

essential_db = 6.0

def test_phase_stability_margin_over_6db():
    db = phase_stability_margin_db(kp=1.0, plant_gain=10.0, noise_floor=0.5)
    assert db > essential_db
