import math
import os
import sys

# Ensure src/ is on path for imports when running tests locally
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from supraluminal_prototype.power import compute_smearing_energy


def test_energy_increases_with_ramp_duration():
    P_peak = 25e6
    # Cruise time for lunar leg at Warp 1 (~2.56 s)
    t_cruise = 2.56
    E30 = compute_smearing_energy(P_peak, 30.0, t_cruise)
    E120 = compute_smearing_energy(P_peak, 120.0, t_cruise)
    E300 = compute_smearing_energy(P_peak, 300.0, t_cruise)
    assert E30 < E120 < E300


def test_numerical_values_match_analysis():
    P_peak = 25e6
    t_cruise = 2.56
    # Two linear ramps: 2 * (0.5 * t_ramp * P_peak) + P_peak * t_cruise
    E30 = compute_smearing_energy(P_peak, 30.0, t_cruise)
    assert math.isclose(E30, 2 * 0.5 * 30.0 * P_peak + P_peak * t_cruise, rel_tol=1e-9)


def test_invalid_inputs():
    try:
        compute_smearing_energy(-1, 30, 2.56)
        assert False
    except ValueError:
        pass
    try:
        compute_smearing_energy(25e6, -1, 2.56)
        assert False
    except ValueError:
        pass
    try:
        compute_smearing_energy(25e6, 30, -1)
        assert False
    except ValueError:
        pass

def test_triangle_shape_equivalence():
    # 'linear' accel+decel equals the 'triangle' total by construction
    E_lin = compute_smearing_energy(25e6, 30.0, 2.56, shape='linear')
    E_tri = compute_smearing_energy(25e6, 30.0, 2.56, shape='triangle')
    assert math.isclose(E_lin, E_tri, rel_tol=1e-12)
