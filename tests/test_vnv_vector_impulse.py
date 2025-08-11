#!/usr/bin/env python3
"""
V&V tests for vectorized impulse engine:
- Energy scales approximately with v_max^2
- Trajectory accuracy improves as n_steps increases
"""
import numpy as np
import pytest

from src.simulation.simulate_vector_impulse import (
    Vector3D,
    VectorImpulseProfile,
    WarpBubbleVector,
    simulate_vector_impulse_maneuver,
)
import numpy as _np


def test_vector_impulse_energy_scales_quadratic():
    target = Vector3D(10.0, 0.0, 0.0)
    warp = WarpBubbleVector(shape_params=_np.array([1.0, 2.0, 0.5]))

    base = VectorImpulseProfile(
        target_displacement=target,
        v_max=1e-5,
        t_up=2.0,
        t_hold=4.0,
        t_down=2.0,
        n_steps=200,
    )

    res1 = simulate_vector_impulse_maneuver(base, warp, enable_progress=False)
    E1 = res1["total_energy"]

    # Double v_max, keep timings
    base.v_max = 2e-5
    res2 = simulate_vector_impulse_maneuver(base, warp, enable_progress=False)
    E2 = res2["total_energy"]

    ratio = E2 / (E1 + 1e-30)
    # Expect near 4x; allow tolerance for discretization and model factors
    assert E2 > E1
    assert 3.3 < ratio < 4.7


def test_trajectory_accuracy_improves_with_steps():
    target = Vector3D(100.0, 0.0, 0.0)
    warp = WarpBubbleVector(shape_params=_np.array([1.0, 2.0, 0.5]))

    coarse = VectorImpulseProfile(
        target_displacement=target,
        v_max=5e-5,
        t_up=5.0,
        t_hold=10.0,
        t_down=5.0,
        n_steps=100,
    )

    fine = VectorImpulseProfile(
        target_displacement=target,
        v_max=5e-5,
        t_up=5.0,
        t_hold=10.0,
        t_down=5.0,
        n_steps=400,
    )

    res_coarse = simulate_vector_impulse_maneuver(coarse, warp, enable_progress=False)
    res_fine = simulate_vector_impulse_maneuver(fine, warp, enable_progress=False)

    err_coarse = res_coarse["trajectory_error"]
    err_fine = res_fine["trajectory_error"]

    # Fine discretization should be as good or better
    assert err_fine <= err_coarse * 1.05
