import numpy as np

from src.supraluminal_prototype.warp_generator import (
    GridSpec,
    plasma_density,
    field_synthesis,
    target_soliton_envelope,
    compute_envelope_error,
)


def test_plasma_density_bounds_and_units():
    grid = GridSpec()
    n0 = 3e20
    res = plasma_density({"grid": grid, "n0": n0})
    n = res["n"]
    assert float(n.min()) >= 0.0
    assert np.isfinite(n).all()
    # peak should be within ~1% of n0 on the coarse grid
    assert np.isclose(float(n.max()), n0, rtol=1e-2)


def test_field_synthesis_bench_against_simplified_target():
    grid = GridSpec()
    analytic = target_soliton_envelope({"grid": grid, "r0": 0.0, "sigma": 0.5 * grid.extent})[
        "envelope"
    ]
    syn = field_synthesis(np.array([1, 1, 1, 1], dtype=float), {"grid": grid, "sigma": 0.2 * grid.extent})["envelope"]
    err = compute_envelope_error(syn, analytic, norm="l2")
    # Coarse grid tolerance; we'll refine as the fitter improves
    assert 0.0 <= err <= 0.6
