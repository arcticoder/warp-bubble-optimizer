import os, sys
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
from supraluminal_prototype.warp_generator import build_metric, expansion_scalar, GridSpec
from supraluminal_prototype.warp_generator import plasma_density, field_synthesis
from supraluminal_prototype.warp_generator import target_soliton_envelope, compute_envelope_error, tune_ring_amplitudes_uniform
from supraluminal_prototype.control import sync_rings
from supraluminal_prototype.hardware import CoilDriver


def test_zero_expansion_metric():
    metric = build_metric({'R': 2.5, 'grid': GridSpec(nx=16, ny=16, nz=16, extent=1.0)})
    theta = expansion_scalar(metric)
    # Discrete zero-divergence up to numerical tolerance on coarse grids
    assert np.nanmax(np.abs(theta)) < 5e-2


def test_ring_sync_tolerance():
    phases = np.array([0.0, 0.2e-6, 0.5e-6, 0.8e-6])
    assert bool(sync_rings(phases, jitter=1e-6))
    assert not bool(sync_rings(phases, jitter=0.5e-6))


def test_coil_driver_linearity():
    drv = CoilDriver(max_current=5000.0, hysteresis=0.0)
    currents = [drv.command(i/30.0) for i in range(31)]
    diffs = [currents[i]-currents[i-1] for i in range(1, len(currents))]
    # Nearly constant step
    assert max(abs(d - diffs[0]) for d in diffs[1:]) < 1e-6


def test_plasma_density_shell_profile():
    gs = GridSpec(nx=16, ny=16, nz=16, extent=1.0)
    res = plasma_density({'grid': gs, 'n0': 3e20, 'R_shell': 0.6, 'width': 0.15})
    n = res['n']
    assert n.shape == (16, 16, 16)
    assert n.max() <= 3e20 + 1e10
    assert n.min() >= 0.0


def test_field_synthesis_envelope_bounds():
    gs = GridSpec(nx=16, ny=16, nz=16, extent=1.0)
    res = field_synthesis([1.0, 0.5, 0.5, 0.2], {'grid': gs, 'sigma': 0.2})
    env = res['envelope']
    assert env.shape == (16, 16, 16)
    assert env.max() <= 1.0 + 1e-9
    assert env.min() >= 0.0


def test_envelope_fit_error_monotonicity_uniform():
    gs = GridSpec(nx=16, ny=16, nz=16, extent=1.0)
    params = {'grid': gs, 'sigma': 0.25}
    target = target_soliton_envelope({'grid': gs, 'r0': 0.0, 'sigma': 0.5}).get('envelope')
    env_lo = field_synthesis([0.1, 0.1, 0.1, 0.1], params)['envelope']
    env_hi = field_synthesis([0.9, 0.9, 0.9, 0.9], params)['envelope']
    err_lo = compute_envelope_error(env_lo, target)
    err_hi = compute_envelope_error(env_hi, target)
    # Higher amplitude should generally reduce error relative to a nonzero target profile
    assert err_hi <= err_lo + 1e-6


def test_tune_ring_amplitudes_uniform_returns_best_controls():
    gs = GridSpec(nx=16, ny=16, nz=16, extent=1.0)
    params = {'grid': gs, 'sigma': 0.25}
    target = target_soliton_envelope({'grid': gs, 'r0': 0.0, 'sigma': 0.5}).get('envelope')
    rc0, e0 = tune_ring_amplitudes_uniform(np.zeros(4), params, target, n_steps=9)
    assert rc0.shape == (4,)
    assert 0.0 <= rc0.max() <= 1.0
    assert e0 >= 0.0
