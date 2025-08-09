import os, sys
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
from supraluminal_prototype.warp_generator import build_metric, expansion_scalar, GridSpec
from supraluminal_prototype.warp_generator import plasma_density, field_synthesis
from supraluminal_prototype.warp_generator import target_soliton_envelope, compute_envelope_error, tune_ring_amplitudes_uniform
from supraluminal_prototype.warp_generator import synthesize_shift_with_envelope, optimize_energy
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


def test_envelope_to_shift_coupling_divergence_small():
    gs = GridSpec(nx=16, ny=16, nz=16, extent=1.0)
    res = synthesize_shift_with_envelope({'grid': gs, 'R': 2.5, 'sigma': 0.25, 'ring_controls': [1.0, 0.7, 0.7, 0.5]})
    theta = expansion_scalar(res)
    assert np.nanmax(np.abs(theta)) < 5e-2


def test_optimize_energy_stub_outputs():
    gs = GridSpec(nx=16, ny=16, nz=16, extent=1.0)
    out = optimize_energy({'grid': gs, 'P_peak': 25e6, 't_ramp': 30.0, 't_cruise': 2.56, 'sigma': 0.25})
    assert 'E' in out and 'best_controls' in out and 'fit_error' in out
    assert out['E'] > 0.0
    assert out['best_controls'].shape == (4,)
    assert out['fit_error'] >= 0.0


def test_battery_feasibility_flag():
    gs = GridSpec(nx=8, ny=8, nz=8, extent=1.0)
    # Compute reference energy and set capacity just above it
    from supraluminal_prototype.power import compute_smearing_energy
    E_ref = compute_smearing_energy(25e6, 30.0, 2.56)
    out_ok = optimize_energy({'grid': gs, 'P_peak': 25e6, 't_ramp': 30.0, 't_cruise': 2.56,
                              'battery_capacity_J': E_ref * 1.01, 'battery_eta0': 1.0, 'battery_eta_slope': 0.0})
    out_bad = optimize_energy({'grid': gs, 'P_peak': 25e6, 't_ramp': 30.0, 't_cruise': 2.56,
                               'battery_capacity_J': E_ref * 0.99, 'battery_eta0': 1.0, 'battery_eta_slope': 0.0})
    assert out_ok['feasible'] is True
    assert out_bad['feasible'] is False

def test_zero_expansion_tolerance_vs_resolution():
    # Coarser grids should generally have higher divergence magnitude than finer grids
    from supraluminal_prototype.warp_generator import build_metric, expansion_scalar, GridSpec
    theta8 = expansion_scalar(build_metric({'R': 2.5, 'grid': GridSpec(nx=8, ny=8, nz=8, extent=1.0)}))
    theta16 = expansion_scalar(build_metric({'R': 2.5, 'grid': GridSpec(nx=16, ny=16, nz=16, extent=1.0)}))
    theta32 = expansion_scalar(build_metric({'R': 2.5, 'grid': GridSpec(nx=32, ny=32, nz=32, extent=1.0)}))
    m8 = float(np.nanmax(np.abs(theta8)))
    m16 = float(np.nanmax(np.abs(theta16)))
    m32 = float(np.nanmax(np.abs(theta32)))
    assert m8 >= m16 * 0.8  # allow slack
    assert m16 >= m32 * 0.8

def test_discharge_efficiency_affects_feasibility():
    gs = GridSpec(nx=8, ny=8, nz=8, extent=1.0)
    from supraluminal_prototype.power import compute_smearing_energy
    E_ref = compute_smearing_energy(25e6, 30.0, 2.56)
    # With high efficiency, capacity just above E_ref should be feasible
    out_hi = optimize_energy({'grid': gs, 'P_peak': 25e6, 't_ramp': 30.0, 't_cruise': 2.56,
                              'battery_capacity_J': E_ref * 1.02, 'battery_eta0': 1.0, 'battery_eta_slope': 0.0})
    # With lower efficiency and same capacity, should flip to infeasible
    out_lo = optimize_energy({'grid': gs, 'P_peak': 25e6, 't_ramp': 30.0, 't_cruise': 2.56,
                              'battery_capacity_J': E_ref * 1.02, 'battery_eta0': 0.85, 'battery_eta_slope': 0.1})
    assert out_hi['feasible'] is True
    assert out_lo['feasible'] is False
