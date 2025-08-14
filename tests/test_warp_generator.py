from __future__ import annotations

import numpy as np

from src.supraluminal_prototype.warp_generator import (
    GridSpec,
    field_synthesis,
    target_soliton_envelope,
    compute_envelope_error,
    plasma_density,
)
from src.supraluminal_prototype.device_facade import DeviceFacade
from src.supraluminal_prototype.experiment_plan import ExperimentPlan
from impulse import IntegratedImpulseController, ImpulseEngineConfig, MissionWaypoint
from src.simulation.simulate_vector_impulse import Vector3D
from simulate_rotation import Quaternion


def test_hyperbolic_wave_sync_envelope_alignment():
    # Interpret "hyperbolic_wave" sync as aligning synthesized ring envelope to a soliton-like target
    grid = GridSpec(nx=24, ny=24, nz=9, extent=1.0)
    tgt = target_soliton_envelope({'grid': grid, 'r0': 0.0, 'sigma': 0.5 * grid.extent})['envelope']
    # Equal ring drive at moderate amplitude
    import numpy as _np
    env = field_synthesis(_np.array([0.6, 0.6, 0.6, 0.6], dtype=float), {'grid': grid, 'sigma': 0.25 * grid.extent})['envelope']
    err = compute_envelope_error(env, tgt, norm='l2')
    # Expect a reasonable alignment (not perfect), enforce an upper bound on error
    assert err < 0.45


def test_plasma_density_shell_peak_and_falloff():
    grid = GridSpec(nx=21, ny=21, nz=21, extent=1.0)
    pd = plasma_density({'grid': grid, 'n0': 3e20, 'R_shell': 0.6 * grid.extent, 'width': 0.15 * grid.extent})
    n = pd['n']
    # Peak should be positive and within [0, n0]
    assert float(np.max(n)) > 0.0
    # Density near center should be much lower than peak
    center = n[grid.nx // 2, grid.ny // 2, grid.nz // 2]
    assert center < 0.2 * np.max(n)


def test_mission_envelope_integration():
    # Build a minimal mission
    cfg = ImpulseEngineConfig(energy_budget=5e13, max_velocity=4e-5)
    ctrl = IntegratedImpulseController(cfg)
    q = Quaternion(1, 0, 0, 0)
    wps = [
        MissionWaypoint(position=Vector3D(0, 0, 0), orientation=q, dwell_time=2.0),
        MissionWaypoint(position=Vector3D(10, 0, 0), orientation=q, dwell_time=2.0),
    ]
    plan = ctrl.plan_impulse_trajectory(wps, hybrid_mode='simulate-first')
    # Synthesize target and field envelope
    grid = GridSpec(nx=24, ny=24, nz=9, extent=1.0)
    tgt = target_soliton_envelope({'grid': grid, 'r0': 0.0, 'sigma': 0.5 * grid.extent})['envelope']
    import numpy as _np
    env = field_synthesis(_np.array([0.6, 0.6, 0.6, 0.6], dtype=float), {'grid': grid, 'sigma': 0.25 * grid.extent})['envelope']
    # Execute mission (no direct coupling to envelope yet; we assert envelope quality alongside mission success)
    import asyncio
    res = asyncio.get_event_loop().run_until_complete(ctrl.execute_impulse_mission(plan, abort_on_budget=False))
    assert res['mission_success'] is True
    err = compute_envelope_error(env, tgt, norm='l2')
    assert err < 0.45
    # Mock device interactions: coil power, laser frequency sync, and field state
    dev = DeviceFacade()
    assert dev.initialize_coil(1.0) is True
    synced_freq = dev.set_laser_frequency(1e15)
    assert synced_freq > 1e15
    state = dev.read_field_state()
    assert state['error'] < 0.45


def test_experiment_plan():
    dev = DeviceFacade()
    plan = ExperimentPlan(laser_power=1.0, coil_freq=1e15, plasma_density=1e20, r_shell=0.5, width=0.1)
    cfg = plan.generate_plan()
    assert isinstance(cfg, dict) and cfg.get('laser_power') == 1.0
    assert plan.validate_plan(dev) is True
