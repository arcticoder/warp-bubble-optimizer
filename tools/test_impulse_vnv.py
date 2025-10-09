"""V&V and UQ tests for Integrated Impulse Controller.

Implements tasks:
1. Mission energy accounting within 5% of planned
2. Trajectory segment dwell & timing adherence (segment duration tolerance)
3. Velocity & angular velocity caps enforcement (translation peak check)
4. Mission abort on energy budget depletion
5. Translation energy estimate upper bound vs simulated (estimate ≥ simulated - tol)
11. Monotonicity of _estimate_translation_energy vs displacement & v_max
14. Controller config injection (override gains)
18. Safety margin check: planned*(1+margin) ≤ budget when feasible
19. JSON export structured mission execution (basic schema presence)

Note: Test values chosen for speed & determinism.
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import numpy as np

try:  # local import path fix for test environment
    from impulse import (  # type: ignore
        IntegratedImpulseController, MissionWaypoint, ImpulseEngineConfig
    )
except Exception:  # pragma: no cover
    import sys, pathlib
    root = pathlib.Path(__file__).parent
    sys.path.insert(0, str(root))
    from impulse import (  # type: ignore
        IntegratedImpulseController, MissionWaypoint, ImpulseEngineConfig
    )
from src.simulation.simulate_vector_impulse import Vector3D


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _simple_waypoints(distances):
    wps = [MissionWaypoint(position=Vector3D(0, 0, 0), orientation=None)]
    acc = 0.0
    for d in distances:
        acc += d
        wps.append(MissionWaypoint(position=Vector3D(acc, 0, 0), orientation=None, dwell_time=5.0))
    return wps


def test_mission_energy_accounting_within_5pct(tmp_path):
    config = ImpulseEngineConfig(energy_budget=5e12, max_velocity=5e-5)
    ctrl = IntegratedImpulseController(config)
    waypoints = _simple_waypoints([50.0, 30.0])
    plan = ctrl.plan_impulse_trajectory(waypoints)
    json_path = tmp_path / "mission.json"
    results = run(ctrl.execute_impulse_mission(plan, json_export_path=str(json_path)))
    planned = plan['total_energy_estimate']
    actual = results['performance_metrics']['total_energy_used']
    assert planned > 0
    rel_err = abs(actual - planned) / planned
    assert rel_err <= 0.05, f"Energy accounting outside 5% (planned={planned}, actual={actual})"
    # JSON export presence
    data = json.loads(Path(json_path).read_text())
    assert 'plan' in data and 'results' in data


def test_segment_timing_adherence():
    config = ImpulseEngineConfig(energy_budget=5e12, max_velocity=4e-5)
    ctrl = IntegratedImpulseController(config)
    waypoints = _simple_waypoints([20.0, 40.0])
    plan = ctrl.plan_impulse_trajectory(waypoints)
    results = run(ctrl.execute_impulse_mission(plan))
    segs = plan['segments']
    sim_segments = results['segment_results']
    for planned_seg, sim_seg in zip(segs, sim_segments):
        planned_time = planned_seg['estimated_time']
        actual_time = sim_seg['segment_time']
        # Allow 15% tolerance due to discretization & velocity shaping differences
        assert abs(actual_time - planned_time) / (planned_time + 1e-9) <= 0.15


def test_velocity_caps_enforced():
    max_v = 3e-5
    config = ImpulseEngineConfig(energy_budget=5e12, max_velocity=max_v)
    ctrl = IntegratedImpulseController(config)
    waypoints = _simple_waypoints([25.0])
    plan = ctrl.plan_impulse_trajectory(waypoints)
    results = run(ctrl.execute_impulse_mission(plan))
    seg = results['segment_results'][0]
    vel_mags = seg['translation_results']['velocity_magnitudes']
    assert np.max(vel_mags) <= max_v * 1.05  # 5% numerical margin


def test_budget_depletion_aborts():
    # Force underestimate by constraining budget very low
    config = ImpulseEngineConfig(energy_budget=1e9, max_velocity=5e-5)
    ctrl = IntegratedImpulseController(config)
    waypoints = _simple_waypoints([200.0])
    plan = ctrl.plan_impulse_trajectory(waypoints)
    # Ensure plan itself exceeds budget
    assert plan['total_energy_estimate'] > config.energy_budget
    results = run(ctrl.execute_impulse_mission(plan, abort_on_budget=True))
    assert results['mission_success'] is False
    # Last segment should include abort_reason
    assert any('abort_reason' in s for s in results['segment_results'])


def test_translation_energy_upper_bound():
    config = ImpulseEngineConfig(energy_budget=5e12, max_velocity=4e-5)
    ctrl = IntegratedImpulseController(config)
    waypoints = _simple_waypoints([15.0])
    plan = ctrl.plan_impulse_trajectory(waypoints)
    results = run(ctrl.execute_impulse_mission(plan))
    est = plan['segments'][0]['estimated_energy']
    sim = results['segment_results'][0]['segment_energy']
    # Analytical model tends to over-estimate slightly; allow 10% under margin
    assert est + 0.1 * est >= sim, f"Estimate not an upper bound within tolerance (est={est}, sim={sim})"


def test_energy_estimate_monotonicity_displacement_velocity():
    config = ImpulseEngineConfig()
    ctrl = IntegratedImpulseController(config)
    from src.simulation.simulate_vector_impulse import VectorImpulseProfile, Vector3D
    # Displacement monotonicity
    prev = 0.0
    for d in [10, 20, 40, 80]:
        profile = VectorImpulseProfile(target_displacement=Vector3D(d, 0, 0), v_max=2e-5, t_up=5, t_hold=10, t_down=5)
        e = ctrl._translate_energy(profile)  # type: ignore
        assert e >= prev - 1e-6
        prev = e
    # Velocity monotonicity
    prev = 0.0
    for v in [1e-5, 2e-5, 3e-5, 4e-5]:
        profile = VectorImpulseProfile(target_displacement=Vector3D(50, 0, 0), v_max=v, t_up=5, t_hold=10, t_down=5)
        e = ctrl._translate_energy(profile)  # type: ignore
        assert e >= prev - 1e-6
        prev = e


def test_controller_config_injection_and_safety_margin():
    config = ImpulseEngineConfig(energy_budget=2e12, max_velocity=5e-5, safety_margin=0.25)
    overrides = {'controller': {'kp': 1.2, 'ki': 0.2, 'kd': 0.08}}
    ctrl = IntegratedImpulseController(config, controller_overrides=overrides)
    assert ctrl.control_config['controller'].kp == 1.2  # type: ignore
    wps = _simple_waypoints([40.0])
    plan = ctrl.plan_impulse_trajectory(wps)
    assert plan['feasible'] == (plan['total_energy_estimate'] * (1 + config.safety_margin) <= config.energy_budget)
