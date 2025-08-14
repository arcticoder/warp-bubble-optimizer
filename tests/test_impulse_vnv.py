"""V&V and UQ tests for Integrated Impulse Controller (collected by pytest)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import numpy as np

import sys, pathlib
repo_root = pathlib.Path(__file__).parent.parent
src_root = repo_root / 'src'
if str(src_root) not in sys.path:
	sys.path.insert(0, str(src_root))
if str(repo_root) not in sys.path:
	sys.path.insert(0, str(repo_root))
from impulse import (  # type: ignore
	IntegratedImpulseController, MissionWaypoint, ImpulseEngineConfig
)
from src.simulation.simulate_vector_impulse import Vector3D
from simulate_rotation import Quaternion
from impulse.seed_utils import set_seed


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
	data = json.loads(Path(json_path).read_text())
	assert 'plan' in data and 'results' in data


def test_segment_timing_adherence():
	config = ImpulseEngineConfig(energy_budget=5e12, max_velocity=4e-5)
	ctrl = IntegratedImpulseController(config)
	waypoints = _simple_waypoints([20.0, 40.0])
	plan = ctrl.plan_impulse_trajectory(waypoints)
	results = run(ctrl.execute_impulse_mission(plan))
	for planned_seg, sim_seg in zip(plan['segments'], results['segment_results']):
		planned_time = planned_seg['estimated_time']
		actual_time = sim_seg['segment_time']
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
	assert np.max(vel_mags) <= max_v * 1.05


def test_angular_velocity_caps_enforced():
	max_omega = 0.2
	config = ImpulseEngineConfig(energy_budget=5e12, max_velocity=4e-5, max_angular_velocity=max_omega)
	ctrl = IntegratedImpulseController(config)
	# Two waypoints with pure rotation change at same position
	wp0 = MissionWaypoint(position=Vector3D(0, 0, 0), orientation=Quaternion(1, 0, 0, 0))
	target_q = Quaternion.from_euler(0.0, 0.0, 0.5)  # 0.5 rad yaw
	wp1 = MissionWaypoint(position=Vector3D(0, 0, 0), orientation=target_q, dwell_time=2.0)
	plan = ctrl.plan_impulse_trajectory([wp0, wp1])
	results = run(ctrl.execute_impulse_mission(plan))
	seg = results['segment_results'][0]
	rot = seg.get('rotation_results')
	assert rot is not None, "Rotation results missing"
	omegas = rot['angular_velocity_profile']
	assert float(np.max(omegas)) <= max_omega * 1.05


def test_budget_depletion_aborts():
	config = ImpulseEngineConfig(energy_budget=1e9, max_velocity=5e-5)
	ctrl = IntegratedImpulseController(config)
	waypoints = _simple_waypoints([200.0])
	plan = ctrl.plan_impulse_trajectory(waypoints)
	assert plan['total_energy_estimate'] > config.energy_budget
	results = run(ctrl.execute_impulse_mission(plan, abort_on_budget=True))
	assert results['mission_success'] is False
	assert any('abort_reason' in s for s in results['segment_results'])


def test_translation_energy_upper_bound():
	config = ImpulseEngineConfig(energy_budget=5e12, max_velocity=4e-5)
	ctrl = IntegratedImpulseController(config)
	waypoints = _simple_waypoints([15.0])
	plan = ctrl.plan_impulse_trajectory(waypoints)
	results = run(ctrl.execute_impulse_mission(plan))
	est = plan['segments'][0]['estimated_energy']
	sim = results['segment_results'][0]['segment_energy']
	assert est + 0.1 * est >= sim


def test_energy_estimate_monotonicity_displacement_velocity():
	config = ImpulseEngineConfig()
	ctrl = IntegratedImpulseController(config)
	from src.simulation.simulate_vector_impulse import VectorImpulseProfile, Vector3D
	prev = 0.0
	for d in [10, 20, 40, 80]:
		profile = VectorImpulseProfile(target_displacement=Vector3D(d, 0, 0), v_max=2e-5, t_up=5, t_hold=10, t_down=5)
		e = ctrl._translate_energy(profile)  # type: ignore
		assert e >= prev - 1e-6
		prev = e
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
	plan = ctrl.plan_impulse_trajectory(_simple_waypoints([40.0]))
	assert plan['feasible'] == (plan['total_energy_estimate'] * (1 + config.safety_margin) <= config.energy_budget)


def test_safety_margin_infeasible_even_if_raw_estimate_fits():
	# Choose budget to be between raw estimate and margin-adjusted estimate
	cfg = ImpulseEngineConfig(energy_budget=2.0e11, max_velocity=5e-5, safety_margin=0.5)
	ctrl = IntegratedImpulseController(cfg)
	plan = ctrl.plan_impulse_trajectory(_simple_waypoints([30.0]))
	assert plan['total_energy_estimate'] <= cfg.energy_budget
	assert plan['total_energy_estimate'] * (1 + cfg.safety_margin) > cfg.energy_budget
	assert plan['feasible'] is False


def test_mission_json_schema_version(tmp_path):
	config = ImpulseEngineConfig(energy_budget=5e12, max_velocity=4e-5)
	ctrl = IntegratedImpulseController(config)
	plan = ctrl.plan_impulse_trajectory(_simple_waypoints([10.0]))
	json_path = tmp_path / "mission.json"
	run(ctrl.execute_impulse_mission(plan, json_export_path=str(json_path)))
	data = json.loads(json_path.read_text())
	assert 'schema' in data and data['schema']['id'] == 'impulse.mission.v1'
	assert int(data['schema']['version']) == 1


def test_json_schema_validation_if_available(tmp_path):
	try:
		import jsonschema  # type: ignore
	except Exception:
		return
	config = ImpulseEngineConfig(energy_budget=5e12, max_velocity=4e-5)
	ctrl = IntegratedImpulseController(config)
	plan = ctrl.plan_impulse_trajectory(_simple_waypoints([10.0]))
	json_path = tmp_path / "mission.json"
	run(ctrl.execute_impulse_mission(plan, json_export_path=str(json_path)))
	data = json.loads(json_path.read_text())
	schema = {
		"type": "object",
		"required": ["plan", "results", "schema"],
		"properties": {
			"schema": {
				"type": "object",
				"required": ["id", "version"],
				"properties": {"id": {"type": "string"}, "version": {"type": "number"}},
			}
		},
	}
	jsonschema.validate(instance=data, schema=schema)


def test_performance_guard_plan_execute_5_segments():
	set_seed(42)
	config = ImpulseEngineConfig(energy_budget=5e13, max_velocity=5e-5)
	ctrl = IntegratedImpulseController(config)
	wps = _simple_waypoints([20.0, 20.0, 20.0, 20.0, 20.0])
	import time
	t0 = time.perf_counter()
	plan = ctrl.plan_impulse_trajectory(wps)
	_ = run(ctrl.execute_impulse_mission(plan))
	dt_ms = (time.perf_counter() - t0) * 1000
	assert dt_ms < 300.0


def test_rotation_strategy_upper_bound():
	from src.impulse.energy_strategies import QuadraticOmegaStrategy, DutyWeightedOmegaStrategy
	cfg = ImpulseEngineConfig()
	strategy = DutyWeightedOmegaStrategy(QuadraticOmegaStrategy(k_factor=1e17))
	ctrl = IntegratedImpulseController(cfg, rotation_energy_strategy=lambda p: strategy.estimate(p))
	from simulate_rotation import RotationProfile, WarpBubbleRotational, simulate_rotation_maneuver
	params = WarpBubbleRotational()
	for omega in [0.05, 0.1]:
		prof = RotationProfile(target_orientation=Quaternion.from_euler(0, 0, 0.3), omega_max=omega, t_up=3, t_hold=5, t_down=3, n_steps=200)
		est = strategy.estimate(prof)
		sim = simulate_rotation_maneuver(prof, params, enable_progress=False)
		assert est >= sim['total_energy'] * 0.9


def test_deprecation_import_warning_root_shim():
	import warnings, importlib
	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter('always', DeprecationWarning)
		importlib.invalidate_caches()
		mod = importlib.import_module('integrated_impulse_control')
		assert any(issubclass(rec.category, DeprecationWarning) for rec in w)
		assert hasattr(mod, 'IntegratedImpulseController')
