from __future__ import annotations

import asyncio
import random

from impulse import IntegratedImpulseController, ImpulseEngineConfig, MissionWaypoint, set_seed
from src.simulation.simulate_vector_impulse import Vector3D
from simulate_rotation import Quaternion


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _rand_waypoints(n_segments: int, rng: random.Random) -> list[MissionWaypoint]:
    # Translation-only segments (orientation=None)
    wps = [MissionWaypoint(position=Vector3D(0,0,0), orientation=None, dwell_time=2.0)]
    acc = 0.0
    for _ in range(n_segments):
        acc += rng.uniform(5.0, 30.0)
        wps.append(MissionWaypoint(position=Vector3D(acc,0,0), orientation=None, dwell_time=2.0))
    return wps


def test_property_energy_accounting_randomized_samples():
    # Not using hypothesis to keep deps light; simple randomized trials instead
    base_seed = 20250814
    set_seed(base_seed)
    rng = random.Random(base_seed)
    cfg = ImpulseEngineConfig(energy_budget=5e13, max_velocity=5e-5)
    ctrl = IntegratedImpulseController(cfg)
    for _ in range(10):
        n = rng.randint(1, 5)
        wps = _rand_waypoints(n, rng)
        plan = ctrl.plan_impulse_trajectory(wps, hybrid_mode='simulate-first')
        res = run(ctrl.execute_impulse_mission(plan))
        planned = plan['total_energy_estimate']
        actual = res['performance_metrics']['total_energy_used']
    # Allow a tolerance (simulation vs estimate); enforce main V&V bound (5%)
        assert planned > 0
    assert abs(actual - planned) / planned <= 0.05
