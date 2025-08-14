from __future__ import annotations

import sys, pathlib
repo_root = pathlib.Path(__file__).parent.parent
src_root = repo_root / 'src'
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from impulse import IntegratedImpulseController, ImpulseEngineConfig, MissionWaypoint  # type: ignore
from src.simulation.simulate_vector_impulse import Vector3D
from simulate_rotation import Quaternion
import numpy as np


def test_six_dof_smoke_runtime_and_basic_outputs():
    cfg = ImpulseEngineConfig(energy_budget=5e12, max_velocity=4e-5, max_angular_velocity=0.15)
    ctrl = IntegratedImpulseController(cfg)
    wp0 = MissionWaypoint(position=Vector3D(0,0,0), orientation=Quaternion(1,0,0,0), dwell_time=2.0)
    # Combined translation + small rotation
    wp1 = MissionWaypoint(position=Vector3D(20,0,0), orientation=Quaternion.from_euler(0.0, 0.0, 0.2), dwell_time=2.0)
    plan = ctrl.plan_impulse_trajectory([wp0, wp1], hybrid_mode='simulate-first')
    res = ctrl.execute_impulse_mission(plan)
    import asyncio
    res = asyncio.get_event_loop().run_until_complete(res)
    assert res['mission_success'] in (True, False)
    assert 'performance_metrics' in res and res['performance_metrics']['total_energy_used'] >= 0.0
    seg = res['segment_results'][0]
    # ensure rotation results present
    assert seg['rotation_results'] is not None
    # ensure translation distance sane
    assert seg['translation_results']['total_distance'] > 0.0
