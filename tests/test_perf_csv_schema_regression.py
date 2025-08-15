from __future__ import annotations
import json
import pandas as pd
from importlib import resources
import asyncio
from impulse import IntegratedImpulseController, ImpulseEngineConfig, MissionWaypoint
from src.simulation.simulate_vector_impulse import Vector3D
from simulate_rotation import Quaternion


def test_perf_csv_schema_regression(tmp_path):
    cfg = ImpulseEngineConfig(energy_budget=5e12, max_velocity=4e-5)
    ctrl = IntegratedImpulseController(cfg)
    q = Quaternion(1, 0, 0, 0)
    wps = [
        MissionWaypoint(position=Vector3D(0,0,0), orientation=q, dwell_time=2.0),
        MissionWaypoint(position=Vector3D(10,0,0), orientation=q, dwell_time=2.0),
    ]
    plan = ctrl.plan_impulse_trajectory(wps)
    csv_path = tmp_path / 'perf.csv'
    asyncio.get_event_loop().run_until_complete(
        ctrl.execute_impulse_mission(plan, perf_csv_path=str(csv_path))
    )
    df = pd.read_csv(csv_path)
    with resources.files('warp_bubble_optimizer').joinpath('schemas/perf.csv.schema.json').open('rb') as f:
        schema = json.load(f)
    # Validate required columns preserved
    required = set(schema['required'])
    assert required.issubset(set(df.columns))
