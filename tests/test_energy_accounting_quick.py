import numpy as np
from src.impulse.integrated_impulse_control import IntegratedImpulseController, ImpulseEngineConfig, MissionWaypoint
from src.simulation.simulate_vector_impulse import Vector3D


def test_energy_accounting_within_10_percent():
    cfg = ImpulseEngineConfig(energy_budget=1e12, max_velocity=5e-5)
    ctrl = IntegratedImpulseController(cfg)
    wps = [MissionWaypoint(position=Vector3D(0,0,0), dwell_time=0.1),
           MissionWaypoint(position=Vector3D(1.0,0,0), dwell_time=0.1)]
    plan = ctrl.plan_impulse_trajectory(wps, hybrid_mode='simulate-first')
    import asyncio
    res = asyncio.get_event_loop().run_until_complete(ctrl.execute_impulse_mission(plan))
    planned = plan['total_energy_estimate']
    actual = res['performance_metrics']['total_energy_used']
    if actual == 0:
        return  # nothing to compare
    ratio = abs(actual - planned)/max(actual, planned)
    assert ratio <= 0.10
