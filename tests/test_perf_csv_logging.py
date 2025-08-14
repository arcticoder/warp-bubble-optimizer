from __future__ import annotations

import asyncio

from impulse import IntegratedImpulseController, ImpulseEngineConfig, MissionWaypoint
from src.simulation.simulate_vector_impulse import Vector3D
from simulate_rotation import Quaternion


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_perf_csv_logging_smoke(tmp_path):
    cfg = ImpulseEngineConfig(energy_budget=5e12, max_velocity=4e-5)
    ctrl = IntegratedImpulseController(cfg)
    q = Quaternion(1, 0, 0, 0)
    wps = [
        MissionWaypoint(position=Vector3D(0,0,0), orientation=q, dwell_time=2.0),
        MissionWaypoint(position=Vector3D(10,0,0), orientation=q, dwell_time=2.0),
    ]
    plan = ctrl.plan_impulse_trajectory(wps)
    out_csv = tmp_path / 'perf.csv'
    _ = run(ctrl.execute_impulse_mission(plan, perf_csv_path=str(out_csv)))
    text = out_csv.read_text()
    assert 'segment_index' in text and 'translation' in text
    assert len(text.splitlines()) >= 2


def test_perf_summary_plot_generation(tmp_path):
    # Reuse the existing flow to produce a perf.csv, then generate a plot
    cfg = ImpulseEngineConfig(energy_budget=5e12, max_velocity=4e-5)
    ctrl = IntegratedImpulseController(cfg)
    q = Quaternion(1, 0, 0, 0)
    wps = [
        MissionWaypoint(position=Vector3D(0,0,0), orientation=q, dwell_time=2.0),
        MissionWaypoint(position=Vector3D(10,0,0), orientation=q, dwell_time=2.0),
    ]
    plan = ctrl.plan_impulse_trajectory(wps)
    out_csv = tmp_path / 'perf.csv'
    _ = run(ctrl.execute_impulse_mission(plan, perf_csv_path=str(out_csv)))
    # Plot
    from subprocess import run as srun
    out_png = tmp_path / 'perf_summary.png'
    r = srun(['python', 'bin/plot_perf_csv.py', '--csv', str(out_csv), '--out', str(out_png)])
    assert r.returncode == 0
    assert out_png.exists() and out_png.stat().st_size > 0
