#!/usr/bin/env python3
from __future__ import annotations

import csv
import time
from pathlib import Path

from impulse import IntegratedImpulseController, ImpulseEngineConfig, MissionWaypoint
from src.simulation.simulate_vector_impulse import Vector3D
from simulate_rotation import Quaternion


def waypoints(n_segments: int):
    q = Quaternion(1,0,0,0)
    wps = [MissionWaypoint(position=Vector3D(0,0,0), orientation=q, dwell_time=1.0)]
    for i in range(n_segments):
        wps.append(MissionWaypoint(position=Vector3D(10*(i+1),0,0), orientation=q, dwell_time=1.0))
    return wps


def main():
    cfg = ImpulseEngineConfig(energy_budget=5e13, max_velocity=5e-5)
    ctrl = IntegratedImpulseController(cfg)
    out = Path('bench_plan_execute.csv')
    with out.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['segments','plan_ms','execute_ms','total_ms','energy_J'])
        for n in [1,2,3,4,5,8,10]:
            wps = waypoints(n)
            t0 = time.perf_counter()
            plan = ctrl.plan_impulse_trajectory(wps)
            t1 = time.perf_counter()
            import asyncio
            res = asyncio.get_event_loop().run_until_complete(ctrl.execute_impulse_mission(plan))
            t2 = time.perf_counter()
            writer.writerow([n, (t1-t0)*1000, (t2-t1)*1000, (t2-t0)*1000, res['performance_metrics']['total_energy_used']])
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
