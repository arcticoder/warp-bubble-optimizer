#!/usr/bin/env python3
"""Parameter sweep benchmark: planning time vs number of segments.

Runs mission planning for N segments with fixed distances to detect regressions.
"""
import time
import statistics as stats
from pathlib import Path

import sys, pathlib
repo_root = pathlib.Path(__file__).parent
src_root = repo_root / 'src'
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from impulse import IntegratedImpulseController, ImpulseEngineConfig, MissionWaypoint  # type: ignore
from src.simulation.simulate_vector_impulse import Vector3D


def make_waypoints(n_segments: int, step: float = 20.0):
    wps = [MissionWaypoint(position=Vector3D(0,0,0), orientation=None, dwell_time=1.0)]
    x = 0.0
    for _ in range(n_segments):
        x += step
        wps.append(MissionWaypoint(position=Vector3D(x,0,0), orientation=None, dwell_time=1.0))
    return wps


def run_once(n_segments: int) -> float:
    cfg = ImpulseEngineConfig(max_velocity=4e-5, energy_budget=1e13)
    ctrl = IntegratedImpulseController(cfg)
    wps = make_waypoints(n_segments)
    t0 = time.perf_counter()
    ctrl.plan_impulse_trajectory(wps, hybrid_mode=True)
    return (time.perf_counter() - t0) * 1000.0


def main():
    sizes = [1, 2, 3, 5, 8]
    print("Segments, avg_ms, p95_ms")
    for n in sizes:
        samples = [run_once(n) for _ in range(5)]
        print(f"{n}, {stats.mean(samples):.2f}, {stats.quantiles(samples, n=20)[-1]:.2f}")


if __name__ == "__main__":
    main()
