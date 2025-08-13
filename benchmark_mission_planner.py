#!/usr/bin/env python3
"""Lightweight performance benchmark for mission planning.

Usage:
    python benchmark_mission_planner.py --segments 5 --repeat 3

Reports average planning time; used as a sanity check so regression
tests can watch for >2× slowdowns.
"""
from __future__ import annotations

import argparse
import time
from statistics import mean, stdev

try:
    from impulse import IntegratedImpulseController, MissionWaypoint, ImpulseEngineConfig  # type: ignore
except Exception:  # pragma: no cover
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from impulse import IntegratedImpulseController, MissionWaypoint, ImpulseEngineConfig  # type: ignore
from src.simulation.simulate_vector_impulse import Vector3D


def build_waypoints(n_segments: int, distance: float = 20.0):
    wps = [MissionWaypoint(position=Vector3D(0,0,0), orientation=None)]
    pos = 0.0
    for _ in range(n_segments):
        pos += distance
        wps.append(MissionWaypoint(position=Vector3D(pos,0,0), orientation=None))
    return wps


def bench(segments: int, repeat: int):
    cfg = ImpulseEngineConfig()
    ctrl = IntegratedImpulseController(cfg)
    wps = build_waypoints(segments)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        ctrl.plan_impulse_trajectory(wps)
        times.append(time.perf_counter() - t0)
    avg = mean(times)
    sd = stdev(times) if len(times) > 1 else 0.0
    print(f"Segments: {segments}, repeat: {repeat}")
    print(f"Avg planning time: {avg*1000:.2f} ms (σ={sd*1000:.2f} ms)")
    # Simple threshold: flag if > 500 ms for given size
    if avg > 0.5:
        print("⚠️  Planning time exceeded 500 ms threshold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    bench(args.segments, args.repeat)

if __name__ == "__main__":
    main()
