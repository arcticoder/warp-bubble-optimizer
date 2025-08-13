#!/usr/bin/env python3
"""Statistical UQ harness for impulse translation energy estimate variance.

Runs randomized waypoint sets N times; computes variance of planned vs
simulated energy usage and prints summary metrics.

Usage:
    python uq_impulse_energy_variance.py --runs 20 --segments 3
"""
from __future__ import annotations

import argparse
import asyncio
import random
from statistics import mean, pstdev

try:
    from impulse import IntegratedImpulseController, ImpulseEngineConfig, MissionWaypoint  # type: ignore
except Exception:  # pragma: no cover
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from impulse import IntegratedImpulseController, ImpulseEngineConfig, MissionWaypoint  # type: ignore
from src.simulation.simulate_vector_impulse import Vector3D


def build_random_waypoints(n_segments: int, max_distance: float = 50.0):
    pos = 0.0
    wps = [MissionWaypoint(position=Vector3D(0,0,0), orientation=None)]
    for _ in range(n_segments):
        step = random.uniform(max_distance * 0.2, max_distance)
        pos += step
        wps.append(MissionWaypoint(position=Vector3D(pos,0,0), orientation=None))
    return wps


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def sample_once(n_segments: int):
    cfg = ImpulseEngineConfig()
    ctrl = IntegratedImpulseController(cfg)
    wps = build_random_waypoints(n_segments)
    plan = ctrl.plan_impulse_trajectory(wps)
    results = run(ctrl.execute_impulse_mission(plan))
    return plan['total_energy_estimate'], results['performance_metrics']['total_energy_used']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--segments', type=int, default=3)
    args = ap.parse_args()
    planned_vals = []
    actual_vals = []
    for _ in range(args.runs):
        p, a = sample_once(args.segments)
        planned_vals.append(p)
        actual_vals.append(a)
    plan_mean = mean(planned_vals)
    act_mean = mean(actual_vals)
    plan_std = pstdev(planned_vals)
    act_std = pstdev(actual_vals)
    print(f"Runs: {args.runs}")
    print(f"Planned Energy Mean: {plan_mean:.3e} ± {plan_std:.2e}")
    print(f"Actual  Energy Mean: {act_mean:.3e} ± {act_std:.2e}")
    bias = act_mean - plan_mean
    print(f"Bias (actual - planned): {bias:.2e} ({bias/plan_mean*100:.2f}%)")

if __name__ == '__main__':
    main()
