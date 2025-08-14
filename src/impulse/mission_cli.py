#!/usr/bin/env python3
"""CLI wrapper for mission planning and execution.

Usage:
  python -m impulse.mission_cli --waypoints waypoints.json --export out.json

Waypoints JSON schema:
{
  "waypoints": [
    {"x": 0, "y": 0, "z": 0},
    {"x": 50, "y": 0, "z": 0}
  ],
  "dwell": 5.0
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .integrated_impulse_control import IntegratedImpulseController, ImpulseEngineConfig, MissionWaypoint, Quaternion
from src.simulation.simulate_vector_impulse import Vector3D


def load_waypoints(path: str, dwell: float | None = None):
  data = json.loads(Path(path).read_text())
  dflt_dwell = dwell if dwell is not None else float(data.get('dwell', 5.0))
  wps = []
  for p in data['waypoints']:
    wps.append(
      MissionWaypoint(
        position=Vector3D(float(p['x']), float(p['y']), float(p['z'])),
        orientation=Quaternion(1, 0, 0, 0),
        dwell_time=dflt_dwell,
      )
    )
  return wps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--waypoints', required=True)
    ap.add_argument('--export', required=False)
    ap.add_argument('--budget', type=float, default=1e13)
    ap.add_argument('--vmax', type=float, default=5e-5)
    ap.add_argument('--hybrid', action='store_true')
    args = ap.parse_args()

    cfg = ImpulseEngineConfig(energy_budget=args.budget, max_velocity=args.vmax)
    ctrl = IntegratedImpulseController(cfg)
    wps = load_waypoints(args.waypoints)
    plan = ctrl.plan_impulse_trajectory(wps, hybrid_mode=args.hybrid)
    import asyncio
    res = asyncio.get_event_loop().run_until_complete(ctrl.execute_impulse_mission(plan, json_export_path=args.export))
    print(json.dumps({
        'planned_GJ': plan['total_energy_estimate']/1e9,
        'actual_GJ': res['performance_metrics']['total_energy_used']/1e9,
        'success': res['mission_success'],
        'segments': len(res['segment_results'])
    }, indent=2))


if __name__ == '__main__':
    main()
