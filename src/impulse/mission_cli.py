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
from .seed_utils import set_seed


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


def main(argv: list[str] | None = None):
  ap = argparse.ArgumentParser()
  ap.add_argument('--waypoints', required=True)
  ap.add_argument('--export', required=False)
  ap.add_argument('--budget', type=float, default=1e13)
  ap.add_argument('--vmax', type=float, default=5e-5)
  ap.add_argument('--hybrid', nargs='?', const='simulate-first', default='off', choices=['off','simulate-first','estimate-first'], help='Hybrid planning mode')
  ap.add_argument('--threshold', type=float, default=0.3, help='Estimate-first threshold fraction of budget')
  ap.add_argument('--raise-on-infeasible', action='store_true')
  ap.add_argument('--raise-on-abort', action='store_true')
  ap.add_argument('--verbose-export', action='store_true')
  ap.add_argument('--export-cache', action='store_true')
  ap.add_argument('--perf-csv', type=str, default=None, help='Optional path to write per-segment performance CSV')
  ap.add_argument('--error-codes', action='store_true', help='Return non-zero exit codes on infeasible planning or budget abort')
  ap.add_argument('--seed', type=int, default=None, help='Set deterministic seed (also sets WARP_SEED/PYTHONHASHSEED)')
  ap.add_argument('--rehearsal', action='store_true', help='Rehearsal mode: plan only, timeline gating, no execution')
  ap.add_argument('--dry-run-abort', action='store_true', help='Simulate abort on threshold crossing without execution')
  ap.add_argument('--timeline-log', type=str, default=None, help='Optional CSV/JSONL path to log timeline events (planned vs executed)')
  args = ap.parse_args(argv)

  # Seed plumbing for reproducibility
  if args.seed is not None:
    set_seed(int(args.seed))
    # Set WARP_SEED for downstream meta persistence
    import os as _os
    _os.environ['WARP_SEED'] = str(int(args.seed))

  cfg = ImpulseEngineConfig(energy_budget=args.budget, max_velocity=args.vmax)
  ctrl = IntegratedImpulseController(cfg)
  wps = load_waypoints(args.waypoints)
  plan = ctrl.plan_impulse_trajectory(wps, hybrid_mode=args.hybrid, estimate_first_threshold=args.threshold, raise_on_infeasible=args.raise_on_infeasible)

  # Optional timeline: write planning event
  def _write_timeline(path: str, rec: dict):
    from datetime import datetime, timezone
    import csv, os, json as _json
    now_iso = datetime.now(timezone.utc).isoformat()
    rec = {**rec, 'iso_time': now_iso}
    # Decide CSV vs JSONL by extension
    if path.lower().endswith('.jsonl'):
      with open(path, 'a') as fh:
        fh.write(_json.dumps(rec) + "\n")
    else:
      exists = os.path.exists(path)
      with open(path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['iso_time','t_rel_s','event','segment_id','planned_value','actual_value'])
        if not exists:
          writer.writeheader()
        writer.writerow({k: rec.get(k) for k in ['iso_time','t_rel_s','event','segment_id','planned_value','actual_value']})

  if args.timeline_log:
    _write_timeline(args.timeline_log, {
      't_rel_s': 0.0,
      'event': 'plan_created',
      'segment_id': None,
      'planned_value': float(plan.get('total_energy_estimate', 0.0)),
      'actual_value': None,
    })
  if args.error_codes and not plan.get('feasible', False):
    # Exit code 2 for infeasible plan
    print(json.dumps({ 'error': 'infeasible_plan' }))
    return 2
  # Rehearsal or dry-run path: do not execute
  if args.rehearsal or args.dry_run_abort:
    result = {
      'planned_GJ': plan['total_energy_estimate']/1e9,
      'actual_GJ': None,
      'success': plan.get('feasible', True),
      'segments': len(plan.get('segments', [])),
      'mode': 'rehearsal' if args.rehearsal else 'dry-run-abort'
    }
    # If export requested, emit a minimal mission JSON that conforms to the schema
    if args.export:
      export = {
        'schema': 'impulse.mission.v1',
        'version': 1,
        'meta': {
          'hybrid_mode': args.hybrid,
          'estimate_first_threshold': args.threshold,
          'rehearsal': bool(args.rehearsal),
        },
        'waypoints': [
          {
            'position': {'x': float(w.position.x), 'y': float(w.position.y), 'z': float(w.position.z)},
            'orientation': None,
            'dwell_time': float(w.dwell_time),
          } for w in wps
        ]
      }
      Path(args.export).write_text(json.dumps(export, indent=2))
    # Timeline finalize event
    if args.timeline_log:
      _write_timeline(args.timeline_log, {
        't_rel_s': 0.0,
        'event': 'rehearsal_complete' if args.rehearsal else 'dry_run_abort_complete',
        'segment_id': None,
        'planned_value': float(plan.get('total_energy_estimate', 0.0)),
        'actual_value': None,
      })
    print(json.dumps(result, indent=2))
    return 0 if (not args.error_codes or result['success']) else 2

  import asyncio
  res = asyncio.get_event_loop().run_until_complete(ctrl.execute_impulse_mission(
    plan,
    json_export_path=args.export,
    raise_on_abort=args.raise_on_abort,
    verbose_export=args.verbose_export,
    export_cache=args.export_cache,
    perf_csv_path=args.perf_csv
  ))
  if args.timeline_log:
    # Record mission completion with actual energy
    try:
      actual = float(res['performance_metrics']['total_energy_used'])
    except Exception:
      actual = None
    _write_timeline(args.timeline_log, {
      't_rel_s': None,
      'event': 'mission_complete',
      'segment_id': None,
      'planned_value': float(plan.get('total_energy_estimate', 0.0)),
      'actual_value': actual,
    })
  if args.error_codes and not res.get('mission_success', True):
    # Exit code 3 for budget abort during execution
    print(json.dumps({ 'error': 'budget_abort' }))
    return 3
  print(json.dumps({
    'planned_GJ': plan['total_energy_estimate']/1e9,
    'actual_GJ': res['performance_metrics']['total_energy_used']/1e9,
    'success': res['mission_success'],
    'segments': len(res['segment_results'])
  }, indent=2))


if __name__ == '__main__':
  main()
