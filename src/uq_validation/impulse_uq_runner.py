from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Avoid heavy imports at module import time
    from impulse import IntegratedImpulseController, ImpulseEngineConfig, MissionWaypoint  # type: ignore
    from src.simulation.simulate_vector_impulse import Vector3D  # type: ignore
    from simulate_rotation import Quaternion  # type: ignore


@dataclass
class ImpulseUQConfig:
    samples: int = 25
    distance_profile: Optional[List[float]] = None
    dwell_time_mean: float = 5.0
    dwell_time_std: float = 1.0
    approach_speed_mean: float = 4e-5
    approach_speed_std: float = 5e-6
    hybrid_mode: str = 'simulate-first'
    seed: Optional[int] = 123
    energy_budget: float = 5e12
    max_velocity: float = 5e-5
    max_angular_velocity: float = 0.2

    def __post_init__(self):
        if self.distance_profile is None:
            self.distance_profile = [10.0, 20.0, 15.0]


def _waypoints_from_profile(dists: List[float], dwell: float, v_approach: float) -> List["MissionWaypoint"]:
    # Local imports to keep module import light for tests that only need CSV parsing
    from impulse import MissionWaypoint  # type: ignore
    from src.simulation.simulate_vector_impulse import Vector3D  # type: ignore
    from simulate_rotation import Quaternion  # type: ignore

    wps = [MissionWaypoint(position=Vector3D(0, 0, 0), orientation=Quaternion(1, 0, 0, 0), dwell_time=dwell, approach_speed=v_approach)]
    acc = 0.0
    for d in dists:
        acc += d
        wps.append(MissionWaypoint(position=Vector3D(acc, 0, 0), orientation=Quaternion(1, 0, 0, 0), dwell_time=dwell, approach_speed=v_approach))
    return wps


def _load_distance_profile(path: Optional[str]) -> Optional[List[float]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    # Try JSON first
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return [float(x) for x in data]
        if isinstance(data, dict) and 'distances' in data:
            return [float(x) for x in data['distances']]
    except Exception:
        pass
    # Fallback CSV: one value per line or comma-separated; allow inline comments starting with '#'
    try:
        text = p.read_text().strip()
        raw = [s.strip() for line in text.splitlines() for s in line.split(',')]
        parts = []
        for tok in raw:
            if not tok:
                continue
            if tok.startswith('#'):
                continue
            if '#' in tok:
                tok = tok.split('#', 1)[0].strip()
                if not tok:
                    continue
            parts.append(tok)
        return [float(x) for x in parts]
    except Exception:
        return None


def parse_csv(path: str) -> List[float]:
    """Public helper to parse a CSV/JSON distance profile.

    Returns a list of floats or raises ValueError if parsing fails.
    """
    vals = _load_distance_profile(path)
    if vals is None:
        raise ValueError(f"Malformed or unreadable distance profile: {path}")
    return vals


def run_impulse_uq(cfg: ImpulseUQConfig) -> Dict[str, Any]:
    # Localize heavy imports to avoid paying cost during module import when only helpers are needed
    from impulse import IntegratedImpulseController, ImpulseEngineConfig  # type: ignore

    if cfg.seed is not None:
        random.seed(cfg.seed)
    config = ImpulseEngineConfig(
        energy_budget=cfg.energy_budget,
        max_velocity=cfg.max_velocity,
        max_angular_velocity=cfg.max_angular_velocity,
    )
    ctrl = IntegratedImpulseController(config)
    records = []
    infeasible = 0
    for i in range(cfg.samples):
        dwell = max(0.1, random.gauss(cfg.dwell_time_mean, cfg.dwell_time_std))
        v_app = max(1e-6, random.gauss(cfg.approach_speed_mean, cfg.approach_speed_std))
        dists = cfg.distance_profile if cfg.distance_profile is not None else [10.0, 20.0, 15.0]
        wps = _waypoints_from_profile(dists, dwell, v_app)
        plan = ctrl.plan_impulse_trajectory(wps, hybrid_mode=cfg.hybrid_mode)
        feasible = bool(plan['feasible'])
        if not feasible:
            infeasible += 1
        results = {
            'planned_energy': float(plan['total_energy_estimate']),
            'feasible': feasible,
            'hybrid_mode': plan.get('hybrid_mode'),
            'dwell_time': float(dwell),
            'approach_speed': float(v_app),
        }
        records.append(results)
    energies = [r['planned_energy'] for r in records]
    mean_e = sum(energies) / max(1, len(energies))
    var_e = sum((e - mean_e)**2 for e in energies) / max(1, (len(energies) - 1))
    stdev_e = math.sqrt(var_e) if var_e > 0 else 0.0
    return {
        'samples': len(records),
        'infeasible_count': infeasible,
        'feasible_fraction': (len(records) - infeasible) / max(1, len(records)),
        'energy_mean': mean_e,
        'energy_std': stdev_e,
        'energy_cv': (stdev_e / mean_e) if mean_e > 0 else 0.0,
        'records': records,
        'distance_profile': cfg.distance_profile,
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Impulse UQ runner")
    ap.add_argument('--samples', type=int, default=25)
    ap.add_argument('--out', type=str, default=None, help='Optional JSON output path')
    ap.add_argument('--hybrid', type=str, default='simulate-first')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--jsonl-out', type=str, default=None, help='Optional JSONLines output path for per-sample records')
    ap.add_argument('--dist-profile', type=str, default=None, help='Path to CSV/JSON of distances to use for all samples')
    args = ap.parse_args(argv)
    dist_profile = _load_distance_profile(args.dist_profile)
    cfg = ImpulseUQConfig(samples=args.samples, seed=args.seed, hybrid_mode=args.hybrid, distance_profile=dist_profile)
    summary = run_impulse_uq(cfg)
    if args.jsonl_out:
        with open(args.jsonl_out, 'w') as fh:
            for rec in summary.get('records', []):
                fh.write(json.dumps(rec) + "\n")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"Wrote UQ summary to {args.out}")
    else:
        print(json.dumps({k: summary[k] for k in ['samples', 'infeasible_count', 'feasible_fraction', 'energy_mean', 'energy_std', 'energy_cv']}, indent=2))
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
