from __future__ import annotations

import json
from pathlib import Path

from impulse.mission_cli import main as mission_main
from impulse.mission_cli import main as mission_main


def test_mission_cli_smoke(tmp_path):
    waypoints = {
        "dwell": 2.0,
        "waypoints": [
            {"x": 0, "y": 0, "z": 0},
            {"x": 10, "y": 0, "z": 0}
        ]
    }
    wp_path = tmp_path / 'wp.json'
    wp_path.write_text(json.dumps(waypoints))
    out_json = tmp_path / 'mission.json'
    perf_csv = tmp_path / 'perf.csv'
    rc = mission_main([
        '--waypoints', str(wp_path),
        '--export', str(out_json),
        '--perf-csv', str(perf_csv),
        '--hybrid', 'simulate-first'
    ])
    assert rc is None or rc == 0
    assert out_json.exists() and perf_csv.exists()
    assert 'segment_index' in perf_csv.read_text()


def test_mission_cli_seed_consistency(tmp_path):
    waypoints = {
        "dwell": 2.0,
        "waypoints": [
            {"x": 0, "y": 0, "z": 0},
            {"x": 10, "y": 0, "z": 0}
        ]
    }
    wp_path = tmp_path / 'wp.json'
    wp_path.write_text(json.dumps(waypoints))
    out1 = tmp_path / 'm1.json'
    out2 = tmp_path / 'm2.json'
    rc1 = mission_main(['--waypoints', str(wp_path), '--export', str(out1), '--hybrid', 'simulate-first', '--seed', '123', '--verbose-export'])
    rc2 = mission_main(['--waypoints', str(wp_path), '--export', str(out2), '--hybrid', 'simulate-first', '--seed', '123', '--verbose-export'])
    assert (rc1 == 0 or rc1 is None) and (rc2 == 0 or rc2 is None)
    d1 = json.loads(out1.read_text())
    d2 = json.loads(out2.read_text())
    # Compare key metrics for determinism
    assert d1['plan']['total_energy_estimate'] == d2['plan']['total_energy_estimate']
    assert d1['results']['performance_metrics']['total_energy_used'] == d2['results']['performance_metrics']['total_energy_used']
    # Check meta randomization seeds persisted
    assert 'meta' in d1 and 'randomization' in d1['meta'] and 'env_seed' in d1['meta']['randomization']
