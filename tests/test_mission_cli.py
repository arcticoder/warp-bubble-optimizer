from __future__ import annotations

import json
from pathlib import Path

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
