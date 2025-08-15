from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys


def run_cmd(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True)


def seed_waypoints(tmp: Path) -> Path:
    wp = tmp / "simple.json"
    wp.write_text(
        json.dumps({
            "waypoints": [
                {"x": 0, "y": 0, "z": 0},
                {"x": 50, "y": 0, "z": 0}
            ],
            "dwell": 1.0
        })
    )
    return wp


def test_timeline_log_quick(tmp_path):
    # Mark as quick by name; pytest.ini already defines markers
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    wp = seed_waypoints(tmp_path)
    timeline = tmp_path / "timeline.csv"
    # Rehearsal with timeline log
    cp = subprocess.run([
        sys.executable, "-m", "impulse.mission_cli",
        "--waypoints", str(wp),
        "--rehearsal",
        "--timeline-log", str(timeline),
        "--hybrid", "simulate-first",
        "--seed", "123"
    ], env=env, text=True, capture_output=True)
    assert cp.returncode == 0, cp.stderr
    assert timeline.exists(), "Timeline CSV not created"
    # Must contain header and at least one row
    lines = timeline.read_text().strip().splitlines()
    assert len(lines) >= 2
    assert lines[0].split(',')[0] == 'iso_time'


def test_rehearsal_vs_execution_schema(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    wp = seed_waypoints(tmp_path)
    # Rehearsal export
    rehearsal = tmp_path / "mission_rehearsal.json"
    cp1 = subprocess.run([
        sys.executable, "-m", "impulse.mission_cli",
        "--waypoints", str(wp),
        "--export", str(rehearsal),
        "--rehearsal",
        "--hybrid", "simulate-first",
        "--seed", "123"
    ], env=env, text=True, capture_output=True)
    assert cp1.returncode == 0, cp1.stderr
    # Execution export
    exec_export = tmp_path / "mission_exec.json"
    cp2 = subprocess.run([
        sys.executable, "-m", "impulse.mission_cli",
        "--waypoints", str(wp),
        "--export", str(exec_export),
        "--hybrid", "simulate-first",
        "--seed", "123"
    ], env=env, text=True, capture_output=True)
    assert cp2.returncode == 0, cp2.stderr
    # Validate rehearsal JSON against schema (plan-level schema)
    val_reh = subprocess.run([sys.executable, "bin/validate_mission_json.py", str(rehearsal)], text=True, capture_output=True)
    assert val_reh.returncode == 0, val_reh.stderr
    # Structural checks
    reh = json.loads(rehearsal.read_text())
    exe = json.loads(exec_export.read_text())
    # Rehearsal export should contain waypoints (plan-level)
    assert 'waypoints' in reh and isinstance(reh['waypoints'], list) and len(reh['waypoints']) >= 1
    # Execution export should include segment results
    assert 'segments' in exe or 'segment_results' in exe.get('results', {})
    # Execution should have additional performance metrics under results
    assert 'results' in exe and 'performance_metrics' in exe['results']
