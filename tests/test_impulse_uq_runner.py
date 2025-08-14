from __future__ import annotations

import json
from pathlib import Path

from src.uq_validation.impulse_uq_runner import ImpulseUQConfig, run_impulse_uq
from src.uq_validation.impulse_uq_runner import main as uq_main


def test_impulse_uq_runner_smoke(tmp_path):
    cfg = ImpulseUQConfig(samples=5, seed=7)
    summary = run_impulse_uq(cfg)
    assert summary['samples'] == 5
    assert 0.0 <= summary['feasible_fraction'] <= 1.0
    assert summary['energy_mean'] >= 0.0

    out_path = tmp_path / 'uq_summary.json'
    out_path.write_text(json.dumps(summary))
    data = json.loads(out_path.read_text())
    assert data['samples'] == 5


def test_uq_runner_jsonl_and_dist_profile(tmp_path):
    # Create a distance profile CSV
    csvp = tmp_path / 'dist.csv'
    csvp.write_text('5,10,5\n')
    out_json = tmp_path / 'uq_summary.json'
    out_jsonl = tmp_path / 'uq_records.jsonl'
    rc = uq_main(['--samples', '7', '--seed', '42', '--out', str(out_json), '--jsonl-out', str(out_jsonl), '--dist-profile', str(csvp)])
    assert rc == 0 or rc is None
    assert out_json.exists() and out_jsonl.exists()
    # Validate JSONL structure
    lines = out_jsonl.read_text().strip().splitlines()
    assert len(lines) == 7
    import json
    rec0 = json.loads(lines[0])
    assert 'planned_energy' in rec0 and 'feasible' in rec0
    # Also test JSON dist-profile format
    jp = tmp_path / 'dist.json'
    jp.write_text('{"distances": [3,6,9]}')
    out2 = tmp_path / 'uq_summary2.json'
    rc2 = uq_main(['--samples', '3', '--seed', '11', '--out', str(out2), '--dist-profile', str(jp)])
    assert rc2 == 0 or rc2 is None
    assert out2.exists()


def test_uq_runner_malformed_dist_profile(tmp_path):
    bad = tmp_path / 'bad.txt'
    bad.write_text('not, numbers, here')
    # Should not crash; falls back to default profile
    rc = uq_main(['--samples', '2', '--seed', '1', '--dist-profile', str(bad)])
    assert rc == 0 or rc is None
