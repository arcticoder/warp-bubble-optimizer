from __future__ import annotations

import json
from pathlib import Path

from src.uq_validation.impulse_uq_runner import ImpulseUQConfig, run_impulse_uq


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
