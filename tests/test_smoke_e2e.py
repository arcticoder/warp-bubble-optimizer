from __future__ import annotations
from src.smoke_e2e_demo import run_demo

def test_smoke_e2e_demo():
    out = run_demo(steps=5)
    assert out['steps'] == 5
    assert isinstance(out['final_state'], list) and len(out['final_state']) == 2
    assert out['mean_innovation'] >= 0.0
