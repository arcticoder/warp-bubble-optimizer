from src.uq_validation.metamaterial_uq import run_metamaterial_uq
from src.uq_validation.laser_coherence_uq import run_laser_coherence_uq


def test_metamaterial_uq_stats_quick():
    res = run_metamaterial_uq(n=64)
    assert 0.0 < res['cv'] < 0.5


def test_laser_coherence_uq_feasible():
    res = run_laser_coherence_uq(n=64)
    assert res['feasible'] is True
