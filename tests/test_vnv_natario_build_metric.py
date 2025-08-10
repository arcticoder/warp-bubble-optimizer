#!/usr/bin/env python3
"""
V&V: Verify build_metric (Natário-like) generates divergence-free shift field.
"""
import numpy as np
from src.supraluminal_prototype.warp_generator import (
    GridSpec,
    build_metric,
    expansion_scalar,
    synthesize_shift_with_envelope,
)


def test_build_metric_divergence_free():
    grid = GridSpec(nx=16, ny=16, nz=16, extent=1.0)
    metric = build_metric({'grid': grid, 'R': 2.5})
    xs, ys, zs = metric['grid']
    shift = metric['shift']

    # Shape checks
    assert shift.shape == (len(xs), len(ys), len(zs), 3)

    # Divergence (expansion) should be near zero (discrete approx; reflective boundaries)
    theta = expansion_scalar(metric)
    assert theta.shape == (len(xs), len(ys), len(zs))
    # Empirically max can be ~3.7e-2 at coarse 16^3 due to discrete differencing; enforce small mean
    assert float(np.max(np.abs(theta))) < 5e-2
    assert float(np.mean(np.abs(theta))) < 5e-3


def test_envelope_coupled_shift_remains_near_div_free():
    grid = GridSpec(nx=16, ny=16, nz=16, extent=1.0)
    metric_prime = synthesize_shift_with_envelope({'grid': grid, 'R': 2.5, 'sigma': 0.2})
    # reuse expansion_scalar by wrapping into metric-like dict
    theta = expansion_scalar({'grid': metric_prime['grid'], 'shift': metric_prime['shift']})
    assert float(np.max(np.abs(theta))) < 6e-2
    assert float(np.mean(np.abs(theta))) < 6e-3
