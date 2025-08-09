import os, sys
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
from supraluminal_prototype.warp_generator import build_metric, expansion_scalar, GridSpec
from supraluminal_prototype.control import sync_rings
from supraluminal_prototype.hardware import CoilDriver


def test_zero_expansion_metric():
    metric = build_metric({'R': 2.5, 'grid': GridSpec(nx=16, ny=16, nz=16, extent=1.0)})
    theta = expansion_scalar(metric)
    # Discrete zero-divergence up to numerical tolerance on coarse grids
    assert np.nanmax(np.abs(theta)) < 5e-2


def test_ring_sync_tolerance():
    phases = np.array([0.0, 0.2e-6, 0.5e-6, 0.8e-6])
    assert bool(sync_rings(phases, jitter=1e-6))
    assert not bool(sync_rings(phases, jitter=0.5e-6))


def test_coil_driver_linearity():
    drv = CoilDriver(max_current=5000.0, hysteresis=0.0)
    currents = [drv.command(i/30.0) for i in range(31)]
    diffs = [currents[i]-currents[i-1] for i in range(1, len(currents))]
    # Nearly constant step
    assert max(abs(d - diffs[0]) for d in diffs[1:]) < 1e-6
