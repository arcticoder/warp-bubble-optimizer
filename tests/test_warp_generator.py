from __future__ import annotations

import numpy as np

from src.supraluminal_prototype.warp_generator import (
    GridSpec,
    field_synthesis,
    target_soliton_envelope,
    compute_envelope_error,
    plasma_density,
)


def test_hyperbolic_wave_sync_envelope_alignment():
    # Interpret "hyperbolic_wave" sync as aligning synthesized ring envelope to a soliton-like target
    grid = GridSpec(nx=24, ny=24, nz=9, extent=1.0)
    tgt = target_soliton_envelope({'grid': grid, 'r0': 0.0, 'sigma': 0.5 * grid.extent})['envelope']
    # Equal ring drive at moderate amplitude
    import numpy as _np
    env = field_synthesis(_np.array([0.6, 0.6, 0.6, 0.6], dtype=float), {'grid': grid, 'sigma': 0.25 * grid.extent})['envelope']
    err = compute_envelope_error(env, tgt, norm='l2')
    # Expect a reasonable alignment (not perfect), enforce an upper bound on error
    assert err < 0.45


def test_plasma_density_shell_peak_and_falloff():
    grid = GridSpec(nx=21, ny=21, nz=21, extent=1.0)
    pd = plasma_density({'grid': grid, 'n0': 3e20, 'R_shell': 0.6 * grid.extent, 'width': 0.15 * grid.extent})
    n = pd['n']
    # Peak should be positive and within [0, n0]
    assert float(np.max(n)) > 0.0
    # Density near center should be much lower than peak
    center = n[grid.nx // 2, grid.ny // 2, grid.nz // 2]
    assert center < 0.2 * np.max(n)
