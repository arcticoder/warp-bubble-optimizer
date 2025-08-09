from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class GridSpec:
    nx: int = 16
    ny: int = 16
    nz: int = 16
    extent: float = 1.0  # coordinate half-extent in each axis

    def linspaces(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.linspace(-self.extent, self.extent, self.nx)
        ys = np.linspace(-self.extent, self.extent, self.ny)
        zs = np.linspace(-self.extent, self.extent, self.nz)
        return xs, ys, zs


def build_metric(params: Dict) -> Dict:
    """
    Build a simple Natário-inspired metric representation via a divergence-free shift field.

    Returns a dict with keys:
    - 'grid': (xs, ys, zs)
    - 'shift': array shape (nx, ny, nz, 3) representing v(x)

    Construction: v = curl A with A = (0, 0, psi), psi = exp(-r^2 / R^2)
    so v = (∂psi/∂y, -∂psi/∂x, 0), which guarantees ∇·v = 0 analytically.
    """
    R = float(params.get('R', 2.5))
    grid = params.get('grid', GridSpec())
    if not isinstance(grid, GridSpec):
        raise ValueError("grid must be a GridSpec instance")
    xs, ys, zs = grid.linspaces()
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    r2 = X**2 + Y**2 + Z**2
    psi = np.exp(-r2 / (R**2))
    # Analytic derivatives of psi
    dpsi_dx = psi * (-2.0 * X / (R**2))
    dpsi_dy = psi * (-2.0 * Y / (R**2))
    # v = curl A with A=(0,0,psi) => v=(dpsi/dy, -dpsi/dx, 0)
    vx = dpsi_dy
    vy = -dpsi_dx
    vz = np.zeros_like(vx)
    shift = np.stack([vx, vy, vz], axis=-1)
    return {'grid': (xs, ys, zs), 'shift': shift}


def expansion_scalar(metric: Dict, spacing: Tuple[float, float, float] | None = None) -> np.ndarray:
    """
    Compute discrete divergence of the shift vector field (Natário zero-expansion target).
    Returns theta array shape (nx, ny, nz). Expect near-zero up to numerical error.
    """
    xs, ys, zs = metric['grid']
    v = metric['shift']  # (nx, ny, nz, 3)
    nx, ny, nz, _ = v.shape
    if spacing is None:
        dx = float(xs[1] - xs[0]) if nx > 1 else 1.0
        dy = float(ys[1] - ys[0]) if ny > 1 else 1.0
        dz = float(zs[1] - zs[0]) if nz > 1 else 1.0
    else:
        dx, dy, dz = spacing

    # Central differences with reflective boundaries to reduce wrap-around artifacts
    def ddx(a: np.ndarray, h: float) -> np.ndarray:
        ap = np.pad(a, ((1, 1), (0, 0), (0, 0)), mode='edge')
        return (ap[2:, :, :] - ap[:-2, :, :]) / (2.0 * h)

    def ddy(a: np.ndarray, h: float) -> np.ndarray:
        ap = np.pad(a, ((0, 0), (1, 1), (0, 0)), mode='edge')
        return (ap[:, 2:, :] - ap[:, :-2, :]) / (2.0 * h)

    def ddz(a: np.ndarray, h: float) -> np.ndarray:
        ap = np.pad(a, ((0, 0), (0, 0), (1, 1)), mode='edge')
        return (ap[:, :, 2:] - ap[:, :, :-2]) / (2.0 * h)

    dvx_dx = ddx(v[..., 0], dx)
    dvy_dy = ddy(v[..., 1], dy)
    dvz_dz = ddz(v[..., 2], dz)
    theta = dvx_dx + dvy_dy + dvz_dz
    return theta


def plasma_density(params: Dict) -> Dict:
    """
    Build a simple plasma density distribution over the grid.

    Parameters keys:
    - 'grid': GridSpec
    - 'n0': peak density (default 3e20 m^-3)
    - 'R_shell': shell radius in grid coordinates (default 0.6*extent)
    - 'width': shell Gaussian width (default 0.15*extent)

    Returns dict with keys: 'grid', 'n' (nx,ny,nz)
    """
    grid = params.get('grid', GridSpec())
    if not isinstance(grid, GridSpec):
        raise ValueError("grid must be a GridSpec instance")
    n0 = float(params.get('n0', 3e20))
    xs, ys, zs = grid.linspaces()
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    R_shell = float(params.get('R_shell', 0.6 * grid.extent))
    width = float(params.get('width', 0.15 * grid.extent))
    r = np.sqrt(X**2 + Y**2 + Z**2)
    # Gaussian shell centered at R_shell
    n = n0 * np.exp(-((r - R_shell) ** 2) / (2.0 * width ** 2))
    return {'grid': (xs, ys, zs), 'n': n}


def field_synthesis(ring_controls: np.ndarray, params: Dict) -> Dict:
    """
    Map ring control amplitudes/phases to a scalar envelope field in the grid.

    Inputs:
    - ring_controls: array shape (4,) amplitudes in [0,1] for four toroidal rings
    - params: {'grid': GridSpec, 'ring_positions': list of 4 (x,y,z), 'sigma': float}

    Output dict with keys: 'grid', 'envelope' in [0, 1+epsilon]
    """
    grid = params.get('grid', GridSpec())
    if not isinstance(grid, GridSpec):
        raise ValueError("grid must be a GridSpec instance")
    xs, ys, zs = grid.linspaces()
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    sigma = float(params.get('sigma', 0.2 * grid.extent))
    ring_positions = params.get('ring_positions')
    if ring_positions is None:
        # Default: 4 rings at +/-x, +/-y on equatorial plane z=0
        r0 = 0.7 * grid.extent
        ring_positions = [( r0, 0.0, 0.0), (-r0, 0.0, 0.0), (0.0,  r0, 0.0), (0.0, -r0, 0.0)]
    rc = np.asarray(ring_controls, dtype=float).reshape(-1)
    if rc.size != 4:
        raise ValueError("ring_controls must have 4 elements")

    env = np.zeros_like(X)
    for amp, (x0, y0, z0) in zip(rc, ring_positions):
        d2 = (X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2
        env += float(max(0.0, min(1.0, amp))) * np.exp(-d2 / (2.0 * sigma ** 2))
    # Normalize to [0, 1]
    env_max = np.max(env)
    if env_max > 0:
        env = env / env_max
    return {'grid': (xs, ys, zs), 'envelope': env}
