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
