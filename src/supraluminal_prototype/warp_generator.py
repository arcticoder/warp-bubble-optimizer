from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from .power import compute_smearing_energy


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


def synthesize_shift_with_envelope(params: Dict) -> Dict:
    """
    Build a divergence-free shift v' by taking v' = curl( e * A ), where A=(0,0,psi) and e is the synthesized envelope.
    This preserves ∇·v' = 0 up to numerical error.
    Required params: {'grid': GridSpec, 'R': float, 'sigma': float, 'ring_controls': array-like of length 4}
    Returns dict with 'grid' and 'shift'.
    """
    grid = params.get('grid', GridSpec())
    if not isinstance(grid, GridSpec):
        raise ValueError("grid must be a GridSpec instance")
    R = float(params.get('R', 2.5))
    xs, ys, zs = grid.linspaces()
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    r2 = X**2 + Y**2 + Z**2
    psi = np.exp(-r2 / (R**2))
    env = field_synthesis(params.get('ring_controls', [1, 1, 1, 1]), {'grid': grid, 'sigma': params.get('sigma', 0.2 * grid.extent)})['envelope']
    Bz = env * psi
    # v' = curl(B) with B=(0,0,Bz)
    # central differences with reflective boundaries
    def ddx(a: np.ndarray, h: float) -> np.ndarray:
        ap = np.pad(a, ((1, 1), (0, 0), (0, 0)), mode='edge')
        return (ap[2:, :, :] - ap[:-2, :, :]) / (2.0 * h)

    def ddy(a: np.ndarray, h: float) -> np.ndarray:
        ap = np.pad(a, ((0, 0), (1, 1), (0, 0)), mode='edge')
        return (ap[:, 2:, :] - ap[:, :-2, :]) / (2.0 * h)

    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    vx = ddy(Bz, dy)
    vy = -ddx(Bz, dx)
    vz = np.zeros_like(vx)
    shift = np.stack([vx, vy, vz], axis=-1)
    return {'grid': (xs, ys, zs), 'shift': shift}


def optimize_energy(params: Dict) -> Dict:
    """
    Minimal optimization stub: evaluate smearing energy and fit error for uniform ring amplitudes.
    Inputs: {'grid': GridSpec, 'P_peak': float, 't_ramp': float, 't_cruise': float,
             'sigma': float, 'target': np.ndarray (optional)}
    Returns: {'E': float, 'best_controls': np.ndarray, 'fit_error': float}
    """
    P_peak = float(params.get('P_peak', 25e6))
    t_ramp = float(params.get('t_ramp', 30.0))
    t_cruise = float(params.get('t_cruise', 2.56))
    E = compute_smearing_energy(P_peak, t_ramp, t_cruise)
    grid = params.get('grid', GridSpec())
    if not isinstance(grid, GridSpec):
        raise ValueError("grid must be a GridSpec instance")
    target_env = params.get('target')
    if target_env is None:
        target_env = target_soliton_envelope({'grid': grid, 'r0': 0.0, 'sigma': 0.5 * grid.extent})['envelope']
    best_rc, best_err = tune_ring_amplitudes_uniform(np.zeros(4), {'grid': grid, 'sigma': params.get('sigma', 0.2 * grid.extent)}, target_env, n_steps=17)
    # Simple discharge efficiency model vs C-rate: eta = eta0 - k * C_rate
    # Estimate C-rate using P_peak and capacity if available; otherwise assume eta0
    E_cap = params.get('battery_capacity_J')
    eta0 = float(params.get('battery_eta0', 0.95))
    k = float(params.get('battery_eta_slope', 0.05))  # efficiency drop per 1C
    eta = eta0
    if E_cap is not None and P_peak > 0:
        # Approximate pack voltage constant; use energy/capacity to estimate effective C-rate over ramp window
        # Define an equivalent C-rate proxy: C_rate_proxy = (P_peak * t_ramp) / E_cap per ramp
        C_rate_proxy = (P_peak * max(t_ramp, 1e-6)) / float(E_cap)
        eta = max(0.5, min(eta0, eta0 - k * C_rate_proxy))
    E_effective = E / max(eta, 1e-6)
    feasible = True if E_cap is None else (E_effective <= float(E_cap))
    return {'E': E, 'E_effective': E_effective, 'best_controls': best_rc, 'fit_error': best_err, 'feasible': feasible, 'eta': eta}


def target_soliton_envelope(params: Dict) -> Dict:
    """
    Build a simple radial soliton-like target envelope using sech^2((r-r0)/sigma).
    Result normalized to [0,1] for comparison.
    """
    grid = params.get('grid', GridSpec())
    if not isinstance(grid, GridSpec):
        raise ValueError("grid must be a GridSpec instance")
    xs, ys, zs = grid.linspaces()
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r0 = float(params.get('r0', 0.0))
    sigma = float(params.get('sigma', 0.5 * grid.extent))
    def sech(x: np.ndarray) -> np.ndarray:
        return 1.0 / np.cosh(x)
    env = (sech((r - r0) / sigma)) ** 2
    env_max = np.max(env)
    if env_max > 0:
        env = env / env_max
    return {'grid': (xs, ys, zs), 'envelope': env}


def compute_envelope_error(envelope: np.ndarray, target: np.ndarray, norm: str = 'l2') -> float:
    """
    Compute error between synthesized envelope and target.
    - norm='l2': sqrt(mean((a-b)^2))
    - norm='l1': mean(|a-b|)
    """
    a = np.asarray(envelope, dtype=float)
    b = np.asarray(target, dtype=float)
    if a.shape != b.shape:
        raise ValueError("envelope and target must have same shape")
    if norm == 'l2':
        return float(np.sqrt(np.mean((a - b) ** 2)))
    if norm == 'l1':
        return float(np.mean(np.abs(a - b)))
    raise ValueError("Unsupported norm")


def tune_ring_amplitudes_uniform(ring_controls_init: np.ndarray, params: Dict,
                                 target_envelope: np.ndarray, n_steps: int = 21) -> Tuple[np.ndarray, float]:
    """
    One-dimensional line search over a uniform scale factor s in [0,1] applied to a base vector of ones(4).
    Returns (best_controls, best_error).
    """
    base = np.ones(4, dtype=float)
    best_err = float('inf')
    best_controls = np.zeros(4, dtype=float)
    for i in range(n_steps):
        s = i / (n_steps - 1)
        rc = np.clip(s * base, 0.0, 1.0)
        env = field_synthesis(rc, params)['envelope']
        err = compute_envelope_error(env, target_envelope, norm='l2')
        if err < best_err:
            best_err = err
            best_controls = rc
    return best_controls, best_err
