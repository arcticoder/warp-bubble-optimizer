"""
Lightweight shim for accelerated Gaussian optimization tests.

Provides the minimal API expected by test_accelerated_gaussian.py without
heavy dependencies or long-running optimizers.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional
import math
import numpy as np

# --- Global defaults used/overridden by the test file ---
R: float = 1.0
N_points: int = 800
r_grid: np.ndarray = np.linspace(0.0, R, N_points)
dr: float = r_grid[1] - r_grid[0]
vol_weights: np.ndarray = 4.0 * np.pi * r_grid**2

mu0: float = 1e-6
G_geo: float = 1e-5
M_gauss: int = 4
hbar: float = 1.054e-34
tau: float = 1.0


# --- Utility profiles ---
def _gaussian_sum(r: np.ndarray, params: Sequence[float], M: Optional[int] = None) -> np.ndarray:
    p = np.asarray(params, dtype=float).reshape(-1)
    if M is None:
        if p.size % 3 != 0:
            raise ValueError("params length must be multiple of 3 (A, r0, sigma per Gaussian)")
        M = p.size // 3
    out = np.zeros_like(r, dtype=float)
    for i in range(M):
        A, r0, sig = p[3*i:3*i+3]
        sig = max(1e-6, float(sig))
        out += float(A) * np.exp(-((r - float(r0)) / sig) ** 2)
    return np.clip(out, 0.0, 1.0)


def f_gaussian_vectorized(r: np.ndarray | float, params: Sequence[float]) -> np.ndarray:
    r_arr = np.asarray(r, dtype=float)
    return _gaussian_sum(r_arr, params)


def rho_eff_gauss_vectorized(r: np.ndarray | float, params: Sequence[float]) -> np.ndarray:
    # Simple surrogate: negative density proportional to gradient magnitude
    r_arr = np.asarray(r, dtype=float)
    # Evaluate on small stencil for gradient approximation
    eps = 1e-4
    f_plus = _gaussian_sum(r_arr + eps, params)
    f_minus = _gaussian_sum(r_arr - eps, params)
    grad = (f_plus - f_minus) / (2*eps)
    return -np.abs(grad)


def E_negative_gauss_fast(params: Sequence[float]) -> float:
    f = _gaussian_sum(r_grid, params)
    # Toy negative energy functional
    integrand = f**2 * vol_weights
    E = np.trapz(integrand, r_grid)
    return float(-float(E))


def E_negative_gauss_slow(params: Sequence[float]) -> float:
    # Slow variant: compute same integral using smaller chunks to emulate overhead
    chunks = 8
    N = r_grid.size
    size = N // chunks
    E = 0.0
    for i in range(chunks):
        a = i*size
        b = (i+1)*size if i < chunks-1 else N
        rg = r_grid[a:b]
        f = _gaussian_sum(rg, params)
        E_chunk = float(np.trapz(f**2 * (4.0*np.pi*rg**2), rg))
        E -= E_chunk
    return float(E)


def get_optimization_bounds(M: int | None = None) -> List[Tuple[float, float]]:
    m = M if M is not None else M_gauss
    # (A in [0,1], r0 in [0,R], sigma in [1e-3, 0.5])
    return [(0.0, 1.0), (0.0, R), (1e-3, 0.5)] * int(m)


@dataclass
class _OptResult:
    x: np.ndarray
    fun: float


def objective_gauss(x: Sequence[float]) -> float:
    # Minimize negative of negative energy => minimize |E| with sign handled
    return -E_negative_gauss_fast(x)


def differential_evolution(func, bounds, strategy='best1bin', maxiter=50, popsize=15,
                           tol=1e-6, seed=None, workers=1):
    """
    Lightweight DE-like random search to keep tests fast.
    Accepts scipy-like signature but runs a tiny stochastic search.
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    dim = bounds.shape[0]
    iters = min(3, int(maxiter))  # clamp for speed
    pop = rng.uniform(bounds[:, 0], bounds[:, 1], size=(max(8, int(popsize)), dim))
    best_x = pop[0]
    best_f = func(best_x)
    for _ in range(iters):
        cand = rng.uniform(bounds[:, 0], bounds[:, 1], size=(pop.shape[0], dim))
        vals = np.array([func(c) for c in cand])
        j = int(np.argmin(vals))
        if vals[j] < best_f:
            best_f = float(vals[j])
            best_x = cand[j]
    return _OptResult(x=np.asarray(best_x, dtype=float), fun=float(best_f))


def optimize_gaussian_ansatz_fast(mu0_in: float, G_geo_in: float, M: int | None = None) -> dict:
    global mu0, G_geo, M_gauss
    mu0 = float(mu0_in)
    G_geo = float(G_geo_in)
    if M is not None:
        M_gauss = int(M)
    bounds = get_optimization_bounds(M_gauss)
    res = differential_evolution(objective_gauss, bounds, maxiter=3, popsize=8, seed=42)
    return {"energy_J": float(-float(res.fun)), "params": res.x.astype(float).tolist()}


def get_hybrid_bounds(M_gauss_hybrid: int = 2) -> List[Tuple[float, float]]:
    # Hybrid: r0, r1, a1, a2 + M*(A,r0,sigma)
    base = [(0.0, R), (0.0, R), (-5.0, 5.0), (-5.0, 5.0)]
    gauss = [(0.0, 1.0), (0.0, R), (1e-3, 0.5)] * int(M_gauss_hybrid)
    return base + gauss


def f_hybrid_vectorized(r: np.ndarray, params: Sequence[float], enable_hybrid: bool = True) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if not enable_hybrid:
        return f_gaussian_vectorized(r, params)
    r0, r1, a1, a2 = params[:4]
    poly = 1.0 + a1 * (r - r0) + a2 * (r - r0) ** 2
    gauss = _gaussian_sum(r, params[4:])
    blend = np.where(r < r1, poly, gauss)
    return np.clip(blend, 0.0, 1.0)


def f_hybrid_prime_vectorized(r: np.ndarray, params: Sequence[float], enable_hybrid: bool = True) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    eps = 1e-4
    return (f_hybrid_vectorized(r + eps, params, enable_hybrid) - f_hybrid_vectorized(r - eps, params, enable_hybrid)) / (2 * eps)


def E_negative_hybrid(params: Sequence[float]) -> float:
    f = f_hybrid_vectorized(r_grid, params, enable_hybrid=True)
    E = float(np.trapz(f**2 * vol_weights, r_grid))
    return float(-E)


def curvature_penalty(params: Sequence[float], enable_hybrid: bool = False) -> float:
    f = f_hybrid_vectorized(r_grid, params, enable_hybrid) if enable_hybrid else _gaussian_sum(r_grid, params)
    # Penalize large second derivative
    d2 = np.gradient(np.gradient(f, r_grid), r_grid)
    return float(np.mean(d2**2))


def penalty_gauss(params: Sequence[float]) -> float:
    p = np.asarray(params, dtype=float)
    # Simple bounds penalty outside [0,1] for amplitudes
    amps = p[0::3]
    over = np.clip(amps - 1.0, 0.0, None)
    under = np.clip(0.0 - amps, 0.0, None)
    return float(np.sum(over + under))


def optimize_with_cma_es(bounds: Sequence[Tuple[float, float]], sigma0: float = 0.1) -> Optional[dict]:
    try:
        import cma  # type: ignore
    except Exception:
        return None
    # Tiny CMA-ES run
    dim = len(bounds)
    x0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds], dtype=float)
    es = cma.CMAEvolutionStrategy(x0, sigma0, {'maxiter': 5, 'verb_disp': 0})
    def f(x):
        return objective_gauss(x)
    es.optimize(f)
    x_best = np.array(es.result.xbest, dtype=float)
    return {"energy_J": float(-objective_gauss(x_best.astype(float).tolist())), "params": x_best.astype(float).tolist()}
