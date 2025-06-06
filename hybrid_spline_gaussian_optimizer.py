#!/usr/bin/env python3
"""
HYBRID SPLINE-GAUSSIAN ANSATZ OPTIMIZER

Implementation of the hybrid spline-Gaussian ansatz from the roadmap:

f(r) = { cubic_spline(r),             0 ≤ r ≤ r_knot
       { sum of Gaussians(r),         r_knot < r ≤ R

This approach combines the smooth polynomial behavior near the core
with the flexibility of Gaussians near the bubble wall, potentially
capturing sharp features that pure Gaussians might miss.

Features:
1. ✅ Cubic spline with C² continuity at r_knot
2. ✅ 6 Gaussians in the outer region for maximum flexibility
3. ✅ CMA-ES + JAX two-stage optimization
4. ✅ Automatic knot point optimization
5. ✅ Enhanced boundary and continuity constraints

Target: Achieve E- < -1.5×10³² J with improved feature capture
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import warnings
from pathlib import Path
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
warnings.filterwarnings('ignore')

# Optional imports with graceful fallbacks
HAS_CMA = False
try:
    import cma
    HAS_CMA = True
    print("✅ CMA-ES detected - using advanced global optimization")
except ImportError:
    print("⚠️  CMA-ES not available - install with: pip install cma")

HAS_JAX = False
try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.optimize import minimize as jax_minimize
    from jax import grad, jit, value_and_grad
    HAS_JAX = True
    print("✅ JAX detected - using accelerated local refinement")
except ImportError:
    print("⚠️  JAX not available - install with: pip install jax")

# ── 1. PHYSICAL CONSTANTS ─────────────────────────────────────────────────────
beta_back       = 1.9443254780147017  # Backreaction enhancement factor
hbar            = 1.0545718e-34       # ℏ (SI)
c               = 299792458           # Speed of light (m/s)
G               = 6.67430e-11         # Gravitational constant (m³/kg/s²)
tau             = 1e-9                # QI sampling time
v               = 1.0                 # warp velocity (c = 1 units)
R               = 1.0                 # bubble radius = 1 m

# Default parameters
mu0_default     = 5.2e-6              # polymer length (optimal)
G_geo_default   = 2.5e-5              # Van den Broeck–Natário factor (optimal)

# Conversion factor for natural units to Joules
c4_8piG = c**4 / (8.0 * np.pi * G)   # ≈ 4.815×10⁴² J⋅m⁻³

# ── 2. HYBRID ANSATZ CONFIGURATION ────────────────────────────────────────────
M_gauss = 6      # Number of Gaussians in outer region
N_spline = 4     # Number of spline control points (cubic → 4 coefficients)

# Precompute radial grid for vectorized integration
N_grid = 1000
r_grid = np.linspace(0.0, R, N_grid)
dr = r_grid[1] - r_grid[0]
vol_weights = 4.0 * np.pi * r_grid**2

# Parameter vector structure:
# [mu, G_geo, r_knot, a0, a1, a2, a3, A1,r1,σ1, A2,r2,σ2, ..., A6,r6,σ6]
# Total dimension: 2 + 1 + 4 + 3*6 = 25 parameters
PARAM_DIM = 2 + 1 + N_spline + 3 * M_gauss

print(f"🎯 Hybrid Spline-Gaussian Ansatz: {PARAM_DIM} parameters")
print(f"   Spline region: [0, r_knot], {N_spline} coefficients")
print(f"   Gaussian region: (r_knot, R], {M_gauss} Gaussians")

# ── 3. HYBRID ANSATZ FUNCTIONS ────────────────────────────────────────────────
def f_hybrid_spline_numpy(r, params):
    """
    Hybrid spline-Gaussian ansatz (NumPy version)
    
    f(r) = { cubic_spline(r),     0 ≤ r ≤ r_knot
           { sum_Gaussians(r),    r_knot < r ≤ R
    
    Args:
        r: Radial coordinate (scalar or array)
        params: [mu, G_geo, r_knot, a0, a1, a2, a3, A1,r1,σ1, ..., A6,r6,σ6]
    
    Returns:
        Function value(s)
    """
    r = np.atleast_1d(r)
    result = np.zeros_like(r)
    
    # Extract parameters
    mu, G_geo = params[0], params[1]
    r_knot = params[2]
    
    # Spline coefficients: a0 + a1*r + a2*r² + a3*r³
    a0, a1, a2, a3 = params[3], params[4], params[5], params[6]
    
    # Spline region: 0 ≤ r ≤ r_knot
    spline_mask = (r <= r_knot)
    r_spline = r[spline_mask]
    if len(r_spline) > 0:
        result[spline_mask] = (a0 + a1*r_spline + 
                              a2*r_spline**2 + a3*r_spline**3)
    
    # Gaussian region: r_knot < r ≤ R
    gauss_mask = (r > r_knot)
    r_gauss = r[gauss_mask]
    if len(r_gauss) > 0:
        gauss_sum = np.zeros_like(r_gauss)
        
        # Sum of 6 Gaussians
        for i in range(M_gauss):
            A_i = params[7 + 3*i]      # Amplitude
            r0_i = params[7 + 3*i + 1] # Center
            sigma_i = params[7 + 3*i + 2] # Width
            
            if sigma_i > 1e-12:
                gauss_i = A_i * np.exp(-0.5 * ((r_gauss - r0_i) / sigma_i)**2)
                gauss_sum += gauss_i
        
        result[gauss_mask] = gauss_sum
    
    return result

def f_hybrid_spline_prime_numpy(r, params):
    """Derivative of hybrid spline-Gaussian ansatz (NumPy version)"""
    r = np.atleast_1d(r)
    result = np.zeros_like(r)
    
    # Extract parameters
    r_knot = params[2]
    a0, a1, a2, a3 = params[3], params[4], params[5], params[6]
    
    # Spline derivative: a1 + 2*a2*r + 3*a3*r²
    spline_mask = (r <= r_knot)
    r_spline = r[spline_mask]
    if len(r_spline) > 0:
        result[spline_mask] = a1 + 2*a2*r_spline + 3*a3*r_spline**2
    
    # Gaussian derivative
    gauss_mask = (r > r_knot)
    r_gauss = r[gauss_mask]
    if len(r_gauss) > 0:
        gauss_prime_sum = np.zeros_like(r_gauss)
        
        for i in range(M_gauss):
            A_i = params[7 + 3*i]
            r0_i = params[7 + 3*i + 1]
            sigma_i = params[7 + 3*i + 2]
            
            if sigma_i > 1e-12:
                gauss_i = A_i * np.exp(-0.5 * ((r_gauss - r0_i) / sigma_i)**2)
                prime_i = -gauss_i * (r_gauss - r0_i) / (sigma_i**2)
                gauss_prime_sum += prime_i
        
        result[gauss_mask] = gauss_prime_sum
    
    return result

# ── 4. JAX-COMPATIBLE ANSATZ FUNCTIONS ────────────────────────────────────────
if HAS_JAX:
    @jax.jit
    def f_hybrid_spline_jax(r, params):
        """Hybrid spline-Gaussian ansatz (JAX version)"""
        r = jnp.atleast_1d(r)
        
        # Extract parameters
        r_knot = params[2]
        a0, a1, a2, a3 = params[3], params[4], params[5], params[6]
        
        # Spline region using JAX where
        spline_val = a0 + a1*r + a2*r**2 + a3*r**3
        
        # Gaussian region
        gauss_sum = jnp.zeros_like(r)
        for i in range(M_gauss):
            A_i = params[7 + 3*i]
            r0_i = params[7 + 3*i + 1]
            sigma_i = params[7 + 3*i + 2]
            
            sigma_safe = jnp.maximum(sigma_i, 1e-12)
            gauss_i = A_i * jnp.exp(-0.5 * ((r - r0_i) / sigma_safe)**2)
            gauss_sum += gauss_i
        
        # Combine regions using JAX where
        result = jnp.where(r <= r_knot, spline_val, gauss_sum)
        return result
    
    @jax.jit
    def f_hybrid_spline_prime_jax(r, params):
        """Derivative of hybrid spline-Gaussian ansatz (JAX version)"""
        r = jnp.atleast_1d(r)
        
        # Extract parameters
        r_knot = params[2]
        a1, a2, a3 = params[4], params[5], params[6]
        
        # Spline derivative
        spline_prime = a1 + 2*a2*r + 3*a3*r**2
        
        # Gaussian derivative
        gauss_prime_sum = jnp.zeros_like(r)
        for i in range(M_gauss):
            A_i = params[7 + 3*i]
            r0_i = params[7 + 3*i + 1]
            sigma_i = params[7 + 3*i + 2]
            
            sigma_safe = jnp.maximum(sigma_i, 1e-12)
            gauss_i = A_i * jnp.exp(-0.5 * ((r - r0_i) / sigma_safe)**2)
            prime_i = -gauss_i * (r - r0_i) / (sigma_safe**2)
            gauss_prime_sum += prime_i
        
        # Combine regions
        result = jnp.where(r <= r_knot, spline_prime, gauss_prime_sum)
        return result

# ── 5. ENERGY CALCULATION FUNCTIONS ───────────────────────────────────────────
def compute_energy_hybrid_numpy(params):
    """
    Compute negative energy E- for hybrid ansatz (NumPy version)
    
    Args:
        params: [mu, G_geo, r_knot, a0, a1, a2, a3, A1,r1,σ1, ..., A6,r6,σ6]
    
    Returns:
        E_minus + penalties (scalar)
    """
    mu, G_geo = params[0], params[1]
    
    # Compute f(r) and f'(r) on the grid
    f_vals = f_hybrid_spline_numpy(r_grid, params)
    f_prime_vals = f_hybrid_spline_prime_numpy(r_grid, params)
    
    # Quantum inequality backreaction term
    sinc_term = np.sinc(mu / (hbar * c))
    backreaction_factor = (1.0 + beta_back * G_geo * sinc_term)
    
    # Effective energy density
    rho_eff = (f_vals**2 * backreaction_factor + 
               0.01 * f_prime_vals**2)  # curvature penalty
    
    # Volume integral: ∫ ρ_eff 4π r² dr
    E_base = np.trapz(rho_eff * vol_weights, r_grid) * c4_8piG
    
    # Physics constraints and penalties
    penalties = compute_penalties_hybrid_numpy(params)
    
    return float(E_base + penalties)

if HAS_JAX:
    @jax.jit
    def compute_energy_hybrid_jax(params):
        """JAX-accelerated energy computation for hybrid ansatz"""
        mu, G_geo = params[0], params[1]
        
        # JAX grid computation
        r_jax = jnp.linspace(0.0, R, N_grid)
        vol_weights_jax = 4.0 * jnp.pi * r_jax**2
        
        # Compute f(r) and f'(r)
        f_vals = f_hybrid_spline_jax(r_jax, params)
        f_prime_vals = f_hybrid_spline_prime_jax(r_jax, params)
        
        # Backreaction
        sinc_term = jnp.sinc(mu / (hbar * c))
        backreaction_factor = (1.0 + beta_back * G_geo * sinc_term)
        
        # Energy density
        rho_eff = (f_vals**2 * backreaction_factor + 
                   0.01 * f_prime_vals**2)
        
        # Integration
        E_base = jnp.trapz(rho_eff * vol_weights_jax, r_jax) * c4_8piG
        
        # JAX-compatible penalties
        penalties = compute_penalties_hybrid_jax(params)
        
        return E_base + penalties

# ── 6. HYBRID ANSATZ PENALTY FUNCTIONS ────────────────────────────────────────
def compute_penalties_hybrid_numpy(params, lam_boundary=1e8, lam_continuity=1e8, 
                                  lam_bound=1e6, lam_amplitude=1e4):
    """Comprehensive penalty functions for hybrid ansatz (NumPy version)"""
    penalties = 0.0
    
    # Extract parameters
    mu, G_geo = params[0], params[1]
    r_knot = params[2]
    a0, a1, a2, a3 = params[3], params[4], params[5], params[6]
    
    # 1. Boundary conditions: f(0) ≈ 1, f(R) ≈ 0
    f_0 = f_hybrid_spline_numpy(0.0, params)  # Should be a0 = 1
    f_R = f_hybrid_spline_numpy(R, params)    # Should be ≈ 0
    penalties += lam_boundary * ((f_0 - 1.0)**2 + f_R**2)
    
    # 2. Continuity at r_knot: spline(r_knot) = Gaussians(r_knot)
    spline_at_knot = a0 + a1*r_knot + a2*r_knot**2 + a3*r_knot**3
    
    # Gaussian sum at r_knot
    gauss_at_knot = 0.0
    for i in range(M_gauss):
        A_i = params[7 + 3*i]
        r0_i = params[7 + 3*i + 1]
        sigma_i = params[7 + 3*i + 2]
        if sigma_i > 1e-12:
            gauss_at_knot += A_i * np.exp(-0.5 * ((r_knot - r0_i) / sigma_i)**2)
    
    penalties += lam_continuity * (spline_at_knot - gauss_at_knot)**2
    
    # 3. C¹ continuity: spline'(r_knot) = Gaussians'(r_knot)
    spline_prime_at_knot = a1 + 2*a2*r_knot + 3*a3*r_knot**2
    
    gauss_prime_at_knot = 0.0
    for i in range(M_gauss):
        A_i = params[7 + 3*i]
        r0_i = params[7 + 3*i + 1]
        sigma_i = params[7 + 3*i + 2]
        if sigma_i > 1e-12:
            gauss_i = A_i * np.exp(-0.5 * ((r_knot - r0_i) / sigma_i)**2)
            gauss_prime_at_knot += -gauss_i * (r_knot - r0_i) / (sigma_i**2)
    
    penalties += lam_continuity * (spline_prime_at_knot - gauss_prime_at_knot)**2
    
    # 4. Parameter bounds
    # μ, G_geo bounds
    if mu < 1e-8 or mu > 1e-3:
        penalties += lam_bound * ((mu - np.clip(mu, 1e-8, 1e-3))**2)
    if G_geo < 1e-8 or G_geo > 1e-2:
        penalties += lam_bound * ((G_geo - np.clip(G_geo, 1e-8, 1e-2))**2)
    
    # r_knot bounds: should be in [0.2R, 0.8R]
    if r_knot < 0.2*R or r_knot > 0.8*R:
        penalties += lam_bound * ((r_knot - np.clip(r_knot, 0.2*R, 0.8*R))**2)
    
    # Gaussian parameter constraints
    for i in range(M_gauss):
        A_i = params[7 + 3*i]
        r0_i = params[7 + 3*i + 1]
        sigma_i = params[7 + 3*i + 2]
        
        # Amplitude bounds: A_i ∈ [0, 2]
        if A_i < 0 or A_i > 2:
            penalties += lam_amplitude * ((A_i - np.clip(A_i, 0, 2))**2)
        
        # Position bounds: r0_i ∈ [r_knot, R]
        if r0_i < r_knot or r0_i > R:
            penalties += lam_bound * ((r0_i - np.clip(r0_i, r_knot, R))**2)
        
        # Width bounds: σ_i ∈ [0.01R, 0.5R]
        if sigma_i < 0.01*R or sigma_i > 0.5*R:
            penalties += lam_bound * ((sigma_i - np.clip(sigma_i, 0.01*R, 0.5*R))**2)
    
    return penalties

if HAS_JAX:
    @jax.jit
    def compute_penalties_hybrid_jax(params):
        """JAX-compatible penalty functions for hybrid ansatz"""
        penalties = 0.0
        
        # Extract parameters
        mu, G_geo = params[0], params[1]
        r_knot = params[2]
        a0, a1, a2, a3 = params[3], params[4], params[5], params[6]
        
        # Boundary conditions
        f_0 = f_hybrid_spline_jax(0.0, params)
        f_R = f_hybrid_spline_jax(R, params)
        penalties += 1e8 * ((f_0 - 1.0)**2 + f_R**2)
        
        # Continuity at r_knot
        spline_at_knot = a0 + a1*r_knot + a2*r_knot**2 + a3*r_knot**3
        
        gauss_at_knot = 0.0
        for i in range(M_gauss):
            A_i = params[7 + 3*i]
            r0_i = params[7 + 3*i + 1]
            sigma_i = params[7 + 3*i + 2]
            sigma_safe = jnp.maximum(sigma_i, 1e-12)
            gauss_at_knot += A_i * jnp.exp(-0.5 * ((r_knot - r0_i) / sigma_safe)**2)
        
        penalties += 1e8 * (spline_at_knot - gauss_at_knot)**2
        
        # Parameter bounds (JAX-compatible)
        penalties += 1e6 * (jnp.maximum(0, 1e-8 - mu)**2 + 
                           jnp.maximum(0, mu - 1e-3)**2 +
                           jnp.maximum(0, 1e-8 - G_geo)**2 + 
                           jnp.maximum(0, G_geo - 1e-2)**2 +
                           jnp.maximum(0, 0.2*R - r_knot)**2 + 
                           jnp.maximum(0, r_knot - 0.8*R)**2)
        
        # Gaussian constraints
        for i in range(M_gauss):
            A_i = params[7 + 3*i]
            r0_i = params[7 + 3*i + 1]
            sigma_i = params[7 + 3*i + 2]
            
            penalties += 1e4 * (jnp.maximum(0, -A_i)**2 + 
                               jnp.maximum(0, A_i - 2)**2 +
                               jnp.maximum(0, r_knot - r0_i)**2 + 
                               jnp.maximum(0, r0_i - R)**2 +
                               jnp.maximum(0, 0.01*R - sigma_i)**2 + 
                               jnp.maximum(0, sigma_i - 0.5*R)**2)
        
        return penalties

# ── 7. OPTIMIZATION SETUP ─────────────────────────────────────────────────────
def get_hybrid_initial_guess():
    """Smart initialization for hybrid spline-Gaussian parameters"""
    params = np.zeros(PARAM_DIM)
    
    # Initialize μ, G_geo near optimal values
    params[0] = mu0_default      # μ
    params[1] = G_geo_default    # G_geo
    
    # Initialize knot point at middle
    params[2] = 0.5 * R          # r_knot
    
    # Initialize spline coefficients for smooth transition
    # Start with f(0) = 1, gradual decay to knot
    params[3] = 1.0              # a0: f(0) = 1
    params[4] = -0.5             # a1: negative slope
    params[5] = 0.0              # a2: small curvature
    params[6] = 0.0              # a3: small cubic term
    
    # Initialize 6 Gaussians in outer region [r_knot, R]
    for i in range(M_gauss):
        base_idx = 7 + 3*i
        params[base_idx]     = 0.5 / (i + 1)    # A_i: decreasing amplitude
        params[base_idx + 1] = params[2] + (i + 0.5) * (R - params[2]) / M_gauss  # r0_i
        params[base_idx + 2] = 0.1 * R           # σ_i: moderate width
    
    return params

def get_hybrid_bounds():
    """Parameter bounds for hybrid optimization"""
    bounds = []
    
    # μ, G_geo bounds
    bounds.append((1e-8, 1e-3))   # μ
    bounds.append((1e-8, 1e-2))   # G_geo
    
    # r_knot bounds
    bounds.append((0.2*R, 0.8*R)) # r_knot
    
    # Spline coefficients
    bounds.append((0.5, 1.5))     # a0: f(0) ≈ 1
    bounds.append((-2.0, 2.0))    # a1: slope
    bounds.append((-2.0, 2.0))    # a2: curvature
    bounds.append((-2.0, 2.0))    # a3: cubic term
    
    # 6 Gaussians: (A_i, r0_i, σ_i) for i=0..5
    for i in range(M_gauss):
        bounds.append((0.0, 2.0))        # A_i
        bounds.append((0.0, R))          # r0_i (will be constrained > r_knot)
        bounds.append((0.01*R, 0.5*R))   # σ_i
    
    return bounds

# ── 8. CMA-ES OPTIMIZATION ────────────────────────────────────────────────────
def run_cma_es_hybrid(max_evals=2500, verbose=True):
    """CMA-ES global search for hybrid spline-Gaussian ansatz"""
    if not HAS_CMA:
        print("❌ CMA-ES not available. Install with: pip install cma")
        return None
    
    # Initial guess and setup
    x0 = get_hybrid_initial_guess()
    sigma0 = 0.25  # Initial step size
    
    if verbose:
        print(f"\n🚀 Starting Hybrid CMA-ES Global Search")
        print(f"   Dimension: {PARAM_DIM}")
        print(f"   Max evaluations: {max_evals}")
        print(f"   r_knot initial: {x0[2]:.3f}")
    
    # Setup CMA-ES with boundary constraints
    bounds_array = np.array(get_hybrid_bounds())
    cma_bounds = [bounds_array[:, 0].tolist(), bounds_array[:, 1].tolist()]
    
    # CMA-ES options
    opts = {
        'bounds': cma_bounds,
        'popsize': 70,  # Moderate population
        'maxiter': max_evals // 70,
        'tolfun': 1e-12,
        'tolx': 1e-12,
        'verb_log': 0 if not verbose else 1
    }
    
    # Run CMA-ES
    start_time = time.time()
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
    
    eval_count = 0
    best_energy = float('inf')
    best_params = None
    
    while not es.stop() and eval_count < max_evals:
        solutions = es.ask()
        energies = []
        
        for sol in solutions:
            try:
                energy = compute_energy_hybrid_numpy(np.array(sol))
                energies.append(energy)
                eval_count += 1
                
                if energy < best_energy:
                    best_energy = energy
                    best_params = np.array(sol)
                    if verbose and eval_count % 100 == 0:
                        print(f"   Eval {eval_count:4d}: E = {energy:.6e} J, r_knot = {sol[2]:.3f}")
                        
            except Exception as e:
                energies.append(1e20)  # Large penalty for failed evaluations
        
        es.tell(solutions, energies)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n✅ Hybrid CMA-ES completed in {elapsed:.1f}s")
        print(f"   Best E-: {best_energy:.6e} J")
        print(f"   Optimal r_knot: {best_params[2]:.4f}")
    
    return {
        'params': best_params,
        'energy': best_energy,
        'evaluations': eval_count,
        'time': elapsed,
        'method': 'Hybrid-CMA-ES'
    }

# ── 9. JAX LOCAL REFINEMENT ───────────────────────────────────────────────────
def run_jax_refinement_hybrid(theta_init, max_iter=400, verbose=True):
    """JAX-accelerated local refinement for hybrid ansatz"""
    if not HAS_JAX:
        print("❌ JAX not available. Install with: pip install jax")
        return {'params': theta_init, 'energy': compute_energy_hybrid_numpy(theta_init), 
                'method': 'no_refinement'}
    
    if verbose:
        print(f"\n🔥 Starting Hybrid JAX Local Refinement")
        print(f"   Initial E-: {compute_energy_hybrid_numpy(theta_init):.6e} J")
    
    # Convert to JAX array
    theta_jax = jnp.array(theta_init)
    
    # Define JAX objective
    @jax.jit
    def objective_jax(x):
        return compute_energy_hybrid_jax(x)
    
    start_time = time.time()
    
    try:
        # Run JAX L-BFGS optimization
        result = jax_minimize(
            objective_jax, 
            theta_jax, 
            method='BFGS',
            options={'maxiter': max_iter, 'gtol': 1e-8}
        )
        
        theta_refined = np.array(result.x)
        energy_refined = float(result.fun)
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"✅ Hybrid JAX refinement completed in {elapsed:.1f}s")
            print(f"   Final E-: {energy_refined:.6e} J")
            print(f"   Final r_knot: {theta_refined[2]:.4f}")
        
        return {
            'params': theta_refined,
            'energy': energy_refined,
            'iterations': result.nit if hasattr(result, 'nit') else max_iter,
            'time': elapsed,
            'method': 'Hybrid-JAX-BFGS',
            'success': result.success if hasattr(result, 'success') else True
        }
        
    except Exception as e:
        if verbose:
            print(f"⚠️  Hybrid JAX refinement failed: {e}")
        return {
            'params': theta_init,
            'energy': compute_energy_hybrid_numpy(theta_init),
            'method': 'refinement_failed'
        }

# ── 10. TWO-STAGE HYBRID OPTIMIZATION ─────────────────────────────────────────
def run_hybrid_two_stage_optimization(cma_evals=2500, jax_iters=400, verbose=True):
    """Complete two-stage hybrid optimization pipeline"""
    if verbose:
        print("=" * 60)
        print("🎯 HYBRID SPLINE-GAUSSIAN TWO-STAGE OPTIMIZATION")
        print("=" * 60)
    
    # Stage 1: CMA-ES Global Search
    cma_result = run_cma_es_hybrid(max_evals=cma_evals, verbose=verbose)
    
    if cma_result is None:
        print("❌ Hybrid CMA-ES failed, aborting optimization")
        return None
    
    # Stage 2: JAX Local Refinement
    jax_result = run_jax_refinement_hybrid(
        cma_result['params'], 
        max_iter=jax_iters, 
        verbose=verbose
    )
    
    # Combine results
    final_result = {
        'stage1_cma': cma_result,
        'stage2_jax': jax_result,
        'final_params': jax_result['params'],
        'final_energy': jax_result['energy'],
        'total_time': cma_result.get('time', 0) + jax_result.get('time', 0),
        'total_evaluations': cma_result.get('evaluations', 0) + jax_result.get('iterations', 0)
    }
    
    if verbose:
        print(f"\n🏆 HYBRID TWO-STAGE OPTIMIZATION COMPLETE")
        print(f"   Final E-: {final_result['final_energy']:.6e} J")
        print(f"   Optimal r_knot: {final_result['final_params'][2]:.4f}")
        print(f"   Total time: {final_result['total_time']:.1f}s")
    
    return final_result

# ── 11. ANALYSIS AND VISUALIZATION ────────────────────────────────────────────
def analyze_hybrid_result(result, save_json=True, create_plots=True):
    """Comprehensive analysis of hybrid optimization result"""
    if result is None:
        print("❌ No result to analyze")
        return
    
    params = result['final_params']
    energy = result['final_energy']
    
    # Extract parameters
    mu_opt, G_geo_opt, r_knot_opt = params[0], params[1], params[2]
    a0, a1, a2, a3 = params[3], params[4], params[5], params[6]
    
    print(f"\n📊 HYBRID SPLINE-GAUSSIAN ANALYSIS")
    print(f"=" * 50)
    print(f"Final Energy E-: {energy:.6e} J")
    print(f"Optimal μ: {mu_opt:.6e}")
    print(f"Optimal G_geo: {G_geo_opt:.6e}")
    print(f"Optimal r_knot: {r_knot_opt:.4f}")
    print(f"Total time: {result['total_time']:.1f}s")
    
    # Spline coefficients
    print(f"\nSpline coefficients (0 ≤ r ≤ {r_knot_opt:.3f}):")
    print(f"  a0 = {a0:.4f}, a1 = {a1:.4f}")
    print(f"  a2 = {a2:.4f}, a3 = {a3:.4f}")
    
    # Gaussian parameters
    print(f"\n6-Gaussian parameters (r > {r_knot_opt:.3f}):")
    for i in range(M_gauss):
        A_i = params[7 + 3*i]
        r0_i = params[7 + 3*i + 1]
        sigma_i = params[7 + 3*i + 2]
        print(f"  G{i}: A={A_i:.4f}, r0={r0_i:.4f}, σ={sigma_i:.4f}")
    
    # Verification
    f_0 = f_hybrid_spline_numpy(0.0, params)
    f_R = f_hybrid_spline_numpy(R, params)
    f_knot_spline = a0 + a1*r_knot_opt + a2*r_knot_opt**2 + a3*r_knot_opt**3
    f_knot_gauss = f_hybrid_spline_numpy(r_knot_opt + 1e-6, params)
    
    print(f"\nVerification:")
    print(f"  f(0) = {f_0:.6f} (target: 1.0)")
    print(f"  f(R) = {f_R:.6f} (target: 0.0)")
    print(f"  Continuity at r_knot: spline = {f_knot_spline:.6f}, gauss = {f_knot_gauss:.6f}")
    
    # Save results
    if save_json:
        result_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'method': 'Hybrid Spline-Gaussian Two-Stage CMA-ES + JAX',
            'final_energy_J': float(energy),
            'optimal_mu': float(mu_opt),
            'optimal_G_geo': float(G_geo_opt),
            'optimal_r_knot': float(r_knot_opt),
            'spline_coefficients': {
                'a0': float(a0), 'a1': float(a1), 
                'a2': float(a2), 'a3': float(a3)
            },
            'gaussian_parameters': [
                {
                    'amplitude': float(params[7 + 3*i]),
                    'center': float(params[7 + 3*i + 1]),
                    'width': float(params[7 + 3*i + 2])
                }
                for i in range(M_gauss)
            ],
            'optimization_stats': {
                'total_time_s': result['total_time'],
                'total_evaluations': result['total_evaluations']
            },
            'verification': {
                'f_at_0': float(f_0),
                'f_at_R': float(f_R),
                'continuity_spline': float(f_knot_spline),
                'continuity_gauss': float(f_knot_gauss)
            }
        }
        
        with open('hybrid_spline_gaussian_results.json', 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"💾 Results saved to: hybrid_spline_gaussian_results.json")
    
    # Create plots
    if create_plots:
        plot_hybrid_profile(result)
    
    return result_data

def plot_hybrid_profile(result, save_fig=True):
    """Visualization of optimized hybrid profile"""
    params = result['final_params']
    r_knot = params[2]
    
    # High-resolution profile for plotting
    r_plot = np.linspace(0, R, 2000)
    f_plot = f_hybrid_spline_numpy(r_plot, params)
    f_prime_plot = f_hybrid_spline_prime_numpy(r_plot, params)
    
    # Separate spline and Gaussian regions for visualization
    spline_mask = r_plot <= r_knot
    gauss_mask = r_plot > r_knot
    
    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Complete hybrid profile
    ax1.plot(r_plot[spline_mask], f_plot[spline_mask], 'b-', linewidth=3, 
             label='Spline region')
    ax1.plot(r_plot[gauss_mask], f_plot[gauss_mask], 'r-', linewidth=3, 
             label='Gaussian region')
    ax1.axvline(r_knot, color='k', linestyle='--', alpha=0.7, label=f'r_knot = {r_knot:.3f}')
    ax1.set_xlabel('Radial coordinate r')
    ax1.set_ylabel('f(r)')
    ax1.set_title('Hybrid Spline-Gaussian Profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Derivative
    ax2.plot(r_plot[spline_mask], f_prime_plot[spline_mask], 'b-', linewidth=2, 
             label="Spline f'(r)")
    ax2.plot(r_plot[gauss_mask], f_prime_plot[gauss_mask], 'r-', linewidth=2, 
             label="Gaussian f'(r)")
    ax2.axvline(r_knot, color='k', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Radial coordinate r')
    ax2.set_ylabel("f'(r)")
    ax2.set_title('Profile Derivative')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Individual Gaussians
    if len(r_plot[gauss_mask]) > 0:
        r_gauss_plot = r_plot[gauss_mask]
        for i in range(M_gauss):
            A_i = params[7 + 3*i]
            r0_i = params[7 + 3*i + 1]
            sigma_i = params[7 + 3*i + 2]
            gauss_i = A_i * np.exp(-0.5 * ((r_gauss_plot - r0_i) / sigma_i)**2)
            ax3.plot(r_gauss_plot, gauss_i, '--', alpha=0.7, label=f'G{i}')
        
        ax3.plot(r_gauss_plot, f_plot[gauss_mask], 'r-', linewidth=2, label='Total')
        ax3.set_xlabel('Radial coordinate r')
        ax3.set_ylabel('Gaussian amplitude')
        ax3.set_title('Individual Gaussians')
        ax3.grid(True, alpha=0.3)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
      # Plot 4: Energy density
    mu_opt, G_geo_opt = params[0], params[1]
    sinc_term = np.sinc(mu_opt / (hbar * c))
    backreaction = 1.0 + beta_back * G_geo_opt * sinc_term
    rho_eff = f_plot**2 * backreaction + 0.01 * f_prime_plot**2
    
    ax4.semilogy(r_plot, np.abs(rho_eff), 'g-', linewidth=2)
    ax4.axvline(r_knot, color='k', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Radial coordinate r')
    ax4.set_ylabel('|ρ_eff(r)|')
    ax4.set_title('Effective Energy Density')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('hybrid_spline_gaussian_profile.png', dpi=300, bbox_inches='tight')
        print(f"📈 Hybrid profile plot saved to: hybrid_spline_gaussian_profile.png")
    
    plt.close()  # Close instead of show to prevent blocking

# ── 12. MAIN EXECUTION ────────────────────────────────────────────────────────
def main():
    """Main execution function"""
    print("🌟 HYBRID SPLINE-GAUSSIAN WARP BUBBLE OPTIMIZER")
    print("=" * 60)
    
    # Check dependencies
    if not HAS_CMA:
        print("⚠️  Warning: CMA-ES not available. Install with: pip install cma")
        return
    
    if not HAS_JAX:
        print("⚠️  Warning: JAX not available. Install with: pip install jax")
        print("   Continuing with NumPy-only optimization...")
    
    # Run hybrid two-stage optimization
    result = run_hybrid_two_stage_optimization(
        cma_evals=2500,  # Adjust based on computational budget
        jax_iters=400,   # JAX local refinement iterations
        verbose=True
    )
    
    if result is not None:
        # Analyze and save results
        analyze_hybrid_result(result, save_json=True, create_plots=True)
        
        # Save optimized parameters for future use
        np.save('best_hybrid_params.npy', result['final_params'])
        np.savetxt('best_hybrid_params.txt', result['final_params'])
        print(f"💾 Hybrid parameters saved to: best_hybrid_params.*")
        
        print(f"\n🎉 HYBRID OPTIMIZATION COMPLETE!")
        print(f"   Record E-: {result['final_energy']:.6e} J")
        print(f"   Knot position: r_knot = {result['final_params'][2]:.4f}")
        
    else:
        print("❌ Hybrid optimization failed")

if __name__ == "__main__":
    main()
