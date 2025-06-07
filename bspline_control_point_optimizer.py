#!/usr/bin/env python3
"""
B-SPLINE CONTROL-POINT WARP BUBBLE OPTIMIZER

Advanced B-spline control-point ansatz implementation with:
1. ✅ B-spline basis functions for flexible profile representation
2. ✅ Joint optimization of (μ, G_geo) + control points  
3. ✅ Hard stability penalty using maximum growth rate from 3D analysis
4. ✅ Two-stage optimization: CMA-ES global → JAX-accelerated local refinement
5. ✅ Surrogate-assisted optimization proposals (optional)
6. ✅ Non-blocking matplotlib visualizations

Mathematical Framework:
f(r) = Σᵢ cᵢ Bᵢ(r)  where Bᵢ(r) are B-spline basis functions

Target: Achieve E- < -2.0×10⁵³ J while maintaining stability (λ_max < 0)

Author: Advanced Warp Bubble Optimizer
Date: December 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import warnings
from pathlib import Path
from scipy.interpolate import BSpline, splev, splrep
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution
import traceback
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

HAS_STABILITY = False
try:
    from test_3d_stability import analyze_stability_3d, WarpBubble3DStabilityAnalyzer
    HAS_STABILITY = True
    print("✅ 3D stability analysis detected - using real stability penalties")
except ImportError:
    print("⚠️  3D stability analysis not available - using heuristic stability penalties")

HAS_GP = False
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern
    HAS_GP = True
    print("✅ Gaussian Process detected - surrogate optimization available")
except ImportError:
    print("⚠️  Gaussian Process not available - skipping surrogate optimization")

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

# ── 2. B-SPLINE CONFIGURATION ─────────────────────────────────────────────────
N_CONTROL = 12   # Number of control points
SPLINE_ORDER = 3 # Cubic B-splines (k=3)
STABILITY_PENALTY_WEIGHT = 1e6  # Weight for stability penalty

# Generate knot vector for clamped B-spline
# For N_CONTROL points and order k=3, we need N_CONTROL+k+1 knots
N_KNOTS = N_CONTROL + SPLINE_ORDER + 1
knots = np.concatenate([
    np.zeros(SPLINE_ORDER + 1),           # Clamped at start
    np.linspace(0, 1, N_KNOTS - 2*(SPLINE_ORDER + 1) + 2),  # Interior knots
    np.ones(SPLINE_ORDER + 1)             # Clamped at end
])

# Precompute radial grid for integration
N_GRID = 1000
r_grid = np.linspace(0.0, R, N_GRID)
dr = r_grid[1] - r_grid[0]
vol_weights = 4.0 * np.pi * r_grid**2

# Parameter vector structure:
# [mu, G_geo, c₁, c₂, ..., c₁₂]
# Total dimension: 2 + N_CONTROL = 14 parameters
PARAM_DIM = 2 + N_CONTROL

print(f"🎯 B-Spline Control-Point Ansatz: {PARAM_DIM} parameters")
print(f"   Control points: {N_CONTROL}")
print(f"   Spline order: {SPLINE_ORDER} (cubic)")
print(f"   Knot vector length: {len(knots)}")

# ── 3. B-SPLINE BASIS FUNCTIONS ───────────────────────────────────────────────
def evaluate_bspline(r, control_points):
    """
    Evaluate B-spline at given radial coordinates
    
    Args:
        r: Radial coordinate(s) (scalar or array)
        control_points: Array of control point values [c₁, c₂, ..., c₁₂]
    
    Returns:
        B-spline function value(s)
    """
    r = np.atleast_1d(r)
    
    # Map r from [0, R] to [0, 1] for the knot vector
    r_normalized = np.clip(r / R, 0.0, 1.0)
    
    # Evaluate B-spline using scipy
    try:
        # Create the BSpline object
        bspline = BSpline(knots, control_points, SPLINE_ORDER)
        result = bspline(r_normalized)
        return np.atleast_1d(result)
    except Exception as e:
        print(f"Warning: B-spline evaluation failed: {e}")
        # Fallback to linear interpolation
        return np.interp(r_normalized, np.linspace(0, 1, len(control_points)), control_points)

def evaluate_bspline_derivative(r, control_points):
    """
    Evaluate B-spline derivative at given radial coordinates
    
    Args:
        r: Radial coordinate(s) (scalar or array)
        control_points: Array of control point values
    
    Returns:
        B-spline derivative value(s)
    """
    r = np.atleast_1d(r)
    r_normalized = np.clip(r / R, 0.0, 1.0)
    
    try:
        # Create BSpline and compute derivative
        bspline = BSpline(knots, control_points, SPLINE_ORDER)
        derivative_bspline = bspline.derivative()
        result = derivative_bspline(r_normalized) / R  # Chain rule: d/dr = (d/dr_norm) * (dr_norm/dr)
        return np.atleast_1d(result)
    except Exception as e:
        print(f"Warning: B-spline derivative evaluation failed: {e}")
        # Fallback to finite difference
        dr_small = 1e-6
        f_plus = evaluate_bspline(r + dr_small, control_points)
        f_minus = evaluate_bspline(r - dr_small, control_points)
        return (f_plus - f_minus) / (2 * dr_small)

# ── 4. JAX-COMPATIBLE B-SPLINE FUNCTIONS ──────────────────────────────────────
if HAS_JAX:
    @jax.jit
    def evaluate_bspline_jax(r, control_points):
        """JAX-compatible B-spline evaluation (simplified implementation)"""
        r = jnp.atleast_1d(r)
        r_normalized = jnp.clip(r / R, 0.0, 1.0)
        
        # Simplified B-spline evaluation using linear interpolation between control points
        # For production, this should use proper B-spline basis functions
        indices = r_normalized * (len(control_points) - 1)
        i = jnp.floor(indices).astype(int)
        i = jnp.clip(i, 0, len(control_points) - 2)
        alpha = indices - i
        
        result = control_points[i] * (1 - alpha) + control_points[i + 1] * alpha
        return result
    
    @jax.jit
    def evaluate_bspline_derivative_jax(r, control_points):
        """JAX-compatible B-spline derivative (simplified implementation)"""
        r = jnp.atleast_1d(r)
        
        # Finite difference approximation for JAX compatibility
        dr_small = 1e-6
        f_plus = evaluate_bspline_jax(r + dr_small, control_points)
        f_minus = evaluate_bspline_jax(r - dr_small, control_points)
        return (f_plus - f_minus) / (2 * dr_small)

# ── 5. ENERGY CALCULATION FUNCTIONS ───────────────────────────────────────────
def compute_energy_bspline_numpy(params):
    """
    Compute negative energy E- for B-spline ansatz (NumPy version)
    
    Args:
        params: [mu, G_geo, c₁, c₂, ..., c₁₂]
    
    Returns:
        E_minus + penalties (scalar)
    """
    mu, G_geo = params[0], params[1]
    control_points = params[2:]
    
    # Compute f(r) and f'(r) on the grid
    f_vals = evaluate_bspline(r_grid, control_points)
    f_prime_vals = evaluate_bspline_derivative(r_grid, control_points)
    
    # Quantum inequality backreaction term
    sinc_term = np.sinc(mu / (hbar * c))
    backreaction_factor = (1.0 + beta_back * G_geo * sinc_term)
    
    # Effective energy density with Van den Broeck–Natário enhancement
    rho_eff = (f_vals**2 * backreaction_factor + 
               0.01 * f_prime_vals**2)  # curvature penalty
    
    # Volume integral: ∫ ρ_eff 4π r² dr
    E_base = np.trapz(rho_eff * vol_weights, r_grid) * c4_8piG
    
    # Physics constraints and penalties
    penalties = compute_penalties_bspline_numpy(params)
    
    return float(E_base + penalties)

if HAS_JAX:
    @jax.jit
    def compute_energy_bspline_jax(params):
        """JAX-accelerated energy computation for B-spline ansatz"""
        mu, G_geo = params[0], params[1]
        control_points = params[2:]
        
        # JAX grid computation
        r_jax = jnp.linspace(0.0, R, N_GRID)
        vol_weights_jax = 4.0 * jnp.pi * r_jax**2
        
        # Compute f(r) and f'(r)
        f_vals = evaluate_bspline_jax(r_jax, control_points)
        f_prime_vals = evaluate_bspline_derivative_jax(r_jax, control_points)
        
        # Backreaction
        sinc_term = jnp.sinc(mu / (hbar * c))
        backreaction_factor = (1.0 + beta_back * G_geo * sinc_term)
        
        # Energy density
        rho_eff = (f_vals**2 * backreaction_factor + 
                   0.01 * f_prime_vals**2)
        
        # Integration
        E_base = jnp.trapz(rho_eff * vol_weights_jax, r_jax) * c4_8piG
        
        # JAX-compatible penalties
        penalties = compute_penalties_bspline_jax(params)
        
        return E_base + penalties

# ── 6. B-SPLINE PENALTY FUNCTIONS ─────────────────────────────────────────────
def compute_penalties_bspline_numpy(params, lam_boundary=1e8, lam_monotonic=1e8, 
                                   lam_bound=1e6, lam_stability=STABILITY_PENALTY_WEIGHT):
    """Comprehensive penalty functions for B-spline ansatz (NumPy version)"""
    penalties = 0.0
    
    mu, G_geo = params[0], params[1]
    control_points = params[2:]
    
    # 1. Boundary conditions: f(0) ≈ 1, f(R) ≈ 0
    f_0 = evaluate_bspline(np.array([0.0]), control_points)[0]
    f_R = evaluate_bspline(np.array([R]), control_points)[0]
    
    penalties += lam_boundary * (f_0 - 1.0)**2  # f(0) = 1
    penalties += lam_boundary * f_R**2           # f(R) = 0
    
    # 2. Monotonicity constraint: f should be decreasing
    f_test = evaluate_bspline(np.linspace(0, R, 50), control_points)
    df_test = np.diff(f_test)
    monotonic_violations = np.sum(np.maximum(df_test, 0.0)**2)
    penalties += lam_monotonic * monotonic_violations
    
    # 3. Parameter bounds
    penalties += lam_bound * np.sum(np.maximum(mu - 1e-3, 0.0)**2)        # μ upper bound
    penalties += lam_bound * np.sum(np.maximum(-mu, 0.0)**2)              # μ lower bound
    penalties += lam_bound * np.sum(np.maximum(G_geo - 1e-2, 0.0)**2)     # G_geo upper bound
    penalties += lam_bound * np.sum(np.maximum(-G_geo, 0.0)**2)           # G_geo lower bound
    
    # 4. Control point bounds
    penalties += lam_bound * np.sum(np.maximum(control_points - 2.0, 0.0)**2)  # Upper bound
    penalties += lam_bound * np.sum(np.maximum(-control_points - 0.5, 0.0)**2) # Lower bound
    
    # 5. Stability penalty using real 3D analysis or heuristic
    stability_penalty = compute_stability_penalty_bspline(params, lam_stability)
    penalties += stability_penalty
    
    return float(penalties)

if HAS_JAX:
    @jax.jit
    def compute_penalties_bspline_jax(params):
        """JAX-compatible penalty functions for B-spline ansatz"""
        mu, G_geo = params[0], params[1]
        control_points = params[2:]
        
        penalties = 0.0
        lam_boundary = 1e8
        lam_monotonic = 1e8
        lam_bound = 1e6
        
        # Boundary conditions
        f_0 = evaluate_bspline_jax(jnp.array([0.0]), control_points)[0]
        f_R = evaluate_bspline_jax(jnp.array([R]), control_points)[0]
        
        penalties += lam_boundary * (f_0 - 1.0)**2
        penalties += lam_boundary * f_R**2
        
        # Monotonicity constraint
        r_test = jnp.linspace(0, R, 50)
        f_test = evaluate_bspline_jax(r_test, control_points)
        df_test = jnp.diff(f_test)
        monotonic_violations = jnp.sum(jnp.maximum(df_test, 0.0)**2)
        penalties += lam_monotonic * monotonic_violations
        
        # Parameter bounds
        penalties += lam_bound * jnp.sum(jnp.maximum(mu - 1e-3, 0.0)**2)
        penalties += lam_bound * jnp.sum(jnp.maximum(-mu, 0.0)**2)
        penalties += lam_bound * jnp.sum(jnp.maximum(G_geo - 1e-2, 0.0)**2)
        penalties += lam_bound * jnp.sum(jnp.maximum(-G_geo, 0.0)**2)
        
        # Control point bounds
        penalties += lam_bound * jnp.sum(jnp.maximum(control_points - 2.0, 0.0)**2)
        penalties += lam_bound * jnp.sum(jnp.maximum(-control_points - 0.5, 0.0)**2)
        
        # Heuristic stability penalty for JAX (real 3D analysis in penalty function below)
        penalties += compute_stability_penalty_heuristic_jax(params)
        
        return penalties

# ── 7. STABILITY PENALTY FUNCTIONS ────────────────────────────────────────────
def compute_stability_penalty_bspline(params, lam_stability=STABILITY_PENALTY_WEIGHT):
    """
    Compute stability penalty using real 3D analysis or heuristic
    
    Args:
        params: [mu, G_geo, c₁, c₂, ..., c₁₂]
        lam_stability: Penalty weight for stability violations
    
    Returns:
        Stability penalty (scalar)
    """
    if HAS_STABILITY:
        try:
            # Use real 3D stability analysis
            mu, G_geo = params[0], params[1]
            control_points = params[2:]
            
            # Create parameter dictionary for stability analysis
            stability_params = {
                'mu': float(mu),
                'G_geo': float(G_geo),
                'profile_type': 'bspline',
                'control_points': control_points.tolist() if hasattr(control_points, 'tolist') else list(control_points),
                'R_b': R
            }
            
            # Run real 3D stability analysis
            result = analyze_stability_3d(stability_params)
            lambda_max = result.get('max_growth_rate', 0.0)
            
            # Hard penalty for positive growth rates (instability)
            return lam_stability * max(lambda_max, 0.0)**2
            
        except Exception as e:
            print(f"⚠️  Stability analysis failed: {e}, using heuristic")
    
    # Fallback to heuristic stability penalty
    return compute_stability_penalty_heuristic_numpy(params, lam_stability)

def compute_stability_penalty_heuristic_numpy(params, lam_stability):
    """Heuristic stability penalty based on profile characteristics (NumPy)"""
    mu, G_geo = params[0], params[1]
    control_points = params[2:]
    
    penalty = 0.0
    
    # Compute profile characteristics
    r_test = np.linspace(0, R, 100)
    f_vals = evaluate_bspline(r_test, control_points)
    f_prime_vals = evaluate_bspline_derivative(r_test, control_points)
    
    # Penalize excessive derivatives (instability indicator)
    max_derivative = np.max(np.abs(f_prime_vals))
    penalty += lam_stability * 0.1 * max(max_derivative - 5.0, 0.0)**2
    
    # Penalize large geometric parameters (can cause instability)
    penalty += lam_stability * 0.01 * (mu * 1e6)**2
    penalty += lam_stability * 0.01 * (G_geo * 1e4)**2
    
    # Penalize rapid oscillations in control points
    control_diffs = np.diff(control_points)
    oscillation_penalty = np.sum(control_diffs**2)
    penalty += lam_stability * 0.001 * oscillation_penalty
    
    return float(penalty)

if HAS_JAX:
    @jax.jit
    def compute_stability_penalty_heuristic_jax(params):
        """JAX-compatible heuristic stability penalty"""
        mu, G_geo = params[0], params[1]
        control_points = params[2:]
        
        lam_stability = STABILITY_PENALTY_WEIGHT
        penalty = 0.0
        
        # Profile characteristics
        r_test = jnp.linspace(0, R, 50)
        f_vals = evaluate_bspline_jax(r_test, control_points)
        f_prime_vals = evaluate_bspline_derivative_jax(r_test, control_points)
        
        # Excessive derivatives
        max_derivative = jnp.max(jnp.abs(f_prime_vals))
        penalty += lam_stability * 0.1 * jnp.maximum(max_derivative - 5.0, 0.0)**2
        
        # Large parameters
        penalty += lam_stability * 0.01 * (mu * 1e6)**2
        penalty += lam_stability * 0.01 * (G_geo * 1e4)**2
        
        # Control point oscillations
        control_diffs = jnp.diff(control_points)
        oscillation_penalty = jnp.sum(control_diffs**2)
        penalty += lam_stability * 0.001 * oscillation_penalty
        
        return penalty

# ── 8. OPTIMIZATION SETUP ─────────────────────────────────────────────────────
def get_bspline_initial_guess(strategy='physics_informed'):
    """
    Smart initialization for B-spline control-point parameters
    
    Args:
        strategy: Initialization strategy ('physics_informed', 'random', 'smooth_decay')
    
    Returns:
        Initial parameter vector [mu, G_geo, c₁, c₂, ..., c₁₂]
    """
    if strategy == 'physics_informed':
        # Start with proven optimal values for (μ, G_geo)
        mu_init = mu0_default
        G_geo_init = G_geo_default
        
        # Physics-informed control points: smooth decay from 1 to 0
        control_positions = np.linspace(0, 1, N_CONTROL)
        control_values = np.exp(-2.0 * control_positions)  # Exponential decay
        
    elif strategy == 'smooth_decay':
        mu_init = 1e-6
        G_geo_init = 1e-5
        
        # Smooth polynomial decay
        control_positions = np.linspace(0, 1, N_CONTROL)
        control_values = (1 - control_positions)**2
        
    else:  # random
        mu_init = np.random.uniform(1e-8, 1e-4)
        G_geo_init = np.random.uniform(1e-8, 1e-3)
        control_values = np.random.uniform(0.0, 1.0, N_CONTROL)
    
    # Combine parameters
    theta_init = np.concatenate([[mu_init, G_geo_init], control_values])
    
    return theta_init

def get_bspline_bounds():
    """Parameter bounds for B-spline optimization"""
    bounds = []
    
    # μ and G_geo bounds (proven effective ranges)
    bounds.append((1e-8, 1e-3))    # μ
    bounds.append((1e-8, 1e-2))    # G_geo
    
    # Control point bounds
    for i in range(N_CONTROL):
        bounds.append((-0.5, 2.0))  # cᵢ bounds
    
    return bounds

# ── 9. CMA-ES GLOBAL OPTIMIZATION ─────────────────────────────────────────────
def run_cma_es_bspline(max_evals=3000, verbose=True, init_strategy='physics_informed'):
    """CMA-ES global search for B-spline control-point ansatz"""
    if not HAS_CMA:
        print("❌ CMA-ES not available. Install with: pip install cma")
        return None
    
    print(f"\n🔍 CMA-ES Global Optimization for B-Spline Ansatz")
    print(f"   Parameters: {PARAM_DIM}")
    print(f"   Max evaluations: {max_evals}")
    print(f"   Initialization: {init_strategy}")
    
    # Initial guess and bounds
    theta_init = get_bspline_initial_guess(init_strategy)
    bounds = get_bspline_bounds()
    
    # CMA-ES configuration
    sigma_init = 0.3
    opts = {
        'bounds': [list(zip(*bounds))[0], list(zip(*bounds))[1]],
        'maxfevals': max_evals,
        'popsize': 4 + int(3 * np.log(PARAM_DIM)),  # Optimal population size
        'seed': 42,
        'verbose': -1 if not verbose else 1
    }
    
    start_time = time.time()
    
    try:
        # Run CMA-ES
        es = cma.CMAEvolutionStrategy(theta_init, sigma_init, opts)
        es.optimize(compute_energy_bspline_numpy)
        
        result = es.result
        theta_opt = result.xbest
        energy_opt = result.fbest
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ CMA-ES Optimization Complete")
        print(f"   Runtime: {elapsed_time:.1f}s")
        print(f"   Evaluations: {result.evaluations}")
        print(f"   Best Energy: {energy_opt:.3e} J")
        print(f"   μ = {theta_opt[0]:.3e}, G_geo = {theta_opt[1]:.3e}")
        
        # Compute pure energy (without penalties)
        pure_energy = compute_pure_energy_bspline(theta_opt)
        
        return {
            'success': True,
            'theta_opt': theta_opt,
            'energy_opt': energy_opt,
            'pure_energy': pure_energy,
            'evaluations': result.evaluations,
            'runtime': elapsed_time,
            'method': 'CMA-ES',
            'ansatz': 'B-spline'
        }
        
    except Exception as e:
        print(f"❌ CMA-ES failed: {e}")
        traceback.print_exc()
        return None

# ── 10. JAX LOCAL REFINEMENT ──────────────────────────────────────────────────
def run_jax_refinement_bspline(theta_init, max_iter=500, verbose=True):
    """JAX-accelerated local refinement for B-spline ansatz"""
    if not HAS_JAX:
        print("❌ JAX not available. Install with: pip install jax")
        return None
    
    print(f"\n⚡ JAX Local Refinement for B-Spline Ansatz")
    print(f"   Max iterations: {max_iter}")
    
    # Create JAX-compatible objective and gradient
    @jax.jit
    def objective_and_grad(params):
        energy = compute_energy_bspline_jax(params)
        grad_energy = grad(compute_energy_bspline_jax)(params)
        return energy, grad_energy
    
    start_time = time.time()
    
    try:
        # Convert to JAX arrays
        theta_jax = jnp.array(theta_init)
        
        # L-BFGS-B optimization with JAX
        result = minimize(
            fun=lambda x: float(compute_energy_bspline_jax(x)),
            x0=theta_jax,
            method='L-BFGS-B',
            jac=lambda x: np.array(grad(compute_energy_bspline_jax)(x)),
            bounds=get_bspline_bounds(),
            options={'maxiter': max_iter, 'disp': verbose}
        )
        
        elapsed_time = time.time() - start_time
        
        if result.success:
            print(f"\n✅ JAX Refinement Complete")
            print(f"   Runtime: {elapsed_time:.1f}s")
            print(f"   Iterations: {result.nit}")
            print(f"   Final Energy: {result.fun:.3e} J")
            print(f"   μ = {result.x[0]:.3e}, G_geo = {result.x[1]:.3e}")
            
            # Compute pure energy
            pure_energy = compute_pure_energy_bspline(result.x)
            
            return {
                'success': True,
                'theta_opt': result.x,
                'energy_opt': result.fun,
                'pure_energy': pure_energy,
                'iterations': result.nit,
                'runtime': elapsed_time,
                'method': 'JAX-L-BFGS-B',
                'ansatz': 'B-spline'
            }
        else:
            print(f"❌ JAX refinement failed: {result.message}")
            return None
            
    except Exception as e:
        print(f"❌ JAX refinement error: {e}")
        traceback.print_exc()
        return None

# ── 11. TWO-STAGE B-SPLINE OPTIMIZATION ───────────────────────────────────────
def run_bspline_two_stage_optimization(cma_evals=3000, jax_iters=500, verbose=True):
    """
    Two-stage B-spline optimization: CMA-ES global + JAX local refinement
    
    Args:
        cma_evals: Maximum CMA-ES evaluations
        jax_iters: Maximum JAX iterations
        verbose: Print progress information
    
    Returns:
        Optimization results dictionary
    """
    print(f"\n{'='*80}")
    print(f"🚀 TWO-STAGE B-SPLINE OPTIMIZATION PIPELINE")
    print(f"{'='*80}")
    print(f"Target: E- < -2.0×10⁵³ J with stability (λ_max < 0)")
    
    # Stage 1: CMA-ES Global Search
    print(f"\n🌍 STAGE 1: CMA-ES GLOBAL SEARCH")
    cma_result = run_cma_es_bspline(max_evals=cma_evals, verbose=verbose)
    
    if cma_result is None or not cma_result['success']:
        print("❌ CMA-ES stage failed")
        return None
    
    # Stage 2: JAX Local Refinement
    if HAS_JAX:
        print(f"\n⚡ STAGE 2: JAX LOCAL REFINEMENT")
        jax_result = run_jax_refinement_bspline(cma_result['theta_opt'], 
                                               max_iter=jax_iters, verbose=verbose)
        
        if jax_result is not None and jax_result['success']:
            # Use JAX result if it's better
            if jax_result['energy_opt'] < cma_result['energy_opt']:
                final_result = jax_result
                final_result['cma_result'] = cma_result
                print(f"✅ JAX refinement improved result!")
            else:
                final_result = cma_result
                final_result['jax_result'] = jax_result
                print(f"⚠️  CMA-ES result better than JAX refinement")
        else:
            final_result = cma_result
            print(f"⚠️  JAX refinement failed, using CMA-ES result")
    else:
        final_result = cma_result
        print(f"⚠️  JAX not available, using CMA-ES result only")
    
    # Stage 3: Analysis and Validation
    print(f"\n📊 STAGE 3: ANALYSIS AND VALIDATION")
    analyze_bspline_result(final_result)
    
    # Save results
    save_bspline_results(final_result)
    
    return final_result

# ── 12. ANALYSIS AND VISUALIZATION ────────────────────────────────────────────
def compute_pure_energy_bspline(params):
    """Compute pure energy without penalties for analysis"""
    mu, G_geo = params[0], params[1]
    control_points = params[2:]
    
    # Compute f(r) and f'(r) on the grid
    f_vals = evaluate_bspline(r_grid, control_points)
    f_prime_vals = evaluate_bspline_derivative(r_grid, control_points)
    
    # Quantum inequality backreaction term
    sinc_term = np.sinc(mu / (hbar * c))
    backreaction_factor = (1.0 + beta_back * G_geo * sinc_term)
    
    # Pure energy density (no penalties)
    rho_eff = f_vals**2 * backreaction_factor
    
    # Volume integral
    E_pure = np.trapz(rho_eff * vol_weights, r_grid) * c4_8piG
    
    return float(E_pure)

def analyze_bspline_result(result):
    """Comprehensive analysis of B-spline optimization result"""
    theta_opt = result['theta_opt']
    energy_opt = result['energy_opt']
    pure_energy = result.get('pure_energy', compute_pure_energy_bspline(theta_opt))
    
    mu, G_geo = theta_opt[0], theta_opt[1]
    control_points = theta_opt[2:]
    
    print(f"\n{'='*60}")
    print(f"B-SPLINE OPTIMIZATION ANALYSIS")
    print(f"{'='*60}")
    print(f"Energy (with penalties): {energy_opt:.3e} J")
    print(f"Pure Energy E-:          {pure_energy:.3e} J")
    print(f"Geometric Parameters:")
    print(f"  μ =      {mu:.3e}")
    print(f"  G_geo =  {G_geo:.3e}")
    print(f"Control Points: {len(control_points)}")
    for i, cp in enumerate(control_points):
        print(f"  c_{i+1:2d} = {cp:8.4f}")
    
    # Boundary condition analysis
    f_0 = evaluate_bspline(np.array([0.0]), control_points)[0]
    f_R = evaluate_bspline(np.array([R]), control_points)[0]
    print(f"\nBoundary Conditions:")
    print(f"  f(0) = {f_0:.4f} (target: 1.0)")
    print(f"  f(R) = {f_R:.4f} (target: 0.0)")
    
    # Stability analysis if available
    if HAS_STABILITY:
        try:
            stability_penalty = compute_stability_penalty_bspline(theta_opt)
            print(f"\nStability Analysis:")
            print(f"  Stability penalty: {stability_penalty:.3e}")
            
            # Full stability analysis
            stability_params = {
                'mu': float(mu),
                'G_geo': float(G_geo),
                'profile_type': 'bspline',
                'control_points': control_points.tolist(),
                'R_b': R
            }
            stability_result = analyze_stability_3d(stability_params)
            lambda_max = stability_result.get('max_growth_rate', 0.0)
            classification = stability_result.get('classification', 'UNKNOWN')
            
            print(f"  Max growth rate: {lambda_max:.3e}")
            print(f"  Classification: {classification}")
            
        except Exception as e:
            print(f"  Stability analysis failed: {e}")
    
    # Create analysis plots
    create_bspline_plots(theta_opt, energy_opt, pure_energy)

def create_bspline_plots(theta_opt, energy_opt, pure_energy):
    """Create comprehensive analysis plots for B-spline result"""
    mu, G_geo = theta_opt[0], theta_opt[1]
    control_points = theta_opt[2:]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'B-Spline Control-Point Optimization Analysis\n'
                f'E- = {pure_energy:.3e} J, μ = {mu:.2e}, G_geo = {G_geo:.2e}',
                fontsize=14, fontweight='bold')
    
    # Plot 1: Warp profile f(r)
    r_plot = np.linspace(0, R, 200)
    f_plot = evaluate_bspline(r_plot, control_points)
    
    axes[0, 0].plot(r_plot, f_plot, 'b-', linewidth=2, label='B-spline f(r)')
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(x=R, color='g', linestyle='--', alpha=0.7, label=f'Bubble R={R}m')
    axes[0, 0].set_xlabel('Radius r (m)')
    axes[0, 0].set_ylabel('Warp Factor f(r)')
    axes[0, 0].set_title(f'{N_CONTROL}-Point B-Spline Profile')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Control points
    control_positions = np.linspace(0, R, N_CONTROL)
    axes[0, 1].plot(control_positions, control_points, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('Control Point Value')
    axes[0, 1].set_title('B-Spline Control Points')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Derivative df/dr
    f_prime_plot = evaluate_bspline_derivative(r_plot, control_points)
    axes[1, 0].plot(r_plot, f_prime_plot, 'r-', linewidth=2, label="f'(r)")
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Radius r (m)')
    axes[1, 0].set_ylabel("Derivative f'(r)")
    axes[1, 0].set_title('Profile Derivative')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Energy density
    f_vals = evaluate_bspline(r_grid, control_points)
    sinc_term = np.sinc(mu / (hbar * c))
    backreaction_factor = (1.0 + beta_back * G_geo * sinc_term)
    rho_eff = f_vals**2 * backreaction_factor
    
    axes[1, 1].plot(r_grid, rho_eff, 'g-', linewidth=2, label='Energy density')
    axes[1, 1].set_xlabel('Radius r (m)')
    axes[1, 1].set_ylabel('Energy Density')
    axes[1, 1].set_title('Effective Energy Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'bspline_analysis_{time.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Analysis plots saved: {filename}")
    
    # Close to prevent blocking
    plt.close()

def save_bspline_results(result):
    """Save B-spline optimization results to JSON"""
    # Convert numpy arrays to lists for JSON serialization
    result_copy = result.copy()
    if 'theta_opt' in result_copy:
        result_copy['theta_opt'] = result_copy['theta_opt'].tolist()
    
    # Add timestamp and metadata
    result_copy.update({
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'ansatz_type': 'B-spline',
        'n_control_points': N_CONTROL,
        'spline_order': SPLINE_ORDER,
        'parameter_dimension': PARAM_DIM,
        'stability_analysis_available': HAS_STABILITY
    })
    
    filename = f'bspline_results_{time.strftime("%Y%m%d_%H%M%S")}.json'
    
    try:
        with open(filename, 'w') as f:
            json.dump(result_copy, f, indent=2)
        print(f"💾 Results saved: {filename}")
    except Exception as e:
        print(f"⚠️  Failed to save results: {e}")

# ── 13. SURROGATE-ASSISTED OPTIMIZATION (OPTIONAL) ───────────────────────────
def run_surrogate_assisted_bspline(n_initial=50, n_iterations=10, verbose=True):
    """
    Surrogate-assisted optimization using Gaussian Process
    
    Args:
        n_initial: Number of initial random samples
        n_iterations: Number of surrogate-guided iterations
        verbose: Print progress information
    
    Returns:
        Optimization results dictionary
    """
    if not HAS_GP:
        print("❌ Gaussian Process not available. Install sklearn for surrogate optimization.")
        return None
    
    print(f"\n🧠 SURROGATE-ASSISTED B-SPLINE OPTIMIZATION")
    print(f"   Initial samples: {n_initial}")
    print(f"   Surrogate iterations: {n_iterations}")
    
    bounds = get_bspline_bounds()
    
    # Step 1: Generate initial samples
    print(f"\n📊 Generating {n_initial} initial samples...")
    X_samples = []
    y_samples = []
    
    for i in range(n_initial):
        # Random sample within bounds
        theta = np.array([np.random.uniform(low, high) for low, high in bounds])
        energy = compute_energy_bspline_numpy(theta)
        
        X_samples.append(theta)
        y_samples.append(energy)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"   Sample {i+1}/{n_initial}: E = {energy:.3e} J")
    
    X_samples = np.array(X_samples)
    y_samples = np.array(y_samples)
    
    best_idx = np.argmin(y_samples)
    best_theta = X_samples[best_idx]
    best_energy = y_samples[best_idx]
    
    print(f"✅ Initial sampling complete")
    print(f"   Best initial energy: {best_energy:.3e} J")
    
    # Step 2: Surrogate-guided optimization
    print(f"\n🎯 Surrogate-guided optimization...")
    
    for iteration in range(n_iterations):
        print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
        
        # Train Gaussian Process
        kernel = RBF(length_scale=1.0) + Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5)
        gp.fit(X_samples, y_samples)
        
        # Acquisition function: Expected Improvement
        def expected_improvement(theta):
            theta = np.atleast_2d(theta)
            mu, sigma = gp.predict(theta, return_std=True)
            
            # Calculate EI
            improvement = best_energy - mu
            z = improvement / (sigma + 1e-9)
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            
            return -ei  # Minimize negative EI
        
        # Optimize acquisition function
        from scipy.stats import norm
        ei_result = differential_evolution(
            expected_improvement,
            bounds,
            maxiter=100,
            seed=42 + iteration
        )
        
        if ei_result.success:
            candidate_theta = ei_result.x
            candidate_energy = compute_energy_bspline_numpy(candidate_theta)
            
            print(f"   Candidate energy: {candidate_energy:.3e} J")
            
            # Add to training data
            X_samples = np.vstack([X_samples, candidate_theta])
            y_samples = np.append(y_samples, candidate_energy)
            
            # Update best
            if candidate_energy < best_energy:
                best_theta = candidate_theta
                best_energy = candidate_energy
                print(f"   🎉 New best energy: {best_energy:.3e} J")
        else:
            print(f"   ⚠️  Acquisition optimization failed")
    
    print(f"\n✅ Surrogate optimization complete")
    print(f"   Final best energy: {best_energy:.3e} J")
    
    # Compute pure energy
    pure_energy = compute_pure_energy_bspline(best_theta)
    
    return {
        'success': True,
        'theta_opt': best_theta,
        'energy_opt': best_energy,
        'pure_energy': pure_energy,
        'method': 'Surrogate-GP',
        'ansatz': 'B-spline',
        'n_samples': len(X_samples),
        'X_samples': X_samples,
        'y_samples': y_samples
    }

# ── 14. MAIN EXECUTION ─────────────────────────────────────────────────────────
def main():
    """Main execution function for B-spline control-point optimization"""
    print(f"\n{'='*80}")
    print(f"🚀 B-SPLINE CONTROL-POINT WARP BUBBLE OPTIMIZER")
    print(f"{'='*80}")
    print(f"Target: E- < -2.0×10⁵³ J with stability constraint (λ_max < 0)")
    print(f"Ansatz: {N_CONTROL}-point B-spline with joint (μ, G_geo) optimization")
    print(f"Features: Two-stage pipeline + stability penalties + non-blocking plots")
    
    # Check available optimization methods
    print(f"\n🔧 Available Methods:")
    print(f"   CMA-ES:           {'✅' if HAS_CMA else '❌'}")
    print(f"   JAX:              {'✅' if HAS_JAX else '❌'}")
    print(f"   3D Stability:     {'✅' if HAS_STABILITY else '❌'}")
    print(f"   Surrogate (GP):   {'✅' if HAS_GP else '❌'}")
    
    start_time = time.time()
    
    # Run two-stage optimization
    result = run_bspline_two_stage_optimization(
        cma_evals=3000,
        jax_iters=500,
        verbose=True
    )
    
    if result is not None and result['success']:
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"🏆 B-SPLINE OPTIMIZATION COMPLETE!")
        print(f"{'='*80}")
        print(f"Total Runtime: {total_time:.1f}s")
        print(f"Best Energy:   {result['pure_energy']:.3e} J")
        print(f"Method:        {result['method']}")
        print(f"Parameters:    μ = {result['theta_opt'][0]:.3e}, G_geo = {result['theta_opt'][1]:.3e}")
        
        # Check if target was achieved
        target_energy = -2e53
        if result['pure_energy'] < target_energy:
            print(f"🎯 TARGET ACHIEVED! E- < {target_energy:.1e} J")
        else:
            improvement_needed = result['pure_energy'] / target_energy
            print(f"⚠️  Target not reached. Need {improvement_needed:.1f}× improvement.")
        
        # Optional: Run surrogate-assisted optimization for further improvement
        if HAS_GP and result['pure_energy'] > target_energy:
            print(f"\n🧠 Running surrogate-assisted optimization for further improvement...")
            surrogate_result = run_surrogate_assisted_bspline(n_initial=30, n_iterations=5)
            
            if surrogate_result and surrogate_result['pure_energy'] < result['pure_energy']:
                print(f"🎉 Surrogate optimization improved result!")
                print(f"   New best energy: {surrogate_result['pure_energy']:.3e} J")
                save_bspline_results(surrogate_result)
    
    else:
        print(f"\n❌ B-SPLINE OPTIMIZATION FAILED")
        print(f"   Check error messages above for debugging")

if __name__ == "__main__":
    main()
