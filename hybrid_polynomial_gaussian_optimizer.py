#!/usr/bin/env python3
"""
HYBRID POLYNOMIAL + GAUSSIAN ANSATZ OPTIMIZER

Implements the hybrid "Polynomial + Gaussian" ansatz from the roadmap:
f(r) = { 1,                           0 ≤ r ≤ r0
       { polynomial(r),               r0 < r < r1  
       { sum of Gaussians(r),         r1 ≤ r < R
       { 0,                           r ≥ R

The polynomial "ramps" smoothly from 1→(Gaussian tail), reducing wasted 
curvature in the core. This implementation uses a 2nd-order polynomial
transition with 2 Gaussians in the tail for optimal energy minimization.

Target: Achieve E- ≈ -1.25×10³¹ J (~1.2× better than pure Gaussian)
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import json
import time

# Optional CMA-ES support
HAS_CMA = False
try:
    import cma
    HAS_CMA = True
    print("✅ CMA-ES available for hybrid optimization")
except ImportError:
    print("⚠️  CMA-ES not available. Install with: pip install cma")

# ── LQG BOUND ENFORCEMENT ──────────────────────────────────────────────────────
# Import LQG-modified quantum inequality bound enforcement
try:
    import sys
    sys.path.append('src')
    from src.warp_qft.stability import enforce_lqg_bound, lqg_modified_bounds
    HAS_LQG_BOUNDS = True
    print("✅ LQG-modified quantum inequality bounds loaded")
except ImportError:
    HAS_LQG_BOUNDS = False
    print("⚠️  LQG bounds not available - using classical energy computation")
    def enforce_lqg_bound(energy, spatial_scale, flight_time, C_lqg=None):
        """Fallback function when LQG bounds module is not available"""
        return energy  # No enforcement

# ── 1. PHYSICAL CONSTANTS ──────────────────────────────────────────────────────
beta_back = 1.9443254780147017
hbar      = 1.0545718e-34  # ℏ (SI)
c         = 299792458      # Speed of light (m/s)
G         = 6.67430e-11    # Gravitational constant (m³/kg/s²)
tau       = 1e-9           # QI sampling time
v         = 1.0            # warp velocity (c = 1 units)
R         = 1.0            # bubble radius = 1 m

# Default parameters
mu0_default   = 1e-6       # polymer length
G_geo_default = 1e-5       # Van den Broeck–Natário factor

# Conversion factor for natural units to Joules
c4_8piG = c**4 / (8.0 * np.pi * G)  # ≈ 4.815×10⁴² J⋅m⁻³

# ── 2. HYBRID ANSATZ CONFIGURATION ────────────────────────────────────────────
# Hyperparameters for hybrid ansatz
M_gauss_hybrid = 2    # Number of Gaussians in tail region
poly_order = 2        # Order of polynomial transition (quadratic)

# Precompute radial grid for vectorized integration
N_grid = 800
r_grid = np.linspace(0.0, R, N_grid)
dr = r_grid[1] - r_grid[0]
vol_weights = 4.0 * np.pi * r_grid**2

# ── 3. HYBRID ANSATZ FUNCTIONS ────────────────────────────────────────────────
def f_hybrid(r, params):
    """
    Hybrid polynomial + Gaussian ansatz:
    
    params = [r0, r1, a1, a2, A0, r0_0, σ0, A1, r0_1, σ1]
    Total length = 2 + poly_order + 3*M_gauss_hybrid = 2 + 2 + 6 = 10
    
    Regions:
    - [0, r0]: f = 1 (flat core)
    - (r0, r1): f = 1 + a1*x + a2*x² where x = (r-r0)/(r1-r0)
    - [r1, R): f = sum of Gaussians
    - [R, ∞): f = 0
    """
    r = np.asarray(r)
    result = np.zeros_like(r, dtype=np.float64)
    
    # Extract parameters
    r0, r1 = params[0], params[1]
    a1, a2 = params[2], params[3]
    gauss_params = params[4:]  # [A0, r0_0, σ0, A1, r0_1, σ1]
    
    # Region masks
    mask_core = r <= r0
    mask_poly = (r > r0) & (r < r1)
    mask_gauss = (r >= r1) & (r < R)
    
    # (1) Flat core: f = 1
    result[mask_core] = 1.0
    
    # (2) Polynomial transition
    if np.any(mask_poly):
        r_poly = r[mask_poly]
        x = (r_poly - r0) / (r1 - r0)  # Normalize to [0,1]
        poly_val = 1.0 + a1 * x + a2 * x**2
        result[mask_poly] = np.clip(poly_val, 0.0, 1.0)
    
    # (3) Gaussian tail
    if np.any(mask_gauss) and len(gauss_params) >= 6:
        r_gauss = r[mask_gauss]
        gauss_total = np.zeros_like(r_gauss)
        
        for i in range(M_gauss_hybrid):
            Ai = gauss_params[3*i + 0]
            r0_i = gauss_params[3*i + 1]
            sig_i = gauss_params[3*i + 2]
            x = (r_gauss - r0_i) / sig_i
            gauss_total += Ai * np.exp(-0.5 * x*x)
        
        result[mask_gauss] = np.clip(gauss_total, 0.0, 1.0)
    
    return result

def f_hybrid_prime(r, params):
    """Derivative of hybrid ansatz"""
    r = np.asarray(r)
    result = np.zeros_like(r, dtype=np.float64)
    
    # Extract parameters
    r0, r1 = params[0], params[1]
    a1, a2 = params[2], params[3]
    gauss_params = params[4:]
    
    # Region masks
    mask_poly = (r > r0) & (r < r1)
    mask_gauss = (r >= r1) & (r < R)
    
    # (1) Polynomial region derivative
    if np.any(mask_poly):
        r_poly = r[mask_poly]
        x = (r_poly - r0) / (r1 - r0)
        dx_dr = 1.0 / (r1 - r0)
        poly_deriv = (a1 + 2*a2 * x) * dx_dr
        result[mask_poly] = poly_deriv
    
    # (2) Gaussian region derivative
    if np.any(mask_gauss) and len(gauss_params) >= 6:
        r_gauss = r[mask_gauss]
        gauss_deriv = np.zeros_like(r_gauss)
        
        for i in range(M_gauss_hybrid):
            Ai = gauss_params[3*i + 0]
            r0_i = gauss_params[3*i + 1]
            sig_i = gauss_params[3*i + 2]
            x = (r_gauss - r0_i) / sig_i
            pref = Ai * np.exp(-0.5 * x*x)
            gauss_deriv += pref * (-(r_gauss - r0_i) / (sig_i**2))
        
        result[mask_gauss] = gauss_deriv
    
    return result

# ── 4. ENERGY CALCULATION ─────────────────────────────────────────────────────
def E_negative_hybrid(params, mu0=None, G_geo=None):
    """
    Fast vectorized energy calculation for hybrid ansatz.
    """
    if mu0 is None:
        mu0 = mu0_default
    if G_geo is None:
        G_geo = G_geo_default
    
    # Calculate f'(r) on grid
    fp_vals = f_hybrid_prime(r_grid, params)
      # Apply enhancement factors
    sinc_val = np.sinc(mu0 / np.pi) if mu0 > 0 else 1.0
    prefac = - (v**2) / (8.0 * np.pi) * beta_back * sinc_val / G_geo
    rho_vals = prefac * (fp_vals**2)
    
    # Vectorized integration: ∫ ρ(r) × 4πr² dr
    integral = np.sum(rho_vals * vol_weights) * dr
    E_negative = integral * c4_8piG
    
    # ⭐ ENFORCE LQG-MODIFIED QUANTUM INEQUALITY BOUND ⭐
    # Target: Push E₋ as close as possible to -C_LQG/T^4 (stricter than Ford-Roman)
    if HAS_LQG_BOUNDS:
        E_negative = enforce_lqg_bound(E_negative, R, tau)
    
    return E_negative

def rho_eff_hybrid(r, params, mu0=None, G_geo=None):
    """Effective energy density for hybrid ansatz"""
    if mu0 is None:
        mu0 = mu0_default
    if G_geo is None:
        G_geo = G_geo_default
    
    fp = f_hybrid_prime(r, params)
    sinc_val = np.sinc(mu0 / np.pi) if mu0 > 0 else 1.0
    prefac = - (v**2) / (8.0 * np.pi) * beta_back * sinc_val / G_geo
    return prefac * (fp**2)

# ── 5. PENALTY FUNCTIONS ──────────────────────────────────────────────────────
def penalty_hybrid(params, mu0=None, G_geo=None, 
                  lam_qi=1e50, lam_bound=1e4, lam_continuity=1e5, 
                  lam_bounds_check=1e8, lam_envelope=1e4):
    """
    Comprehensive penalty function for hybrid ansatz with all constraints.
    """
    if mu0 is None:
        mu0 = mu0_default
    if G_geo is None:
        G_geo = G_geo_default
    
    r0, r1 = params[0], params[1]
    a1, a2 = params[2], params[3]
    gauss_params = params[4:]
    
    total_penalty = 0.0
    
    # (1) Bounds check: ensure 0 < r0 < r1 < R
    if not (0.0 < r0 < r1 < R):
        bounds_violation = (abs(min(r0, 0.0)) + 
                          abs(max(r0 - r1, 0.0)) + 
                          abs(max(r1 - R, 0.0)))
        total_penalty += lam_bounds_check * bounds_violation**2
    
    # (2) QI penalty at r=0
    try:
        rho0 = rho_eff_hybrid(0.0, params, mu0, G_geo)
        qi_bound = - (hbar * np.sinc(mu0 / np.pi)) / (12.0 * np.pi * tau**2)
        qi_violation = max(0.0, -(rho0 - qi_bound))
        total_penalty += lam_qi * (qi_violation**2)
    except:
        total_penalty += lam_qi  # Large penalty for evaluation errors
    
    # (3) Boundary conditions: f(0) ≈ 1, f(R) ≈ 0
    try:
        f0 = f_hybrid(0.0, params)
        fR = f_hybrid(R, params)
        total_penalty += lam_bound * ((f0 - 1.0)**2 + (fR - 0.0)**2)
    except:
        total_penalty += lam_bound
    
    # (4) Continuity at interfaces
    try:
        # At r0: left side = 1, right side = 1 + a1*0 + a2*0 = 1 (automatic)
        # At r1: left side = 1 + a1 + a2, right side = sum of Gaussians
        poly_val_at_r1 = 1.0 + a1 + a2
        
        gauss_val_at_r1 = 0.0
        if len(gauss_params) >= 6:
            for i in range(M_gauss_hybrid):
                Ai = gauss_params[3*i + 0]
                r0_i = gauss_params[3*i + 1]
                sig_i = gauss_params[3*i + 2]
                x = (r1 - r0_i) / sig_i
                gauss_val_at_r1 += Ai * np.exp(-0.5 * x*x)
        
        continuity_error = (poly_val_at_r1 - gauss_val_at_r1)**2
        total_penalty += lam_continuity * continuity_error
    except:
        total_penalty += lam_continuity
    
    # (5) Envelope constraint: sum of Gaussian amplitudes ≤ 1
    try:
        if len(gauss_params) >= 6:
            A_sum = sum(gauss_params[3*i] for i in range(M_gauss_hybrid))
            envelope_violation = max(0.0, A_sum - 1.0)
            total_penalty += lam_envelope * (envelope_violation**2)
    except:
        total_penalty += lam_envelope
    
    return total_penalty

def curvature_penalty_hybrid(params, lam_curv=1e3):
    """
    Smoothness penalty for hybrid ansatz based on curvature.
    """
    try:
        # Sample f on grid
        f_vals = f_hybrid(r_grid, params)
        
        # Second derivative via central differences
        fpp = np.zeros_like(f_vals)
        fpp[1:-1] = (f_vals[2:] - 2*f_vals[1:-1] + f_vals[:-2]) / (dr**2)
        fpp[0] = fpp[1]
        fpp[-1] = fpp[-2]
        
        # Weighted integral: ∫ fpp² r² dr
        integrand = (fpp**2) * (r_grid**2)
        integral = np.sum(integrand) * dr
        
        return lam_curv * integral
    except:
        return lam_curv * 1e10  # Large penalty for evaluation errors

def monotonicity_penalty_hybrid(params, lam_mono=1e3):
    """
    Enforce f'(r) ≤ 0 by penalizing positive derivatives.
    """
    try:
        fp_vals = f_hybrid_prime(r_grid, params)
        violation = np.maximum(0.0, fp_vals)
        integral = np.sum(violation**2) * dr
        return lam_mono * integral
    except:
        return lam_mono * 1e10

def objective_hybrid(params, mu0=None, G_geo=None, 
                    enable_curvature=True, enable_monotonicity=True):
    """
    Complete objective function for hybrid ansatz optimization.
    """
    try:
        # Energy calculation
        energy = E_negative_hybrid(params, mu0, G_geo)
        
        # Basic penalties
        penalty_basic = penalty_hybrid(params, mu0, G_geo)
        
        # Physics-informed penalties
        penalty_physics = 0.0
        if enable_curvature:
            penalty_physics += curvature_penalty_hybrid(params)
        if enable_monotonicity:
            penalty_physics += monotonicity_penalty_hybrid(params)
        
        return energy + penalty_basic + penalty_physics
    
    except Exception as e:
        print(f"⚠️  Objective evaluation failed: {e}")
        return 1e15  # Very large penalty for failed evaluations

# ── 6. OPTIMIZATION BOUNDS AND INITIALIZATION ─────────────────────────────────
def get_hybrid_bounds():
    """
    Generate bounds for hybrid ansatz parameters.
    params = [r0, r1, a1, a2, A0, r0_0, σ0, A1, r0_1, σ1]
    """
    bounds = []
    
    # Interface bounds with constraint r0 < r1
    bounds.extend([
        (0.1*R, 0.4*R),    # r0 ∈ [0.1R, 0.4R]
        (0.5*R, 0.8*R)     # r1 ∈ [0.5R, 0.8R]
    ])
    
    # Polynomial coefficient bounds
    bounds.extend([
        (-2.0, 2.0),       # a1 ∈ [-2, 2]
        (-2.0, 2.0)        # a2 ∈ [-2, 2]
    ])
    
    # Gaussian parameter bounds (in tail region [r1, R])
    for i in range(M_gauss_hybrid):
        bounds.extend([
            (0.0, 1.0),        # Amplitude Ai ∈ [0, 1]
            (0.5*R, R),        # Position r0_i ∈ [r1, R] approx
            (R/50, R/4)        # Width σ_i ∈ [R/50, R/4]
        ])
    
    return bounds

def get_hybrid_initial_guess():
    """
    Generate physics-informed initial guess for hybrid ansatz.
    """
    # Interface positions
    r0_init = 0.25 * R
    r1_init = 0.65 * R
    
    # Small polynomial coefficients for smooth transition
    a1_init = -0.5   # Slight negative slope
    a2_init = 0.1    # Small curvature
    
    # Gaussian parameters in tail region
    gauss_init = []
    for i in range(M_gauss_hybrid):
        A_i = 0.4 / (i + 1)           # Decreasing amplitudes
        r0_i = r1_init + (i + 0.5) * (R - r1_init) / M_gauss_hybrid
        sig_i = (R - r1_init) / (4 * M_gauss_hybrid)
        gauss_init.extend([A_i, r0_i, sig_i])
    
    return np.array([r0_init, r1_init, a1_init, a2_init] + gauss_init)

# ── 7. OPTIMIZATION ALGORITHMS ────────────────────────────────────────────────
def optimize_with_cma_hybrid(bounds, objective_func, sigma0=0.2, maxiter=300):
    """CMA-ES optimization for hybrid ansatz"""
    if not HAS_CMA:
        raise ImportError("CMA-ES not available. Install with: pip install cma")
    
    # Initial guess and bounds
    x0 = [(b[0] + b[1]) / 2.0 for b in bounds]
    bounds_cma = [[b[0] for b in bounds], [b[1] for b in bounds]]
    
    # CMA options
    opts = {
        'bounds': bounds_cma,
        'popsize': 20,
        'maxiter': maxiter,
        'tolfun': 1e-9,
        'verb_disp': 1
    }
    
    try:
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        es.optimize(objective_func)
        
        return {
            'x': np.array(es.result.xbest),
            'fun': es.result.fbest,
            'success': True,
            'nfev': es.result.evaluations
        }
    except Exception as e:
        print(f"❌ CMA-ES optimization failed: {e}")
        return {'success': False}

def optimize_with_de_hybrid(bounds, objective_func, maxiter=300):
    """Differential Evolution optimization for hybrid ansatz"""
    try:
        result = differential_evolution(
            objective_func,
            bounds,
            strategy='best1bin',
            maxiter=maxiter,
            popsize=20,
            tol=1e-9,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=False,
            workers=-1,  # Parallel processing
            disp=True
        )
        return result
    except Exception as e:
        print(f"❌ Differential Evolution failed: {e}")
        return {'success': False}

# ── 8. MAIN OPTIMIZATION PIPELINE ─────────────────────────────────────────────
def run_hybrid_optimization(mu0=None, G_geo=None, use_cma=True, 
                           enable_curvature=True, enable_monotonicity=True,
                           verbose=True):
    """
    Main optimization pipeline for hybrid polynomial + Gaussian ansatz.
    
    Returns:
        dict with optimization results
    """
    if mu0 is None:
        mu0 = mu0_default
    if G_geo is None:
        G_geo = G_geo_default
    
    if verbose:
        print("🚀 HYBRID POLYNOMIAL + GAUSSIAN ANSATZ OPTIMIZER")
        print("=" * 60)
        print(f"📊 Configuration:")
        print(f"   Parameters: μ₀={mu0:.1e}, G_geo={G_geo:.1e}")
        print(f"   Polynomial order: {poly_order}")
        print(f"   Gaussians in tail: {M_gauss_hybrid}")
        print(f"   Physics constraints: Curvature={enable_curvature}, "
              f"Monotonicity={enable_monotonicity}")
        print(f"   Optimizer: {'CMA-ES' if use_cma and HAS_CMA else 'Differential Evolution'}")
    
    # Get bounds and initial guess
    bounds = get_hybrid_bounds()
    x0 = get_hybrid_initial_guess()
    
    # Define objective function with fixed parameters
    def objective(params):
        return objective_hybrid(params, mu0, G_geo, enable_curvature, enable_monotonicity)
    
    # Global optimization
    start_time = time.time()
    
    if use_cma and HAS_CMA:
        if verbose:
            print(f"\n🔍 Running CMA-ES global optimization...")
        global_result = optimize_with_cma_hybrid(bounds, objective)
    else:
        if verbose:
            print(f"\n🔍 Running Differential Evolution...")
        global_result = optimize_with_de_hybrid(bounds, objective)
    
    if not global_result.get('success', False):
        return {'success': False, 'message': 'Global optimization failed'}
    
    global_time = time.time() - start_time
    
    # Local refinement
    if verbose:
        print(f"✅ Global optimization completed in {global_time:.1f}s")
        print(f"🔧 Refining with L-BFGS-B...")
    
    local_start = time.time()
    local_result = minimize(
        objective,
        x0=global_result['x'],
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 300, 'ftol': 1e-9}
    )
    
    local_time = time.time() - local_start
    total_time = global_time + local_time
    
    # Use best result
    if local_result.success and local_result.fun < global_result['fun']:
        best_params = local_result.x
        best_objective = local_result.fun
    else:
        best_params = global_result['x']
        best_objective = global_result['fun']
    
    # Calculate final metrics
    best_energy = E_negative_hybrid(best_params, mu0, G_geo)
    penalty = penalty_hybrid(best_params, mu0, G_geo)
    
    if verbose:
        print(f"✅ Local refinement completed in {local_time:.1f}s")
        print(f"\n🏆 HYBRID OPTIMIZATION RESULTS:")
        print(f"   Energy E₋: {best_energy:.3e} J")
        print(f"   Penalty: {penalty:.3e}")
        print(f"   Total objective: {best_objective:.3e}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Function evaluations: {global_result.get('nfev', 0)}")
        
        # Parameter breakdown
        r0, r1 = best_params[0], best_params[1]
        a1, a2 = best_params[2], best_params[3]
        print(f"\n📊 OPTIMIZED PARAMETERS:")
        print(f"   Interfaces: r₀={r0:.3f}, r₁={r1:.3f}")
        print(f"   Polynomial: a₁={a1:.3f}, a₂={a2:.3f}")
        
        for i in range(M_gauss_hybrid):
            Ai = best_params[4 + 3*i + 0]
            r0_i = best_params[4 + 3*i + 1]
            sig_i = best_params[4 + 3*i + 2]
            print(f"   Gaussian {i}: A={Ai:.3f}, r₀={r0_i:.3f}, σ={sig_i:.3f}")
    
    return {
        'success': True,
        'ansatz_type': 'hybrid',
        'params': best_params.tolist(),
        'energy_J': best_energy,
        'penalty': penalty,
        'objective': best_objective,
        'mu0': mu0,
        'G_geo': G_geo,
        'optimization_time': total_time,
        'function_evaluations': global_result.get('nfev', 0),
        'interfaces': {'r0': best_params[0], 'r1': best_params[1]},
        'polynomial_coeffs': {'a1': best_params[2], 'a2': best_params[3]},
        'gaussian_params': [
            {'A': best_params[4+3*i], 'r0': best_params[4+3*i+1], 'sigma': best_params[4+3*i+2]}
            for i in range(M_gauss_hybrid)
        ]
    }

# ── 9. ANALYSIS AND VISUALIZATION ─────────────────────────────────────────────
def analyze_hybrid_profile(params, mu0=None, G_geo=None, save_plot=True):
    """
    Analyze and visualize the optimized hybrid profile.
    """
    if mu0 is None:
        mu0 = mu0_default
    if G_geo is None:
        G_geo = G_geo_default
    
    # Generate high-resolution profile
    r_plot = np.linspace(0, R, 1000)
    
    f_vals = f_hybrid(r_plot, params)
    fp_vals = f_hybrid_prime(r_plot, params)
    rho_vals = rho_eff_hybrid(r_plot, params, mu0, G_geo)
    
    # Calculate curvature
    fpp = np.gradient(np.gradient(f_vals, r_plot), r_plot)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Profile with region boundaries
    r0, r1 = params[0], params[1]
    axes[0, 0].plot(r_plot, f_vals, 'b-', linewidth=2, label='f(r)')
    axes[0, 0].axvline(r0, color='r', linestyle='--', alpha=0.7, label=f'r₀={r0:.3f}')
    axes[0, 0].axvline(r1, color='g', linestyle='--', alpha=0.7, label=f'r₁={r1:.3f}')
    axes[0, 0].set_xlabel('r (m)')
    axes[0, 0].set_ylabel('f(r)')
    axes[0, 0].set_title('Hybrid Warp Profile')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Derivative
    axes[0, 1].plot(r_plot, fp_vals, 'r-', linewidth=2)
    axes[0, 1].axvline(r0, color='r', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(r1, color='g', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('r (m)')
    axes[0, 1].set_ylabel("f'(r)")
    axes[0, 1].set_title("Profile Derivative")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy density
    axes[1, 0].plot(r_plot, rho_vals, 'purple', linewidth=2)
    axes[1, 0].axvline(r0, color='r', linestyle='--', alpha=0.7)
    axes[1, 0].axvline(r1, color='g', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('r (m)')
    axes[1, 0].set_ylabel('ρ_eff (J/m³)')
    axes[1, 0].set_title('Effective Energy Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Curvature
    axes[1, 1].plot(r_plot, fpp, 'm-', linewidth=2)
    axes[1, 1].axvline(r0, color='r', linestyle='--', alpha=0.7)
    axes[1, 1].axvline(r1, color='g', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('r (m)')
    axes[1, 1].set_ylabel("f''(r)")
    axes[1, 1].set_title('Profile Curvature')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Calculate and display energy
    energy = E_negative_hybrid(params, mu0, G_geo)
    plt.suptitle(f'Hybrid Ansatz Analysis - E₋ = {energy:.2e} J', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_plot:
        filename = 'hybrid_ansatz_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Hybrid analysis plot saved as '{filename}'")
    
    return fig

def compare_with_gaussian(hybrid_result, gaussian_energy=-1.45e31):
    """
    Compare hybrid result with pure Gaussian baseline.
    """
    print(f"\n🔬 PERFORMANCE COMPARISON")
    print("=" * 40)
    
    hybrid_energy = hybrid_result['energy_J']
    improvement = abs(hybrid_energy / gaussian_energy)
    
    print(f"Gaussian baseline: {gaussian_energy:.2e} J")
    print(f"Hybrid result:     {hybrid_energy:.2e} J")
    print(f"Improvement:       {improvement:.3f}×")
    
    if improvement > 1.0:
        print(f"✅ BREAKTHROUGH: {(improvement-1)*100:.1f}% better than Gaussian!")
    else:
        print(f"📊 Performance: {improvement*100:.1f}% of Gaussian baseline")
    
    # Cost analysis
    cost_hybrid = abs(hybrid_energy) * 0.001 / 1e12  # Rough $/kWh estimate
    cost_gaussian = abs(gaussian_energy) * 0.001 / 1e12
    
    print(f"\nCost estimates:")
    print(f"Gaussian: ~{cost_gaussian:.1e} $")
    print(f"Hybrid:   ~{cost_hybrid:.1e} $")
    print(f"Savings:  ~{abs(cost_gaussian - cost_hybrid):.1e} $")

# ── 10. COMMAND LINE INTERFACE ────────────────────────────────────────────────
def main():
    """Main execution function"""
    print("🌟 HYBRID POLYNOMIAL + GAUSSIAN OPTIMIZER")
    print("🎯 Targeting E₋ ≈ -1.25×10³¹ J (~1.2× better than pure Gaussian)")
    print()
    
    # Run optimization with default parameters
    result = run_hybrid_optimization(
        use_cma=True,
        enable_curvature=True,
        enable_monotonicity=True,
        verbose=True
    )
    
    if result.get('success', False):
        # Analyze the result
        analyze_hybrid_profile(result['params'], result['mu0'], result['G_geo'])
        
        # Compare with Gaussian baseline
        compare_with_gaussian(result)
        
        # Save results
        with open('hybrid_optimization_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to 'hybrid_optimization_result.json'")
        
        # Baseline comparison
        soliton_baseline = -1.584e31
        if result['energy_J'] < soliton_baseline:
            improvement = abs(result['energy_J'] / soliton_baseline)
            print(f"\n🎉 SUCCESS: Exceeded soliton baseline by {(improvement-1)*100:.1f}%!")
        else:
            performance = abs(result['energy_J'] / soliton_baseline) * 100
            print(f"\n📊 Achieved {performance:.1f}% of soliton baseline energy")
    
    else:
        print("❌ Hybrid optimization failed")
        return None
    
    return result

if __name__ == "__main__":
    main()
