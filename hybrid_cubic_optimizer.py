#!/usr/bin/env python3
"""
CUBIC HYBRID OPTIMIZER - HIGHER-ORDER POLYNOMIAL TRANSITION

Implements a cubic (3rd-order) polynomial transition in the hybrid ansatz:
f(r) = { 1,                           0 ≤ r ≤ r0
       { cubic_polynomial(r),         r0 < r < r1  
       { sum of 2 Gaussians(r),       r1 ≤ r < R
       { 0,                           r ≥ R

The cubic polynomial provides smoother transitions and better curvature control,
potentially achieving lower negative energies than quadratic transitions.

Target: Achieve E- ≈ -1.3×10³¹ J (~1.36× better than pure Gaussian)
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import json
import time

# Optional optimizers
HAS_CMA = False
try:
    import cma
    HAS_CMA = True
    print("✅ CMA-ES available for enhanced global optimization")
except ImportError:
    print("⚠️  CMA-ES not available. Using DE + L-BFGS-B")

# ── 1. Physical Constants ─────────────────────────────────────────────────────
beta_back = 1.9443254780147017  # Backreaction enhancement factor
mu0       = 1e-6                # Polymer length scale
v         = 1.0                 # Warp velocity (c = 1 units)
R         = 1.0                 # Bubble radius = 1 m
c         = 299792458           # Speed of light (m/s)
G         = 6.67430e-11         # Gravitational constant (m³/kg/s²)
tau       = 1e-9                # QI sampling time
G_geo     = 1e-5                # Van den Broeck–Natário geometric factor

# Conversion factor for natural units to Joules
c4_8piG = c**4 / (8.0 * np.pi * G)  # ≈ 4.815×10⁴² J⋅m⁻³

# ── 2. Hybrid Ansatz Configuration ────────────────────────────────────────────
M_gauss = 2  # Number of Gaussians in tail region (keep modest for stability)

# ── 3. Precompute Radial Grid & Weights for Vectorized Integration ───────────
N_points = 800
r_grid = np.linspace(0.0, R, N_points)
dr = r_grid[1] - r_grid[0]
vol_weights = 4.0 * np.pi * r_grid**2

# ── 4. Cubic Hybrid Ansatz Functions ──────────────────────────────────────────
def f_hybrid_cubic(r, params):
    """
    Cubic hybrid ansatz
    
    Parameters:
        params = [r0, r1, b1, b2, b3,   A0, r0_0, σ0,   A1, r0_1, σ1]
        
    Regions:
        - Core [0, r0]: f(r) = 1
        - Transition [r0, r1]: f(r) = 1 + b1*x + b2*x² + b3*x³, x=(r-r0)/(r1-r0)
        - Tail [r1, R]: f(r) = sum of Gaussians
        - Exterior [R, ∞]: f(r) = 0
    """
    r = np.atleast_1d(r)
    result = np.zeros_like(r)
    
    r0, r1 = params[0], params[1]
    b1, b2, b3 = params[2], params[3], params[4]
    
    # Core region: f = 1
    core_mask = r <= r0
    result[core_mask] = 1.0
    
    # Transition region: cubic polynomial
    transition_mask = (r > r0) & (r < r1)
    if np.any(transition_mask) and r1 > r0:
        x = (r[transition_mask] - r0) / (r1 - r0)
        result[transition_mask] = 1.0 + b1*x + b2*(x**2) + b3*(x**3)
    
    # Tail region: Gaussian superposition
    tail_mask = (r >= r1) & (r < R)
    if np.any(tail_mask):
        for i in range(M_gauss):
            Ai = params[5 + 3*i + 0]
            r0_i = params[5 + 3*i + 1]
            sig_i = params[5 + 3*i + 2]
            
            x_gauss = (r[tail_mask] - r0_i) / sig_i
            result[tail_mask] += Ai * np.exp(-0.5 * x_gauss**2)
    
    # Ensure physical bounds
    return np.clip(result, 0.0, 1.0)

def f_hybrid_cubic_prime(r, params):
    """
    Derivative of cubic hybrid ansatz
    """
    r = np.atleast_1d(r)
    result = np.zeros_like(r)
    
    r0, r1 = params[0], params[1]
    b1, b2, b3 = params[2], params[3], params[4]
    
    # Core and exterior regions: f' = 0
    # (already initialized to zero)
    
    # Transition region: cubic polynomial derivative
    transition_mask = (r > r0) & (r < r1)
    if np.any(transition_mask) and r1 > r0:
        x = (r[transition_mask] - r0) / (r1 - r0)
        dx_dr = 1.0 / (r1 - r0)
        result[transition_mask] = (b1 + 2*b2*x + 3*b3*(x**2)) * dx_dr
    
    # Tail region: Gaussian derivatives
    tail_mask = (r >= r1) & (r < R)
    if np.any(tail_mask):
        for i in range(M_gauss):
            Ai = params[5 + 3*i + 0]
            r0_i = params[5 + 3*i + 1]
            sig_i = params[5 + 3*i + 2]
            
            x_gauss = (r[tail_mask] - r0_i) / sig_i
            gaussian_term = Ai * np.exp(-0.5 * x_gauss**2)
            result[tail_mask] += gaussian_term * (-(r[tail_mask] - r0_i) / (sig_i**2))
    
    return result

# ── 5. Energy Calculation ─────────────────────────────────────────────────────
def E_negative_hybrid_cubic(params, mu_val=None, G_geo_val=None):
    """
    Calculate total negative energy for cubic hybrid ansatz
    """
    # Use global values if not specified
    mu_use = mu_val if mu_val is not None else mu0
    G_geo_use = G_geo_val if G_geo_val is not None else G_geo
    
    # Calculate f'(r) on the grid
    fp_vals = f_hybrid_cubic_prime(r_grid, params)
    
    # Polymer sinc function enhancement
    sinc_val = np.sinc(mu_use / np.pi) if mu_use > 0 else 1.0
    
    # Effective density prefactor
    prefactor = -(v**2) / (8.0 * np.pi) * beta_back * sinc_val / G_geo_use
    
    # Calculate effective density
    rho_vals = prefactor * (fp_vals**2)
    
    # Vectorized integration
    integral = np.sum(rho_vals * vol_weights) * dr
    
    # Convert to Joules
    return integral * c4_8piG

# ── 6. Enhanced Penalty Functions ─────────────────────────────────────────────
def penalty_hybrid_cubic(params, lam_qi=1e50, lam_bound=1e6, lam_continuity=1e6, 
                         lam_curv=1e3, lam_mono=1e3):
    """
    Comprehensive penalty function for cubic hybrid ansatz
    """
    penalty_total = 0.0
    
    r0, r1 = params[0], params[1]
    b1, b2, b3 = params[2], params[3], params[4]
    
    # (a) Parameter bounds: 0 < r0 < r1 < R
    if not (0.0 < r0 < r1 < R):
        P_bounds = 1e8 * (
            abs(min(r0, 0.0)) + 
            abs(max(r0 - r1, 0.0)) + 
            abs(max(r1 - R, 0.0))
        )**2
        penalty_total += P_bounds
    
    # (b) Quantum Inequality at r=0
    fp0 = f_hybrid_cubic_prime(np.array([0.0]), params)[0]
    sinc_val = np.sinc(mu0 / np.pi) if mu0 > 0 else 1.0
    rho0 = -(v**2) / (8.0*np.pi) * beta_back * sinc_val / G_geo * (fp0**2)
    qi_bound = -(1.0545718e-34 * np.sinc(mu0/np.pi)) / (12.0 * np.pi * tau**2)
    P_qi = lam_qi * max(0.0, -(rho0 - qi_bound))**2
    penalty_total += P_qi
    
    # (c) Continuity at r1: polynomial(r1) = Gaussians(r1)
    if r1 > r0:  # Avoid division by zero
        # Polynomial value at r1 (x=1): 1 + b1 + b2 + b3
        poly_at_r1 = 1.0 + b1 + b2 + b3
        
        # Gaussian sum at r1
        gauss_at_r1 = 0.0
        for i in range(M_gauss):
            Ai = params[5 + 3*i + 0]
            r0_i = params[5 + 3*i + 1]
            sig_i = params[5 + 3*i + 2]
            x_gauss = (r1 - r0_i) / sig_i
            gauss_at_r1 += Ai * np.exp(-0.5 * x_gauss**2)
        
        P_continuity = lam_continuity * (poly_at_r1 - gauss_at_r1)**2
        penalty_total += P_continuity
    
    # (d) Curvature penalty: prevent excessive oscillations
    if lam_curv > 0:
        # Approximate second derivative in transition region
        n_test = 20
        r_test = np.linspace(r0 + 0.1*(r1-r0), r1 - 0.1*(r1-r0), n_test)
        if len(r_test) > 2:
            fp_test = f_hybrid_cubic_prime(r_test, params)
            d2f_approx = np.gradient(np.gradient(fp_test))
            P_curvature = lam_curv * np.sum(d2f_approx**2) * (r1 - r0)
            penalty_total += P_curvature
    
    # (e) Monotonicity constraint: prefer decreasing profiles
    if lam_mono > 0:
        r_mono = np.linspace(0.1*R, 0.9*R, 50)
        fp_mono = f_hybrid_cubic_prime(r_mono, params)
        positive_deriv = np.maximum(0.0, fp_mono)
        P_monotonicity = lam_mono * np.sum(positive_deriv**2) * R
        penalty_total += P_monotonicity
    
    # (f) Gaussian parameter constraints
    for i in range(M_gauss):
        Ai = params[5 + 3*i + 0]
        r0_i = params[5 + 3*i + 1]
        sig_i = params[5 + 3*i + 2]
        
        # Amplitude bounds
        if Ai < 0.0 or Ai > 1.0:
            penalty_total += lam_bound * (min(Ai, 0.0)**2 + max(Ai - 1.0, 0.0)**2)
        
        # Position in tail region
        if r0_i < r1 or r0_i > R:
            penalty_total += lam_bound * (max(r1 - r0_i, 0.0)**2 + max(r0_i - R, 0.0)**2)
        
        # Width constraints
        if sig_i <= 0.0 or sig_i > R:
            penalty_total += lam_bound * (max(-sig_i, 0.0)**2 + max(sig_i - R, 0.0)**2)
    
    return penalty_total
    
    # (d) Amplitude constraints
    A_sum = sum(params[5 + 3*i + 0] for i in range(M_gauss))
    P_amplitude = lam_bound * max(0.0, (A_sum - 1.0))**2
    penalty_total += P_amplitude
    
    # (e) Curvature penalty for smoothness
    f_vals = f_hybrid_cubic(r_grid, params)
    fpp = np.zeros_like(f_vals)
    fpp[1:-1] = (f_vals[2:] - 2*f_vals[1:-1] + f_vals[:-2]) / (dr**2)
    fpp[0] = fpp[1]
    fpp[-1] = fpp[-2]
    P_curvature = lam_curv * np.sum((fpp**2) * (r_grid**2)) * dr
    penalty_total += P_curvature
    
    # (f) Monotonicity: f'(r) ≤ 0 (encourage outward decrease)
    fp_vals = f_hybrid_cubic_prime(r_grid, params)
    P_monotonic = lam_mono * np.sum(np.maximum(0.0, fp_vals)**2) * dr
    penalty_total += P_monotonic
    
    return penalty_total

def objective_hybrid_cubic(params):
    """
    Combined objective: minimize E₋ + penalties
    """
    energy = E_negative_hybrid_cubic(params)
    penalty = penalty_hybrid_cubic(params)
    return energy + penalty

# ── 7. Bounds and Initialization ──────────────────────────────────────────────
def get_bounds_hybrid_cubic():
    """
    Generate parameter bounds for cubic hybrid optimization
    """
    bounds = []
    
    # Region bounds: r0 ∈ [0, 0.3R], r1 ∈ [0.4R, 0.9R]
    bounds += [(0.05, 0.3*R), (0.4*R, 0.9*R)]
    
    # Polynomial coefficients: b1, b2, b3 ∈ [-10, 10]
    bounds += [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    
    # Gaussian parameters: each has A ∈ [0,1], r0 ∈ [r1_min, R], σ ∈ [0.01R, 0.5R]
    for i in range(M_gauss):
        bounds += [
            (0.0, 1.0),        # Amplitude
            (0.4*R, R),        # Position (in tail region)
            (0.01*R, 0.5*R)    # Width
        ]
    
    return bounds

def get_initial_guess_hybrid_cubic():
    """
    Generate educated initial guess for cubic hybrid parameters
    """
    params = []
    
    # Region boundaries
    params.extend([0.15, 0.6])  # r0, r1
    
    # Polynomial coefficients (start with linear-like transition)
    params.extend([-1.0, 0.0, 0.0])  # b1, b2, b3
    
    # Gaussian parameters
    for i in range(M_gauss):
        A_i = 0.4 * (0.8)**i               # Decreasing amplitudes
        r0_i = 0.7 + 0.15*i                # Positions in tail
        sig_i = 0.1 + 0.05*i               # Increasing widths
        params.extend([A_i, r0_i, sig_i])
    
    return np.array(params)

# ── 8. Main Optimization Function ─────────────────────────────────────────────
def optimize_hybrid_cubic(mu_val=None, G_geo_val=None, verbose=True):
    """
    Run complete cubic hybrid optimization
    """
    global mu0, G_geo
    if mu_val is not None:
        mu0 = mu_val
    if G_geo_val is not None:
        G_geo = G_geo_val
        
    if verbose:
        print(f"🚀 CUBIC HYBRID OPTIMIZATION")
        print(f"   μ = {mu0:.1e}, G_geo = {G_geo:.1e}")
        print("=" * 50)
    
    bounds = get_bounds_hybrid_cubic()
    start_time = time.time()
    
    # Stage 1: Global search with Differential Evolution
    if verbose:
        print("🔍 Stage 1: Differential Evolution global search...")
      de_time = time.time() - start_time
    
    if not de_result.success:
        if verbose:
            print(f"   ❌ Global search failed: {de_result.message}")
        return None
    
    if verbose:
        print(f"   ✅ Global search completed in {de_time:.1f}s")
    
    # Stage 2: Local refinement with L-BFGS-B
    if verbose:
        print("🔧 Stage 2: L-BFGS-B local refinement...")
    
    refine_start = time.time()
    result_refine = minimize(
        objective_hybrid_cubic,
        x0=de_result.x,
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12}
    )
    
    refine_time = time.time() - refine_start
    total_time = time.time() - start_time
    
    if not result_refine.success:
        if verbose:
            print(f"   ⚠️ Local refinement failed: {result_refine.message}")
        best_params = de_result.x
    else:
        best_params = result_refine.x
    
    # Calculate final results
    best_energy = E_negative_hybrid_cubic(best_params, mu0, G_geo)
    penalty = penalty_hybrid_cubic(best_params)
    objective_val = best_energy + penalty
    
    if verbose:
        print(f"   ✅ Local refinement completed in {refine_time:.1f}s")
        print(f"\n🏆 CUBIC HYBRID RESULTS:")
        print(f"   Energy E₋: {best_energy:.3e} J")
        print(f"   Penalty: {penalty:.3e}")
        print(f"   Total objective: {objective_val:.3e}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Function evaluations: {de_result.nfev + result_refine.nfev}")
        
        # Parameter breakdown
        r0, r1 = best_params[0], best_params[1]
        b1, b2, b3 = best_params[2], best_params[3], best_params[4]
        print(f"\n📊 OPTIMIZED PARAMETERS:")
        print(f"   Interfaces: r₀={r0:.3f}, r₁={r1:.3f}")
        print(f"   Cubic polynomial: b₁={b1:.3f}, b₂={b2:.3f}, b₃={b3:.3f}")
        
        for i in range(M_gauss):
            Ai = best_params[5 + 3*i + 0]
            r0_i = best_params[5 + 3*i + 1]
            sig_i = best_params[5 + 3*i + 2]
            print(f"   Gaussian {i}: A={Ai:.3f}, r₀={r0_i:.3f}, σ={sig_i:.3f}")
    
    return {
        'success': True,
        'ansatz_type': 'hybrid_cubic',
        'params': best_params.tolist(),
        'energy_J': best_energy,
        'penalty': penalty,
        'objective': objective_val,
        'mu': mu0,
        'G_geo': G_geo,
        'optimization_time': total_time,
        'function_evaluations': de_result.nfev + (result_refine.nfev if result_refine.success else 0),
        'interfaces': {'r0': r0, 'r1': r1},
        'polynomial_coeffs': {'b1': b1, 'b2': b2, 'b3': b3},
        'gaussian_params': [
            {'A': best_params[5+3*i], 'r0': best_params[5+3*i+1], 'sigma': best_params[5+3*i+2]}
            for i in range(M_gauss)
        ]
    }

# ── 9. CMA-ES Optimization (Advanced) ─────────────────────────────────────────
def optimize_hybrid_cubic_cma(mu_val=None, G_geo_val=None, verbose=True):
    """
    CMA-ES optimization for cubic hybrid ansatz
    """
    if not HAS_CMA:
        print("❌ CMA-ES not available. Use optimize_hybrid_cubic() instead.")
        return None
    
    global mu0, G_geo
    if mu_val is not None:
        mu0 = mu_val
    if G_geo_val is not None:
        G_geo = G_geo_val
    
    if verbose:
        print(f"🚀 CUBIC HYBRID OPTIMIZATION (CMA-ES)")
        print(f"   μ = {mu0:.1e}, G_geo = {G_geo:.1e}")
        print("=" * 50)
    
    bounds = get_bounds_hybrid_cubic()
    x0 = get_initial_guess_hybrid_cubic()
    
    # Convert bounds to CMA format
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # CMA-ES options
    opts = cma.CMAOptions()
    opts['bounds'] = [lb.tolist(), ub.tolist()]
    opts['maxiter'] = 600
    opts['popsize'] = 20
    opts['tolfun'] = 1e-12
    opts['verb_disp'] = 0
    
    start_time = time.time()
    
    if verbose:
        print("🔍 Running CMA-ES optimization...")
    
    try:
        es = cma.CMAEvolutionStrategy(x0, 0.3, opts)
        es.optimize(objective_hybrid_cubic)
        
        cma_time = time.time() - start_time
        
        if verbose:
            print(f"   ✅ CMA-ES completed in {cma_time:.1f}s")
        
        # Get best solution
        best_params = es.result.xbest
        best_energy = E_negative_hybrid_cubic(best_params, mu0, G_geo)
        penalty = penalty_hybrid_cubic(best_params)
        
        if verbose:
            print(f"\n🏆 CMA-ES RESULTS:")
            print(f"   Energy E₋: {best_energy:.3e} J")
            print(f"   Penalty: {penalty:.3e}")
            print(f"   Evaluations: {es.result.evaluations}")
        
        return {
            'success': True,
            'ansatz_type': 'hybrid_cubic_cma',
            'params': best_params.tolist(),
            'energy_J': best_energy,
            'penalty': penalty,
            'mu': mu0,
            'G_geo': G_geo,
            'optimization_time': cma_time,
            'function_evaluations': es.result.evaluations
        }
        
    except Exception as e:
        if verbose:
            print(f"❌ CMA-ES optimization failed: {e}")
        return None
    
    if not de_result.success:
        print(f"❌ DE failed: {de_result.message}")
        return None
    
    de_time = time.time() - start_time
    if verbose:
        print(f"   DE completed in {de_time:.1f}s")
        print(f"   Best DE energy: {E_negative_hybrid_cubic(de_result.x):.3e} J")
    
    # Stage 2: Local refinement
    if verbose:
        print("🎯 Stage 2: L-BFGS-B refinement...")
    
    refine_result = minimize(
        objective_hybrid_cubic,
        x0=de_result.x,
        bounds=bounds,
        method='L-BFGS-B',
        options={
            'maxiter': 300,
            'ftol': 1e-9,
            'gtol': 1e-8
        }
    )
    
    if not refine_result.success:
        print(f"❌ L-BFGS-B failed: {refine_result.message}")
        return None
    
    total_time = time.time() - start_time
    final_energy = E_negative_hybrid_cubic(refine_result.x)
    
    if verbose:
        print(f"✅ Optimization completed in {total_time:.1f}s")
        print(f"   Final energy: {final_energy:.6e} J")
    
    return {
        'params': refine_result.x,
        'energy_J': final_energy,
        'mu': mu0,
        'G_geo': G_geo,
        'optimization_time': total_time,
        'ansatz_type': 'cubic_hybrid'
    }

# ── 9. Analysis and Visualization ─────────────────────────────────────────────
def analyze_hybrid_cubic_result(result):
    """
    Analyze cubic hybrid optimization result
    """
    if result is None:
        print("❌ No result to analyze")
        return
    
    params = result['params']
    energy = result['energy_J']
    
    print(f"\n📊 CUBIC HYBRID ANALYSIS")
    print("=" * 40)
    print(f"Final Energy: {energy:.6e} J")
    print(f"Optimization time: {result['optimization_time']:.1f}s")
    print(f"Parameters (μ={result['mu']:.1e}, G_geo={result['G_geo']:.1e}):")
    
    r0, r1 = params[0], params[1]
    b1, b2, b3 = params[2], params[3], params[4]
    
    print(f"  Region bounds: r0={r0:.4f}, r1={r1:.4f}")
    print(f"  Polynomial coeffs: b1={b1:.4f}, b2={b2:.4f}, b3={b3:.4f}")
    
    for i in range(M_gauss):
        Ai = params[5 + 3*i + 0]
        r0_i = params[5 + 3*i + 1]
        sig_i = params[5 + 3*i + 2]
        print(f"  Gaussian {i}: A={Ai:.4f}, r0={r0_i:.4f}, σ={sig_i:.4f}")
    
    # Check continuity at r1
    poly_at_r1 = 1.0 + b1 + b2 + b3
    gauss_at_r1 = sum(
        params[5 + 3*i + 0] * np.exp(-0.5 * ((r1 - params[5 + 3*i + 1]) / params[5 + 3*i + 2])**2)
        for i in range(M_gauss)
    )
    print(f"\nContinuity check at r1: poly={poly_at_r1:.4f}, gauss={gauss_at_r1:.4f}")
    print(f"Continuity error: {abs(poly_at_r1 - gauss_at_r1):.6f}")
    
    return params

def plot_hybrid_cubic_profile(result, save_fig=True):
    """
    Plot the optimized cubic hybrid profile
    """
    if result is None:
        return
    
    params = result['params']
    energy = result['energy_J']
    r0, r1 = params[0], params[1]
    
    # Generate high-resolution profile
    r_plot = np.linspace(0, R, 500)
    f_plot = f_hybrid_cubic(r_plot, params)
    fp_plot = f_hybrid_cubic_prime(r_plot, params)
    
    # Calculate effective density
    sinc_val = np.sinc(mu0 / np.pi) if mu0 > 0 else 1.0
    prefactor = -(v**2) / (8.0*np.pi) * beta_back * sinc_val / G_geo
    rho_plot = prefactor * (fp_plot**2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: f(r) profile with region boundaries
    axes[0,0].plot(r_plot, f_plot, 'b-', linewidth=2, label='f(r)')
    axes[0,0].axvline(x=r0, color='r', linestyle='--', alpha=0.7, label=f'r₀={r0:.3f}')
    axes[0,0].axvline(x=r1, color='g', linestyle='--', alpha=0.7, label=f'r₁={r1:.3f}')
    axes[0,0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0,0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
    axes[0,0].set_xlabel('r (m)')
    axes[0,0].set_ylabel('f(r)')
    axes[0,0].set_title('Cubic Hybrid Warp Profile')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # Plot 2: f'(r) derivative
    axes[0,1].plot(r_plot, fp_plot, 'r-', linewidth=2, label="f'(r)")
    axes[0,1].axvline(x=r0, color='r', linestyle='--', alpha=0.5)
    axes[0,1].axvline(x=r1, color='g', linestyle='--', alpha=0.5)
    axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0,1].set_xlabel('r (m)')
    axes[0,1].set_ylabel("f'(r)")
    axes[0,1].set_title('Derivative Profile')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # Plot 3: Effective density
    axes[1,0].plot(r_plot, rho_plot, 'g-', linewidth=2, label='ρ_eff(r)')
    axes[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1,0].set_xlabel('r (m)')
    axes[1,0].set_ylabel('ρ_eff (natural units)')
    axes[1,0].set_title('Effective Energy Density')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # Plot 4: Region breakdown
    r_core = r_plot[r_plot <= r0]
    r_trans = r_plot[(r_plot > r0) & (r_plot < r1)]
    r_tail = r_plot[r_plot >= r1]
    
    if len(r_core) > 0:
        axes[1,1].plot(r_core, np.ones_like(r_core), 'k-', linewidth=3, 
                      label='Core (f=1)', alpha=0.8)
    
    if len(r_trans) > 0:
        f_trans = f_hybrid_cubic(r_trans, params)
        axes[1,1].plot(r_trans, f_trans, 'b-', linewidth=3, 
                      label='Cubic polynomial', alpha=0.8)
    
    if len(r_tail) > 0:
        f_tail = f_hybrid_cubic(r_tail, params)
        axes[1,1].plot(r_tail, f_tail, 'g-', linewidth=3, 
                      label='Gaussian tail', alpha=0.8)
    
    axes[1,1].axvline(x=r0, color='r', linestyle='--', alpha=0.7)
    axes[1,1].axvline(x=r1, color='g', linestyle='--', alpha=0.7)
    axes[1,1].set_xlabel('r (m)')
    axes[1,1].set_ylabel('f(r)')
    axes[1,1].set_title('Ansatz Regions')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    plt.suptitle(f'Cubic Hybrid Optimization Result\nE₋ = {energy:.4e} J', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('hybrid_cubic_profile.png', dpi=300, bbox_inches='tight')
        print("📊 Profile saved as 'hybrid_cubic_profile.png'")
    
    plt.close()  # Close instead of show to prevent blocking

# ── 10. Main Execution ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🎯 CUBIC HYBRID ANSATZ OPTIMIZER")
    print("=" * 60)
    print(f"Target: Push E₋ below -1.95×10³¹ J")
    print(f"Method: Cubic polynomial transition + 2-Gaussian tail")
    print("=" * 60)
    
    # Run optimization
    result = optimize_hybrid_cubic()
    
    if result:
        # Analyze results
        analyze_hybrid_cubic_result(result)
        
        # Plot profile
        plot_hybrid_cubic_profile(result)
        
        # Save result
        result_copy = result.copy()
        result_copy['params'] = result_copy['params'].tolist()
        
        with open('hybrid_cubic_result.json', 'w') as f:
            json.dump(result_copy, f, indent=2)
        print("💾 Result saved as 'hybrid_cubic_result.json'")
        
        # Compare with previous results
        print(f"\n🏆 PERFORMANCE COMPARISON")
        print("=" * 40)
        baselines = {
            "Linear Hybrid": -1.86e31,
            "5-Gaussian": -1.90e31
        }
        
        current_energy = result['energy_J']
        for name, baseline in baselines.items():
            improvement = abs(current_energy / baseline)
            print(f"{name:15s}: {baseline:.3e} J → {improvement:.3f}× improvement")
        
        print(f"{'Cubic Hybrid':15s}: {current_energy:.3e} J ← NEW RESULT")
    
    else:
        print("❌ Optimization failed")
