#!/usr/bin/env python3
"""
CMA-ES 4-GAUSSIAN OPTIMIZER (STANDALONE)

Advanced CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer
for 4-Gaussian superposition ansatz. CMA-ES is particularly effective for 
multimodal optimization landscapes with correlated parameters.

Target: Achieve E- ≈ -1.8×10³¹ J with superior convergence
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import json
import time

# CMA-ES support
HAS_CMA = False
try:
    import cma
    HAS_CMA = True
    print("✅ CMA-ES available for enhanced global optimization")
except ImportError:
    print("❌ CMA-ES not available. Install with: pip install cma")
    exit(1)

# ── 1. PHYSICAL CONSTANTS ─────────────────────────────────────────────────────
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

# ── 2. ANSATZ CONFIGURATION ──────────────────────────────────────────────────
M_gauss = 4  # Number of Gaussian lumps (optimal balance)

# Precompute radial grid for vectorized integration
N_grid = 800
r_grid = np.linspace(0.0, R, N_grid)
dr = r_grid[1] - r_grid[0]
vol_weights = 4.0 * np.pi * r_grid**2

# ── 3. FOUR-GAUSSIAN ANSATZ FUNCTIONS ────────────────────────────────────────
def f_gaussian_4(r, params):
    """
    4-Gaussian superposition ansatz
    
    f(r) = ∑ᵢ Aᵢ exp(-½((r-r₀ᵢ)/σᵢ)²)
    
    params = [A₀, r₀₀, σ₀, A₁, r₀₁, σ₁, A₂, r₀₂, σ₂, A₃, r₀₃, σ₃]
    """
    r = np.atleast_1d(r)
    result = np.zeros_like(r)
    
    for i in range(M_gauss):
        Ai = params[3*i + 0]
        r0i = params[3*i + 1]
        sigi = params[3*i + 2]
        
        x = (r - r0i) / sigi
        result += Ai * np.exp(-0.5 * x**2)
    
    return result

def f_gaussian_4_prime(r, params):
    """
    Derivative of 4-Gaussian ansatz
    """
    r = np.atleast_1d(r)
    result = np.zeros_like(r)
    
    for i in range(M_gauss):
        Ai = params[3*i + 0]
        r0i = params[3*i + 1]
        sigi = params[3*i + 2]
        
        x = (r - r0i) / sigi
        gaussian_term = Ai * np.exp(-0.5 * x**2)
        derivative_term = -(r - r0i) / (sigi**2)
        result += gaussian_term * derivative_term
    
    return result

# ── 4. ENERGY CALCULATION ────────────────────────────────────────────────────
def E_negative_cma(params, mu0=None, G_geo=None):
    """
    Fast vectorized energy calculation for CMA-ES optimization
    
    E₋ = ∫ ρ_eff(r) * 4π r² dr * c⁴/(8πG)
    """
    if mu0 is None:
        mu0 = mu0_default
    if G_geo is None:
        G_geo = G_geo_default
    
    # Calculate f'(r) on the grid
    fp_vals = f_gaussian_4_prime(r_grid, params)
    
    # Polymer sinc function enhancement
    sinc_val = np.sinc(mu0 / np.pi) if mu0 > 0 else 1.0
    
    # Effective density prefactor
    prefactor = -(v**2) / (8.0 * np.pi) * beta_back * sinc_val / G_geo
    
    # Calculate effective density
    rho_vals = prefactor * (fp_vals**2)
    
    # Vectorized integration: ∫ ρ(r) * 4π r² dr
    integral = np.sum(rho_vals * vol_weights) * dr
    
    # Convert to Joules
    return integral * c4_8piG

# ── 5. ENHANCED PENALTY FUNCTIONS ────────────────────────────────────────────
def penalty_cma(params, mu0=None, G_geo=None, 
                lam_qi=1e50, lam_bound=1e4, lam_curv=1e3, 
                lam_mono=1e3, lam_amplitude=1e4, lam_ordering=1e5):
    """
    Enhanced penalty function designed for CMA-ES optimization
    """
    if mu0 is None:
        mu0 = mu0_default
    if G_geo is None:
        G_geo = G_geo_default
    
    penalty_total = 0.0
    
    # (a) Quantum Inequality constraint at r=0
    fp0 = f_gaussian_4_prime(np.array([0.0]), params)[0]
    sinc_val = np.sinc(mu0 / np.pi) if mu0 > 0 else 1.0
    rho0 = -(v**2) / (8.0*np.pi) * beta_back * sinc_val / G_geo * (fp0**2)
    qi_bound = -(hbar * sinc_val) / (12.0 * np.pi * tau**2)
    P_qi = lam_qi * max(0.0, -(rho0 - qi_bound))**2
    penalty_total += P_qi
    
    # (b) Boundary conditions: f(0) ≈ 1, f(R) ≈ 0
    f0 = f_gaussian_4(np.array([0.0]), params)[0]
    fR = f_gaussian_4(np.array([R]), params)[0]
    P_boundary = lam_bound * ((f0 - 1.0)**2 + (fR - 0.0)**2)
    penalty_total += P_boundary
    
    # (c) Amplitude constraints
    A_total = sum(params[0::3])  # Sum of all amplitudes
    P_amplitude = lam_amplitude * max(0.0, (A_total - 1.2))**2
    penalty_total += P_amplitude
    
    # Individual amplitude bounds (soft constraints)
    for i in range(M_gauss):
        Ai = params[3*i]
        if Ai < 0.0:
            penalty_total += lam_bound * (Ai**2)
        if Ai > 1.0:
            penalty_total += lam_bound * ((Ai - 1.0)**2)
    
    # (d) Position ordering: encourage r₀₀ < r₀₁ < r₀₂ < r₀₃
    for i in range(M_gauss - 1):
        r0_curr = params[3*i + 1]
        r0_next = params[3*(i+1) + 1]
        if r0_curr > r0_next:
            penalty_total += lam_ordering * (r0_curr - r0_next)**2
    
    # (e) Curvature penalty: smooth profiles
    if lam_curv > 0:
        r_test = np.linspace(0.1*R, 0.9*R, 50)
        f_test = f_gaussian_4(r_test, params)
        f_second_deriv = np.gradient(np.gradient(f_test))
        P_curvature = lam_curv * np.sum(f_second_deriv**2) * R / len(r_test)
        penalty_total += P_curvature
    
    # (f) Monotonicity: prefer decreasing profiles
    if lam_mono > 0:
        fp_vals = f_gaussian_4_prime(r_grid, params)
        positive_deriv = np.maximum(0.0, fp_vals)
        P_monotonic = lam_mono * np.sum(positive_deriv**2) * dr
        penalty_total += P_monotonic
    
    return penalty_total

def objective_cma(params, mu0=None, G_geo=None):
    """
    Complete objective function for CMA-ES optimization
    """
    try:
        energy = E_negative_cma(params, mu0, G_geo)
        penalty = penalty_cma(params, mu0, G_geo)
        return energy + penalty
    except Exception as e:
        # Return large penalty for failed evaluations
        return 1e15

# ── 6. OPTIMIZATION BOUNDS AND INITIALIZATION ────────────────────────────────
def get_cma_bounds():
    """
    Generate bounds for CMA-ES 4-Gaussian optimization
    """
    bounds = []
    for i in range(M_gauss):
        bounds += [
            (0.0, 1.0),        # Amplitude A_i ∈ [0,1]
            (0.0, R),          # Position r0_i ∈ [0,R]
            (0.01, 0.4*R)      # Width σ_i ∈ [0.01, 0.4R]
        ]
    return bounds

def get_cma_initial_guess(strategy='physics_informed'):
    """
    Generate initial guess for CMA-ES optimization
    """
    if strategy == 'physics_informed':
        # Educated guess based on physics intuition
        params = []
        for i in range(M_gauss):
            A_i = 0.8 * (0.85)**i            # Decreasing amplitudes
            r0_i = (i + 0.5) * R / M_gauss   # Evenly spaced positions
            sig_i = R / (2 * M_gauss + 1)    # Moderate widths
            params.extend([A_i, r0_i, sig_i])
        return np.array(params)
    
    elif strategy == 'random':
        # Random initialization within bounds
        bounds = get_cma_bounds()
        params = []
        for lb, ub in bounds:
            params.append(np.random.uniform(lb, ub))
        return np.array(params)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# ── 7. CMA-ES OPTIMIZATION ───────────────────────────────────────────────────
def optimize_with_cma_advanced(bounds, objective_func, mu0=None, G_geo=None,
                              sigma0=0.2, maxiter=500, popsize=None, verbose=True):
    """
    Advanced CMA-ES optimization with boundary constraints and adaptive settings
    """
    if not HAS_CMA:
        raise ImportError("CMA-ES not available")
    
    # Convert bounds to CMA format
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Initial guess
    x0 = get_cma_initial_guess('physics_informed')
    
    # Ensure x0 is within bounds
    x0 = np.clip(x0, lb, ub)
    
    # CMA-ES options
    opts = cma.CMAOptions()
    opts['bounds'] = [lb.tolist(), ub.tolist()]
    opts['maxiter'] = maxiter
    if popsize is not None:
        opts['popsize'] = popsize
    opts['tolfun'] = 1e-12
    opts['verb_disp'] = 0  # Suppress CMA output
    
    start_time = time.time()
    
    if verbose:
        print(f"   Starting CMA-ES with σ₀={sigma0}, maxiter={maxiter}")
    
    try:
        # Define objective with fixed parameters
        def obj_fixed(params):
            return objective_func(params, mu0, G_geo)
        
        # Initialize CMA-ES
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        
        # Track best solution
        best_energy = float('inf')
        best_params = None
        
        # Optimization loop
        iteration = 0
        while not es.stop():
            solutions = es.ask()
            fitnesses = [obj_fixed(x) for x in solutions]
            es.tell(solutions, fitnesses)
            
            # Track best solution
            current_best_idx = np.argmin(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]
            
            if current_best_fitness < best_energy:
                best_energy = current_best_fitness
                best_params = solutions[current_best_idx].copy()
            
            iteration += 1
            
            if verbose and iteration % 50 == 0:
                energy_only = E_negative_cma(best_params, mu0, G_geo)
                print(f"   Iteration {iteration}: Best energy = {energy_only:.3e} J")
        
        optimization_time = time.time() - start_time
        
        if verbose:
            print(f"✅ CMA-ES completed in {optimization_time:.1f}s")
            print(f"   Total iterations: {iteration}")
            print(f"   Function evaluations: {es.result.evaluations}")
        
        return {
            'x': es.result.xbest,
            'fun': es.result.fbest,
            'success': True,
            'nfev': es.result.evaluations,
            'nit': iteration,
            'optimization_time': optimization_time
        }
        
    except Exception as e:
        print(f"❌ CMA-ES optimization failed: {e}")
        return {'success': False, 'message': str(e)}

# ── 8. MAIN OPTIMIZATION FUNCTION ────────────────────────────────────────────
def run_cma_optimization(mu0=None, G_geo=None, use_refinement=True, verbose=True):
    """
    Run complete CMA-ES optimization with optional L-BFGS-B refinement
    """
    if mu0 is None:
        mu0 = mu0_default
    if G_geo is None:
        G_geo = G_geo_default
    
    if verbose:
        print(f"🚀 CMA-ES 4-GAUSSIAN OPTIMIZATION")
        print(f"   Parameters: μ₀={mu0:.1e}, G_geo={G_geo:.1e}")
        print("=" * 50)
    
    bounds = get_cma_bounds()
    start_time = time.time()
    
    # Stage 1: CMA-ES global optimization
    if verbose:
        print("🔍 Stage 1: CMA-ES global optimization...")
    
    cma_result = optimize_with_cma_advanced(
        bounds, objective_cma, mu0, G_geo,
        sigma0=0.3,      # Moderate exploration
        maxiter=300,     # Reasonable for efficiency
        popsize=16,      # Good balance
        verbose=verbose
    )
    
    if not cma_result['success']:
        if verbose:
            print("❌ CMA-ES optimization failed")
        return None
    
    # Stage 2: Optional L-BFGS-B refinement
    best_params = cma_result['x']
    if use_refinement:
        if verbose:
            print("🔧 Stage 2: L-BFGS-B refinement...")
        
        refine_start = time.time()
        refine_result = minimize(
            lambda x: objective_cma(x, mu0, G_geo),
            x0=best_params,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        refine_time = time.time() - refine_start
        
        if refine_result.success:
            best_params = refine_result.x
            if verbose:
                print(f"   ✅ Refinement completed in {refine_time:.1f}s")
        else:
            if verbose:
                print(f"   ⚠️ Refinement failed: {refine_result.message}")
    
    total_time = time.time() - start_time
    
    # Calculate final results
    best_energy = E_negative_cma(best_params, mu0, G_geo)
    penalty = penalty_cma(best_params, mu0, G_geo)
    objective_val = best_energy + penalty
    
    if verbose:
        print(f"\n🏆 CMA-ES OPTIMIZATION RESULTS:")
        print(f"   Energy E₋: {best_energy:.3e} J")
        print(f"   Penalty: {penalty:.3e}")
        print(f"   Total objective: {objective_val:.3e}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Function evaluations: {cma_result['nfev']}")
        
        # Parameter breakdown
        print(f"\n📊 OPTIMIZED PARAMETERS:")
        for i in range(M_gauss):
            Ai = best_params[3*i + 0]
            r0_i = best_params[3*i + 1]
            sig_i = best_params[3*i + 2]
            print(f"   Gaussian {i}: A={Ai:.3f}, r₀={r0_i:.3f}, σ={sig_i:.3f}")
    
    return {
        'success': True,
        'ansatz_type': 'cma_4gaussian',
        'params': best_params.tolist(),
        'energy_J': best_energy,
        'penalty': penalty,
        'objective': objective_val,
        'mu': mu0,
        'G_geo': G_geo,
        'optimization_time': total_time,
        'function_evaluations': cma_result['nfev'],
        'gaussian_params': [
            {'A': best_params[3*i], 'r0': best_params[3*i+1], 'sigma': best_params[3*i+2]}
            for i in range(M_gauss)
        ]
    }

# ── 9. ANALYSIS AND VISUALIZATION ────────────────────────────────────────────
def plot_cma_profile(result, save_fig=True):
    """
    Plot the optimized CMA-ES 4-Gaussian profile
    """
    if not result or not result.get('success', False):
        print("❌ No valid result to plot")
        return
    
    params = result['params']
    r_plot = np.linspace(0, R, 1000)
    f_plot = f_gaussian_4(r_plot, params)
    fp_plot = f_gaussian_4_prime(r_plot, params)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Profile plot
    ax1.plot(r_plot, f_plot, 'b-', linewidth=2, label='f(r)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    
    ax1.set_xlabel('Radius r (m)')
    ax1.set_ylabel('f(r)')
    ax1.set_title(f'CMA-ES 4-Gaussian Profile (E₋ = {result["energy_J"]:.3e} J)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Derivative plot
    ax2.plot(r_plot, fp_plot, 'r-', linewidth=2, label="f'(r)")
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    ax2.set_xlabel('Radius r (m)')
    ax2.set_ylabel("f'(r)")
    ax2.set_title('Profile Derivative')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig:
        filename = 'cma_4gaussian_profile.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Profile saved as {filename}")
    
    plt.close()  # Close instead of show to prevent blocking

def save_cma_results(result, filename='cma_4gaussian_results.json'):
    """
    Save CMA-ES optimization results to JSON file
    """
    if result:
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Results saved to {filename}")

# ── 10. MAIN EXECUTION ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 CMA-ES 4-GAUSSIAN SUPERPOSITION OPTIMIZER")
    print("=" * 60)
    print("Target: Push E₋ below -1.8×10³¹ J using CMA-ES global optimization")
    print("=" * 60)
    
    # Run optimization
    result = run_cma_optimization(verbose=True)
    
    if result and result['success']:
        # Save results
        save_cma_results(result)
        
        # Plot profile
        plot_cma_profile(result)
        
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"   CMA-ES 4-Gaussian Energy: {result['energy_J']:.3e} J")
        print(f"   Optimization time: {result['optimization_time']:.1f}s")
        print(f"   Function evaluations: {result['function_evaluations']}")
        
        # Compare to baseline
        baseline_energy = -1.2e31  # Approximate current best
        improvement = abs(result['energy_J']) / abs(baseline_energy)
        print(f"   Improvement over baseline: {improvement:.2f}×")
        
    else:
        print("❌ Optimization failed!")
