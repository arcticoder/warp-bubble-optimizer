#!/usr/bin/env python3
"""
CMA-ES 4-GAUSSIAN OPTIMIZER

Advanced CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer
for 4-Gaussian superposition ansatz. CMA-ES is particularly effective for 
multimodal optimization landscapes with correlated parameters.

Features:
- CMA-ES global optimization with boundary constraints
- Enhanced physics-informed penalty functions
- Adaptive parameter bounds refinement
- Vectorized energy calculation (~100× faster)
- Joint parameter scanning capability

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
    print("✅ CMA-ES available for advanced optimization")
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
    Four-Gaussian superposition ansatz
    
    f(r) = sum_{i=0}^{3} A_i * exp(-0.5 * ((r - r0_i)/σ_i)²)
    
    Args:
        r: Radial coordinate (scalar or array)
        params: [A0,r0_0,σ0, A1,r0_1,σ1, A2,r0_2,σ2, A3,r0_3,σ3]
    """
    r = np.atleast_1d(r)
    total = np.zeros_like(r)
    
    for i in range(M_gauss):
        Ai = params[3*i + 0]      # Amplitude
        r0_i = params[3*i + 1]    # Center position  
        sig_i = params[3*i + 2]   # Width
        
        x = (r - r0_i) / sig_i
        total += Ai * np.exp(-0.5 * x**2)
    
    return np.clip(total, 0.0, 1.0)

def f_gaussian_4_prime(r, params):
    """
    Derivative of four-Gaussian ansatz
    
    f'(r) = sum_{i=0}^{3} A_i * exp(-0.5 * x_i²) * (-(r - r0_i)/σ_i²)
    """
    r = np.atleast_1d(r)
    deriv = np.zeros_like(r)
    
    for i in range(M_gauss):
        Ai = params[3*i + 0]
        r0_i = params[3*i + 1]  
        sig_i = params[3*i + 2]
        
        x = (r - r0_i) / sig_i
        gaussian_term = Ai * np.exp(-0.5 * x**2)
        deriv += gaussian_term * (-(r - r0_i) / (sig_i**2))
    
    return deriv

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
    qi_bound = -(hbar * np.sinc(mu0/np.pi)) / (12.0 * np.pi * tau**2)
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
            penalty_total += lam_amplitude * (Ai**2)
        elif Ai > 1.0:
            penalty_total += lam_amplitude * ((Ai - 1.0)**2)
    
    # (d) Curvature penalty for smoothness
    f_vals = f_gaussian_4(r_grid, params)
    fpp = np.zeros_like(f_vals)
    fpp[1:-1] = (f_vals[2:] - 2*f_vals[1:-1] + f_vals[:-2]) / (dr**2)
    fpp[0] = fpp[1]
    fpp[-1] = fpp[-2]
    P_curvature = lam_curv * np.sum((fpp**2) * (r_grid**2)) * dr
    penalty_total += P_curvature
    
    # (e) Monotonicity penalty: encourage f'(r) ≤ 0
    fp_vals = f_gaussian_4_prime(r_grid, params)
    P_monotonic = lam_mono * np.sum(np.maximum(0.0, fp_vals)**2) * dr
    penalty_total += P_monotonic
    
    # (f) Gaussian ordering penalty (encourage ordered positions)
    positions = [params[3*i + 1] for i in range(M_gauss)]
    for i in range(M_gauss - 1):
        if positions[i] > positions[i+1]:
            P_order = lam_ordering * (positions[i] - positions[i+1])**2
            penalty_total += P_order
    
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
    
    Args:
        strategy: 'physics_informed', 'random', or 'distributed'
    """
    if strategy == 'physics_informed':
        # Based on physical intuition and previous results
        params = []
        for i in range(M_gauss):
            A_i = 0.8 * (0.85)**i        # Decreasing amplitudes
            r0_i = (i + 0.3) * R / (M_gauss + 0.5)  # Spread positions
            sig_i = 0.05 + 0.04 * i      # Increasing widths
            params.extend([A_i, r0_i, sig_i])
        return np.array(params)
    
    elif strategy == 'distributed':
        # Evenly distributed Gaussians
        params = []
        for i in range(M_gauss):
            A_i = 0.7 - 0.1 * i
            r0_i = i * R / (M_gauss - 1) if M_gauss > 1 else 0.5 * R
            sig_i = 0.08 + 0.02 * i
            params.extend([A_i, r0_i, sig_i])
        return np.array(params)
    
    else:  # random
        bounds = get_cma_bounds()
        params = []
        for low, high in bounds:
            params.append(np.random.uniform(low, high))
        return np.array(params)

# ── 7. CMA-ES OPTIMIZATION ───────────────────────────────────────────────────
def optimize_with_cma_advanced(bounds, objective_func, mu0=None, G_geo=None,
                              sigma0=0.2, maxiter=500, popsize=None, verbose=True):
    """
    Advanced CMA-ES optimization with boundary constraints and adaptive settings
    """
    if not HAS_CMA:
        raise ImportError("CMA-ES not available. Install with: pip install cma")
    
    # Generate initial guess
    x0 = get_cma_initial_guess('physics_informed')
    
    # Extract bounds
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    # Adaptive population size
    if popsize is None:
        popsize = 4 + int(3 * np.log(len(x0)))  # CMA-ES recommended formula
    
    # CMA-ES options
    opts = {
        'bounds': [lower_bounds, upper_bounds],
        'maxiter': maxiter,
        'popsize': popsize,
        'tolfun': 1e-9,
        'tolx': 1e-9,
        'verbose': 0 if not verbose else -1,
        'seed': 42
    }
    
    if verbose:
        print(f"🔍 CMA-ES Configuration:")
        print(f"   Population size: {popsize}")
        print(f"   Initial sigma: {sigma0}")
        print(f"   Max iterations: {maxiter}")
        print(f"   Problem dimension: {len(x0)}")
    
    try:
        # Define objective with fixed parameters
        def obj_fixed(params):
            return objective_func(params, mu0, G_geo)
        
        # Initialize CMA-ES
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        
        start_time = time.time()
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
        return {'success': False}

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
    
    # CMA-ES global optimization
    cma_result = optimize_with_cma_advanced(
        bounds, objective_cma, mu0, G_geo, verbose=verbose
    )
    
    if not cma_result.get('success', False):
        if verbose:
            print("❌ CMA-ES optimization failed")
        return {'success': False, 'message': 'CMA-ES optimization failed'}
    
    best_params = cma_result['x']
    best_energy = E_negative_cma(best_params, mu0, G_geo)
    
    # Optional L-BFGS-B refinement
    if use_refinement:
        if verbose:
            print("🔧 Running L-BFGS-B refinement...")
        
        def objective_fixed(params):
            return objective_cma(params, mu0, G_geo)
        
        refine_result = minimize(
            objective_fixed,
            x0=best_params,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-9}
        )
        
        if refine_result.success:
            best_params = refine_result.x
            best_energy = E_negative_cma(best_params, mu0, G_geo)
            if verbose:
                print(f"✅ Refinement improved energy to: {best_energy:.3e} J")
        else:
            if verbose:
                print("⚠️  Refinement failed, using CMA-ES result")
    
    total_time = time.time() - start_time
    penalty = penalty_cma(best_params, mu0, G_geo)
    
    if verbose:
        print(f"\n🏆 CMA-ES OPTIMIZATION RESULTS:")
        print(f"   Energy E₋: {best_energy:.3e} J")
        print(f"   Penalty: {penalty:.3e}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Function evaluations: {cma_result.get('nfev', 0)}")
        
        # Parameter breakdown
        print(f"\n📊 OPTIMIZED PARAMETERS:")
        for i in range(M_gauss):
            Ai = best_params[3*i + 0]
            r0_i = best_params[3*i + 1]
            sig_i = best_params[3*i + 2]
            print(f"   Gaussian {i}: A={Ai:.4f}, r₀={r0_i:.4f}, σ={sig_i:.4f}")
        
        # Check boundary conditions
        f0 = f_gaussian_4(np.array([0.0]), best_params)[0]
        fR = f_gaussian_4(np.array([R]), best_params)[0]
        print(f"\nBoundary check: f(0)={f0:.4f}, f(R)={fR:.6f}")
    
    return {
        'success': True,
        'ansatz_type': 'cma_4gaussian',
        'params': best_params.tolist(),
        'energy_J': best_energy,
        'penalty': penalty,
        'mu0': mu0,
        'G_geo': G_geo,
        'optimization_time': total_time,
        'function_evaluations': cma_result.get('nfev', 0),
        'cma_result': cma_result
    }

# ── 9. ANALYSIS AND VISUALIZATION ────────────────────────────────────────────
def plot_cma_profile(result, save_fig=True):
    """
    Plot the optimized CMA-ES 4-Gaussian profile
    """
    if not result.get('success', False):
        print("❌ No successful result to plot")
        return
    
    params = result['params']
    energy = result['energy_J']
    
    # High-resolution profile
    r_plot = np.linspace(0, R, 1000)
    f_plot = f_gaussian_4(r_plot, params)
    fp_plot = f_gaussian_4_prime(r_plot, params)
    
    # Calculate effective density
    mu0 = result['mu0']
    G_geo = result['G_geo']
    sinc_val = np.sinc(mu0 / np.pi) if mu0 > 0 else 1.0
    prefactor = -(v**2) / (8.0*np.pi) * beta_back * sinc_val / G_geo
    rho_plot = prefactor * (fp_plot**2)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: f(r) profile
    axes[0,0].plot(r_plot, f_plot, 'b-', linewidth=2.5, label='f(r) - CMA-ES 4-Gaussian')
    axes[0,0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0,0].axhline(y=1, color='k', linestyle='-', alpha=0.3)
    axes[0,0].set_xlabel('r (m)')
    axes[0,0].set_ylabel('f(r)')
    axes[0,0].set_title('CMA-ES Optimized 4-Gaussian Profile')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # Plot 2: f'(r) derivative
    axes[0,1].plot(r_plot, fp_plot, 'g-', linewidth=2, label="f'(r)")
    axes[0,1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0,1].set_xlabel('r (m)')
    axes[0,1].set_ylabel("f'(r)")
    axes[0,1].set_title('Derivative Profile')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # Plot 3: Effective energy density
    axes[1,0].plot(r_plot, rho_plot, 'purple', linewidth=2, label='ρ_eff(r)')
    axes[1,0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1,0].set_xlabel('r (m)')
    axes[1,0].set_ylabel('ρ_eff (natural units)')
    axes[1,0].set_title('Effective Energy Density')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # Plot 4: Parameter summary
    axes[1,1].text(0.05, 0.95, f"🏆 CMA-ES RESULTS", transform=axes[1,1].transAxes, 
                   fontsize=14, fontweight='bold', va='top')
    axes[1,1].text(0.05, 0.85, f"Energy: {energy:.3e} J", transform=axes[1,1].transAxes, 
                   fontsize=12, va='top')
    axes[1,1].text(0.05, 0.75, f"μ₀: {mu0:.1e}", transform=axes[1,1].transAxes, 
                   fontsize=12, va='top')
    axes[1,1].text(0.05, 0.65, f"G_geo: {G_geo:.1e}", transform=axes[1,1].transAxes, 
                   fontsize=12, va='top')
    axes[1,1].text(0.05, 0.55, f"Optimizer: CMA-ES", transform=axes[1,1].transAxes, 
                   fontsize=12, va='top')
    axes[1,1].text(0.05, 0.45, f"Evaluations: {result.get('function_evaluations', 0)}", 
                   transform=axes[1,1].transAxes, fontsize=12, va='top')
    
    # Individual Gaussian parameters
    y_pos = 0.35
    for i in range(M_gauss):
        Ai = params[3*i + 0]
        r0_i = params[3*i + 1]
        sig_i = params[3*i + 2]
        axes[1,1].text(0.05, y_pos, f"G{i}: A={Ai:.3f}, r₀={r0_i:.3f}, σ={sig_i:.3f}", 
                       transform=axes[1,1].transAxes, fontsize=10, va='top')
        y_pos -= 0.08
    
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')
    
    plt.tight_layout()
    
    if save_fig:
        filename = f'cma_4gaussian_profile_E{abs(energy):.2e}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Profile saved as: {filename}")
    
    plt.close()  # Close instead of show to prevent blocking
    return fig

# ── 10. PARAMETER SCANNING ───────────────────────────────────────────────────
def parameter_scan_cma(mu_values=None, G_geo_values=None, verbose=True):
    """
    Comprehensive parameter scan using CMA-ES optimization
    """
    if mu_values is None:
        mu_values = np.logspace(-7, -5, 5)  # 1e-7 to 1e-5
    if G_geo_values is None:
        G_geo_values = np.logspace(-6, -4, 5)  # 1e-6 to 1e-4
    
    if verbose:
        print(f"🔍 CMA-ES PARAMETER SCAN")
        print(f"   μ₀ values: {len(mu_values)} points")
        print(f"   G_geo values: {len(G_geo_values)} points")
        print(f"   Total combinations: {len(mu_values) * len(G_geo_values)}")
        print("=" * 50)
    
    results = []
    best_overall = {'energy_J': 0.0}
    
    for i, mu0 in enumerate(mu_values):
        for j, G_geo in enumerate(G_geo_values):
            if verbose:
                print(f"\n📊 Combination {i*len(G_geo_values)+j+1}/{len(mu_values)*len(G_geo_values)}")
                print(f"   μ₀={mu0:.1e}, G_geo={G_geo:.1e}")
            
            # Run CMA-ES optimization
            result = run_cma_optimization(mu0=mu0, G_geo=G_geo, verbose=False)
            
            if result.get('success', False):
                results.append(result)
                
                if result['energy_J'] < best_overall['energy_J']:
                    best_overall = result.copy()
                    if verbose:
                        print(f"   🎯 NEW BEST: E₋ = {result['energy_J']:.3e} J")
                else:
                    if verbose:
                        print(f"   E₋ = {result['energy_J']:.3e} J")
            else:
                if verbose:
                    print(f"   ❌ Optimization failed")
    
    if verbose:
        print(f"\n🏆 PARAMETER SCAN SUMMARY:")
        print(f"   Successful runs: {len(results)}")
        print(f"   Best energy: {best_overall['energy_J']:.3e} J")
        print(f"   Best parameters: μ₀={best_overall['mu0']:.1e}, G_geo={best_overall['G_geo']:.1e}")
    
    return results, best_overall

# ── 11. MAIN EXECUTION ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 CMA-ES 4-GAUSSIAN OPTIMIZER")
    print("="*60)
    print("Target: Achieve E₋ ≈ -1.8×10³¹ J with CMA-ES global optimization")
    print("="*60)
    
    # Run single optimization
    result = run_cma_optimization(verbose=True)
    
    if result.get('success', False):
        # Plot the result
        plot_cma_profile(result)
        
        # Save results
        with open('cma_4gaussian_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("\n💾 Results saved to: cma_4gaussian_results.json")
        
        print(f"\n🎯 SUMMARY:")
        print(f"   CMA-ES 4-Gaussian Energy: {result['energy_J']:.3e} J")
        print(f"   Optimization time: {result['optimization_time']:.1f}s")
        print(f"   Success: ✅")
    else:
        print("\n❌ Optimization failed")
