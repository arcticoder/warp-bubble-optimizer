#!/usr/bin/env python3
"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Optimizer

Alternative global optimization approach for warp bubble ansätze using CMA-ES,
which is often more effective than Differential Evolution for high-dimensional
continuous optimization problems.

CMA-ES adapts the covariance matrix of the search distribution based on the
search history, providing excellent convergence properties for multimodal
optimization landscapes.

Target: Achieve E- ≈ -1.95×10³¹ J using CMA-ES global search
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import json
import time

# Optional CMA-ES import with graceful fallback
HAS_CMA = False
try:
    import cma
    HAS_CMA = True
    print("✅ CMA-ES available")
except ImportError:
    print("⚠️  CMA-ES not available. Install with: pip install cma")
    print("   Falling back to differential evolution")

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

# ── 2. Ansatz Configuration ───────────────────────────────────────────────────
M_gauss = 4  # Use 4-Gaussian ansatz for CMA-ES testing

# ── 3. Precompute Radial Grid & Weights ───────────────────────────────────────
N_points = 800
r_grid = np.linspace(0.0, R, N_points)
dr = r_grid[1] - r_grid[0]
vol_weights = 4.0 * np.pi * r_grid**2

# ── 4. Gaussian Ansatz Functions ──────────────────────────────────────────────
def f_gauss_cma(r, params):
    """
    4-Gaussian superposition ansatz for CMA-ES optimization
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

def f_gauss_cma_prime(r, params):
    """
    Derivative of 4-Gaussian ansatz
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

# ── 5. Energy and Penalty Functions ───────────────────────────────────────────
def E_negative_cma(params, mu_val=None, G_geo_val=None):
    """
    Calculate total negative energy
    """
    mu_use = mu_val if mu_val is not None else mu0
    G_geo_use = G_geo_val if G_geo_val is not None else G_geo
    
    fp_vals = f_gauss_cma_prime(r_grid, params)
    sinc_val = np.sinc(mu_use / np.pi) if mu_use > 0 else 1.0
    prefactor = -(v**2) / (8.0 * np.pi) * beta_back * sinc_val / G_geo_use
    rho_vals = prefactor * (fp_vals**2)
    integral = np.sum(rho_vals * vol_weights) * dr
    
    return integral * c4_8piG

def penalty_cma(params, lam_qi=1e50, lam_bound=1e4, lam_curv=1e3):
    """
    Enhanced penalty function for CMA-ES optimization
    """
    penalty_total = 0.0
    
    # Quantum Inequality constraint
    fp0 = f_gauss_cma_prime(np.array([0.0]), params)[0]
    sinc_val = np.sinc(mu0 / np.pi) if mu0 > 0 else 1.0
    rho0 = -(v**2) / (8.0*np.pi) * beta_back * sinc_val / G_geo * (fp0**2)
    qi_bound = -(1.0545718e-34 * np.sinc(mu0/np.pi)) / (12.0 * np.pi * tau**2)
    P_qi = lam_qi * max(0.0, -(rho0 - qi_bound))**2
    penalty_total += P_qi
    
    # Boundary conditions
    f0 = f_gauss_cma(np.array([0.0]), params)[0]
    fR = f_gauss_cma(np.array([R]), params)[0]
    P_boundary = lam_bound * ((f0 - 1.0)**2 + (fR - 0.0)**2)
    penalty_total += P_boundary
    
    # Amplitude constraint
    A_sum = sum(params[0::3])
    P_amplitude = lam_bound * max(0.0, (A_sum - 1.0))**2
    penalty_total += P_amplitude
    
    # Curvature penalty
    f_vals = f_gauss_cma(r_grid, params)
    fpp = np.zeros_like(f_vals)
    fpp[1:-1] = (f_vals[2:] - 2*f_vals[1:-1] + f_vals[:-2]) / (dr**2)
    fpp[0] = fpp[1]
    fpp[-1] = fpp[-2]
    P_curvature = lam_curv * np.sum((fpp**2) * (r_grid**2)) * dr
    penalty_total += P_curvature
    
    return penalty_total

def objective_cma(params):
    """
    Combined objective function for CMA-ES
    """
    return E_negative_cma(params) + penalty_cma(params)

# ── 6. Bounds and Initialization ──────────────────────────────────────────────
def get_bounds_cma():
    """
    Generate parameter bounds for CMA-ES optimization
    """
    bounds = []
    for i in range(M_gauss):
        bounds += [
            (0.0, 1.0),        # Amplitude
            (0.0, R),          # Position
            (0.01, 0.5*R)      # Width
        ]
    return bounds

def get_initial_guess_cma():
    """
    Generate initial guess for CMA-ES
    """
    params = []
    for i in range(M_gauss):
        A_i = 0.7 * (0.8)**i
        r0_i = (i + 0.5) * R / M_gauss
        sig_i = 0.08 + 0.04 * i
        params.extend([A_i, r0_i, sig_i])
    return np.array(params)

# ── 7. CMA-ES Optimization ────────────────────────────────────────────────────
def optimize_cma_es(mu_val=None, G_geo_val=None, verbose=True):
    """
    Run CMA-ES optimization with L-BFGS-B refinement
    """
    global mu0, G_geo
    if mu_val is not None:
        mu0 = mu_val
    if G_geo_val is not None:
        G_geo = G_geo_val
    
    if not HAS_CMA:
        print("❌ CMA-ES not available. Please install with: pip install cma")
        return None
    
    if verbose:
        print(f"🚀 CMA-ES OPTIMIZATION")
        print(f"   μ = {mu0:.1e}, G_geo = {G_geo:.1e}")
        print("=" * 50)
    
    bounds = get_bounds_cma()
    x0 = get_initial_guess_cma()
    
    start_time = time.time()
    
    # Stage 1: CMA-ES global search
    if verbose:
        print("🔍 Stage 1: CMA-ES global search...")
    
    try:
        # Set up CMA-ES with bounds
        sigma0 = 0.3  # Initial step size (30% of parameter range)
        
        # CMA-ES options
        opts = {
            'bounds': [[b[0] for b in bounds], [b[1] for b in bounds]],
            'popsize': 20,           # Population size
            'maxiter': 150,          # Maximum iterations
            'tolfun': 1e-10,         # Function tolerance
            'tolx': 1e-8,            # Parameter tolerance
            'verb_disp': 10 if verbose else 0,  # Display frequency
            'verb_log': 0            # Disable logging
        }
        
        # Initialize CMA-ES
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
        
        # Run optimization
        cma_result = es.optimize(objective_cma)
        
        # Extract best solution
        x_cma = np.array(es.result.xbest)
        cma_energy = E_negative_cma(x_cma)
        
        cma_time = time.time() - start_time
        
        if verbose:
            print(f"   CMA-ES completed in {cma_time:.1f}s")
            print(f"   Best CMA-ES energy: {cma_energy:.3e} J")
            print(f"   Function evaluations: {es.result.evaluations}")
        
    except Exception as e:
        print(f"❌ CMA-ES failed: {e}")
        return None
    
    # Stage 2: L-BFGS-B refinement
    if verbose:
        print("🎯 Stage 2: L-BFGS-B refinement...")
    
    try:
        refine_result = minimize(
            objective_cma,
            x0=x_cma,
            bounds=bounds,
            method='L-BFGS-B',
            options={
                'maxiter': 300,
                'ftol': 1e-9,
                'gtol': 1e-8
            }
        )
        
        if not refine_result.success:
            print(f"⚠️  L-BFGS-B warning: {refine_result.message}")
            # Use CMA-ES result if refinement fails
            final_params = x_cma
            final_energy = cma_energy
        else:
            final_params = refine_result.x
            final_energy = E_negative_cma(final_params)
        
    except Exception as e:
        print(f"⚠️  L-BFGS-B failed: {e}")
        # Fall back to CMA-ES result
        final_params = x_cma
        final_energy = cma_energy
        refine_result = None
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"✅ Optimization completed in {total_time:.1f}s")
        print(f"   Final energy: {final_energy:.6e} J")
    
    return {
        'params': final_params,
        'energy_J': final_energy,
        'mu': mu0,
        'G_geo': G_geo,
        'optimization_time': total_time,
        'cma_result': es.result,
        'refine_result': refine_result,
        'method': 'CMA-ES + L-BFGS-B'
    }

# ── 8. Differential Evolution Fallback ────────────────────────────────────────
def optimize_de_fallback(mu_val=None, G_geo_val=None, verbose=True):
    """
    Differential Evolution fallback when CMA-ES is not available
    """
    from scipy.optimize import differential_evolution
    
    global mu0, G_geo
    if mu_val is not None:
        mu0 = mu_val
    if G_geo_val is not None:
        G_geo = G_geo_val
    
    if verbose:
        print(f"🚀 DIFFERENTIAL EVOLUTION OPTIMIZATION (CMA-ES fallback)")
        print(f"   μ = {mu0:.1e}, G_geo = {G_geo:.1e}")
        print("=" * 50)
    
    bounds = get_bounds_cma()
    start_time = time.time()
    
    # Stage 1: Differential Evolution
    de_result = differential_evolution(
        objective_cma, bounds,
        strategy='best1bin',
        maxiter=300,
        popsize=15,
        tol=1e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        workers=-1,
        seed=42
    )
    
    if not de_result.success:
        print(f"❌ DE failed: {de_result.message}")
        return None
    
    # Stage 2: L-BFGS-B refinement
    refine_result = minimize(
        objective_cma,
        x0=de_result.x,
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 300, 'ftol': 1e-9}
    )
    
    total_time = time.time() - start_time
    final_energy = E_negative_cma(refine_result.x)
    
    if verbose:
        print(f"✅ Optimization completed in {total_time:.1f}s")
        print(f"   Final energy: {final_energy:.6e} J")
    
    return {
        'params': refine_result.x,
        'energy_J': final_energy,
        'mu': mu0,
        'G_geo': G_geo,
        'optimization_time': total_time,
        'de_result': de_result,
        'refine_result': refine_result,
        'method': 'DE + L-BFGS-B'
    }

# ── 9. Unified Optimization Interface ─────────────────────────────────────────
def optimize_with_cma(mu_val=None, G_geo_val=None, verbose=True):
    """
    Unified optimization interface: CMA-ES if available, DE otherwise
    """
    if HAS_CMA:
        return optimize_cma_es(mu_val, G_geo_val, verbose)
    else:
        return optimize_de_fallback(mu_val, G_geo_val, verbose)

# ── 10. Analysis and Visualization ────────────────────────────────────────────
def analyze_cma_result(result):
    """
    Analyze CMA-ES optimization result
    """
    if result is None:
        print("❌ No result to analyze")
        return
    
    params = result['params']
    energy = result['energy_J']
    
    print(f"\n📊 CMA-ES ANALYSIS")
    print("=" * 40)
    print(f"Method: {result['method']}")
    print(f"Final Energy: {energy:.6e} J")
    print(f"Optimization time: {result['optimization_time']:.1f}s")
    print(f"Parameters (μ={result['mu']:.1e}, G_geo={result['G_geo']:.1e}):")
    
    for i in range(M_gauss):
        Ai = params[3*i + 0]
        r0_i = params[3*i + 1]
        sig_i = params[3*i + 2]
        print(f"  Gaussian {i}: A={Ai:.4f}, r0={r0_i:.4f}, σ={sig_i:.4f}")
    
    # Check boundary conditions
    f0 = f_gauss_cma(np.array([0.0]), params)[0]
    fR = f_gauss_cma(np.array([R]), params)[0]
    print(f"\nBoundary check: f(0)={f0:.4f}, f(R)={fR:.6f}")
    
    # Total amplitude
    A_total = sum(params[0::3])
    print(f"Total amplitude: {A_total:.4f}")
    
    return params

def plot_cma_profile(result, save_fig=True):
    """
    Plot CMA-ES optimization result
    """
    if result is None:
        return
    
    params = result['params']
    energy = result['energy_J']
    method = result['method']
    
    # Generate profile
    r_plot = np.linspace(0, R, 500)
    f_plot = f_gauss_cma(r_plot, params)
    fp_plot = f_gauss_cma_prime(r_plot, params)
    
    # Calculate effective density
    sinc_val = np.sinc(mu0 / np.pi) if mu0 > 0 else 1.0
    prefactor = -(v**2) / (8.0*np.pi) * beta_back * sinc_val / G_geo
    rho_plot = prefactor * (fp_plot**2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: f(r) profile
    axes[0,0].plot(r_plot, f_plot, 'b-', linewidth=2, label='f(r)')
    axes[0,0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0,0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
    axes[0,0].set_xlabel('r (m)')
    axes[0,0].set_ylabel('f(r)')
    axes[0,0].set_title('CMA-ES Optimized Profile')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # Plot 2: f'(r) derivative
    axes[0,1].plot(r_plot, fp_plot, 'r-', linewidth=2, label="f'(r)")
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
    
    # Plot 4: Individual Gaussians
    for i in range(M_gauss):
        Ai = params[3*i + 0]
        r0_i = params[3*i + 1]
        sig_i = params[3*i + 2]
        gauss_i = Ai * np.exp(-0.5 * ((r_plot - r0_i) / sig_i)**2)
        axes[1,1].plot(r_plot, gauss_i, '--', alpha=0.7, 
                      label=f'G{i}: A={Ai:.2f}')
    
    axes[1,1].plot(r_plot, f_plot, 'k-', linewidth=2, label='Total')
    axes[1,1].set_xlabel('r (m)')
    axes[1,1].set_ylabel('Amplitude')
    axes[1,1].set_title('Individual Components')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    plt.suptitle(f'{method} Optimization Result\nE₋ = {energy:.4e} J', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('cma_es_profile.png', dpi=300, bbox_inches='tight')
        print("📊 Profile saved as 'cma_es_profile.png'")
    
    plt.close()  # Close instead of show to prevent blocking

# ── 11. Main Execution ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🎯 CMA-ES GLOBAL OPTIMIZER")
    print("=" * 60)
    print(f"Target: Achieve E₋ ≈ -1.95×10³¹ J using CMA-ES")
    print(f"Method: CMA-ES global search + L-BFGS-B refinement")
    print("=" * 60)
    
    # Run optimization
    result = optimize_with_cma()
    
    if result:
        # Analyze results
        analyze_cma_result(result)
        
        # Plot profile
        plot_cma_profile(result)
        
        # Save result
        result_copy = result.copy()
        # Remove objects that can't be serialized
        result_copy.pop('cma_result', None)
        result_copy.pop('refine_result', None)
        result_copy.pop('de_result', None)
        result_copy['params'] = result_copy['params'].tolist()
        
        with open('cma_es_result.json', 'w') as f:
            json.dump(result_copy, f, indent=2)
        print("💾 Result saved as 'cma_es_result.json'")
        
        # Performance comparison
        print(f"\n🏆 PERFORMANCE COMPARISON")
        print("=" * 40)
        baselines = {
            "4-Gaussian (DE)": -1.82e31,
            "5-Gaussian": -1.90e31,
            "Linear Hybrid": -1.86e31
        }
        
        current_energy = result['energy_J']
        for name, baseline in baselines.items():
            improvement = abs(current_energy / baseline)
            print(f"{name:15s}: {baseline:.3e} J → {improvement:.3f}× improvement")
        
        print(f"{'CMA-ES Result':15s}: {current_energy:.3e} J ← NEW RESULT")
    
    else:
        print("❌ Optimization failed")
