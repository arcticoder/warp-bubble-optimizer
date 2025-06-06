#!/usr/bin/env python3
"""
Six-Gaussian Superposition Ansatz Optimizer

Extension of the 5-Gaussian ansatz to 6 Gaussians to push E- even lower.
Target: Achieve E- ≈ -1.95×10³¹ J (improvement over current -1.90×10³¹ J)

Each additional Gaussian provides 3 more optimization parameters:
- Amplitude A_i ∈ [0,1]  
- Position r0_i ∈ [0,R]
- Width σ_i ∈ [0.01R, 0.5R]

Uses accelerated vectorized integration with enhanced physics constraints.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import json
import time

# ── 1. Physical Constants ─────────────────────────────────────────────────────
beta_back = 1.9443254780147017  # Backreaction enhancement factor
mu0       = 1e-6                # Polymer length scale (can be scanned)
v         = 1.0                 # Warp velocity (c = 1 units)
R         = 1.0                 # Bubble radius = 1 m
c         = 299792458           # Speed of light (m/s)
G         = 6.67430e-11         # Gravitational constant (m³/kg/s²)
tau       = 1e-9                # QI sampling time
G_geo     = 1e-5                # Van den Broeck–Natário geometric factor

# Conversion factor for natural units to Joules
c4_8piG = c**4 / (8.0 * np.pi * G)  # ≈ 4.815×10⁴² J⋅m⁻³

# ── 2. Ansatz Configuration ───────────────────────────────────────────────────
M_gauss = 6  # Number of Gaussian lumps (increased from 5)

# ── 3. Precompute Radial Grid & Weights for Vectorized Integration ───────────
N_points = 800  # Optimized grid size for speed vs accuracy
r_grid = np.linspace(0.0, R, N_points)
dr = r_grid[1] - r_grid[0]
vol_weights = 4.0 * np.pi * r_grid**2  # Volume elements for ∫4π r² dr

# ── 4. Six-Gaussian Ansatz Functions ──────────────────────────────────────────
def f_gauss_M6(r, params):
    """
    Six-Gaussian superposition ansatz
    
    f(r) = sum_{i=0}^{5} A_i * exp(-0.5 * ((r - r0_i)/σ_i)²)
    
    Args:
        r: Radial coordinate (scalar or array)
        params: Parameter vector [A0,r0_0,σ0, A1,r0_1,σ1, ..., A5,r0_5,σ5]
    
    Returns:
        Function value(s) clipped to [0,1] for physicality
    """
    r = np.atleast_1d(r)
    total = np.zeros_like(r)
    
    for i in range(M_gauss):
        Ai = params[3*i + 0]      # Amplitude
        r0_i = params[3*i + 1]    # Center position  
        sig_i = params[3*i + 2]   # Width
        
        x = (r - r0_i) / sig_i
        total += Ai * np.exp(-0.5 * x**2)
    
    return np.clip(total, 0.0, 1.0)  # Enforce physical bounds

def f_gauss_M6_prime(r, params):
    """
    Derivative of six-Gaussian ansatz
    
    f'(r) = sum_{i=0}^{5} A_i * exp(-0.5 * x_i²) * (-(r - r0_i)/σ_i²)
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

# ── 5. Vectorized Energy Calculation ──────────────────────────────────────────
def E_negative_M6(params, mu_val=None, G_geo_val=None):
    """
    Calculate total negative energy using vectorized integration
    
    E₋ = ∫ ρ_eff(r) * 4π r² dr * c⁴/(8πG)
    
    where ρ_eff = -(v²/8π) * β_back * sinc(μ) / G_geo * (f'(r))²
    """
    # Use global values if not specified
    mu_use = mu_val if mu_val is not None else mu0
    G_geo_use = G_geo_val if G_geo_val is not None else G_geo
    
    # Calculate f'(r) on the grid
    fp_vals = f_gauss_M6_prime(r_grid, params)
    
    # Polymer sinc function enhancement
    sinc_val = np.sinc(mu_use / np.pi) if mu_use > 0 else 1.0
    
    # Effective density prefactor
    prefactor = -(v**2) / (8.0 * np.pi) * beta_back * sinc_val / G_geo_use
    
    # Calculate effective density
    rho_vals = prefactor * (fp_vals**2)
    
    # Vectorized integration: ∫ ρ(r) * 4π r² dr
    integral = np.sum(rho_vals * vol_weights) * dr
    
    # Convert to Joules
    return integral * c4_8piG

# ── 6. Enhanced Physics Penalties ─────────────────────────────────────────────
def penalty_M6(params, lam_qi=1e50, lam_bound=1e4, lam_curv=1e3, lam_mono=1e3):
    """
    Enhanced penalty function with multiple physics constraints
    """
    penalty_total = 0.0
    
    # (a) Quantum Inequality constraint at r=0
    fp0 = f_gauss_M6_prime(np.array([0.0]), params)[0]
    sinc_val = np.sinc(mu0 / np.pi) if mu0 > 0 else 1.0
    rho0 = -(v**2) / (8.0*np.pi) * beta_back * sinc_val / G_geo * (fp0**2)
    
    # QI bound: ρ ≥ -ℏsinc(μ/π)/(12π τ²)
    qi_bound = -(1.0545718e-34 * np.sinc(mu0/np.pi)) / (12.0 * np.pi * tau**2)
    P_qi = lam_qi * max(0.0, -(rho0 - qi_bound))**2
    penalty_total += P_qi
    
    # (b) Boundary conditions: f(0) ≈ 1, f(R) ≈ 0
    f0 = f_gauss_M6(np.array([0.0]), params)[0]
    fR = f_gauss_M6(np.array([R]), params)[0]
    P_boundary = lam_bound * ((f0 - 1.0)**2 + (fR - 0.0)**2)
    penalty_total += P_boundary
    
    # (c) Amplitude constraint: sum(A_i) ≤ 1 (avoid over-normalization)
    A_sum = sum(params[0::3])  # Extract all amplitudes
    P_amplitude = lam_bound * max(0.0, (A_sum - 1.0))**2
    penalty_total += P_amplitude
    
    # (d) Curvature penalty: ∫(f''(r))² r² dr (promotes smoothness)
    f_vals = f_gauss_M6(r_grid, params)
    fpp = np.zeros_like(f_vals)
    fpp[1:-1] = (f_vals[2:] - 2*f_vals[1:-1] + f_vals[:-2]) / (dr**2)
    fpp[0] = fpp[1]      # Boundary handling
    fpp[-1] = fpp[-2]
    P_curvature = lam_curv * np.sum((fpp**2) * (r_grid**2)) * dr
    penalty_total += P_curvature
    
    # (e) Monotonicity penalty: f'(r) ≤ 0 (enforce decrease outward)
    fp_vals = f_gauss_M6_prime(r_grid, params)
    P_monotonic = lam_mono * np.sum(np.maximum(0.0, fp_vals)**2) * dr
    penalty_total += P_monotonic
    
    return penalty_total

def objective_M6(params):
    """
    Combined objective: minimize E₋ + penalties
    """
    energy = E_negative_M6(params)
    penalty = penalty_M6(params)
    return energy + penalty

# ── 7. Optimization Bounds and Initialization ─────────────────────────────────
def get_bounds_M6():
    """
    Generate parameter bounds for 6-Gaussian optimization
    """
    bounds = []
    for i in range(M_gauss):
        bounds += [
            (0.0, 1.0),        # Amplitude A_i ∈ [0,1]
            (0.0, R),          # Position r0_i ∈ [0,R]  
            (0.01, 0.5*R)      # Width σ_i ∈ [0.01R, 0.5R]
        ]
    return bounds

def get_initial_guess_M6():
    """
    Generate educated initial guess for 6-Gaussian parameters
    """
    params = []
    for i in range(M_gauss):
        # Distribute Gaussians across the bubble with decreasing amplitudes
        A_i = 0.8 * (0.8)**i        # Exponentially decreasing amplitudes
        r0_i = (i + 0.5) * R / M_gauss  # Evenly spaced positions
        sig_i = 0.1 + 0.05 * i       # Gradually increasing widths
        params.extend([A_i, r0_i, sig_i])
    return np.array(params)

# ── 8. Main Optimization Function ─────────────────────────────────────────────
def optimize_M6(mu_val=None, G_geo_val=None, verbose=True):
    """
    Run complete 6-Gaussian optimization: DE global search + L-BFGS refinement
    """
    global mu0, G_geo
    if mu_val is not None:
        mu0 = mu_val
    if G_geo_val is not None:
        G_geo = G_geo_val
        
    if verbose:
        print(f"🚀 6-GAUSSIAN OPTIMIZATION")
        print(f"   μ = {mu0:.1e}, G_geo = {G_geo:.1e}")
        print("=" * 50)
    
    bounds = get_bounds_M6()
    start_time = time.time()
    
    # Stage 1: Global search with Differential Evolution
    if verbose:
        print("🔍 Stage 1: Differential Evolution global search...")
    
    de_result = differential_evolution(
        objective_M6, bounds,
        strategy='best1bin',
        maxiter=300,           # Moderate iterations for speed
        popsize=12,            # Population size
        tol=1e-7,
        mutation=(0.5, 1.0),   # Mutation factor range
        recombination=0.7,     # Crossover probability
        polish=False,          # Don't polish (we'll do L-BFGS separately)
        workers=-1,            # Use all CPU cores
        seed=42                # Reproducible results
    )
    
    if not de_result.success:
        print(f"❌ DE failed: {de_result.message}")
        return None
    
    de_time = time.time() - start_time
    if verbose:
        print(f"   DE completed in {de_time:.1f}s")
        print(f"   Best DE energy: {E_negative_M6(de_result.x):.3e} J")
    
    # Stage 2: Local refinement with L-BFGS-B
    if verbose:
        print("🎯 Stage 2: L-BFGS-B refinement...")
    
    refine_result = minimize(
        objective_M6, 
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
    final_energy = E_negative_M6(refine_result.x)
    
    if verbose:
        print(f"✅ Optimization completed in {total_time:.1f}s")
        print(f"   Final energy: {final_energy:.6e} J")
        print(f"   Function evaluations: DE={de_result.nfev}, L-BFGS={refine_result.nfev}")
    
    return {
        'params': refine_result.x,
        'energy_J': final_energy,
        'mu': mu0,
        'G_geo': G_geo,
        'optimization_time': total_time,
        'de_result': de_result,
        'refine_result': refine_result
    }

# ── 9. Analysis and Visualization ─────────────────────────────────────────────
def analyze_M6_result(result):
    """
    Analyze and display the 6-Gaussian optimization result
    """
    if result is None:
        print("❌ No result to analyze")
        return
    
    params = result['params']
    energy = result['energy_J']
    
    print(f"\n📊 6-GAUSSIAN ANALYSIS")
    print("=" * 40)
    print(f"Final Energy: {energy:.6e} J")
    print(f"Optimization time: {result['optimization_time']:.1f}s")
    print(f"Parameters (μ={result['mu']:.1e}, G_geo={result['G_geo']:.1e}):")
    
    for i in range(M_gauss):
        Ai = params[3*i + 0]
        r0_i = params[3*i + 1]
        sig_i = params[3*i + 2]
        print(f"  Lump {i}: A={Ai:.4f}, r0={r0_i:.4f}, σ={sig_i:.4f}")
    
    # Check boundary conditions
    f0 = f_gauss_M6(np.array([0.0]), params)[0]
    fR = f_gauss_M6(np.array([R]), params)[0]
    print(f"\nBoundary check: f(0)={f0:.4f}, f(R)={fR:.6f}")
    
    # Calculate total amplitude
    A_total = sum(params[0::3])
    print(f"Total amplitude: {A_total:.4f}")
    
    return params

def plot_M6_profile(result, save_fig=True):
    """
    Plot the optimized 6-Gaussian profile
    """
    if result is None:
        return
    
    params = result['params']
    energy = result['energy_J']
    
    # Generate high-resolution profile
    r_plot = np.linspace(0, R, 500)
    f_plot = f_gauss_M6(r_plot, params)
    fp_plot = f_gauss_M6_prime(r_plot, params)
    
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
    axes[0,0].set_title('6-Gaussian Warp Profile')
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
    
    # Plot 4: Individual Gaussian components
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
    axes[1,1].set_title('Individual Gaussian Components')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    plt.suptitle(f'6-Gaussian Optimization Result\nE₋ = {energy:.4e} J', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('gaussian_M6_profile.png', dpi=300, bbox_inches='tight')
        print("📊 Profile saved as 'gaussian_M6_profile.png'")
    
    plt.close()  # Close instead of show to prevent blocking

# ── 10. Main Execution ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🎯 SIX-GAUSSIAN SUPERPOSITION OPTIMIZER")
    print("=" * 60)
    print(f"Target: Push E₋ below -1.95×10³¹ J")
    print(f"Current best: ~-1.90×10³¹ J (5-Gaussian)")
    print("=" * 60)
    
    # Run optimization with default parameters
    result = optimize_M6()
    
    if result:
        # Analyze results
        analyze_M6_result(result)
        
        # Plot profile
        plot_M6_profile(result)
        
        # Save result for later use
        result_copy = result.copy()
        # Remove scipy objects that can't be serialized
        result_copy.pop('de_result', None)
        result_copy.pop('refine_result', None)
        result_copy['params'] = result_copy['params'].tolist()
        
        with open('gaussian_M6_result.json', 'w') as f:
            json.dump(result_copy, f, indent=2)
        print("💾 Result saved as 'gaussian_M6_result.json'")
        
        # Compare with previous results
        print(f"\n🏆 PERFORMANCE COMPARISON")
        print("=" * 40)
        baselines = {
            "2-Lump Soliton": -1.584e31,
            "3-Gaussian": -1.732e31, 
            "4-Gaussian": -1.82e31,
            "Hybrid (poly+2G)": -1.86e31,
            "5-Gaussian": -1.90e31
        }
        
        current_energy = result['energy_J']
        for name, baseline in baselines.items():
            improvement = abs(current_energy / baseline)
            print(f"{name:15s}: {baseline:.3e} J → {improvement:.3f}× improvement")
        
        print(f"{'6-Gaussian':15s}: {current_energy:.3e} J ← NEW RESULT")
        
        # Cost estimate
        cost_per_joule = 2.78e-10  # $/J at $0.001/kWh
        total_cost = abs(current_energy) * cost_per_joule
        print(f"\nEstimated cost: ${total_cost:.2e} at $0.001/kWh")
    
    else:
        print("❌ Optimization failed")
