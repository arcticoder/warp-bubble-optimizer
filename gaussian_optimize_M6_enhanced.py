#!/usr/bin/env python3
"""
ENHANCED SIX-GAUSSIAN SUPERPOSITION OPTIMIZER

Building on the baseline 6-Gaussian implementation with:
1. Enhanced global search strategies
2. Multiple optimization backends (DE + CMA-ES + JAX)
3. Joint parameter scanning over μ and G_geo
4. Physics-informed constraints and penalties
5. Comprehensive analysis and benchmarking

Target: E₋ ≈ -1.95×10³¹ J → -2.0×10³¹ J
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Optional imports with graceful fallbacks
HAS_CMA = False
try:
    import cma
    HAS_CMA = True
    print("✅ CMA-ES available for enhanced global optimization")
except ImportError:
    print("⚠️  CMA-ES not available. Install with: pip install cma")

HAS_JAX = False
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit
    HAS_JAX = True
    print("✅ JAX available for gradient-based optimization")
except ImportError:
    print("⚠️  JAX not available. Install with: pip install jax")

# ── 1. ENHANCED CONSTANTS ─────────────────────────────────────────────────────
beta_back = 1.9443254780147017  # Backreaction enhancement factor
mu0       = 1e-6                # Polymer length scale (will be scanned)
v         = 1.0                 # Warp velocity (c = 1 units)
R         = 1.0                 # Bubble radius = 1 m
c         = 299792458           # Speed of light (m/s)
G         = 6.67430e-11         # Gravitational constant (m³/kg/s²)
tau       = 1e-9                # QI sampling time
G_geo     = 1e-5                # Van den Broeck–Natário factor (will be scanned)

# Conversion factor
c4_8piG = c**4 / (8.0 * np.pi * G)

# Enhanced ansatz configuration
M_gauss = 6  # Number of Gaussian lumps

# ── 2. ENHANCED GRID & INTEGRATION ────────────────────────────────────────────
N_points = 800  # Optimized for speed vs accuracy
r_grid = np.linspace(0.0, R, N_points)
dr = r_grid[1] - r_grid[0]
vol_weights = 4.0 * np.pi * r_grid**2

# ── 3. ENHANCED ANSATZ FUNCTIONS ──────────────────────────────────────────────
def f_gauss_enhanced(r, params):
    """Enhanced 6-Gaussian with improved boundary handling"""
    r = np.atleast_1d(r)
    total = np.zeros_like(r)
    
    for i in range(M_gauss):
        Ai = params[3*i + 0]
        r0_i = params[3*i + 1]
        sig_i = params[3*i + 2]
        
        # Enhanced Gaussian with boundary smoothing
        x = (r - r0_i) / sig_i
        gaussian = Ai * np.exp(-0.5 * x**2)
        
        # Apply soft boundary constraints
        boundary_factor = np.where(r <= R, 1.0, np.exp(-(r - R)**2 / (0.1*R)**2))
        total += gaussian * boundary_factor
    
    return np.clip(total, 0.0, 1.0)

def f_gauss_enhanced_prime(r, params):
    """Enhanced derivative with boundary corrections"""
    r = np.atleast_1d(r)
    deriv = np.zeros_like(r)
    
    for i in range(M_gauss):
        Ai = params[3*i + 0]
        r0_i = params[3*i + 1]
        sig_i = params[3*i + 2]
        
        x = (r - r0_i) / sig_i
        gaussian = Ai * np.exp(-0.5 * x**2)
        gauss_deriv = gaussian * (-(r - r0_i) / (sig_i**2))
        
        # Apply boundary factor and its derivative
        boundary_factor = np.where(r <= R, 1.0, np.exp(-(r - R)**2 / (0.1*R)**2))
        boundary_deriv = np.where(r <= R, 0.0, 
                                 boundary_factor * (-2*(r - R) / (0.1*R)**2))
        
        deriv += gauss_deriv * boundary_factor + gaussian * boundary_deriv
    
    return deriv

# ── 4. ENHANCED ENERGY CALCULATION ────────────────────────────────────────────
def E_negative_enhanced(params, mu_val=None, G_geo_val=None):
    """Enhanced energy calculation with improved numerics"""
    mu_use = mu_val if mu_val is not None else mu0
    G_geo_use = G_geo_val if G_geo_val is not None else G_geo
    
    # Calculate derivative
    fp_vals = f_gauss_enhanced_prime(r_grid, params)
    
    # Enhanced sinc function with numerical stability
    if mu_use > 0:
        sinc_val = np.sinc(mu_use / np.pi)
    else:
        sinc_val = 1.0
    
    # Enhanced prefactor calculation
    prefactor = -(v**2) / (8.0 * np.pi) * beta_back * sinc_val / G_geo_use
    
    # Calculate effective density with regularization
    fp_squared = fp_vals**2
    # Add small regularization to prevent numerical instability
    fp_squared_reg = fp_squared + 1e-20
    
    rho_vals = prefactor * fp_squared_reg
    
    # Enhanced integration with adaptive weighting
    weights_adaptive = vol_weights * (1.0 + 0.1 * np.exp(-r_grid / (0.1*R)))
    integral = np.sum(rho_vals * weights_adaptive) * dr
    
    return integral * c4_8piG

# ── 5. ENHANCED PENALTY FUNCTIONS ─────────────────────────────────────────────
def penalty_enhanced(params, mu_val=None, G_geo_val=None, 
                    lam_qi=1e50, lam_bound=1e4, lam_curv=1e3, 
                    lam_mono=1e3, lam_symmetry=1e2):
    """Enhanced penalty with additional physics constraints"""
    mu_use = mu_val if mu_val is not None else mu0
    G_geo_use = G_geo_val if G_geo_val is not None else G_geo
    
    penalty_total = 0.0
    
    # (a) Enhanced Quantum Inequality constraint
    fp0 = f_gauss_enhanced_prime(np.array([0.0]), params)[0]
    sinc_val = np.sinc(mu_use / np.pi) if mu_use > 0 else 1.0
    rho0 = -(v**2) / (8.0*np.pi) * beta_back * sinc_val / G_geo_use * (fp0**2)
    
    qi_bound = -(1.0545718e-34 * np.sinc(mu_use/np.pi)) / (12.0 * np.pi * tau**2)
    P_qi = lam_qi * max(0.0, -(rho0 - qi_bound))**2
    penalty_total += P_qi
    
    # (b) Enhanced boundary conditions with soft penalties
    f0 = f_gauss_enhanced(np.array([0.0]), params)[0]
    fR = f_gauss_enhanced(np.array([R]), params)[0]
    P_boundary = lam_bound * ((f0 - 1.0)**2 + (fR - 0.0)**2)
    
    # Additional boundary smoothness
    fp0_boundary = f_gauss_enhanced_prime(np.array([0.0]), params)[0]
    fpR_boundary = f_gauss_enhanced_prime(np.array([R]), params)[0]
    P_boundary += 0.1 * lam_bound * (fp0_boundary**2 + fpR_boundary**2)
    penalty_total += P_boundary
    
    # (c) Enhanced amplitude constraint with redistribution penalty
    A_sum = sum(params[0::3])
    P_amplitude = lam_bound * max(0.0, (A_sum - 1.0))**2
    
    # Penalize extreme amplitude imbalances
    amplitudes = np.array(params[0::3])
    amplitude_variance = np.var(amplitudes)
    P_amplitude += 0.5 * lam_bound * amplitude_variance
    penalty_total += P_amplitude
    
    # (d) Enhanced curvature penalty with adaptive weighting
    f_vals = f_gauss_enhanced(r_grid, params)
    fpp = np.zeros_like(f_vals)
    fpp[1:-1] = (f_vals[2:] - 2*f_vals[1:-1] + f_vals[:-2]) / (dr**2)
    fpp[0] = fpp[1]
    fpp[-1] = fpp[-2]
    
    # Adaptive curvature penalty (higher weight near boundaries)
    curvature_weights = 1.0 + 2.0 * np.exp(-r_grid / (0.2*R))
    P_curvature = lam_curv * np.sum((fpp**2) * (r_grid**2) * curvature_weights) * dr
    penalty_total += P_curvature
    
    # (e) Enhanced monotonicity with smooth violations
    fp_vals = f_gauss_enhanced_prime(r_grid, params)
    P_monotonic = lam_mono * np.sum(np.maximum(0.0, fp_vals)**2) * dr
    penalty_total += P_monotonic
    
    # (f) NEW: Symmetry/correlation penalty for overlapping Gaussians
    for i in range(M_gauss):
        for j in range(i+1, M_gauss):
            r0_i, sig_i = params[3*i + 1], params[3*i + 2]
            r0_j, sig_j = params[3*j + 1], params[3*j + 2]
            
            # Penalize overly similar Gaussians
            distance = abs(r0_i - r0_j)
            width_avg = 0.5 * (sig_i + sig_j)
            if distance < width_avg:
                overlap_penalty = np.exp(-(distance / width_avg)**2)
                P_symmetry += lam_symmetry * overlap_penalty
    
    penalty_total += P_symmetry
    
    return penalty_total

def objective_enhanced(params, mu_val=None, G_geo_val=None):
    """Enhanced objective combining energy and penalties"""
    energy = E_negative_enhanced(params, mu_val, G_geo_val)
    penalty = penalty_enhanced(params, mu_val, G_geo_val)
    return energy + penalty

# ── 6. ENHANCED OPTIMIZATION STRATEGIES ───────────────────────────────────────
def get_bounds_enhanced():
    """Enhanced parameter bounds with physics-informed constraints"""
    bounds = []
    for i in range(M_gauss):
        # Amplitude bounds with reduced overlap
        A_max = 0.8 if i == 0 else 0.6  # Reduce later amplitudes
        bounds += [
            (0.0, A_max),              # Amplitude A_i
            (i * R / (M_gauss + 1), 
             (i + 2) * R / (M_gauss + 1)),  # Position r0_i (spread out)
            (0.01, 0.4*R)              # Width σ_i (slightly reduced max)
        ]
    return bounds

def get_smart_initial_guess_enhanced():
    """Physics-informed initial guess with optimal spacing"""
    params = []
    
    # Use golden ratio for optimal spacing
    phi = (1 + np.sqrt(5)) / 2
    
    for i in range(M_gauss):
        # Decreasing amplitudes with smooth progression
        A_i = 0.7 * (0.85)**i
        
        # Non-uniform spacing for better coverage
        if i < M_gauss // 2:
            r0_i = (i + 0.3) * R / (M_gauss + 1)
        else:
            r0_i = (i + 0.7) * R / (M_gauss + 1)
        
        # Adaptive widths (smaller in center, larger outside)
        sig_i = 0.08 + 0.03 * i + 0.02 * (r0_i / R)
        
        params.extend([A_i, r0_i, sig_i])
    
    return np.array(params)

# ── 7. MULTI-STRATEGY OPTIMIZATION ────────────────────────────────────────────
def optimize_with_differential_evolution(mu_val=None, G_geo_val=None, verbose=True):
    """Enhanced DE optimization"""
    bounds = get_bounds_enhanced()
    
    def objective(params):
        return objective_enhanced(params, mu_val, G_geo_val)
    
    if verbose:
        print("🔍 Running Enhanced Differential Evolution...")
    
    start_time = time.time()
    result = differential_evolution(
        objective, bounds,
        strategy='best1bin',
        maxiter=400,        # Increased for better convergence
        popsize=15,         # Larger population for 6-Gaussians
        tol=1e-8,          # Tighter tolerance
        mutation=(0.3, 1.2), # Broader mutation range
        recombination=0.8,   # Higher recombination
        polish=False,
        workers=-1,
        seed=42
    )
    
    de_time = time.time() - start_time
    
    if verbose and result.success:
        energy = E_negative_enhanced(result.x, mu_val, G_geo_val)
        print(f"   ✅ DE completed in {de_time:.1f}s: E₋ = {energy:.3e} J")
    
    return result

def optimize_with_cma_es(mu_val=None, G_geo_val=None, verbose=True):
    """CMA-ES optimization (if available)"""
    if not HAS_CMA:
        return None
    
    bounds = get_bounds_enhanced()
    x0 = get_smart_initial_guess_enhanced()
    
    def objective(params):
        return objective_enhanced(params, mu_val, G_geo_val)
    
    if verbose:
        print("🧬 Running CMA-ES optimization...")
    
    start_time = time.time()
    
    # CMA-ES setup
    sigma0 = 0.2
    opts = {
        'bounds': [[b[0] for b in bounds], [b[1] for b in bounds]],
        'popsize': 20,
        'maxiter': 200,
        'verb_disp': 0
    }
    
    try:
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
        es.optimize(objective)
        
        result_x = np.array(es.result.xbest)
        result_fun = es.result.fbest
        success = True
        
        cma_time = time.time() - start_time
        
        if verbose:
            energy = E_negative_enhanced(result_x, mu_val, G_geo_val)
            print(f"   ✅ CMA-ES completed in {cma_time:.1f}s: E₋ = {energy:.3e} J")
        
        return {
            'x': result_x,
            'fun': result_fun,
            'success': success,
            'nfev': es.result.evaluations,
            'time': cma_time
        }
        
    except Exception as e:
        if verbose:
            print(f"   ❌ CMA-ES failed: {e}")
        return None

def optimize_enhanced_M6(mu_val=None, G_geo_val=None, method='auto', verbose=True):
    """Main enhanced optimization with multiple strategies"""
    if verbose:
        print(f"🚀 ENHANCED 6-GAUSSIAN OPTIMIZATION")
        print(f"   Method: {method}")
        print(f"   μ = {mu_val or mu0:.1e}, G_geo = {G_geo_val or G_geo:.1e}")
        print("=" * 50)
    
    bounds = get_bounds_enhanced()
    best_result = None
    best_energy = float('inf')
    
    # Strategy 1: Enhanced Differential Evolution
    if method in ['auto', 'de', 'all']:
        de_result = optimize_with_differential_evolution(mu_val, G_geo_val, verbose)
        if de_result and de_result.success:
            energy = E_negative_enhanced(de_result.x, mu_val, G_geo_val)
            if energy < best_energy:
                best_energy = energy
                best_result = {
                    'method': 'DE',
                    'params': de_result.x,
                    'energy_J': energy,
                    'details': de_result
                }
    
    # Strategy 2: CMA-ES (if available)
    if method in ['auto', 'cma', 'all'] and HAS_CMA:
        cma_result = optimize_with_cma_es(mu_val, G_geo_val, verbose)
        if cma_result and cma_result['success']:
            energy = E_negative_enhanced(cma_result['x'], mu_val, G_geo_val)
            if energy < best_energy:
                best_energy = energy
                best_result = {
                    'method': 'CMA-ES',
                    'params': cma_result['x'],
                    'energy_J': energy,
                    'details': cma_result
                }
    
    # Final L-BFGS-B refinement on best result
    if best_result is not None:
        if verbose:
            print(f"🎯 Refining best result ({best_result['method']}) with L-BFGS-B...")
        
        def objective_for_refine(params):
            return objective_enhanced(params, mu_val, G_geo_val)
        
        refine_result = minimize(
            objective_for_refine,
            x0=best_result['params'],
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-9}
        )
        
        if refine_result.success:
            final_energy = E_negative_enhanced(refine_result.x, mu_val, G_geo_val)
            if final_energy < best_energy:
                best_result['params'] = refine_result.x
                best_result['energy_J'] = final_energy
                best_result['method'] += '+L-BFGS-B'
                best_result['refine_details'] = refine_result
    
    # Package final result
    if best_result:
        final_result = {
            'params': best_result['params'],
            'energy_J': best_result['energy_J'],
            'optimization_method': best_result['method'],
            'mu': mu_val or mu0,
            'G_geo': G_geo_val or G_geo,
            'success': True
        }
        
        if verbose:
            print(f"✅ Enhanced optimization complete")
            print(f"   Best method: {best_result['method']}")
            print(f"   Final energy: {best_result['energy_J']:.6e} J")
        
        return final_result
    else:
        if verbose:
            print("❌ All optimization strategies failed")
        return None

# ── 8. JOINT PARAMETER SCANNING ───────────────────────────────────────────────
def joint_parameter_scan(mu_values=None, G_geo_values=None, verbose=True):
    """Comprehensive scan over μ and G_geo parameter space"""
    if mu_values is None:
        mu_values = np.logspace(-8, -3, 6)  # [1e-8, ..., 1e-3]
    if G_geo_values is None:
        G_geo_values = np.logspace(-7, -3, 5)  # [1e-7, ..., 1e-3]
    
    if verbose:
        print(f"🔍 JOINT PARAMETER SCAN")
        print(f"   μ values: {len(mu_values)} points from {mu_values[0]:.1e} to {mu_values[-1]:.1e}")
        print(f"   G_geo values: {len(G_geo_values)} points from {G_geo_values[0]:.1e} to {G_geo_values[-1]:.1e}")
        print(f"   Total combinations: {len(mu_values) * len(G_geo_values)}")
        print("=" * 60)
    
    results = []
    best_overall = {'energy_J': 0.0}
    
    total_combinations = len(mu_values) * len(G_geo_values)
    current_combination = 0
    
    for mu_val in mu_values:
        for G_geo_val in G_geo_values:
            current_combination += 1
            
            if verbose:
                print(f"\n📊 Combination {current_combination}/{total_combinations}")
                print(f"   μ = {mu_val:.1e}, G_geo = {G_geo_val:.1e}")
            
            # Run optimization
            result = optimize_enhanced_M6(mu_val, G_geo_val, method='auto', verbose=False)
            
            if result and result['success']:
                results.append(result)
                
                if result['energy_J'] < best_overall['energy_J']:
                    best_overall = result.copy()
                    if verbose:
                        print(f"   🎯 NEW BEST: E₋ = {result['energy_J']:.3e} J")
                else:
                    if verbose:
                        print(f"   ✅ E₋ = {result['energy_J']:.3e} J")
            else:
                if verbose:
                    print(f"   ❌ Optimization failed")
    
    if verbose:
        print(f"\n🏆 SCAN COMPLETE")
        print(f"   Successful combinations: {len(results)}/{total_combinations}")
        if best_overall['energy_J'] < 0:
            print(f"   Best overall: μ={best_overall['mu']:.1e}, G_geo={best_overall['G_geo']:.1e}")
            print(f"   Best energy: {best_overall['energy_J']:.6e} J")
    
    return results, best_overall

# ── 9. ANALYSIS FUNCTIONS ─────────────────────────────────────────────────────
def analyze_enhanced_result(result, save_json=True):
    """Comprehensive analysis of enhanced optimization result"""
    if not result or not result.get('success'):
        print("❌ No valid result to analyze")
        return None
    
    params = result['params']
    energy = result['energy_J']
    
    print(f"\n📊 ENHANCED 6-GAUSSIAN ANALYSIS")
    print("=" * 50)
    print(f"Optimization method: {result.get('optimization_method', 'Unknown')}")
    print(f"Final Energy: {energy:.6e} J")
    print(f"Parameters: μ={result['mu']:.1e}, G_geo={result['G_geo']:.1e}")
    
    print(f"\n🎯 GAUSSIAN PARAMETERS:")
    total_amplitude = 0.0
    for i in range(M_gauss):
        Ai = params[3*i + 0]
        r0_i = params[3*i + 1]
        sig_i = params[3*i + 2]
        total_amplitude += Ai
        print(f"  Lump {i}: A={Ai:.4f}, r0={r0_i:.4f}, σ={sig_i:.4f}")
    
    print(f"\n🔍 PHYSICS VALIDATION:")
    f0 = f_gauss_enhanced(np.array([0.0]), params)[0]
    fR = f_gauss_enhanced(np.array([R]), params)[0]
    print(f"  Boundary conditions: f(0)={f0:.4f}, f(R)={fR:.6f}")
    print(f"  Total amplitude: {total_amplitude:.4f}")
    
    # Check QI compliance
    fp0 = f_gauss_enhanced_prime(np.array([0.0]), params)[0]
    sinc_val = np.sinc(result['mu'] / np.pi) if result['mu'] > 0 else 1.0
    rho0 = -(v**2) / (8.0*np.pi) * beta_back * sinc_val / result['G_geo'] * (fp0**2)
    qi_bound = -(1.0545718e-34 * np.sinc(result['mu']/np.pi)) / (12.0 * np.pi * tau**2)
    qi_compliant = rho0 >= qi_bound
    print(f"  QI compliance: {'✅' if qi_compliant else '❌'} (ρ₀={rho0:.2e}, bound={qi_bound:.2e})")
    
    # Performance comparison
    print(f"\n🏆 PERFORMANCE COMPARISON:")
    baselines = {
        "2-Lump Soliton": -1.584e31,
        "3-Gaussian": -1.732e31,
        "4-Gaussian": -1.82e31,
        "Hybrid (poly+2G)": -1.86e31,
        "5-Gaussian": -1.90e31,
        "6-Gaussian (baseline)": -1.95e31
    }
    
    for name, baseline in baselines.items():
        improvement = abs(energy / baseline)
        symbol = "🚀" if improvement > 1.05 else "✅" if improvement > 1.0 else "⚠️"
        print(f"  {symbol} vs {name:20s}: {improvement:.3f}× improvement")
    
    # Cost estimate
    cost_per_joule = 2.78e-10  # $/J at $0.001/kWh
    total_cost = abs(energy) * cost_per_joule
    print(f"\n💰 COST ESTIMATE: ${total_cost:.2e} at $0.001/kWh")
    
    if save_json:
        result_copy = result.copy()
        result_copy['params'] = result_copy['params'].tolist()
        # Remove non-serializable objects
        result_copy.pop('details', None)
        result_copy.pop('refine_details', None)
        
        with open('enhanced_M6_result.json', 'w') as f:
            json.dump(result_copy, f, indent=2)
        print(f"💾 Result saved as 'enhanced_M6_result.json'")
    
    return result

# ── 10. MAIN EXECUTION ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🎯 ENHANCED SIX-GAUSSIAN SUPERPOSITION OPTIMIZER")
    print("=" * 70)
    print("Target: Push E₋ below -1.95×10³¹ J using enhanced strategies")
    print("Methods: DE + CMA-ES + L-BFGS-B + Joint parameter scanning")
    print("=" * 70)
    
    # Single optimization with default parameters
    print("\n🚀 SINGLE OPTIMIZATION (Enhanced)")
    result_single = optimize_enhanced_M6(method='all', verbose=True)
    
    if result_single:
        analyze_enhanced_result(result_single)
    
    # Joint parameter scan (reduced scope for demonstration)
    print("\n🔍 JOINT PARAMETER SCAN (Reduced scope)")
    mu_scan = np.logspace(-7, -5, 3)     # [1e-7, 1e-6, 1e-5]
    G_geo_scan = np.logspace(-6, -4, 3)  # [1e-6, 1e-5, 1e-4]
    
    scan_results, best_scan = joint_parameter_scan(mu_scan, G_geo_scan, verbose=True)
    
    if best_scan and best_scan.get('energy_J', 0) < 0:
        print(f"\n🏆 BEST FROM SCAN:")
        analyze_enhanced_result(best_scan, save_json=False)
        
        # Save scan results
        scan_summary = {
            'best_result': best_scan,
            'all_results': scan_results,
            'scan_parameters': {
                'mu_values': mu_scan.tolist(),
                'G_geo_values': G_geo_scan.tolist()
            }
        }
        
        # Make serializable
        for result in scan_summary['all_results'] + [scan_summary['best_result']]:
            if 'params' in result:
                result['params'] = result['params'].tolist()
            result.pop('details', None)
            result.pop('refine_details', None)
        
        with open('enhanced_M6_parameter_scan.json', 'w') as f:
            json.dump(scan_summary, f, indent=2)
        print(f"💾 Scan results saved as 'enhanced_M6_parameter_scan.json'")
    
    print("\n" + "=" * 70)
    print("🏁 ENHANCED 6-GAUSSIAN OPTIMIZATION COMPLETE")
    print("=" * 70)
