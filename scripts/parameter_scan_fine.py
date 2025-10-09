#!/usr/bin/env python3
"""
Fine Parameter Scan for Joint (μ, G_geo) Optimization

Comprehensive 2D parameter sweep to find optimal physical parameters
for both 6-Gaussian and hybrid cubic ansätze. This addresses the limitation
of fixed μ=10⁻⁶ and G_geo=10⁻⁵ by exploring a broader parameter space.

Target: Find optimal (μ, G_geo) combinations that push E- even lower
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Import optimization functions (will be modified for parameter scanning)
import importlib.util
import sys
import os

# ── 1. Enhanced Parameter Scan Configuration ─────────────────────────────────
# Parameter ranges for comprehensive scan
MU_RANGE = np.logspace(-8, -3, 8)      # [1e-8, 1e-7, ..., 1e-3]
G_GEO_RANGE = np.logspace(-7, -3, 6)   # [1e-7, 1e-6, ..., 1e-3]

print(f"Parameter scan configuration:")
print(f"  μ range: {MU_RANGE}")
print(f"  G_geo range: {G_GEO_RANGE}")
print(f"  Total combinations: {len(MU_RANGE) * len(G_GEO_RANGE)}")

# ── 2. Enhanced 6-Gaussian Implementation ────────────────────────────────────
def optimize_6gauss_with_params(mu_val, G_geo_val):
    """6-Gaussian optimization with specific (μ, G_geo) values"""
    from scipy.optimize import differential_evolution
    
    # Constants
    beta_back = 1.9443254780147017
    v = 1.0
    R = 1.0
    c = 299792458
    G = 6.67430e-11
    tau = 1e-9
    c4_8piG = c**4 / (8.0 * np.pi * G)
    
    # Grid
    N = 600  # Slightly reduced for speed
    r_grid = np.linspace(0.0, R, N)
    dr = r_grid[1] - r_grid[0]
    vol_weights = 4.0 * np.pi * r_grid**2
    M_gauss = 6
    
    def f_6g(r, p):
        total = np.zeros_like(r)
        for i in range(M_gauss):
            Ai = p[3*i + 0]
            r0_i = p[3*i + 1]
            sig_i = p[3*i + 2]
            x = (r - r0_i) / sig_i
            total += Ai * np.exp(-0.5 * x*x)
        return np.clip(total, 0.0, 1.0)
    
    def f_6g_prime(r, p):
        deriv = np.zeros_like(r)
        for i in range(M_gauss):
            Ai = p[3*i + 0]
            r0_i = p[3*i + 1]
            sig_i = p[3*i + 2]
            x = (r - r0_i) / sig_i
            pref = Ai * np.exp(-0.5 * x*x)
            deriv += pref * (-(r - r0_i) / (sig_i**2))
        return deriv
    
    def E_neg_6g(p):
        fp_vals = f_6g_prime(r_grid, p)
        sinc_val = np.sinc(mu_val/np.pi) if mu_val > 0 else 1.0
        prefac = - (v**2)/(8.0*np.pi) * beta_back * sinc_val / G_geo_val
        rho_vals = prefac * (fp_vals**2)
        integral = np.sum(rho_vals * vol_weights) * dr
        return integral * c4_8piG
    
    def penalty_6g(p):
        penalty = 0.0
        
        # QI constraint
        fp0 = f_6g_prime(np.array([0.0]), p)[0]
        sinc_val = np.sinc(mu_val/np.pi) if mu_val > 0 else 1.0
        rho0 = - (v**2)/(8.0*np.pi) * beta_back * sinc_val / G_geo_val * (fp0**2)
        qi_bound = - (1.0545718e-34 * np.sinc(mu_val/np.pi)) / (12.0 * np.pi * tau**2)
        penalty += 1e50 * max(0.0, -(rho0 - qi_bound))**2
        
        # Boundary conditions
        f0 = f_6g(np.array([0.0]), p)[0]
        fR = f_6g(np.array([R]), p)[0]
        penalty += 1e4 * ((f0 - 1.0)**2 + (fR - 0.0)**2)
        
        # Amplitude constraint
        A_sum = sum(p[0::3])
        penalty += 1e4 * max(0.0, (A_sum - 1.0))**2
        
        return penalty
    
    def obj_6g(p):
        return E_neg_6g(p) + penalty_6g(p)
    
    # Bounds
    bounds = []
    for _ in range(M_gauss):
        bounds += [(0.0, 1.0), (0.0, R), (0.01, 0.5*R)]
    
    try:
        # Fast DE optimization
        result = differential_evolution(
            obj_6g, bounds,
            strategy='best1bin', maxiter=500,  # Reduced for speed
            popsize=15, tol=1e-6,
            mutation=(0.5, 1), recombination=0.7,
            polish=False, workers=1  # Single worker to avoid conflicts
        )
        
        if result.success:
            return E_neg_6g(result.x), result.x
        else:
            return None, None
            
    except Exception as e:
        return None, None

def optimize_hybrid_cubic_with_params(mu_val, G_geo_val):
    """Hybrid cubic optimization with specific (μ, G_geo) values"""
    from scipy.optimize import differential_evolution
    
    # Constants (using provided values)
    beta_back = 1.9443254780147017
    v = 1.0
    R = 1.0
    c = 299792458
    G = 6.67430e-11
    tau = 1e-9
    c4_8piG = c**4 / (8.0 * np.pi * G)
    M_g = 2
    
    # Grid
    N = 600
    r_grid = np.linspace(0.0, R, N)
    dr = r_grid[1] - r_grid[0]
    vol_weights = 4.0 * np.pi * r_grid**2
    
    def f_hc(r, p):
        r0, r1 = p[0], p[1]
        b1, b2, b3 = p[2], p[3], p[4]
        
        if isinstance(r, (int, float)):
            if r <= r0:
                return 1.0
            elif r < r1:
                x = (r - r0)/(r1 - r0)
                val = 1.0 + b1*x + b2*(x**2) + b3*(x**3)
                return np.clip(val, 0.0, 1.0)
            else:
                total = 0.0
                for i in range(M_g):
                    Ai = p[5 + 3*i + 0]
                    r0_i = p[5 + 3*i + 1]
                    sig_i = p[5 + 3*i + 2]
                    x = (r - r0_i)/sig_i
                    total += Ai * np.exp(-0.5*x*x)
                return np.clip(total, 0.0, 1.0)
        else:
            result = np.ones_like(r)
            mask_trans = (r > r0) & (r < r1)
            if np.any(mask_trans):
                x = (r[mask_trans] - r0)/(r1 - r0)
                poly_vals = 1.0 + b1*x + b2*(x**2) + b3*(x**3)
                result[mask_trans] = np.clip(poly_vals, 0.0, 1.0)
            
            mask_gauss = (r >= r1)
            if np.any(mask_gauss):
                r_gauss = r[mask_gauss]
                gauss_total = np.zeros_like(r_gauss)
                for i in range(M_g):
                    Ai = p[5 + 3*i + 0]
                    r0_i = p[5 + 3*i + 1]
                    sig_i = p[5 + 3*i + 2]
                    x = (r_gauss - r0_i)/sig_i
                    gauss_total += Ai * np.exp(-0.5*x*x)
                result[mask_gauss] = np.clip(gauss_total, 0.0, 1.0)
            
            return result
    
    def f_hc_prime(r, p):
        r0, r1 = p[0], p[1]
        b1, b2, b3 = p[2], p[3], p[4]
        
        if isinstance(r, (int, float)):
            if r <= r0 or r >= R:
                return 0.0
            elif r < r1:
                x = (r - r0)/(r1 - r0)
                dx_dr = 1.0/(r1 - r0)
                return (b1 + 2*b2*x + 3*b3*(x**2)) * dx_dr
            else:
                deriv = 0.0
                for i in range(M_g):
                    Ai = p[5 + 3*i + 0]
                    r0_i = p[5 + 3*i + 1]
                    sig_i = p[5 + 3*i + 2]
                    x = (r - r0_i)/sig_i
                    pref = Ai * np.exp(-0.5*x*x)
                    deriv += pref * (-(r - r0_i)/(sig_i**2))
                return deriv
        else:
            result = np.zeros_like(r)
            mask_trans = (r > r0) & (r < r1)
            if np.any(mask_trans):
                x = (r[mask_trans] - r0)/(r1 - r0)
                dx_dr = 1.0/(r1 - r0)
                result[mask_trans] = (b1 + 2*b2*x + 3*b3*(x**2)) * dx_dr
            
            mask_gauss = (r >= r1) & (r < R)
            if np.any(mask_gauss):
                r_gauss = r[mask_gauss]
                for i in range(M_g):
                    Ai = p[5 + 3*i + 0]
                    r0_i = p[5 + 3*i + 1]
                    sig_i = p[5 + 3*i + 2]
                    x = (r_gauss - r0_i)/sig_i
                    pref = Ai * np.exp(-0.5*x*x)
                    result[mask_gauss] += pref * (-(r_gauss - r0_i)/(sig_i**2))
            
            return result
    
    def E_neg_hc(p):
        fp = f_hc_prime(r_grid, p)
        sinc_val = np.sinc(mu_val/np.pi) if mu_val > 0 else 1.0
        prefac = - (v**2)/(8.0*np.pi) * beta_back * sinc_val / G_geo_val
        rho_vals = prefac * (fp**2)
        integral = np.sum(rho_vals * vol_weights) * dr
        return integral * c4_8piG
    
    def penalty_hc(p):
        r0, r1 = p[0], p[1]
        penalty = 0.0
        
        # Bounds
        if not (0.0 < r0 < r1 < R):
            penalty += 1e6 * ((max(0, -r0))**2 + (max(0, r0-r1))**2 + (max(0, r1-R))**2)
        
        # QI constraint
        fp0 = f_hc_prime(0.0, p)
        sinc_val = np.sinc(mu_val/np.pi) if mu_val > 0 else 1.0
        rho0 = - (v**2)/(8.0*np.pi) * beta_back * sinc_val / G_geo_val * (fp0**2)
        qi_bound = - (1.0545718e-34 * np.sinc(mu_val/np.pi)) / (12.0 * np.pi * tau**2)
        penalty += 1e50 * max(0.0, -(rho0 - qi_bound))**2
        
        # Continuity
        b1, b2, b3 = p[2], p[3], p[4]
        poly_at_r1 = 1.0 + b1 + b2 + b3
        gauss_at_r1 = 0.0
        for i in range(M_g):
            Ai = p[5+3*i+0]
            r0_i = p[5+3*i+1]
            sig_i = p[5+3*i+2]
            gauss_at_r1 += Ai * np.exp(-0.5*((r1 - r0_i)/sig_i)**2)
        penalty += 1e6 * (poly_at_r1 - gauss_at_r1)**2
        
        return penalty
    
    def obj_hc(p):
        return E_neg_hc(p) + penalty_hc(p)
    
    # Bounds
    bounds = [(0.05*R, 0.35*R), (0.4*R, 0.8*R)]  # r0, r1
    bounds += [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]  # b1, b2, b3
    for _ in range(M_g):
        bounds += [(0.0, 1.0), (0.4*R, R), (0.02*R, 0.4*R)]
    
    try:
        result = differential_evolution(
            obj_hc, bounds,
            strategy='best1bin', maxiter=500,
            popsize=15, tol=1e-6,
            mutation=(0.5, 1), recombination=0.8,
            polish=False, workers=1
        )
        
        if result.success:
            return E_neg_hc(result.x), result.x
        else:
            return None, None
            
    except Exception as e:
        return None, None

# ── 3. Parallel Parameter Scanning ───────────────────────────────────────────
def scan_single_combination(args):
    """Single parameter combination scan (for parallel processing)"""
    mu_val, G_geo_val = args
    
    results = {
        'mu': mu_val,
        'G_geo': G_geo_val,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(f"🔍 Scanning μ={mu_val:.2e}, G_geo={G_geo_val:.2e}")
    
    # Test 6-Gaussian
    try:
        energy_6g, params_6g = optimize_6gauss_with_params(mu_val, G_geo_val)
        if energy_6g is not None:
            results['6gauss'] = {
                'energy': float(energy_6g),
                'parameters': params_6g.tolist() if params_6g is not None else None,
                'success': True
            }
        else:
            results['6gauss'] = {'success': False}
    except Exception as e:
        results['6gauss'] = {'success': False, 'error': str(e)}
    
    # Test Hybrid Cubic
    try:
        energy_hc, params_hc = optimize_hybrid_cubic_with_params(mu_val, G_geo_val)
        if energy_hc is not None:
            results['hybrid_cubic'] = {
                'energy': float(energy_hc),
                'parameters': params_hc.tolist() if params_hc is not None else None,
                'success': True
            }
        else:
            results['hybrid_cubic'] = {'success': False}
    except Exception as e:
        results['hybrid_cubic'] = {'success': False, 'error': str(e)}
    
    return results

def run_comprehensive_parameter_scan():
    """Run comprehensive 2D parameter scan with parallel processing"""
    print("🚀 Comprehensive Fine Parameter Scan")
    print("="*60)
    print(f"Scanning {len(MU_RANGE)} × {len(G_GEO_RANGE)} = {len(MU_RANGE) * len(G_GEO_RANGE)} combinations")
    print(f"Available CPU cores: {cpu_count()}")
    
    # Prepare parameter combinations
    param_combinations = [(mu, G_geo) for mu in MU_RANGE for G_geo in G_GEO_RANGE]
    
    # Run parallel scan
    start_time = time.time()
    
    # Use fewer processes to avoid memory issues
    n_processes = min(4, cpu_count())
    print(f"Using {n_processes} parallel processes")
    
    with Pool(processes=n_processes) as pool:
        all_results = pool.map(scan_single_combination, param_combinations)
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Parameter scan completed in {elapsed:.1f}s")
    
    # Analyze results
    analyze_scan_results(all_results)
    
    return all_results

def analyze_scan_results(all_results):
    """Comprehensive analysis of parameter scan results"""
    print(f"\n📊 PARAMETER SCAN ANALYSIS")
    print("="*60)
    
    # Extract successful results
    successful_6g = []
    successful_hc = []
    
    for result in all_results:
        if result.get('6gauss', {}).get('success', False):
            successful_6g.append({
                'mu': result['mu'],
                'G_geo': result['G_geo'],
                'energy': result['6gauss']['energy']
            })
        
        if result.get('hybrid_cubic', {}).get('success', False):
            successful_hc.append({
                'mu': result['mu'],
                'G_geo': result['G_geo'],
                'energy': result['hybrid_cubic']['energy']
            })
    
    print(f"Successful optimizations:")
    print(f"  6-Gaussian: {len(successful_6g)}/{len(all_results)}")
    print(f"  Hybrid Cubic: {len(successful_hc)}/{len(all_results)}")
    
    # Find best results
    best_overall = {'energy': 0, 'type': None}
    
    if successful_6g:
        best_6g = min(successful_6g, key=lambda x: x['energy'])
        print(f"\nBest 6-Gaussian result:")
        print(f"  Energy: {best_6g['energy']:.4e} J")
        print(f"  Parameters: μ={best_6g['mu']:.2e}, G_geo={best_6g['G_geo']:.2e}")
        
        if best_6g['energy'] < best_overall['energy']:
            best_overall.update(best_6g)
            best_overall['type'] = '6-Gaussian'
    
    if successful_hc:
        best_hc = min(successful_hc, key=lambda x: x['energy'])
        print(f"\nBest Hybrid Cubic result:")
        print(f"  Energy: {best_hc['energy']:.4e} J")
        print(f"  Parameters: μ={best_hc['mu']:.2e}, G_geo={best_hc['G_geo']:.2e}")
        
        if best_hc['energy'] < best_overall['energy']:
            best_overall.update(best_hc)
            best_overall['type'] = 'Hybrid Cubic'
    
    if best_overall['type']:
        print(f"\n🏆 BEST OVERALL RESULT:")
        print(f"  Type: {best_overall['type']}")
        print(f"  Energy: {best_overall['energy']:.4e} J")
        print(f"  Parameters: μ={best_overall['mu']:.2e}, G_geo={best_overall['G_geo']:.2e}")
        
        # Check if target achieved
        if best_overall['energy'] < -2.0e31:
            print(f"  🎯 TARGET ACHIEVED! E < -2.0×10³¹ J")
        elif best_overall['energy'] < -1.95e31:
            print(f"  🔥 Excellent! Very close to target")
        elif best_overall['energy'] < -1.90e31:
            print(f"  ✅ Great progress toward target")
    
    # Save comprehensive results
    scan_summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_combinations': len(all_results),
        'successful_6g': len(successful_6g),
        'successful_hc': len(successful_hc),
        'best_6g': best_6g if successful_6g else None,
        'best_hc': best_hc if successful_hc else None,
        'best_overall': best_overall if best_overall['type'] else None,
        'full_results': all_results
    }
    
    with open('parameter_scan_comprehensive_results.json', 'w') as f:
        json.dump(scan_summary, f, indent=2)
    
    # Generate visualization
    plot_parameter_scan_results(successful_6g, successful_hc)
    
    print(f"\n📁 Results saved to: parameter_scan_comprehensive_results.json")
    print(f"📊 Visualization saved to: parameter_scan_heatmap.png")

def plot_parameter_scan_results(results_6g, results_hc):
    """Create comprehensive visualization of parameter scan results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Create meshgrid for heatmaps
    mu_grid, G_geo_grid = np.meshgrid(MU_RANGE, G_GEO_RANGE, indexing='ij')
    
    # 6-Gaussian heatmap
    energy_6g_grid = np.full_like(mu_grid, np.nan)
    for result in results_6g:
        mu_idx = np.argmin(np.abs(MU_RANGE - result['mu']))
        G_geo_idx = np.argmin(np.abs(G_GEO_RANGE - result['G_geo']))
        energy_6g_grid[mu_idx, G_geo_idx] = result['energy']
    
    im1 = ax1.imshow(energy_6g_grid, extent=[np.log10(G_GEO_RANGE[0]), np.log10(G_GEO_RANGE[-1]),
                                            np.log10(MU_RANGE[0]), np.log10(MU_RANGE[-1])],
                    aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    ax1.set_xlabel('log₁₀(G_geo)')
    ax1.set_ylabel('log₁₀(μ)')
    ax1.set_title('6-Gaussian Energy Landscape')
    plt.colorbar(im1, ax=ax1, label='Energy (J)')
    
    # Hybrid Cubic heatmap
    energy_hc_grid = np.full_like(mu_grid, np.nan)
    for result in results_hc:
        mu_idx = np.argmin(np.abs(MU_RANGE - result['mu']))
        G_geo_idx = np.argmin(np.abs(G_GEO_RANGE - result['G_geo']))
        energy_hc_grid[mu_idx, G_geo_idx] = result['energy']
    
    im2 = ax2.imshow(energy_hc_grid, extent=[np.log10(G_GEO_RANGE[0]), np.log10(G_GEO_RANGE[-1]),
                                            np.log10(MU_RANGE[0]), np.log10(MU_RANGE[-1])],
                    aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    ax2.set_xlabel('log₁₀(G_geo)')
    ax2.set_ylabel('log₁₀(μ)')
    ax2.set_title('Hybrid Cubic Energy Landscape')
    plt.colorbar(im2, ax=ax2, label='Energy (J)')
    
    # Best results comparison
    methods = ['6-Gaussian', 'Hybrid Cubic']
    best_energies = []
    
    if results_6g:
        best_6g = min(results_6g, key=lambda x: x['energy'])['energy']
        best_energies.append(best_6g)
    else:
        best_energies.append(0)
    
    if results_hc:
        best_hc = min(results_hc, key=lambda x: x['energy'])['energy']
        best_energies.append(best_hc)
    else:
        best_energies.append(0)
    
    bars = ax3.bar(methods, best_energies, color=['blue', 'orange'], alpha=0.7)
    ax3.axhline(-2.0e31, color='red', linestyle='--', label='Target: -2.0×10³¹ J')
    ax3.set_ylabel('Energy (J)')
    ax3.set_title('Best Results Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Parameter distribution of successful results
    if results_6g:
        mu_vals_6g = [r['mu'] for r in results_6g]
        G_geo_vals_6g = [r['G_geo'] for r in results_6g]
        energies_6g = [r['energy'] for r in results_6g]
        
        scatter = ax4.scatter(np.log10(mu_vals_6g), np.log10(G_geo_vals_6g), 
                             c=energies_6g, s=50, alpha=0.7, cmap='viridis', 
                             label='6-Gaussian')
        
    if results_hc:
        mu_vals_hc = [r['mu'] for r in results_hc]
        G_geo_vals_hc = [r['G_geo'] for r in results_hc]
        energies_hc = [r['energy'] for r in results_hc]
        
        scatter = ax4.scatter(np.log10(mu_vals_hc), np.log10(G_geo_vals_hc), 
                             c=energies_hc, s=50, alpha=0.7, cmap='plasma', 
                             marker='^', label='Hybrid Cubic')
    
    ax4.set_xlabel('log₁₀(μ)')
    ax4.set_ylabel('log₁₀(G_geo)')
    ax4.set_title('Successful Parameter Combinations')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_scan_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to prevent blocking

# ── 4. Main Execution ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Fine Parameter Scan for Joint (μ, G_geo) Optimization")
    print("="*60)
    
    # Run comprehensive scan
    results = run_comprehensive_parameter_scan()
    
    print("\n🎯 Parameter scan completed!")
    print("Check parameter_scan_comprehensive_results.json for detailed results")
    print("Check parameter_scan_heatmap.png for visualization")
