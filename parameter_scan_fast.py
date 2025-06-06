#!/usr/bin/env python3
"""
FAST PARAMETER SCAN - Two-Stage Optimization Strategy
====================================================

Implements a high-performance two-stage parameter scanning approach:
1. Coarse scan: 48 combinations with N=400, DE(popsize=8, maxiter=150) in parallel
2. Full polish: Top 3 candidates with N=800, DE(popsize=12, maxiter=300)

Expected speedup: ~30× faster than serial approach (2.5 min vs 72 min)

Author: Advanced Warp Bubble Optimizer
Date: June 2025
"""

import numpy as np
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import os

warnings.filterwarnings('ignore')

# ── LIGHTWEIGHT GAUSSIAN OPTIMIZER ────────────────────────────────────────────
def optimize_gaussian_fast(mu, G_geo, grid_N=400, popsize=8, maxiter=150):
    """
    Fast 6-Gaussian optimizer with reduced resolution for parameter scanning.
    
    Args:
        mu: Polymer length parameter
        G_geo: Van den Broeck-Natário geometric factor
        grid_N: Grid resolution (reduced for speed)
        popsize: DE population size (reduced for speed)
        maxiter: DE max iterations (reduced for speed)
    
    Returns:
        dict: {'success': bool, 'energy_J': float, 'params': array, 'mu': float, 'G_geo': float}
    """
    from scipy.optimize import differential_evolution
    
    try:
        # Physical constants
        beta_back = 1.9443254780147017
        v = 1.0
        R = 1.0
        c = 299792458
        G = 6.67430e-11
        tau = 1e-9
        c4_8piG = c**4 / (8.0 * np.pi * G)
        
        # Fast grid setup
        r_grid = np.linspace(0.0, R, grid_N)
        dr = r_grid[1] - r_grid[0]
        vol_weights = 4.0 * np.pi * r_grid**2
        M_gauss = 6
        
        def f_6g_fast(r, p):
            """Fast 6-Gaussian profile evaluation"""
            profile = np.zeros_like(r)
            for i in range(M_gauss):
                A_i = p[3*i]
                sigma_i = max(p[3*i + 1], 0.01*R)  # Ensure positive width
                r0_i = p[3*i + 2]
                profile += A_i * np.exp(-0.5 * ((r - r0_i) / sigma_i)**2)
            return profile
        
        def f_6g_prime_fast(r, p):
            """Fast derivative computation"""
            df = np.zeros_like(r)
            for i in range(M_gauss):
                A_i = p[3*i]
                sigma_i = max(p[3*i + 1], 0.01*R)
                r0_i = p[3*i + 2]
                gaussian = A_i * np.exp(-0.5 * ((r - r0_i) / sigma_i)**2)
                df += gaussian * (-(r - r0_i) / sigma_i**2)
            return df
        
        def E_neg_fast(p):
            """Fast energy evaluation with vectorized integration"""
            # Compute profile and derivatives
            f_val = f_6g_fast(r_grid, p)
            df_dr = f_6g_prime_fast(r_grid, p)
            
            # QI calculation with reduced precision
            f_at_R = f_val[-1] if len(f_val) > 0 else 0
            f_R_desired = 0.0
            QI = beta_back * abs(f_at_R - f_R_desired) / tau
            
            # Energy density (simplified calculation for speed)
            T_rr = (v**2 * df_dr**2) / (2 * r_grid + 1e-12)
            
            # Fast integration
            E_negative = c4_8piG * np.sum(T_rr * vol_weights) * dr
            
            # Quantum improvement term
            if mu > 0 and G_geo > 0:
                delta_E_qi = -QI * mu * G_geo * c**2
                E_negative += delta_E_qi
            
            return E_negative
        
        def penalty_fast(p):
            """Fast penalty computation"""
            penalty = 0.0
            
            # QI constraint (simplified)
            f_val = f_6g_fast(r_grid, p)
            f_at_R = f_val[-1] if len(f_val) > 0 else 0
            QI = beta_back * abs(f_at_R) / tau
            if QI > 1e36:
                penalty += 1e50 * (QI - 1e36)**2
            
            # Bounds constraints
            for i in range(M_gauss):
                A_i = p[3*i]
                sigma_i = p[3*i + 1]
                r0_i = p[3*i + 2]
                
                # Amplitude bounds
                if A_i < 0 or A_i > 1:
                    penalty += 1e4 * max(0, -A_i)**2 + 1e4 * max(0, A_i - 1)**2
                
                # Width bounds
                if sigma_i < 0.01*R or sigma_i > 0.5*R:
                    penalty += 1e4 * max(0, 0.01*R - sigma_i)**2 + 1e4 * max(0, sigma_i - 0.5*R)**2
                
                # Position bounds
                if r0_i < 0 or r0_i > R:
                    penalty += 1e4 * max(0, -r0_i)**2 + 1e4 * max(0, r0_i - R)**2
            
            return penalty
        
        def objective_fast(p):
            """Fast objective function"""
            return E_neg_fast(p) + penalty_fast(p)
        
        # Fast optimization bounds
        bounds = []
        for i in range(M_gauss):
            bounds.extend([
                (0.0, 1.0),      # Amplitude
                (0.01*R, 0.5*R), # Width
                (0.0, R)         # Position
            ])
        
        # Fast differential evolution
        result = differential_evolution(
            objective_fast,
            bounds,
            popsize=popsize,
            maxiter=maxiter,
            seed=42,
            workers=1,  # Single worker per process to avoid nested parallelism
            atol=1e-6,
            tol=1e-6
        )
        
        if result.success:
            energy_J = E_neg_fast(result.x)
            return {
                'success': True,
                'energy_J': float(energy_J),
                'params': result.x.tolist(),
                'mu': float(mu),
                'G_geo': float(G_geo),
                'iterations': result.nit,
                'function_evals': result.nfev
            }
        else:
            return {
                'success': False,
                'energy_J': np.nan,
                'params': None,
                'mu': float(mu),
                'G_geo': float(G_geo),
                'error': 'Optimization failed'
            }
    
    except Exception as e:
        return {
            'success': False,
            'energy_J': np.nan,
            'params': None,
            'mu': float(mu),
            'G_geo': float(G_geo),
            'error': str(e)
        }

def optimize_gaussian_full(mu, G_geo, grid_N=800, popsize=12, maxiter=300):
    """
    Full-resolution 6-Gaussian optimizer for final polishing.
    
    Args:
        mu: Polymer length parameter
        G_geo: Van den Broeck-Natário geometric factor
        grid_N: Full grid resolution
        popsize: Full DE population size
        maxiter: Full DE max iterations
    
    Returns:
        dict: Full optimization result
    """
    from scipy.optimize import differential_evolution, minimize
    
    try:
        # Physical constants
        beta_back = 1.9443254780147017
        v = 1.0
        R = 1.0
        c = 299792458
        G = 6.67430e-11
        tau = 1e-9
        c4_8piG = c**4 / (8.0 * np.pi * G)
        
        # Full grid setup
        r_grid = np.linspace(0.0, R, grid_N)
        dr = r_grid[1] - r_grid[0]
        vol_weights = 4.0 * np.pi * r_grid**2
        M_gauss = 6
        
        def f_6g_full(r, p):
            """Full 6-Gaussian profile evaluation"""
            profile = np.zeros_like(r)
            for i in range(M_gauss):
                A_i = p[3*i]
                sigma_i = max(p[3*i + 1], 0.01*R)
                r0_i = p[3*i + 2]
                profile += A_i * np.exp(-0.5 * ((r - r0_i) / sigma_i)**2)
            return profile
        
        def f_6g_prime_full(r, p):
            """Full derivative computation"""
            df = np.zeros_like(r)
            for i in range(M_gauss):
                A_i = p[3*i]
                sigma_i = max(p[3*i + 1], 0.01*R)
                r0_i = p[3*i + 2]
                gaussian = A_i * np.exp(-0.5 * ((r - r0_i) / sigma_i)**2)
                df += gaussian * (-(r - r0_i) / sigma_i**2)
            return df
        
        def E_neg_full(p):
            """Full energy evaluation"""
            # Compute profile and derivatives
            f_val = f_6g_full(r_grid, p)
            df_dr = f_6g_prime_full(r_grid, p)
            
            # Full QI calculation
            f_at_R = f_val[-1] if len(f_val) > 0 else 0
            f_R_desired = 0.0
            QI = beta_back * abs(f_at_R - f_R_desired) / tau
            
            # Full energy density calculation
            T_rr = (v**2 * df_dr**2) / (2 * r_grid + 1e-12)
            
            # Enhanced curvature terms
            d2f_dr2 = np.gradient(df_dr, dr)
            T_rr += (v**2 * f_val * d2f_dr2) / (r_grid + 1e-12)
            T_rr += (v**2 * f_val * df_dr) / (r_grid**2 + 1e-12)
            
            # Full integration
            E_negative = c4_8piG * np.sum(T_rr * vol_weights) * dr
            
            # Quantum improvement term
            if mu > 0 and G_geo > 0:
                delta_E_qi = -QI * mu * G_geo * c**2
                E_negative += delta_E_qi
            
            return E_negative
        
        def penalty_full(p):
            """Full penalty computation with enhanced constraints"""
            penalty = 0.0
            
            # QI constraint
            f_val = f_6g_full(r_grid, p)
            f_at_R = f_val[-1] if len(f_val) > 0 else 0
            QI = beta_back * abs(f_at_R) / tau
            if QI > 1e36:
                penalty += 1e50 * (QI - 1e36)**2
            
            # Enhanced physics constraints
            df_dr = f_6g_prime_full(r_grid, p)
            d2f_dr2 = np.gradient(df_dr, dr)
            
            # Curvature penalty
            penalty += 1e3 * np.mean(d2f_dr2**2)
            
            # Smoothness penalty
            penalty += 1e2 * np.mean(np.diff(df_dr)**2)
            
            # Bounds constraints
            for i in range(M_gauss):
                A_i = p[3*i]
                sigma_i = p[3*i + 1]
                r0_i = p[3*i + 2]
                
                if A_i < 0 or A_i > 1:
                    penalty += 1e4 * max(0, -A_i)**2 + 1e4 * max(0, A_i - 1)**2
                if sigma_i < 0.01*R or sigma_i > 0.5*R:
                    penalty += 1e4 * max(0, 0.01*R - sigma_i)**2 + 1e4 * max(0, sigma_i - 0.5*R)**2
                if r0_i < 0 or r0_i > R:
                    penalty += 1e4 * max(0, -r0_i)**2 + 1e4 * max(0, r0_i - R)**2
            
            return penalty
        
        def objective_full(p):
            """Full objective function"""
            return E_neg_full(p) + penalty_full(p)
        
        # Full optimization bounds
        bounds = []
        for i in range(M_gauss):
            bounds.extend([
                (0.0, 1.0),      # Amplitude
                (0.01*R, 0.5*R), # Width
                (0.0, R)         # Position
            ])
        
        # Full differential evolution
        result_de = differential_evolution(
            objective_full,
            bounds,
            popsize=popsize,
            maxiter=maxiter,
            seed=42,
            workers=1,
            atol=1e-8,
            tol=1e-8
        )
        
        # L-BFGS-B refinement
        if result_de.success:
            result_refine = minimize(
                objective_full,
                result_de.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if result_refine.success and result_refine.fun < result_de.fun:
                final_params = result_refine.x
                final_energy = E_neg_full(final_params)
            else:
                final_params = result_de.x
                final_energy = E_neg_full(final_params)
        else:
            return {
                'success': False,
                'energy_J': np.nan,
                'params': None,
                'mu': float(mu),
                'G_geo': float(G_geo),
                'error': 'DE optimization failed'
            }
        
        return {
            'success': True,
            'energy_J': float(final_energy),
            'params': final_params.tolist(),
            'mu': float(mu),
            'G_geo': float(G_geo),
            'de_iterations': result_de.nit,
            'de_function_evals': result_de.nfev,
            'refinement_success': result_refine.success if 'result_refine' in locals() else False
        }
    
    except Exception as e:
        return {
            'success': False,
            'energy_J': np.nan,
            'params': None,
            'mu': float(mu),
            'G_geo': float(G_geo),
            'error': str(e)
        }

# ── MAIN SCANNING PIPELINE ────────────────────────────────────────────────────
def run_fast_parameter_scan():
    """Execute the two-stage fast parameter scan"""
    
    print("🚀 FAST PARAMETER SCAN - Two-Stage Strategy")
    print("=" * 60)
    
    # ── 1. Build the scan grid ────────────────────────────────────────────────
    mu_list = np.logspace(-8, -3, 8)    # 8 values: 1e-8 … 1e-3
    Gg_list = np.logspace(-7, -3, 6)    # 6 values: 1e-7 … 1e-3
    
    # All combinations
    combos = [(mu, Gg) for mu in mu_list for Gg in Gg_list]
    
    print(f"Parameter scan configuration:")
    print(f"  μ range: {mu_list}")
    print(f"  G_geo range: {Gg_list}")
    print(f"  Total combinations: {len(combos)}")
    print(f"  Using 12 parallel workers")
    print()
    
    # ── 2. Execute the coarse scan in parallel ────────────────────────────────
    print("Stage 1: Coarse scan (N=400, DE popsize=8, maxiter=150)")
    print("-" * 50)
    
    results_coarse = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=12) as exe:
        # Submit all jobs
        futures = {
            exe.submit(optimize_gaussian_fast, mu, Gg, 400, 8, 150): (mu, Gg)
            for mu, Gg in combos
        }
        
        completed = 0
        for fut in as_completed(futures):
            mu, Gg = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    'success': False, 
                    'energy_J': np.nan, 
                    'params': None,
                    'mu': mu, 
                    'G_geo': Gg, 
                    'error': str(e)
                }
            
            results_coarse.append(res)
            completed += 1
            
            status = "✓" if res.get('success', False) else "✗"
            energy_str = f"{res.get('energy_J', np.nan):.3e}" if not np.isnan(res.get('energy_J', np.nan)) else "FAILED"
            
            print(f"  [{completed:2d}/48] μ={mu:.1e}, G_geo={Gg:.1e} → E₋ = {energy_str} J  {status}")
    
    end_time = time.time()
    coarse_time = end_time - start_time
    print(f"\nCoarse scan completed in {coarse_time:.1f}s")
    
    # ── 3. Pick Top-K for full polish ─────────────────────────────────────────
    # Sort by energy_J (most negative first)
    valid_results = [r for r in results_coarse if r.get('success', False)]
    
    if not valid_results:
        print("❌ No successful runs in coarse scan—exiting.")
        return None, None
    
    print(f"\n✅ {len(valid_results)}/{len(combos)} combinations successful in coarse scan")
    
    # Extract the 3 best
    valid_results.sort(key=lambda r: r['energy_J'])
    top_k = valid_results[:3]
    
    print("\nTop 3 candidates from coarse scan:")
    for idx, r in enumerate(top_k, 1):
        print(f"  {idx}) μ={r['mu']:.1e}, G_geo={r['G_geo']:.1e}, E₋={r['energy_J']:.3e} J")
    
    # ── 4. Full-Resolution Polish on Top-K ────────────────────────────────────
    print(f"\nStage 2: Full polish (N=800, DE popsize=12, maxiter=300)")
    print("-" * 50)
    
    results_full = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=3) as exe:
        futures_full = {
            exe.submit(optimize_gaussian_full, r['mu'], r['G_geo'], 800, 12, 300): r
            for r in top_k
        }
        
        for fut in as_completed(futures_full):
            r0 = futures_full[fut]
            try:
                r_full = fut.result()
            except Exception as e:
                r_full = {
                    'success': False, 
                    'energy_J': np.nan, 
                    'params': None,
                    'mu': r0['mu'], 
                    'G_geo': r0['G_geo'], 
                    'error': str(e)
                }
            
            results_full.append(r_full)
            
            status = "✓" if r_full.get('success', False) else "✗"
            energy_str = f"{r_full.get('energy_J', np.nan):.3e}" if not np.isnan(r_full.get('energy_J', np.nan)) else "FAILED"
            
            # Compare with coarse result
            coarse_energy = r0['energy_J']
            improvement = ""
            if r_full.get('success', False) and not np.isnan(r_full.get('energy_J', np.nan)):
                improvement_factor = abs(r_full['energy_J'] / coarse_energy)
                improvement = f" ({improvement_factor:.3f}× refined)"
            
            print(f"  Polished μ={r_full['mu']:.1e}, G_geo={r_full['G_geo']:.1e} → E₋ = {energy_str} J  {status}{improvement}")
    
    end_time = time.time()
    polish_time = end_time - start_time
    total_time = coarse_time + polish_time
    
    print(f"\nFull polish completed in {polish_time:.1f}s")
    print(f"Total scan time: {total_time:.1f}s (~{total_time/60:.1f} minutes)")
    
    # ── 5. Final Results Analysis ─────────────────────────────────────────────
    successful_full = [r for r in results_full if r.get('success', False)]
    
    if successful_full:
        best_result = min(successful_full, key=lambda r: r['energy_J'])
        
        print(f"\n🎯 BEST RESULT:")
        print(f"  μ = {best_result['mu']:.3e}")
        print(f"  G_geo = {best_result['G_geo']:.3e}")
        print(f"  E₋ = {best_result['energy_J']:.6e} J")
        
        # Compare with baseline
        baseline_energy = -1.95e31  # Previous best estimate
        if best_result['energy_J'] < baseline_energy:
            improvement_factor = abs(best_result['energy_J'] / baseline_energy)
            print(f"  🚀 {improvement_factor:.3f}× improvement over baseline!")
        
    else:
        print("❌ No successful full optimizations")
        best_result = None
    
    # ── 6. Save All Results ───────────────────────────────────────────────────
    all_results = {
        'scan_metadata': {
            'mu_range': mu_list.tolist(),
            'G_geo_range': Gg_list.tolist(),
            'total_combinations': len(combos),
            'coarse_time_s': coarse_time,
            'polish_time_s': polish_time,
            'total_time_s': total_time
        },
        'coarse_scan': results_coarse,
        'full_polish': results_full,
        'best_result': best_result
    }
    
    with open('fast_parameter_scan_results.json', 'w') as fp:
        json.dump(all_results, fp, indent=2)
    
    print(f"\n💾 Results saved to: fast_parameter_scan_results.json")
    
    return all_results, best_result

def main():
    """Main execution"""
    try:
        all_results, best_result = run_fast_parameter_scan()
        
        if best_result:
            print(f"\n🏁 Fast parameter scan completed successfully!")
            print(f"   Best energy: {best_result['energy_J']:.6e} J")
            print(f"   Optimal μ: {best_result['mu']:.3e}")
            print(f"   Optimal G_geo: {best_result['G_geo']:.3e}")
        else:
            print(f"\n❌ Parameter scan failed to find valid solutions")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Scan interrupted by user")
    except Exception as e:
        print(f"\n❌ Scan failed with error: {e}")

if __name__ == "__main__":
    main()
