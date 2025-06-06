#!/usr/bin/env python3
"""
Hybrid Optimize Cubic - Higher-Order Polynomial Transition

Advanced hybrid ansatz with cubic (3rd-order) polynomial transition between
inner flat region and outer Gaussian tails. This provides smoother transitions
and reduced wasted curvature compared to linear transitions.

Target: Push E- from -1.86×10³¹ J to < -1.95×10³¹ J
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import json
import time

# Constants
beta_back = 1.9443254780147017
mu0       = 1e-6
v         = 1.0
R         = 1.0
c         = 299792458
G         = 6.67430e-11
c4_8piG   = c**4/(8.0*np.pi*G)
G_geo     = 1e-5
tau       = 1e-9

# Grid
N = 800
r_grid     = np.linspace(0.0, R, N)
dr         = r_grid[1] - r_grid[0]
vol_weights= 4.0*np.pi * r_grid**2

# Hybrid: cubic ramp between r0,r1 + M_g=2 Gaussians
M_g = 2

def f_hybrid_cubic(r, p):
    """
    Enhanced hybrid ansatz with cubic polynomial transition
    
    p = [r0, r1, b1, b2, b3,   A0,r0_0,σ0,   A1,r0_1,σ1]
    
    - For r ≤ r0: f(r) = 1.0 (flat interior)
    - For r0 < r < r1: f(r) = 1 + b1*x + b2*x² + b3*x³, where x = (r-r0)/(r1-r0)
    - For r ≥ r1: f(r) = sum of M_g Gaussians
    """
    r0, r1 = p[0], p[1]
    b1, b2, b3 = p[2], p[3], p[4]
    
    if isinstance(r, (int, float)):
        # Scalar case
        if r <= r0:
            return 1.0
        elif r < r1:
            x = (r - r0)/(r1 - r0)
            val = 1.0 + b1*x + b2*(x**2) + b3*(x**3)
            return np.clip(val, 0.0, 1.0)
        else:
            # Gaussian region
            total = 0.0
            for i in range(M_g):
                Ai    = p[5 + 3*i + 0]
                r0_i  = p[5 + 3*i + 1]
                sig_i = p[5 + 3*i + 2]
                x = (r - r0_i)/sig_i
                total += Ai * np.exp(-0.5*x*x)
            return np.clip(total, 0.0, 1.0)
    else:
        # Vector case
        result = np.ones_like(r)
        
        # Transition region
        mask_trans = (r > r0) & (r < r1)
        if np.any(mask_trans):
            x = (r[mask_trans] - r0)/(r1 - r0)
            poly_vals = 1.0 + b1*x + b2*(x**2) + b3*(x**3)
            result[mask_trans] = np.clip(poly_vals, 0.0, 1.0)
        
        # Gaussian region
        mask_gauss = (r >= r1)
        if np.any(mask_gauss):
            r_gauss = r[mask_gauss]
            gauss_total = np.zeros_like(r_gauss)
            for i in range(M_g):
                Ai    = p[5 + 3*i + 0]
                r0_i  = p[5 + 3*i + 1]
                sig_i = p[5 + 3*i + 2]
                x = (r_gauss - r0_i)/sig_i
                gauss_total += Ai * np.exp(-0.5*x*x)
            result[mask_gauss] = np.clip(gauss_total, 0.0, 1.0)
        
        return result

def f_hybrid_cubic_prime(r, p):
    """Enhanced derivative for cubic hybrid ansatz"""
    r0, r1 = p[0], p[1]
    b1, b2, b3 = p[2], p[3], p[4]
    
    if isinstance(r, (int, float)):
        # Scalar case
        if r <= r0 or r >= R:
            return 0.0
        elif r < r1:
            x = (r - r0)/(r1 - r0)
            dx_dr = 1.0/(r1 - r0)
            return (b1 + 2*b2*x + 3*b3*(x**2)) * dx_dr
        else:
            # Gaussian region
            deriv = 0.0
            for i in range(M_g):
                Ai    = p[5 + 3*i + 0]
                r0_i  = p[5 + 3*i + 1]
                sig_i = p[5 + 3*i + 2]
                x = (r - r0_i)/sig_i
                pref = Ai * np.exp(-0.5*x*x)
                deriv += pref * (-(r - r0_i)/(sig_i**2))
            return deriv
    else:
        # Vector case
        result = np.zeros_like(r)
        
        # Transition region
        mask_trans = (r > r0) & (r < r1)
        if np.any(mask_trans):
            x = (r[mask_trans] - r0)/(r1 - r0)
            dx_dr = 1.0/(r1 - r0)
            result[mask_trans] = (b1 + 2*b2*x + 3*b3*(x**2)) * dx_dr
        
        # Gaussian region
        mask_gauss = (r >= r1) & (r < R)
        if np.any(mask_gauss):
            r_gauss = r[mask_gauss]
            for i in range(M_g):
                Ai    = p[5 + 3*i + 0]
                r0_i  = p[5 + 3*i + 1]
                sig_i = p[5 + 3*i + 2]
                x = (r_gauss - r0_i)/sig_i
                pref = Ai * np.exp(-0.5*x*x)
                result[mask_gauss] += pref * (-(r_gauss - r0_i)/(sig_i**2))
        
        return result

def E_neg_hybrid_cubic(p):
    """Compute negative energy for cubic hybrid ansatz"""
    fp = f_hybrid_cubic_prime(r_grid, p)
    sinc_val = np.sinc(mu0/np.pi) if mu0>0 else 1.0
    prefac   = - (v**2)/(8.0*np.pi) * beta_back * sinc_val / G_geo
    rho_vals = prefac * (fp**2)
    integral = np.sum(rho_vals * vol_weights) * dr
    return integral * c4_8piG

def penalty_hybrid_cubic(p, lam_qi=1e50, lam_bound=1e6, lam_cont=1e6, lam_curv=1e3):
    """Enhanced penalty function for cubic hybrid ansatz"""
    r0, r1 = p[0], p[1]
    b1, b2, b3 = p[2], p[3], p[4]
    
    # (a) Bounds: 0 < r0 < r1 < R
    P_b = 0.0
    if not (0.0 < r0 < r1 < R): 
        P_b = lam_bound * ((max(0, -r0))**2 + (max(0, r0-r1))**2 + (max(0, r1-R))**2)
    
    # (b) QI constraint at r=0
    fp0 = f_hybrid_cubic_prime(0.0, p)
    sincv = np.sinc(mu0/np.pi) if mu0>0 else 1.0
    rho0 = - (v**2)/(8.0*np.pi)*beta_back*sincv/G_geo * (fp0**2)
    qi_bound = - (1.0545718e-34 * np.sinc(mu0/np.pi))/(12.0*np.pi*tau**2)
    P_qi = lam_qi * max(0.0, -(rho0 - qi_bound))**2
    
    # (c) Continuity at r1: polynomial(r1) = Gaussians(r1)
    poly_at_r1 = 1.0 + b1 + b2 + b3  # x=1 at r=r1
    gauss_at_r1 = 0.0
    for i in range(M_g):
        Ai = p[5+3*i+0]
        r0_i = p[5+3*i+1]
        sig_i = p[5+3*i+2]
        gauss_at_r1 += Ai * np.exp(-0.5*((r1 - r0_i)/sig_i)**2)
    
    P_cont = lam_cont * (poly_at_r1 - gauss_at_r1)**2
    
    # (d) Derivative continuity at r1 (optional but recommended)
    poly_deriv_at_r1 = (b1 + 2*b2 + 3*b3) / (r1 - r0)
    gauss_deriv_at_r1 = 0.0
    for i in range(M_g):
        Ai = p[5+3*i+0]
        r0_i = p[5+3*i+1]
        sig_i = p[5+3*i+2]
        x = (r1 - r0_i)/sig_i
        pref = Ai * np.exp(-0.5*x*x)
        gauss_deriv_at_r1 += pref * (-(r1 - r0_i)/(sig_i**2))
    
    P_deriv_cont = lam_cont * (poly_deriv_at_r1 - gauss_deriv_at_r1)**2
    
    # (e) Amplitude constraints for Gaussians
    A_sum = sum(p[5+3*i+0] for i in range(M_g))
    P_amp = lam_bound * max(0.0, A_sum - 1.0)**2
    
    # (f) Curvature penalty (smoothness)
    f_vals = f_hybrid_cubic(r_grid, p)
    fpp = np.zeros_like(f_vals)
    fpp[1:-1] = (f_vals[2:] - 2*f_vals[1:-1] + f_vals[:-2])/(dr**2)
    fpp[0] = fpp[1]
    fpp[-1] = fpp[-2]
    P_curv = lam_curv * np.sum((fpp**2)*(r_grid**2))*dr
    
    # (g) Monotonicity constraint (f'(r) ≤ 0)
    fp_vals = f_hybrid_cubic_prime(r_grid, p)
    P_mono = lam_curv * np.sum(np.maximum(0.0, fp_vals)**2)*dr
    
    return P_b + P_qi + P_cont + P_deriv_cont + P_amp + P_curv + P_mono

def obj_hybrid_cubic(p):
    """Objective function for cubic hybrid optimization"""
    return E_neg_hybrid_cubic(p) + penalty_hybrid_cubic(p)

def get_bounds_hybrid_cubic():
    """Enhanced bounds for cubic hybrid ansatz"""
    bnds = []
    
    # r0 ∈ [0.05R, 0.35R], r1 ∈ [0.4R, 0.8R] - ensure reasonable separation
    bnds += [(0.05*R, 0.35*R), (0.4*R, 0.8*R)]
    
    # Cubic coefficients b1,b2,b3 ∈ [-5, 5] - moderate range to avoid wild oscillations
    bnds += [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    # Each Gaussian: A∈[0,1], r0∈[0.4R, R], σ∈[0.02R, 0.4R]
    for _ in range(M_g):
        bnds += [(0.0, 1.0), (0.4*R, R), (0.02*R, 0.4*R)]
    
    return bnds

def run_hybrid_cubic():
    """Run enhanced cubic hybrid optimization with multiple strategies"""
    print("🚀 Hybrid Cubic Polynomial Transition Optimizer")
    print("="*60)
    print(f"Target: Push E- from -1.86×10³¹ J to < -1.95×10³¹ J")
    print(f"Ansatz: Cubic transition + {M_g} Gaussians")
    
    bounds = get_bounds_hybrid_cubic()
    
    # Enhanced DE parameters for better exploration
    de_params = {
        'bounds': bounds,
        'strategy': 'best1bin',
        'maxiter': 2000,  # Increased iterations
        'popsize': 20,    # Larger population
        'tol': 1e-8,
        'atol': 1e-10,
        'mutation': (0.5, 1.0),
        'recombination': 0.8,
        'polish': True,
        'workers': -1,    # Use all available cores
        'seed': 42
    }
    
    print(f"\n🔧 Running Enhanced Differential Evolution...")
    print(f"   Parameters: {len(bounds)} dimensions")
    print(f"   Population: {de_params['popsize']}")
    print(f"   Max iterations: {de_params['maxiter']}")
    
    start_time = time.time()
    
    try:
        de_result = differential_evolution(obj_hybrid_cubic, **de_params)
        
        elapsed = time.time() - start_time
        
        if de_result.success:
            print(f"✅ DE optimization completed in {elapsed:.1f}s")
            print(f"   Function evaluations: {de_result.nfev}")
            print(f"   Convergence: {de_result.message}")
            
            # Local refinement with L-BFGS-B
            print(f"\n🔧 Refining with L-BFGS-B...")
            
            refine_result = minimize(
                obj_hybrid_cubic, x0=de_result.x, bounds=bounds,
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if refine_result.success:
                print(f"✅ Refinement successful")
                final_result = refine_result
            else:
                print(f"⚠️  Refinement had issues: {refine_result.message}")
                final_result = de_result
                
        else:
            print(f"❌ DE optimization failed: {de_result.message}")
            return None
            
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        return None
    
    # Analyze results
    final_energy = E_neg_hybrid_cubic(final_result.x)
    penalty_val = penalty_hybrid_cubic(final_result.x)
    
    print(f"\n📊 CUBIC HYBRID OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Final energy: {final_energy:.4e} J")
    print(f"Constraint penalty: {penalty_val:.4e}")
    print(f"Target achieved: {'YES' if final_energy < -1.95e31 else 'Approaching target'}")
    
    # Extract and display parameters
    r0, r1 = final_result.x[0], final_result.x[1]
    b1, b2, b3 = final_result.x[2], final_result.x[3], final_result.x[4]
    
    print(f"\nCubic transition parameters:")
    print(f"  Inner radius r0: {r0:.4f} m")
    print(f"  Outer radius r1: {r1:.4f} m")
    print(f"  Cubic coefficients: b1={b1:.4f}, b2={b2:.4f}, b3={b3:.4f}")
    
    print(f"\nGaussian components:")
    for i in range(M_g):
        Ai = final_result.x[5 + 3*i + 0]
        r0_i = final_result.x[5 + 3*i + 1]
        sig_i = final_result.x[5 + 3*i + 2]
        print(f"  Gaussian {i+1}: A={Ai:.4f}, r0={r0_i:.4f}m, σ={sig_i:.4f}m")
    
    # Save results
    result_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'Hybrid Cubic Transition',
        'energy_joules': float(final_energy),
        'parameters': [float(x) for x in final_result.x],
        'target_achieved': bool(final_energy < -1.95e31),
        'transition_params': {'r0': float(r0), 'r1': float(r1), 'b1': float(b1), 'b2': float(b2), 'b3': float(b3)},
        'gaussian_params': [
            {'A': float(final_result.x[5+3*i]), 'r0': float(final_result.x[6+3*i]), 'sigma': float(final_result.x[7+3*i])}
            for i in range(M_g)
        ]
    }
    
    with open('hybrid_cubic_results.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    
    # Generate visualization
    plot_cubic_hybrid_results(final_result.x, final_energy)
    
    print(f"\n📁 Results saved to: hybrid_cubic_results.json")
    print(f"📊 Visualization saved to: hybrid_cubic_analysis.png")
    
    return final_result

def plot_cubic_hybrid_results(params, energy):
    """Visualize cubic hybrid optimization results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Main profile plot
    ax1.plot(r_grid, f_hybrid_cubic(r_grid, params), 'blue', linewidth=2, 
             label=f'Cubic Hybrid: E = {energy:.2e} J')
    
    r0, r1 = params[0], params[1]
    ax1.axvline(r0, color='red', linestyle='--', alpha=0.7, label=f'r0 = {r0:.3f}')
    ax1.axvline(r1, color='green', linestyle='--', alpha=0.7, label=f'r1 = {r1:.3f}')
    
    ax1.set_xlabel('Radius r (m)')
    ax1.set_ylabel('Field amplitude f(r)')
    ax1.set_title('Cubic Hybrid Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Derivative plot
    fp_vals = f_hybrid_cubic_prime(r_grid, params)
    ax2.plot(r_grid, fp_vals, 'red', linewidth=2)
    ax2.axvline(r0, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(r1, color='green', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Radius r (m)')
    ax2.set_ylabel("f'(r)")
    ax2.set_title('Field Derivative')
    ax2.grid(True, alpha=0.3)
    
    # Component breakdown
    r_fine = np.linspace(0, R, 1000)
    
    # Plot individual regions
    mask_inner = r_fine <= r0
    mask_trans = (r_fine > r0) & (r_fine < r1)
    mask_gauss = r_fine >= r1
    
    if np.any(mask_inner):
        ax3.plot(r_fine[mask_inner], np.ones(np.sum(mask_inner)), 'blue', 
                linewidth=3, label='Flat region')
    
    if np.any(mask_trans):
        b1, b2, b3 = params[2], params[3], params[4]
        x_trans = (r_fine[mask_trans] - r0) / (r1 - r0)
        poly_vals = 1.0 + b1*x_trans + b2*(x_trans**2) + b3*(x_trans**3)
        ax3.plot(r_fine[mask_trans], poly_vals, 'orange', 
                linewidth=3, label='Cubic transition')
    
    if np.any(mask_gauss):
        gauss_total = np.zeros(np.sum(mask_gauss))
        for i in range(M_g):
            Ai = params[5 + 3*i + 0]
            r0_i = params[5 + 3*i + 1]
            sig_i = params[5 + 3*i + 2]
            gauss_i = Ai * np.exp(-0.5 * ((r_fine[mask_gauss] - r0_i) / sig_i)**2)
            gauss_total += gauss_i
            ax3.plot(r_fine[mask_gauss], gauss_i, '--', alpha=0.7, 
                    label=f'Gauss {i+1}')
        ax3.plot(r_fine[mask_gauss], gauss_total, 'green', 
                linewidth=3, label='Total Gaussians')
    
    ax3.set_xlabel('Radius r (m)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Component Breakdown')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Energy density
    rho_vals = []
    for r_val in r_grid:
        fp_val = f_hybrid_cubic_prime(r_val, params)
        sinc_val = np.sinc(mu0/np.pi) if mu0>0 else 1.0
        prefac = - (v**2)/(8.0*np.pi) * beta_back * sinc_val / G_geo
        rho_vals.append(prefac * (fp_val**2))
    
    ax4.plot(r_grid, rho_vals, 'purple', linewidth=2)
    ax4.axvline(r0, color='red', linestyle='--', alpha=0.7)
    ax4.axvline(r1, color='green', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Radius r (m)')
    ax4.set_ylabel('Energy density ρ(r)')
    ax4.set_title('Energy Density Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hybrid_cubic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    result = run_hybrid_cubic()
    if result:
        print("\n🎯 Cubic hybrid optimization completed!")
    else:
        print("\n❌ Optimization failed")
