#!/usr/bin/env python3
"""
Enhanced soliton ansatz optimizer with improved robustness
Based on debug findings, using better initial conditions and constraints
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution
import json
import matplotlib.pyplot as plt
import os

# Physical constants
hbar = 1.0545718e-34  # J⋅s
c = 299792458         # m/s
G = 6.67430e-11       # m³/(kg⋅s²)
v = 1.0               # Warp velocity (fraction of c)
R = 1.0               # Characteristic length scale (m)

# Derived constants
c4_8piG = c**4 / (8.0 * np.pi * G)  # ≈ 4.815×10⁴² J⋅m⁻³

# Soliton configuration
M_soliton = 2  # Number of soliton terms

def f_soliton(r, params):
    """Enhanced soliton ansatz with robust numerical handling"""
    if len(params) != 3 * M_soliton:
        raise ValueError(f"Expected {3*M_soliton} parameters, got {len(params)}")
    
    total = 0.0
    for i in range(M_soliton):
        Ai = params[3*i + 0]
        r0_i = params[3*i + 1] 
        sig_i = max(params[3*i + 2], 1e-8)  # Prevent division by zero
        
        x = (r - r0_i) / sig_i
        # Use robust sech² calculation
        if abs(x) > 20:  # Prevent numerical overflow
            sech2 = 0.0
        else:
            cosh_x = np.cosh(x)
            sech2 = 1.0 / (cosh_x * cosh_x)
        
        total += Ai * sech2
    
    return np.clip(total, 0.0, 1.0)  # Enforce physical bounds

def f_soliton_prime(r, params):
    """Enhanced derivative with robust numerical handling"""
    deriv = 0.0
    for i in range(M_soliton):
        Ai = params[3*i + 0]
        r0_i = params[3*i + 1]
        sig_i = max(params[3*i + 2], 1e-8)
        
        x = (r - r0_i) / sig_i
        if abs(x) > 20:  # Prevent numerical overflow
            continue
            
        cosh_x = np.cosh(x)
        tanh_x = np.tanh(x)
        sech2 = 1.0 / (cosh_x * cosh_x)
        
        deriv += Ai * (-2.0) * sech2 * tanh_x / sig_i
    
    return deriv

def calculate_enhancement_factors(mu, R_ratio):
    """Calculate enhancement factors for given parameters"""
    # Backreaction enhancement (from previous optimization)
    beta_backreaction = 1.9443254780147017
    
    # Polymer sinc function
    sinc_polymer = np.sinc(mu / np.pi) if mu > 0 else 1.0
    
    # Geometric reduction
    geometric_reduction = R_ratio
    
    # Polymer scale factor
    polymer_scale = mu
    
    return {
        'beta_backreaction': beta_backreaction,
        'sinc_polymer': sinc_polymer, 
        'geometric_reduction': geometric_reduction,
        'polymer_scale': polymer_scale
    }

def calculate_total_energy(params, mu=1e-6, R_ratio=1e-5):
    """Calculate total negative energy with enhanced robustness"""
    try:
        # Calculate enhancement factors
        factors = calculate_enhancement_factors(mu, R_ratio)
        
        # Energy density function
        def rho_eff(r):
            if r <= 0:
                return 0.0
            f_val = f_soliton(r, params)
            f_prime_val = f_soliton_prime(r, params)
            
            # Enhanced negative energy density calculation
            rho = -factors['beta_backreaction'] * factors['sinc_polymer'] * \
                  factors['geometric_reduction'] * factors['polymer_scale'] * \
                  c4_8piG * (f_prime_val**2) / (r**2)
                  
            return rho if np.isfinite(rho) else 0.0
        
        # Integrate over the bubble volume with robust error handling
        def integrand(r):
            return 4 * np.pi * r**2 * rho_eff(r)
        
        result, error = quad(integrand, 1e-6, 10*R, 
                           limit=1000, epsabs=1e-12, epsrel=1e-10)
        
        energy = result if np.isfinite(result) else 0.0
        
        # Enforce LQG-modified QI bound
        try:
            from src.warp_qft.stability import enforce_lqg_bound
            # Use travel time estimate
            flight_time = R / c  # Time for light to cross bubble
            energy = enforce_lqg_bound(energy, R, flight_time)
        except ImportError:
            # Fallback for standalone use
            print("⚠️  LQG bound enforcement unavailable - using raw energy")
        
        return energy
        
    except Exception as e:
        print(f"Warning: Energy calculation failed: {e}")
        return 0.0

def objective_function(params, mu=1e-6, R_ratio=1e-5):
    """Enhanced objective function with penalty terms"""
    # Calculate energy (we want to minimize, so negate it)
    energy = calculate_total_energy(params, mu, R_ratio)
    
    # Physical constraint penalties
    penalty = 0.0
    
    # Ensure parameters are physically reasonable
    for i in range(M_soliton):
        Ai = params[3*i + 0]
        r0_i = params[3*i + 1]
        sig_i = params[3*i + 2]
        
        # Penalty for negative amplitudes
        if Ai < 0:
            penalty += 1e6 * abs(Ai)
        
        # Penalty for extreme widths
        if sig_i < 1e-3 or sig_i > 1.0:
            penalty += 1e6 * (abs(sig_i - 0.1)**2)
        
        # Penalty for centers outside reasonable range
        if r0_i < 0 or r0_i > 2*R:
            penalty += 1e6 * (abs(r0_i - R)**2)
    
    # Boundary condition penalties
    f_0 = f_soliton(0, params)
    f_R = f_soliton(R, params)
    
    # Penalty for not satisfying f(0) ≈ 0, f(R) ≈ 0
    penalty += 1e3 * (f_0**2 + f_R**2)
    
    # Penalty for f > 1 anywhere (non-physical)
    r_test = np.linspace(0, 2*R, 100)
    f_test = [f_soliton(r, params) for r in r_test]
    for f_val in f_test:
        if f_val > 1.0:
            penalty += 1e6 * (f_val - 1.0)**2
    
    return -energy + penalty

def optimize_soliton_enhanced(mu=1e-6, R_ratio=1e-5, method='differential_evolution'):
    """Enhanced optimization with multiple strategies"""
    
    print(f"🚀 Starting enhanced soliton optimization")
    print(f"   μ = {mu:.2e}, R_ratio = {R_ratio:.2e}")
    
    # Define bounds based on debug findings
    bounds = []
    for i in range(M_soliton):
        bounds.extend([
            (0.01, 2.0),      # Amplitude Ai
            (0.0, 2*R),       # Center r0_i  
            (0.01, 0.5)       # Width sig_i
        ])
    
    # Multiple starting points based on debug findings
    initial_guesses = [
        [0.1, 0.0, 0.01, 0.1, 0.0, 0.01],      # Debug solution
        [1.0, 0.3, 0.1, 0.5, 0.7, 0.1],        # Original test
        [0.5, 0.2, 0.05, 0.8, 0.8, 0.05],      # Symmetric
        [2.0, 0.1, 0.02, 1.0, 0.9, 0.03],      # High amplitude
    ]
    
    best_result = None
    best_energy = float('inf')
    
    if method == 'differential_evolution':
        # Use differential evolution for global optimization
        print("Using differential evolution...")
        result = differential_evolution(
            lambda x: objective_function(x, mu, R_ratio),
            bounds=bounds,
            maxiter=500,
            popsize=15,
            atol=1e-8,
            seed=42
        )
        
        if result.success:
            energy = calculate_total_energy(result.x, mu, R_ratio)
            return result.x, energy, True
        else:
            print(f"Differential evolution failed: {result.message}")
    
    # Try multiple local optimizations
    print("Trying multiple local optimizations...")
    for i, x0 in enumerate(initial_guesses):
        try:
            result = minimize(
                lambda x: objective_function(x, mu, R_ratio),
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if result.success:
                energy = calculate_total_energy(result.x, mu, R_ratio)
                print(f"  Attempt {i+1}: Energy = {energy:.3e} J")
                
                if energy < best_energy:
                    best_energy = energy
                    best_result = result
                    
        except Exception as e:
            print(f"  Attempt {i+1}: Failed with {e}")
            continue
    
    if best_result is not None:
        return best_result.x, best_energy, True
    else:
        return None, 0.0, False

def run_comprehensive_optimization():
    """Run comprehensive optimization with multiple parameter sets"""
    
    print("🎯 ENHANCED SOLITON OPTIMIZATION")
    print("=" * 60)
    
    # Test parameter combinations
    mu_values = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
    R_ratios = [1e-6, 1e-5, 1e-4]
    
    results = []
    best_overall = None
    best_energy_overall = float('inf')
    
    for mu in mu_values:
        for R_ratio in R_ratios:
            print(f"\n📍 Testing μ={mu:.1e}, R_ratio={R_ratio:.1e}")
            
            params, energy, success = optimize_soliton_enhanced(mu, R_ratio)
            
            if success and energy < 0:  # Valid negative energy
                print(f"✅ Success! Energy = {energy:.3e} J")
                
                result = {
                    'mu': mu,
                    'R_ratio': R_ratio,
                    'parameters': params.tolist(),
                    'energy_J': energy,
                    'enhancement_factors': calculate_enhancement_factors(mu, R_ratio)
                }
                results.append(result)
                
                if energy < best_energy_overall:
                    best_energy_overall = energy
                    best_overall = result
            else:
                print(f"❌ Failed or positive energy")
    
    # Save results
    output = {
        'best_result': best_overall,
        'all_results': results,
        'summary': {
            'total_attempts': len(mu_values) * len(R_ratios),
            'successful': len(results),
            'best_energy_J': best_energy_overall if best_overall else None
        }
    }
    
    with open('enhanced_soliton_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n🏆 OPTIMIZATION COMPLETE")
    print(f"Successful optimizations: {len(results)}/{len(mu_values) * len(R_ratios)}")
    if best_overall:
        print(f"Best energy: {best_energy_overall:.3e} J")
        print(f"Best params: μ={best_overall['mu']:.1e}, R_ratio={best_overall['R_ratio']:.1e}")
    
    return best_overall, results

def plot_best_profile(result):
    """Plot the best soliton profile"""
    if result is None:
        print("No result to plot")
        return
        
    params = result['parameters']
    mu = result['mu']
    R_ratio = result['R_ratio']
    
    r_vals = np.linspace(0, 2*R, 1000)
    f_vals = [f_soliton(r, params) for r in r_vals]
    fp_vals = [f_soliton_prime(r, params) for r in r_vals]
    
    plt.figure(figsize=(12, 8))
    
    # Profile plot
    plt.subplot(2, 2, 1)
    plt.plot(r_vals, f_vals, 'b-', linewidth=2, label='f(r)')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.3, label='Physical limit')
    plt.xlabel('r/R')
    plt.ylabel('f(r)')
    plt.title('Enhanced Soliton Profile')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Derivative plot
    plt.subplot(2, 2, 2)
    plt.plot(r_vals, fp_vals, 'r-', linewidth=2, label="f'(r)")
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('r/R')
    plt.ylabel("f'(r)")
    plt.title('Soliton Derivative')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Energy density
    plt.subplot(2, 2, 3)
    factors = calculate_enhancement_factors(mu, R_ratio)
    rho_vals = []
    for r in r_vals[1:]:  # Skip r=0 to avoid singularity
        f_prime = f_soliton_prime(r, params)
        rho = -factors['beta_backreaction'] * factors['sinc_polymer'] * \
              factors['geometric_reduction'] * factors['polymer_scale'] * \
              c4_8piG * (f_prime**2) / (r**2)
        rho_vals.append(rho)
    
    plt.plot(r_vals[1:], rho_vals, 'g-', linewidth=2)
    plt.xlabel('r/R')
    plt.ylabel('ρ (J/m³)')
    plt.title('Energy Density')
    plt.grid(True, alpha=0.3)
    plt.yscale('symlog')
    
    # Parameter info
    plt.subplot(2, 2, 4)
    plt.axis('off')
    info_text = f"""Enhanced Soliton Results
    
μ = {mu:.2e}
R_ratio = {R_ratio:.2e}
Energy = {result['energy_J']:.3e} J

Parameters:
"""
    for i in range(M_soliton):
        A = params[3*i]
        r0 = params[3*i+1]
        sig = params[3*i+2]
        info_text += f"  Soliton {i+1}: A={A:.3f}, r₀={r0:.3f}, σ={sig:.3f}\n"
    
    plt.text(0.1, 0.9, info_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('enhanced_soliton_profile.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to prevent blocking

if __name__ == "__main__":
    # Run comprehensive optimization
    best_result, all_results = run_comprehensive_optimization()
    
    # Plot best result
    if best_result:
        plot_best_profile(best_result)
        
        # Compare with polynomial baseline
        print(f"\n📊 COMPARISON WITH POLYNOMIAL BASELINE")
        print(f"Soliton energy:   {best_result['energy_J']:.3e} J")
        
        # Try to read polynomial baseline for comparison
        try:
            with open('soliton_optimization_results.json', 'r') as f:
                poly_data = json.load(f)
                poly_energy = poly_data['polynomial_energy_J']
                print(f"Polynomial energy: {poly_energy:.3e} J")
                
                if best_result['energy_J'] < poly_energy:
                    improvement = abs(best_result['energy_J'] / poly_energy)
                    print(f"🎉 Soliton is {improvement:.1f}× better!")
                else:
                    ratio = abs(poly_energy / best_result['energy_J'])
                    print(f"📉 Polynomial is {ratio:.1f}× better")
        except:
            print("Could not load polynomial baseline for comparison")
    else:
        print("❌ No successful optimization found")
