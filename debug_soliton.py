#!/usr/bin/env python3
"""
Debug soliton optimization to understand why it's failing
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Simplified constants for debugging
beta_back = 1.9443254780147017
G_geo = 1e-5
mu0 = 1e-6
hbar = 1.0545718e-34
c = 299792458
G = 6.67430e-11
v = 1.0
R = 1.0
M_soliton = 2

# Conversion factor
c4_8piG = c**4 / (8.0 * np.pi * G)

def f_soliton_debug(r, params):
    """Debug version of soliton ansatz"""
    total = 0.0
    for i in range(M_soliton):
        Ai = params[3*i + 0]
        r0_i = params[3*i + 1]
        sig_i = params[3*i + 2]
        if sig_i > 0:  # Avoid division by zero
            total += Ai * (1.0 / np.cosh((r - r0_i)/sig_i))**2
    return np.clip(total, 0.0, 2.0)  # Allow f > 1 for debugging

def f_soliton_prime_debug(r, params):
    """Debug version of derivative"""
    deriv = 0.0
    for i in range(M_soliton):
        Ai = params[3*i + 0]
        r0_i = params[3*i + 1]
        sig_i = params[3*i + 2]
        if sig_i > 0:
            x = (r - r0_i) / sig_i
            sech2 = 1.0 / np.cosh(x)**2
            deriv += Ai * (-2.0) * sech2 * np.tanh(x) / sig_i
    return deriv

def analyze_test_params():
    """Analyze specific test parameters"""
    test_params = np.array([1.0, 0.3, 0.1, 0.5, 0.7, 0.1])
    
    print("🔍 DEBUGGING SOLITON ANSATZ")
    print("=" * 50)
    print(f"Test parameters: {test_params}")
    
    # Test individual components
    r_test = np.linspace(0, R, 100)
    f_vals = [f_soliton_debug(r, test_params) for r in r_test]
    fp_vals = [f_soliton_prime_debug(r, test_params) for r in r_test]
    
    print(f"\nFunction values:")
    print(f"  f(0) = {f_soliton_debug(0, test_params):.4f}")
    print(f"  f(R) = {f_soliton_debug(R, test_params):.4f}")
    print(f"  max(f) = {max(f_vals):.4f}")
    print(f"  min(f) = {min(f_vals):.4f}")
    
    print(f"\nDerivative values:")
    print(f"  f'(0) = {f_soliton_prime_debug(0, test_params):.4f}")
    print(f"  f'(R) = {f_soliton_prime_debug(R, test_params):.4f}")
    print(f"  max(|f'|) = {max(np.abs(fp_vals)):.4f}")
    
    # Calculate energy density at a few points
    sinc_val = np.sinc(mu0)
    print(f"\nEnhancement factors:")
    print(f"  β_backreaction = {beta_back:.6f}")
    print(f"  sinc(μ) = {sinc_val:.10f}")
    print(f"  G_geo = {G_geo:.1e}")
    print(f"  c⁴/(8πG) = {c4_8piG:.3e}")
    
    # Energy density at center
    fp_center = f_soliton_prime_debug(0.5, test_params)
    rho_center = - c4_8piG * (v**2 / c**2) * (fp_center**2) * beta_back * sinc_val / G_geo
    print(f"\nEnergy density at r=0.5:")
    print(f"  f'(0.5) = {fp_center:.6f}")
    print(f"  ρ(0.5) = {rho_center:.3e} J/m³")
    
    # Total energy integral
    def integrand(rr):
        fp = f_soliton_prime_debug(rr, test_params)
        rho = - c4_8piG * (v**2 / c**2) * (fp**2) * beta_back * sinc_val / G_geo
        return rho * 4.0 * np.pi * rr**2
    
    try:
        total_energy, _ = quad(integrand, 0.0, R, limit=200)
        print(f"\nTotal energy: {total_energy:.3e} J")
    except Exception as e:
        print(f"\nIntegration error: {e}")
    
    # Create diagnostic plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(r_test, f_vals, 'b-', linewidth=2)
    plt.xlabel('r (m)')
    plt.ylabel('f(r)')
    plt.title('Soliton Profile')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(r_test, fp_vals, 'r-', linewidth=2)
    plt.xlabel('r (m)')
    plt.ylabel("f'(r)")
    plt.title('Soliton Derivative')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    rho_vals = []
    for r in r_test:
        fp = f_soliton_prime_debug(r, test_params)
        rho = - c4_8piG * (v**2 / c**2) * (fp**2) * beta_back * sinc_val / G_geo
        rho_vals.append(rho)
    
    plt.plot(r_test, rho_vals, 'g-', linewidth=2)
    plt.xlabel('r (m)')
    plt.ylabel('ρ (J/m³)')
    plt.title('Energy Density')
    plt.yscale('symlog')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('soliton_debug_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show to prevent blocking
    
    return total_energy

def test_simple_optimization():
    """Test a simplified optimization"""
    print("\n" + "=" * 50)
    print("🔧 TESTING SIMPLIFIED OPTIMIZATION")
    
    def simple_objective(params):
        try:
            # Simple energy calculation without penalties
            def integrand(rr):
                fp = f_soliton_prime_debug(rr, params)
                return fp**2 * rr**2  # Simplified integrand
            
            energy, _ = quad(integrand, 0.0, R, limit=100)
            return energy  # Minimize this (should be positive)
            
        except Exception as e:
            print(f"Error in objective: {e}")
            return 1e10  # Large penalty for failed calculations
    
    # Simple bounds
    bounds = [
        (0.1, 2.0), (0.0, R), (0.01, 0.5),  # First lump
        (0.1, 2.0), (0.0, R), (0.01, 0.5)   # Second lump
    ]
    
    init_params = np.array([1.0, 0.3, 0.1, 0.5, 0.7, 0.1])
    
    print(f"Initial parameters: {init_params}")
    print(f"Initial objective: {simple_objective(init_params):.6f}")
    
    try:
        res = minimize(
            simple_objective,
            x0=init_params,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 100}
        )
        
        print(f"Optimization success: {res.success}")
        print(f"Final parameters: {res.x}")
        print(f"Final objective: {res.fun:.6f}")
        
        return res.x if res.success else None
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None

if __name__ == "__main__":
    # Debug analysis
    total_energy = analyze_test_params()
    
    # Test simple optimization
    optimal_params = test_simple_optimization()
    
    if optimal_params is not None:
        print(f"\n✅ Found working parameters!")
        print(f"Consider using these as starting point for full optimization")
    else:
        print(f"\n❌ Even simplified optimization failed")
        print(f"Need to investigate the ansatz formulation")
