import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution

# === 1. Constants (match optimize.py) ===
beta_back    = 1.9443254780147017
G_geo        = 1e-5           # Van den Broeck–Natário factor (R_ext/R_int)^-3
mu0          = 1e-6           # polymer length (unchanged)
hbar         = 1.0545718e-34  # ℏ
c            = 299792458      # Speed of light (m/s)
G            = 6.67430e-11    # Gravitational constant (m³/kg/s²)
tau          = 1e-9           # QI sampling time
v            = 1.0            # warp velocity (warp 1, c=1)
R            = 1.0            # bubble radius = 1 m
r0           = 0.1 * R        # inner flat region
M_soliton    = 2              # number of sech² lumps
# Conversion factor for natural units to Joules
c4_8piG      = c**4 / (8.0 * np.pi * G)  # Converts natural units to J/m³

# === 2. Soliton Ansätz Definition ===
def f_soliton(r, params):
    """
    Soliton (Lentz‐style) ansatz:
      f(r) = sum_{i=0..M-1} A[i] * sech^2((r - r0_i)/sigma[i])
    param vector length = 3*M: [A0, r0_0, sigma0, A1, r0_1, sigma1, ...]
    Enforce f(0)=1 and f(R)=0 via penalty in objective.
    """
    total = 0.0
    for i in range(M_soliton):
        Ai    = params[3*i + 0]
        r0_i  = params[3*i + 1]
        sig_i = params[3*i + 2]
        total += Ai * (1.0 / np.cosh((r - r0_i)/sig_i))**2
    return np.clip(total, 0.0, 1.0)  # ensure 0≤f≤1

def f_soliton_prime(r, params):
    """
    Derivative df/dr for soliton ansatz.
    d/dr [A * sech²((r-r0)/σ)] = 
      A * (-2) * sech²(x) * tanh(x) * (1/σ), with x=(r-r0)/σ.
    """
    deriv = 0.0
    for i in range(M_soliton):
        Ai    = params[3*i + 0]
        r0_i  = params[3*i + 1]
        sig_i = params[3*i + 2]
        x = (r - r0_i) / sig_i
        sech2 = 1.0 / np.cosh(x)**2
        deriv += Ai * (-2.0) * sech2 * np.tanh(x) / sig_i
    return deriv

# === 3. Effective Density ρ_eff(r; params) ===
def rho_eff_soliton(r, params):
    fp = f_soliton_prime(r, params)
    sinc_val = np.sinc(mu0)  # np.sinc(x) = sin(πx)/(πx)
    
    # Calculate effective energy density using proper relativistic formula
    # ρ_eff = -(c⁴/8πG) * (v²/c²) * (df/dr)² * corrections
    rho_classical = - c4_8piG * (v**2 / c**2) * (fp**2)
    
    # Apply all correction factors: polymer, backreaction, geometric
    rho_corrected = rho_classical * beta_back * sinc_val / G_geo
    
    return rho_corrected  # Energy density in J/m³

# === 4. Total Negative‐Energy Integral E₋ ===
def E_negative_soliton(params):
    integrand = lambda rr: rho_eff_soliton(rr, params) * 4.0 * np.pi * rr**2
    val, _ = quad(integrand, 0.0, R, limit=200)
    return val  # Energy in Joules

# === 5. QI Penalty & Boundary Penalty ===
def penalty_soliton(params, lam_qi=1e50, lam_bound=1e6):
    # (a) QI penalty: test ρ_eff(0)
    rho0 = rho_eff_soliton(0.0, params)
    qi_bound = - (hbar * np.sinc(mu0)) / (12.0 * np.pi * tau**2)
    qi_violation = max(0.0, -(rho0 - qi_bound))
    P_qi = lam_qi * (qi_violation**2)

    # (b) Boundary conditions: f(0) should be ≈1, f(R) ≈0
    f0 = f_soliton(0.0, params)
    fR = f_soliton(R, params)
    P_bound = lam_bound * ((f0 - 1.0)**2 + (fR - 0.0)**2)
    
    # Additional penalty to prevent trivial solutions
    non_trivial = 1e5 * np.exp(-1e3 * abs(E_negative_soliton(params)))
    
    return P_qi + P_bound + non_trivial

# === 6. Objective Function ===
def objective_soliton(params):
    Eneg = E_negative_soliton(params)
    Ptot = penalty_soliton(params)
    return Eneg + Ptot

# === 7. Initial Guess & Bounds ===
#   We choose 2 lumps. For initial guess:
#   - A0=1.0, r0_0=0.3, sigma0=0.1
#   - A1=0.5, r0_1=0.7, sigma1=0.1
init_params = np.array([1.0, 0.3, 0.1,   0.5, 0.7, 0.1])
# Bounds: Ai∈[0,2], r0_i∈[0,R], sigma_i∈[0.01,R/2]
bounds = []
for i in range(M_soliton):
    bounds += [
        (0.0, 2.0),        # Ai
        (0.0, R),          # r0_i
        (0.01, R*0.5)      # sigma_i
    ]

# === 8. Polynomial Ansatz Comparison ===
def f_polynomial(r, n=4):
    """Simple polynomial ansatz for comparison: f(r) = (1 - r/R)^n"""
    return max(0.0, (1.0 - r/R)**n)

def f_polynomial_prime(r, n=4):
    """Derivative of polynomial ansatz"""
    if r >= R:
        return 0.0
    return -n * (1.0 - r/R)**(n-1) / R

def E_negative_polynomial(n=4):
    """Calculate energy for polynomial ansatz"""
    def integrand(rr):
        fp = f_polynomial_prime(rr, n)
        sinc_val = np.sinc(mu0)
        rho_classical = - c4_8piG * (v**2 / c**2) * (fp**2)
        rho_corrected = rho_classical * beta_back * sinc_val / G_geo
        return rho_corrected * 4.0 * np.pi * rr**2
    
    val, _ = quad(integrand, 0.0, R, limit=200)
    return val

def analyze_profile(params, name="Soliton"):
    """Analyze and visualize the optimized profile"""
    import matplotlib.pyplot as plt
    
    r_vals = np.linspace(0, R, 100)
    f_vals = [f_soliton(r, params) for r in r_vals]
    rho_vals = [rho_eff_soliton(r, params) for r in r_vals]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(r_vals, f_vals, 'b-', linewidth=2, label=f'{name} f(r)')
    plt.xlabel('Radius r (m)')
    plt.ylabel('f(r)')
    plt.title(f'{name} Metric Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(r_vals, rho_vals, 'r-', linewidth=2, label=f'{name} ρ_eff(r)')
    plt.xlabel('Radius r (m)')
    plt.ylabel('Energy Density (J/m³)')
    plt.title(f'{name} Energy Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('symlog')
    
    plt.tight_layout()
    plt.savefig(f'{name.lower()}_profile_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return f_vals, rho_vals

# === 9. Run the Minimization ===
if __name__ == "__main__":
    print("🚀 Starting Soliton vs Polynomial Ansatz Comparison")
    print("=" * 60)
    
    # First calculate polynomial baseline for comparison
    print("📊 Calculating Polynomial Ansatz Baseline...")
    E_poly = E_negative_polynomial(n=4)
    print(f"   Polynomial (n=4): E₋ = {E_poly:.3e} J")
    
    # Now optimize soliton ansatz
    print("\n🔧 Optimizing Soliton Ansatz...")
    print("   Running global optimization with differential evolution...")
    res_global = differential_evolution(
        objective_soliton,
        bounds=bounds,
        maxiter=30,
        popsize=20,
        tol=1e-6,
        seed=42  # For reproducibility
    )
    
    print("   Running local optimization to refine solution...")
    # Then refine with L-BFGS-B
    res = minimize(
        objective_soliton,
        x0=res_global.x,  # Use global optimum as starting point
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12}
    )

    print("\n" + "=" * 60)
    
    if not res.success:
        print("❌ Soliton optimization failed:", res.message)
    else:
        params_opt = res.x
        Eopt = E_negative_soliton(params_opt)
        
        print("✅ OPTIMIZATION RESULTS")
        print("-" * 30)
        print("🎯 Soliton Ansatz (optimized):")
        print("   → Optimal parameters [A, r0, σ] for each lump:")
        for i in range(M_soliton):
            Ai, r0_i, sig_i = params_opt[3*i:3*i+3]
            print(f"     Lump {i}: A={Ai:.4f}, r0={r0_i:.4f}m, σ={sig_i:.4f}m")
        print(f"   → Minimum negative energy: E₋ = {Eopt:.3e} J")
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"   Polynomial ansatz: {E_poly:.3e} J")
        print(f"   Soliton ansatz:    {Eopt:.3e} J")
        
        if abs(Eopt) < abs(E_poly):
            improvement = abs(E_poly) / abs(Eopt)
            print(f"   🏆 Soliton is {improvement:.2f}× better than polynomial!")
        else:
            degradation = abs(Eopt) / abs(E_poly)
            print(f"   📉 Soliton is {degradation:.2f}× worse than polynomial")
        
        # Analyze the optimized profile
        print(f"\n🔍 Generating profile analysis...")
        try:
            analyze_profile(params_opt, "Soliton")
            print("   → Profile plots saved as 'soliton_profile_analysis.png'")
        except ImportError:
            print("   → Matplotlib not available, skipping plots")
        except Exception as e:
            print(f"   → Plot generation failed: {e}")
            
        # Save results
        print(f"\n💾 Saving optimization results...")
        results = {
            'optimization_success': True,
            'optimal_parameters': params_opt.tolist(),
            'optimal_energy_J': float(Eopt),
            'polynomial_energy_J': float(E_poly),
            'improvement_factor': float(abs(E_poly) / abs(Eopt)) if abs(Eopt) > 0 else float('inf'),
            'enhancement_factors': {
                'beta_backreaction': beta_back,
                'sinc_polymer': float(np.sinc(mu0)),
                'geometric_reduction': G_geo,
                'polymer_scale': mu0
            }
        }
        
        import json
        with open('soliton_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("   → Results saved to 'soliton_optimization_results.json'")
        
    print("\n" + "=" * 60)
    print("🏁 Analysis Complete!")

# === 9. Polynomial Ansatz Comparison ===
def f_polynomial(r, n=4):
    """Simple polynomial ansatz for comparison: f(r) = (1 - r/R)^n"""
    return max(0.0, (1.0 - r/R)**n)

def f_polynomial_prime(r, n=4):
    """Derivative of polynomial ansatz"""
    if r >= R:
        return 0.0
    return -n * (1.0 - r/R)**(n-1) / R

def E_negative_polynomial(n=4):
    """Calculate energy for polynomial ansatz"""
    def integrand(rr):
        fp = f_polynomial_prime(rr, n)
        sinc_val = np.sinc(mu0)
        rho_classical = - c4_8piG * (v**2 / c**2) * (fp**2)
        rho_corrected = rho_classical * beta_back * sinc_val / G_geo
        return rho_corrected * 4.0 * np.pi * rr**2
    
    val, _ = quad(integrand, 0.0, R, limit=200)
    return val

def analyze_profile(params, name="Soliton"):
    """Analyze and visualize the optimized profile"""
    import matplotlib.pyplot as plt
    
    r_vals = np.linspace(0, R, 100)
    f_vals = [f_soliton(r, params) for r in r_vals]
    rho_vals = [rho_eff_soliton(r, params) for r in r_vals]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(r_vals, f_vals, 'b-', linewidth=2, label=f'{name} f(r)')
    plt.xlabel('Radius r (m)')
    plt.ylabel('f(r)')
    plt.title(f'{name} Metric Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(r_vals, rho_vals, 'r-', linewidth=2, label=f'{name} ρ_eff(r)')
    plt.xlabel('Radius r (m)')
    plt.ylabel('Energy Density (J/m³)')
    plt.title(f'{name} Energy Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('symlog')
    
    plt.tight_layout()
    plt.savefig(f'{name.lower()}_profile_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return f_vals, rho_vals
