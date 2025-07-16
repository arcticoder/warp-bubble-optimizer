#!/usr/bin/env python3
"""
Hybrid Ansatz Optimizer

Combines the best features of different ansatz types to potentially achieve
even lower negative energies. Implements:

1. Polynomial + Soliton hybrid
2. Gaussian + Polynomial hybrid  
3. Multi-ansatz superposition with optimized weights

The goal is to leverage the polynomial's flexibility in the inner region
with the soliton's localized energy concentration in the outer region.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution, minimize
import json
import matplotlib.pyplot as plt

# Physical constants
beta_back    = 1.9443254780147017
G_geo        = 1e-5           
mu0          = 1e-6           
hbar         = 1.0545718e-34  
c            = 299792458      
G            = 6.67430e-11    
tau          = 1e-9           
v            = 1.0            
R            = 1.0            

# Conversion factor
c4_8piG = c**4 / (8.0 * np.pi * G)

class HybridAnsatzOptimizer:
    """
    Hybrid ansatz optimizer combining multiple ansatz types
    """
    
    def __init__(self, hybrid_type='poly_soliton'):
        """
        Initialize hybrid optimizer
        
        Args:
            hybrid_type: Type of hybrid ('poly_soliton', 'gauss_poly', 'multi_super')
        """
        self.hybrid_type = hybrid_type
        print(f"🔧 Initialized {hybrid_type} hybrid optimizer")
    
    # ── Individual Ansatz Components ──────────────────────────────────────────
    def f_polynomial(self, r, poly_params, max_degree=4):
        """
        Polynomial component: f_poly(r) = sum(a_i * (r/R)^i)
        """
        r_norm = r / R
        result = 0.0
        for i, coeff in enumerate(poly_params[:max_degree+1]):
            result += coeff * (r_norm ** i)
        return result
    
    def f_polynomial_prime(self, r, poly_params, max_degree=4):
        """
        Derivative of polynomial component
        """
        r_norm = r / R
        result = 0.0
        for i, coeff in enumerate(poly_params[:max_degree+1]):
            if i > 0:
                result += i * coeff * (r_norm ** (i-1)) / R
        return result
    
    def f_soliton(self, r, soliton_params, M_soliton=2):
        """
        Soliton component: sum of sech^2 lumps
        """
        total = 0.0
        for i in range(M_soliton):
            if 3*i+2 < len(soliton_params):
                Ai = soliton_params[3*i + 0]
                r0i = soliton_params[3*i + 1]
                sigi = soliton_params[3*i + 2]
                
                arg = (r - r0i) / sigi
                if abs(arg) < 10:  # Avoid overflow
                    sech_sq = (1.0 / np.cosh(arg))**2
                    total += Ai * sech_sq
        return total
    
    def f_soliton_prime(self, r, soliton_params, M_soliton=2):
        """
        Derivative of soliton component
        """
        deriv = 0.0
        for i in range(M_soliton):
            if 3*i+2 < len(soliton_params):
                Ai = soliton_params[3*i + 0]
                r0i = soliton_params[3*i + 1]
                sigi = soliton_params[3*i + 2]
                
                arg = (r - r0i) / sigi
                if abs(arg) < 10:
                    sech_val = 1.0 / np.cosh(arg)
                    tanh_val = np.tanh(arg)
                    deriv += -2.0 * Ai * (sech_val**2) * tanh_val / sigi
        return deriv
    
    def f_gaussian(self, r, gauss_params, M_gauss=3):
        """
        Gaussian component: sum of Gaussian lumps
        """
        total = 0.0
        for i in range(M_gauss):
            if 3*i+2 < len(gauss_params):
                Ai = gauss_params[3*i + 0]
                r0i = gauss_params[3*i + 1]
                sigi = gauss_params[3*i + 2]
                
                x = (r - r0i) / sigi
                total += Ai * np.exp(-0.5 * x*x)
        return total
    
    def f_gaussian_prime(self, r, gauss_params, M_gauss=3):
        """
        Derivative of Gaussian component
        """
        deriv = 0.0
        for i in range(M_gauss):
            if 3*i+2 < len(gauss_params):
                Ai = gauss_params[3*i + 0]
                r0i = gauss_params[3*i + 1]
                sigi = gauss_params[3*i + 2]
                
                x = (r - r0i) / sigi
                pref = Ai * np.exp(-0.5 * x*x)
                deriv += pref * (-(r - r0i) / (sigi**2))
        return deriv
    
    # ── Hybrid Ansatz Definitions ─────────────────────────────────────────────
    def f_hybrid(self, r, params):
        """
        Hybrid ansatz function based on type
        """
        if self.hybrid_type == 'poly_soliton':
            return self._f_poly_soliton(r, params)
        elif self.hybrid_type == 'gauss_poly':
            return self._f_gauss_poly(r, params)
        elif self.hybrid_type == 'multi_super':
            return self._f_multi_super(r, params)
        else:
            raise ValueError(f"Unknown hybrid type: {self.hybrid_type}")
    
    def f_hybrid_prime(self, r, params):
        """
        Derivative of hybrid ansatz
        """
        if self.hybrid_type == 'poly_soliton':
            return self._f_poly_soliton_prime(r, params)
        elif self.hybrid_type == 'gauss_poly':
            return self._f_gauss_poly_prime(r, params)
        elif self.hybrid_type == 'multi_super':
            return self._f_multi_super_prime(r, params)
        else:
            raise ValueError(f"Unknown hybrid type: {self.hybrid_type}")
    
    def _f_poly_soliton(self, r, params):
        """
        Polynomial + Soliton hybrid: f = w1*poly + w2*soliton
        params = [w1, w2, poly_params..., soliton_params...]
        """
        w1, w2 = params[0], params[1]
        
        # Split parameters
        n_poly = 5  # degree 4 polynomial = 5 coefficients
        poly_params = params[2:2+n_poly]
        soliton_params = params[2+n_poly:]
        
        f_poly = self.f_polynomial(r, poly_params)
        f_sol = self.f_soliton(r, soliton_params, M_soliton=2)
        
        result = w1 * f_poly + w2 * f_sol
        return np.clip(result, 0.0, 1.0)
    
    def _f_poly_soliton_prime(self, r, params):
        """
        Derivative of polynomial + soliton hybrid
        """
        w1, w2 = params[0], params[1]
        
        n_poly = 5
        poly_params = params[2:2+n_poly]
        soliton_params = params[2+n_poly:]
        
        fp_poly = self.f_polynomial_prime(r, poly_params)
        fp_sol = self.f_soliton_prime(r, soliton_params, M_soliton=2)
        
        return w1 * fp_poly + w2 * fp_sol
    
    def _f_gauss_poly(self, r, params):
        """
        Gaussian + Polynomial hybrid
        """
        w1, w2 = params[0], params[1]
        
        n_poly = 5
        poly_params = params[2:2+n_poly]
        gauss_params = params[2+n_poly:]
        
        f_poly = self.f_polynomial(r, poly_params)
        f_gauss = self.f_gaussian(r, gauss_params, M_gauss=2)
        
        result = w1 * f_poly + w2 * f_gauss
        return np.clip(result, 0.0, 1.0)
    
    def _f_gauss_poly_prime(self, r, params):
        """
        Derivative of Gaussian + polynomial hybrid
        """
        w1, w2 = params[0], params[1]
        
        n_poly = 5
        poly_params = params[2:2+n_poly]
        gauss_params = params[2+n_poly:]
        
        fp_poly = self.f_polynomial_prime(r, poly_params)
        fp_gauss = self.f_gaussian_prime(r, gauss_params, M_gauss=2)
        
        return w1 * fp_poly + w2 * fp_gauss
    
    def _f_multi_super(self, r, params):
        """
        Multi-ansatz superposition: f = w1*poly + w2*gauss + w3*soliton
        """
        w1, w2, w3 = params[0], params[1], params[2]
        
        n_poly = 5
        n_gauss = 6  # 2 Gaussians = 6 params
        poly_params = params[3:3+n_poly]
        gauss_params = params[3+n_poly:3+n_poly+n_gauss]
        soliton_params = params[3+n_poly+n_gauss:]
        
        f_poly = self.f_polynomial(r, poly_params)
        f_gauss = self.f_gaussian(r, gauss_params, M_gauss=2)
        f_sol = self.f_soliton(r, soliton_params, M_soliton=2)
        
        result = w1 * f_poly + w2 * f_gauss + w3 * f_sol
        return np.clip(result, 0.0, 1.0)
    
    def _f_multi_super_prime(self, r, params):
        """
        Derivative of multi-ansatz superposition
        """
        w1, w2, w3 = params[0], params[1], params[2]
        
        n_poly = 5
        n_gauss = 6
        poly_params = params[3:3+n_poly]
        gauss_params = params[3+n_poly:3+n_poly+n_gauss]
        soliton_params = params[3+n_poly+n_gauss:]
        
        fp_poly = self.f_polynomial_prime(r, poly_params)
        fp_gauss = self.f_gaussian_prime(r, gauss_params, M_gauss=2)
        fp_sol = self.f_soliton_prime(r, soliton_params, M_soliton=2)
        
        return w1 * fp_poly + w2 * fp_gauss + w3 * fp_sol
    
    # ── Energy and Optimization ───────────────────────────────────────────────
    def rho_eff_hybrid(self, r, params):
        """
        Effective energy density for hybrid ansatz        """
        fp = self.f_hybrid_prime(r, params)
        sinc_val = np.sinc(mu0 / np.pi) if mu0 > 0 else 1.0
        prefac = - (v**2) / (8.0 * np.pi) * beta_back * sinc_val / G_geo
        return prefac * (fp**2)
        
    def E_negative_hybrid(self, params):
        """
        Total negative energy for hybrid ansatz
        """
        def integrand(rr):
            rho_val = self.rho_eff_hybrid(rr, params)
            return rho_val * 4.0 * np.pi * (rr**2)
        
        val, _ = quad(integrand, 0.0, R, limit=200)
        energy = val * c4_8piG
        
        # Enforce LQG-modified QI bound
        try:
            from src.warp_qft.stability import enforce_lqg_bound
            energy = enforce_lqg_bound(energy, R, tau)
        except ImportError:
            # Fallback for standalone use
            print("⚠️  LQG bound enforcement unavailable - using raw energy")
        
        return energy
    
    def penalty_hybrid(self, params, lam_qi=1e50, lam_bound=1e6, flight_time=None):
        """
        Penalty function for hybrid ansatz using LQG-modified QI bound
        """
        if flight_time is None:
            flight_time = tau
            
        # Import LQG bounds from stability module
        from src.warp_qft.stability import lqg_modified_bounds
        
        # LQG-modified QI penalty (stricter than Ford-Roman)
        rho0 = self.rho_eff_hybrid(0.0, params)
        
        # Use LQG-modified bound: E_- ≥ -C_LQG / T^4
        lqg_bounds = lqg_modified_bounds(rho0, R, flight_time)
        lqg_bound = lqg_bounds["lqg_bound"]
        
        qi_violation = max(0.0, -(rho0 - lqg_bound))
        P_qi = lam_qi * (qi_violation**2)
        
        # Boundary conditions
        f0 = self.f_hybrid(0.0, params)
        fR = self.f_hybrid(R, params)
        P_bound = lam_bound * ((f0 - 1.0)**2 + (fR - 0.0)**2)
        
        # Weight normalization (weights should sum to ~1)
        if self.hybrid_type == 'poly_soliton':
            w1, w2 = params[0], params[1]
            P_weight = lam_bound * (w1 + w2 - 1.0)**2
        elif self.hybrid_type == 'gauss_poly':
            w1, w2 = params[0], params[1]
            P_weight = lam_bound * (w1 + w2 - 1.0)**2
        elif self.hybrid_type == 'multi_super':
            w1, w2, w3 = params[0], params[1], params[2]
            P_weight = lam_bound * (w1 + w2 + w3 - 1.0)**2
        else:
            P_weight = 0.0
        
        return P_qi + P_bound + P_weight
    
    def objective_hybrid(self, params):
        """
        Objective function for hybrid optimization
        """
        energy = self.E_negative_hybrid(params)
        penalty = self.penalty_hybrid(params)
        return energy + penalty
    
    def get_bounds(self):
        """
        Get optimization bounds for hybrid parameters
        """
        bounds = []
        
        if self.hybrid_type == 'poly_soliton':
            # [w1, w2, poly_coeffs(5), soliton_params(6)]
            bounds.extend([(0.0, 1.0), (0.0, 1.0)])  # weights
            bounds.extend([(-2.0, 2.0)] * 5)  # polynomial coefficients
            bounds.extend([(0.0, 2.0), (0.0, R), (0.01, R/2)] * 2)  # 2 solitons
            
        elif self.hybrid_type == 'gauss_poly':
            # [w1, w2, poly_coeffs(5), gauss_params(6)]
            bounds.extend([(0.0, 1.0), (0.0, 1.0)])  # weights
            bounds.extend([(-2.0, 2.0)] * 5)  # polynomial coefficients
            bounds.extend([(0.0, 1.0), (0.0, R), (R/50, R/2)] * 2)  # 2 Gaussians
            
        elif self.hybrid_type == 'multi_super':
            # [w1, w2, w3, poly_coeffs(5), gauss_params(6), soliton_params(6)]
            bounds.extend([(0.0, 1.0)] * 3)  # weights
            bounds.extend([(-2.0, 2.0)] * 5)  # polynomial coefficients
            bounds.extend([(0.0, 1.0), (0.0, R), (R/50, R/2)] * 2)  # 2 Gaussians
            bounds.extend([(0.0, 2.0), (0.0, R), (0.01, R/2)] * 2)  # 2 solitons
        
        return bounds
    
    def optimize(self, mu_val=1e-6, G_geo_val=1e-5):
        """
        Run hybrid optimization
        """
        global mu0, G_geo
        mu0 = mu_val
        G_geo = G_geo_val
        
        print(f"   Optimizing {self.hybrid_type} with μ={mu_val:.2e}, G_geo={G_geo_val:.2e}")
        
        bounds = self.get_bounds()
        
        # Global search
        print("     Running global search...")
        result_de = differential_evolution(
            self.objective_hybrid,
            bounds,
            strategy='best1bin',
            maxiter=600,  # More iterations for complex optimization
            popsize=25,   # Larger population
            tol=1e-8,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=False,
            disp=False
        )
        
        if not result_de.success:
            print(f"     ❌ Global search failed: {result_de.message}")
            return None
        
        # Local refinement
        print("     Running local refinement...")
        res_final = minimize(
            self.objective_hybrid,
            x0=result_de.x,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 600, 'ftol': 1e-12}
        )
        
        if not res_final.success:
            print(f"     ❌ Local refinement failed: {res_final.message}")
            return None
        
        params_opt = res_final.x
        Eopt = self.E_negative_hybrid(params_opt)
        
        return {
            'hybrid_type': self.hybrid_type,
            'params': params_opt,
            'energy_J': Eopt,
            'mu': mu_val,
            'G_geo': G_geo_val,
            'success': True
        }

def run_hybrid_comparison():
    """
    Compare different hybrid ansatz types
    """
    print("🚀 HYBRID ANSATZ OPTIMIZATION")
    print("=" * 60)
    
    hybrid_types = ['poly_soliton', 'gauss_poly', 'multi_super']
    
    # Focus on known good parameter region
    mu_values = [1e-6, 5e-6]
    G_geo_values = [1e-5, 1e-4]
    
    results = {}
    
    for hybrid_type in hybrid_types:
        print(f"\n🔧 Testing {hybrid_type} hybrid ansatz")
        print("-" * 40)
        
        optimizer = HybridAnsatzOptimizer(hybrid_type=hybrid_type)
        type_results = []
        best_result = None
        best_energy = 0
        
        for mu_val in mu_values:
            for G_geo_val in G_geo_values:
                result = optimizer.optimize(mu_val, G_geo_val)
                
                if result and result['success']:
                    energy_val = result['energy_J']
                    print(f"   μ={mu_val:.1e}, G_geo={G_geo_val:.1e}: E₋ = {energy_val:.3e} J")
                    
                    type_results.append(result)
                    
                    if energy_val < best_energy:
                        best_energy = energy_val
                        best_result = result
                else:
                    print(f"   μ={mu_val:.1e}, G_geo={G_geo_val:.1e}: ❌ Failed")
        
        results[hybrid_type] = {
            'all_results': type_results,
            'best_result': best_result,
            'best_energy': best_energy
        }
        
        if best_result:
            print(f"   🏆 Best {hybrid_type}: E₋ = {best_energy:.3e} J")
        else:
            print(f"   ❌ No successful {hybrid_type} optimizations")
    
    return results

def analyze_hybrid_results(results):
    """
    Analyze hybrid optimization results
    """
    print("\n" + "=" * 60)
    print("📊 HYBRID ANSATZ COMPARISON RESULTS")
    print("=" * 60)
    
    soliton_baseline = -1.584e31
    
    print("\n🏆 BEST RESULTS BY HYBRID TYPE:")
    print("-" * 50)
    
    for hybrid_type in results.keys():
        best_result = results[hybrid_type]['best_result']
        
        if best_result:
            energy = best_result['energy_J']
            improvement = abs(energy) / abs(soliton_baseline)
            
            print(f"   {hybrid_type}: E₋ = {energy:.3e} J (×{improvement:.3f})")
            print(f"     μ={best_result['mu']:.2e}, G_geo={best_result['G_geo']:.2e}")
            
            # Show weights
            params = best_result['params']
            if hybrid_type in ['poly_soliton', 'gauss_poly']:
                w1, w2 = params[0], params[1]
                print(f"     Weights: w1={w1:.3f}, w2={w2:.3f}")
            elif hybrid_type == 'multi_super':
                w1, w2, w3 = params[0], params[1], params[2]
                print(f"     Weights: w1={w1:.3f}, w2={w2:.3f}, w3={w3:.3f}")
        else:
            print(f"   {hybrid_type}: ❌ No successful optimization")
    
    # Find overall best
    overall_best = None
    overall_best_energy = 0
    
    for hybrid_type, data in results.items():
        if data['best_result'] and data['best_energy'] < overall_best_energy:
            overall_best_energy = data['best_energy']
            overall_best = data['best_result']
    
    if overall_best:
        improvement = abs(overall_best_energy) / abs(soliton_baseline)
        print(f"\n🎯 OVERALL CHAMPION HYBRID:")
        print(f"   Type: {overall_best['hybrid_type']}")
        print(f"   E₋ = {overall_best_energy:.3e} J")
        print(f"   Improvement: ×{improvement:.3f} vs 2-lump soliton")
        
        if improvement > 1.0:
            print(f"   🎉 SUCCESS: {improvement:.2f}× better than baseline!")
        
        return overall_best
    else:
        print("\n❌ No successful hybrid optimizations found")
        return None

def plot_hybrid_profiles(results):
    """
    Plot the best hybrid profiles
    """
    r_plot = np.linspace(0, R, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green']
    
    plot_idx = 0
    for hybrid_type in results.keys():
        if results[hybrid_type]['best_result'] and plot_idx < 3:
            best_result = results[hybrid_type]['best_result']
            optimizer = HybridAnsatzOptimizer(hybrid_type=hybrid_type)
            
            f_vals = [optimizer.f_hybrid(r, best_result['params']) for r in r_plot]
            
            ax = axes[plot_idx]
            ax.plot(r_plot, f_vals, color=colors[plot_idx], linewidth=2,
                   label=f'{hybrid_type} hybrid')
            ax.set_xlabel('Radial distance r (m)')
            ax.set_ylabel('f(r)')
            ax.set_title(f'{hybrid_type.replace("_", " ").title()} Hybrid\nE₋ = {best_result["energy_J"]:.2e} J')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plot_idx += 1
    
    # Hide unused subplot
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('hybrid_ansatz_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Hybrid ansatz plot saved as 'hybrid_ansatz_comparison.png'")

def save_hybrid_results(results, filename='hybrid_ansatz_results.json'):
    """
    Save hybrid results to JSON
    """
    with open(filename, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"💾 Hybrid results saved to '{filename}'")

def main():
    """
    Main execution function
    """
    print("🌟 HYBRID ANSATZ OPTIMIZATION")
    print("🎯 Combining polynomial, Gaussian, and soliton ansätze")
    print("🔍 Target: Ultimate E₋ minimization")
    print()
    
    # Run hybrid comparison
    results = run_hybrid_comparison()
    
    # Analyze results
    best_hybrid = analyze_hybrid_results(results)
    
    # Plot profiles
    plot_hybrid_profiles(results)
    
    # Save results
    save_hybrid_results(results)
    
    print("\n" + "=" * 60)
    print("🏁 HYBRID ANSATZ OPTIMIZATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
