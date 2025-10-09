#!/usr/bin/env python3
"""
Enhanced Multi-Soliton Optimizer (M=3,4)

Extension of the current 2-lump soliton ansatz to 3 and 4 lumps to explore
whether additional degrees of freedom can push E- even lower. Each additional
lump provides 3 more parameters (amplitude, position, width) for optimization.

Current baseline: 2-lump soliton E- = -1.584×10³¹ J
Target: Explore if 3-4 lumps can achieve E- < -2.0×10³¹ J
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution, minimize
import json
import matplotlib.pyplot as plt

# Physical constants (match current soliton_optimize.py)
beta_back    = 1.9443254780147017
G_geo        = 1e-5           # Van den Broeck–Natário factor (R_ext/R_int)^-3
mu0          = 1e-6           # polymer length
hbar         = 1.0545718e-34  # ℏ
c            = 299792458      # Speed of light (m/s)  
G            = 6.67430e-11    # Gravitational constant (m³/kg/s²)
tau          = 1e-9           # QI sampling time
v            = 1.0            # warp velocity (warp 1, c=1)
R            = 1.0            # bubble radius = 1 m
r0           = 0.1 * R        # inner flat region

# Conversion factor for natural units to Joules
c4_8piG = c**4 / (8.0 * np.pi * G)  # ≈ 4.815×10⁴² J⋅m⁻³

class MultiSolitonOptimizer:
    """
    Multi-soliton ansatz optimizer with configurable number of lumps
    """
    
    def __init__(self, M_soliton=3):
        """
        Initialize optimizer
        
        Args:
            M_soliton: Number of soliton lumps (2, 3, or 4)
        """
        self.M_soliton = M_soliton
        print(f"🔧 Initialized {M_soliton}-lump soliton optimizer")
    
    def f_soliton(self, r, params):
        """
        Multi-soliton ansatz: f(r) = sum_{i=0..M-1} A[i] * sech^2((r - r0_i)/sigma[i])
        
        Args:
            r: Radial distance
            params: [A0, r0_0, sigma0, A1, r0_1, sigma1, ...] (length = 3*M)
        
        Returns:
            Profile function value, clipped to [0,1]
        """
        total = 0.0
        for i in range(self.M_soliton):
            Ai    = params[3*i + 0]
            r0_i  = params[3*i + 1]
            sig_i = params[3*i + 2]
            
            # Robust sech^2 calculation
            arg = (r - r0_i) / sig_i
            if abs(arg) > 10:  # Avoid overflow
                sech_sq = 0.0
            else:
                sech_sq = (1.0 / np.cosh(arg))**2
            
            total += Ai * sech_sq
        
        return np.clip(total, 0.0, 1.0)  # ensure 0≤f≤1

    def f_soliton_prime(self, r, params):
        """
        Derivative of multi-soliton ansatz
        
        d/dr [A * sech^2((r-r0)/σ)] = -2A * sech^2((r-r0)/σ) * tanh((r-r0)/σ) * (1/σ)
        """
        deriv = 0.0
        for i in range(self.M_soliton):
            Ai    = params[3*i + 0]
            r0_i  = params[3*i + 1]
            sig_i = params[3*i + 2]
            
            arg = (r - r0_i) / sig_i
            if abs(arg) > 10:  # Avoid overflow
                continue
            
            sech_val = 1.0 / np.cosh(arg)
            tanh_val = np.tanh(arg)
            deriv += -2.0 * Ai * (sech_val**2) * tanh_val / sig_i
        
        return deriv

    def rho_eff_soliton(self, r, params):
        """
        Effective energy density with all enhancement factors
        """
        fp = self.f_soliton_prime(r, params)
        sinc_val = np.sinc(mu0 / np.pi) if mu0 > 0 else 1.0
        prefac = - (v**2) / (8.0 * np.pi) * beta_back * sinc_val / G_geo
        return prefac * (fp**2)

    def E_negative_soliton(self, params):
        """
        Total negative energy integral
        """
        def integrand(rr):
            rho_val = self.rho_eff_soliton(rr, params)
            return rho_val * 4.0 * np.pi * (rr**2)
        
        val, _ = quad(integrand, 0.0, R, limit=200)
        return val * c4_8piG

    def penalty_soliton(self, params, lam_qi=1e50, lam_bound=1e6):
        """
        Penalty function for physical constraints
        """
        # (a) QI penalty: test ρ_eff(0)
        rho0 = self.rho_eff_soliton(0.0, params)
        qi_bound = - (hbar * np.sinc(mu0 / np.pi)) / (12.0 * np.pi * tau**2)
        qi_violation = max(0.0, -(rho0 - qi_bound))
        P_qi = lam_qi * (qi_violation**2)

        # (b) Boundary conditions: f(0) ≈ 1, f(R) ≈ 0
        f0 = self.f_soliton(0.0, params)
        fR = self.f_soliton(R, params)
        P_bound = lam_bound * ((f0 - 1.0)**2 + (fR - 0.0)**2)

        # (c) Amplitude constraint: individual A_i ≤ 1
        P_amp = 0.0
        for i in range(self.M_soliton):
            Ai = params[3*i]
            if Ai > 1.0:
                P_amp += lam_bound * (Ai - 1.0)**2

        # (d) Prevent overlap issues: ensure some separation between lumps
        P_sep = 0.0
        for i in range(self.M_soliton):
            for j in range(i+1, self.M_soliton):
                r0_i = params[3*i + 1]
                r0_j = params[3*j + 1]
                sig_i = params[3*i + 2]
                sig_j = params[3*j + 2]
                min_separation = 0.5 * (sig_i + sig_j)
                actual_separation = abs(r0_i - r0_j)
                if actual_separation < min_separation:
                    P_sep += lam_bound * (min_separation - actual_separation)**2

        return P_qi + P_bound + P_amp + P_sep

    def objective_soliton(self, params):
        """
        Objective function: minimize E_negative + penalties
        """
        energy = self.E_negative_soliton(params)
        penalty = self.penalty_soliton(params)
        return energy + penalty

    def get_bounds(self):
        """
        Get optimization bounds for parameters
        """
        bounds = []
        for i in range(self.M_soliton):
            # Ai ∈ [0, 2], r0_i ∈ [0, R], sigma_i ∈ [0.01, R/2]
            bounds.append((0.0, 2.0))      # Amplitude
            bounds.append((0.0, R))        # Center position
            bounds.append((0.01, R/2))     # Width
        return bounds

    def get_initial_guess(self):
        """
        Generate reasonable initial guess based on number of lumps
        """
        params = []
        
        if self.M_soliton == 2:
            # Based on successful 2-lump optimization
            params = [1.0, 0.3, 0.1, 0.5, 0.7, 0.1]
        elif self.M_soliton == 3:
            # Distribute 3 lumps across the bubble
            params = [0.8, 0.2, 0.1, 0.6, 0.5, 0.12, 0.4, 0.8, 0.1]
        elif self.M_soliton == 4:
            # Distribute 4 lumps across the bubble
            params = [0.7, 0.15, 0.08, 0.5, 0.35, 0.1, 0.4, 0.6, 0.12, 0.3, 0.85, 0.08]
        else:
            # Generic distribution
            for i in range(self.M_soliton):
                A_i = 1.0 / (i + 1)  # Decreasing amplitudes
                r0_i = (i + 1) * R / (self.M_soliton + 1)  # Evenly spaced
                sig_i = 0.1  # Standard width
                params.extend([A_i, r0_i, sig_i])
        
        return np.array(params)

    def optimize(self, mu_val=1e-6, G_geo_val=1e-5):
        """
        Run optimization for given parameters
        """
        global mu0, G_geo
        mu0 = mu_val  
        G_geo = G_geo_val
        
        print(f"   Optimizing {self.M_soliton}-lump soliton with μ={mu_val:.2e}, G_geo={G_geo_val:.2e}")
        
        bounds = self.get_bounds()
        
        # Global search with differential evolution
        print("     Running global search...")
        result_de = differential_evolution(
            self.objective_soliton,
            bounds,
            strategy='best1bin',
            maxiter=500,
            popsize=20,  # Larger population for more complex optimization
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
            self.objective_soliton,
            x0=result_de.x,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12}
        )

        if not res_final.success:
            print(f"     ❌ Local refinement failed: {res_final.message}")
            return None

        params_opt = res_final.x
        Eopt = self.E_negative_soliton(params_opt)
        
        return {
            'M_soliton': self.M_soliton,
            'params': params_opt,
            'energy_J': Eopt,
            'mu': mu_val,
            'G_geo': G_geo_val,
            'success': True
        }

def run_multi_soliton_comparison():
    """
    Compare performance across different numbers of soliton lumps
    """
    print("🚀 MULTI-SOLITON ANSATZ COMPARISON")
    print("=" * 60)
    
    # Test different numbers of lumps
    M_values = [2, 3, 4]
    
    # Parameter values to test (focus on known good region)
    mu_values = [1e-6, 5e-6, 1e-5]
    G_geo_values = [1e-5, 1e-4]
    
    results = {}
    
    for M in M_values:
        print(f"\n🔧 Testing {M}-lump soliton ansatz")
        print("-" * 40)
        
        optimizer = MultiSolitonOptimizer(M_soliton=M)
        M_results = []
        best_result = None
        best_energy = 0
        
        for mu_val in mu_values:
            for G_geo_val in G_geo_values:
                result = optimizer.optimize(mu_val, G_geo_val)
                
                if result and result['success']:
                    energy_val = result['energy_J']
                    print(f"   μ={mu_val:.1e}, G_geo={G_geo_val:.1e}: E₋ = {energy_val:.3e} J")
                    
                    M_results.append(result)
                    
                    if energy_val < best_energy:
                        best_energy = energy_val
                        best_result = result
                else:
                    print(f"   μ={mu_val:.1e}, G_geo={G_geo_val:.1e}: ❌ Failed")
        
        results[M] = {
            'all_results': M_results,
            'best_result': best_result,
            'best_energy': best_energy
        }
        
        if best_result:
            print(f"   🏆 Best {M}-lump: E₋ = {best_energy:.3e} J")
        else:
            print(f"   ❌ No successful {M}-lump optimizations")
    
    return results

def analyze_multi_soliton_results(results):
    """
    Analyze and compare multi-soliton results
    """
    print("\n" + "=" * 60)
    print("📊 MULTI-SOLITON COMPARISON RESULTS")
    print("=" * 60)
    
    soliton_baseline = -1.584e31  # Current 2-lump result
    
    print("\n🏆 BEST RESULTS BY NUMBER OF LUMPS:")
    print("-" * 50)
    
    for M in sorted(results.keys()):
        best_result = results[M]['best_result']
        
        if best_result:
            energy = best_result['energy_J']
            improvement = abs(energy) / abs(soliton_baseline)
            
            print(f"   {M}-lump: E₋ = {energy:.3e} J (×{improvement:.3f})")
            
            # Show optimal parameters
            params = best_result['params']
            print(f"     Parameters: μ={best_result['mu']:.2e}, G_geo={best_result['G_geo']:.2e}")
            print(f"     Lumps: ", end="")
            for i in range(M):
                Ai, r0i, sigi = params[3*i:3*i+3]
                print(f"[A={Ai:.3f}, r0={r0i:.3f}, σ={sigi:.3f}] ", end="")
            print()
        else:
            print(f"   {M}-lump: ❌ No successful optimization")
    
    # Find overall best
    overall_best = None
    overall_best_energy = 0
    
    for M, data in results.items():
        if data['best_result'] and data['best_energy'] < overall_best_energy:
            overall_best_energy = data['best_energy']
            overall_best = data['best_result']
    
    if overall_best:
        improvement = abs(overall_best_energy) / abs(soliton_baseline)
        print(f"\n🎯 OVERALL CHAMPION:")
        print(f"   {overall_best['M_soliton']}-lump soliton")
        print(f"   E₋ = {overall_best_energy:.3e} J")
        print(f"   Improvement: ×{improvement:.3f} vs 2-lump baseline")
        
        if improvement > 1.0:
            print(f"   🎉 SUCCESS: {improvement:.2f}× better than baseline!")
        
        return overall_best
    else:
        print("\n❌ No successful optimizations found")
        return None

def plot_multi_soliton_profiles(results):
    """
    Plot profiles for different numbers of lumps
    """
    r_plot = np.linspace(0, R, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange']
    
    plot_idx = 0
    for M in sorted(results.keys()):
        if results[M]['best_result'] and plot_idx < 4:
            best_result = results[M]['best_result']
            optimizer = MultiSolitonOptimizer(M_soliton=M)
            
            f_vals = [optimizer.f_soliton(r, best_result['params']) for r in r_plot]
            
            ax = axes[plot_idx]
            ax.plot(r_plot, f_vals, color=colors[plot_idx], linewidth=2, 
                   label=f'{M}-lump soliton')
            ax.set_xlabel('Radial distance r (m)')
            ax.set_ylabel('f(r)')
            ax.set_title(f'{M}-Lump Soliton Profile\nE₋ = {best_result["energy_J"]:.2e} J')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('multi_soliton_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Multi-soliton comparison plot saved as 'multi_soliton_comparison.png'")

def save_multi_soliton_results(results, filename='multi_soliton_results.json'):
    """
    Save multi-soliton results to JSON file
    """
    with open(filename, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"💾 Multi-soliton results saved to '{filename}'")

def main():
    """
    Main execution function
    """
    print("🌟 ENHANCED MULTI-SOLITON OPTIMIZATION")
    print("🎯 Exploring 2, 3, and 4-lump soliton ansätze")
    print("🔍 Target: Push beyond E₋ = -1.584×10³¹ J")
    print()
    
    # Run multi-soliton comparison
    results = run_multi_soliton_comparison()
    
    # Analyze results
    best_overall = analyze_multi_soliton_results(results)
    
    # Plot profiles
    plot_multi_soliton_profiles(results)
    
    # Save results
    save_multi_soliton_results(results)
    
    print("\n" + "=" * 60)
    print("🏁 MULTI-SOLITON OPTIMIZATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
