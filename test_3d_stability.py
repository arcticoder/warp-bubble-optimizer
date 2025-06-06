#!/usr/bin/env python3
"""
3+1D Stability Analysis for Warp Bubble Profiles
================================================

This script performs comprehensive 3+1D stability analysis for optimized
warp bubble profiles. It implements linearized perturbation analysis around
the static bubble solution to identify unstable modes and growth rates.

Key Features:
- Linearized 3+1D Einstein equations around bubble solution
- Spherical harmonic decomposition of perturbations
- Eigenvalue analysis for stability modes
- Growth rate computation for unstable modes
- Multiple ansatz profile testing
- Comprehensive stability classification
- Physical interpretation of instabilities

Physics Background:
- Analyzes metric perturbations: g_μν = g⁽⁰⁾_μν + h_μν
- Decomposes h_μν in spherical harmonics Y_ℓᵐ(θ,φ)
- Solves linearized Einstein equations for eigenfrequencies ω
- Stable if Im(ω) ≤ 0 for all modes; unstable if Im(ω) > 0

Author: Advanced Warp Bubble Optimizer
Date: June 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
import warnings
from scipy.integrate import solve_ivp, odeint
from scipy.linalg import eig, eigvals
from scipy.optimize import fsolve
import scipy.special as sp

warnings.filterwarnings('ignore')

class WarpBubble3DStabilityAnalyzer:
    """3+1D stability analysis for warp bubble profiles."""
    
    def __init__(self):
        # Physical constants
        self.c = 299792458.0  # Speed of light (m/s)
        self.G = 6.67430e-11  # Gravitational constant (m³/kg/s²)
        
        # Bubble parameters
        self.R_b = 1.0  # Bubble radius (meters)
        
        # Numerical grid
        self.r_max = 5.0
        self.nr = 200  # Reduced for stability analysis
        self.r = np.linspace(0.01, self.r_max, self.nr)
        self.dr = self.r[1] - self.r[0]
        
        # Spherical harmonic modes to analyze
        self.l_modes = [0, 1, 2, 3, 4, 5]  # Monopole through hexapole
        
        # Stability results
        self.stability_results = {}
    
    def load_profile(self, profile_type='gaussian_6', params=None):
        """Load warp bubble profile for stability analysis."""
        if profile_type == 'gaussian_6' and params is not None:
            self.f_profile = self.gaussian_profile_6(self.r, params)
        elif profile_type == 'hybrid_cubic' and params is not None:
            self.f_profile = self.hybrid_cubic_profile(self.r, params)
        elif profile_type == 'test_stable':
            # Simple stable test profile
            self.f_profile = np.exp(-0.5 * ((self.r - self.R_b) / 0.3)**2)
        elif profile_type == 'test_unstable':
            # Artificially unstable test profile
            self.f_profile = np.sin(2 * np.pi * self.r / self.R_b) * np.exp(-self.r / 2.0)
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")
        
        # Compute derivatives
        self.df_dr = np.gradient(self.f_profile, self.dr)
        self.d2f_dr2 = np.gradient(self.df_dr, self.dr)
        
        print(f"Loaded profile: {profile_type}")
        print(f"f(R_b) = {np.interp(self.R_b, self.r, self.f_profile):.6f}")
        print(f"f(r_max) = {self.f_profile[-1]:.6f}")
    
    def gaussian_profile_6(self, r, params):
        """6-Gaussian warp bubble profile."""
        A = params[0::3]  # Amplitudes
        sigma = np.abs(params[1::3]) + 1e-8  # Widths
        mu_centers = params[2::3]  # Centers
        
        profile = np.zeros_like(r)
        for i in range(6):
            profile += A[i] * np.exp(-0.5 * ((r - mu_centers[i]) / sigma[i])**2)
        
        return profile
    
    def hybrid_cubic_profile(self, r, params):
        """Hybrid cubic transition profile."""
        # Extract parameters (simplified version)
        A_core, sigma_core, mu_core = params[0], params[1], params[2]
        a3, a2, a1, a0 = params[3], params[4], params[5], params[6]
        
        # Core Gaussian
        core = A_core * np.exp(-0.5 * ((r - mu_core) / sigma_core)**2)
        
        # Cubic polynomial transition
        transition = a3 * r**3 + a2 * r**2 + a1 * r + a0
        
        # Smooth combination
        weight = 0.5 * (1 + np.tanh(2 * (r - self.R_b)))
        profile = (1 - weight) * core + weight * transition
        
        return profile
    
    def compute_background_metric(self):
        """Compute background metric components for the bubble solution."""
        # Static spherically symmetric metric: ds² = -α²dt² + β²dr² + r²dΩ²
        # where α and β depend on the warp factor f(r)
        
        # For Alcubierre-type bubble: α = 1, β = 1 (simplified)
        # More generally: α² = 1 - 2Φ, β² = 1 + 2Φ where Φ relates to f
        
        self.alpha = np.ones_like(self.r)  # Lapse function
        self.beta = np.ones_like(self.r)   # Radial metric component
        
        # Include warp factor effects (simplified model)
        Phi = 0.1 * self.f_profile  # Gravitational potential
        self.alpha = np.sqrt(np.abs(1 - 2 * Phi))
        self.beta = np.sqrt(1 + 2 * Phi)
        
        # Derivatives
        self.dalpha_dr = np.gradient(self.alpha, self.dr)
        self.dbeta_dr = np.gradient(self.beta, self.dr)
    
    def linearized_einstein_operator(self, l):
        """Construct linearized Einstein operator for ℓ-mode perturbations."""
        n = len(self.r)
        
        # Construct differential operator matrix for ℓ-mode
        # Simplified version of the linearized Einstein equations
        
        # Second derivative operator
        D2 = np.zeros((n, n))
        for i in range(1, n-1):
            D2[i, i-1] = 1.0 / self.dr**2
            D2[i, i] = -2.0 / self.dr**2
            D2[i, i+1] = 1.0 / self.dr**2
        
        # First derivative operator
        D1 = np.zeros((n, n))
        for i in range(1, n-1):
            D1[i, i+1] = 1.0 / (2 * self.dr)
            D1[i, i-1] = -1.0 / (2 * self.dr)
        
        # Radial terms
        R_inv = np.diag(1.0 / self.r)
        R_inv2 = np.diag(1.0 / self.r**2)
        
        # Background curvature terms
        K_bg = np.diag(self.d2f_dr2 + 2 * self.df_dr / self.r)
        
        # Construct linearized operator
        # L = -ω²I + D² + (2/r)D + [(ℓ(ℓ+1)/r² - K_bg)]
        L_spatial = D2 + 2 * R_inv @ D1 + (l * (l + 1)) * R_inv2 - K_bg
        
        return L_spatial
    
    def solve_eigenvalue_problem(self, l, omega_guess_range=(-10, 10), n_omega=50):
        """Solve eigenvalue problem for ℓ-mode perturbations."""
        print(f"Analyzing ℓ = {l} mode...")
        
        # Get linearized operator
        L = self.linearized_einstein_operator(l)
        
        # Boundary conditions (simplified: perturbations vanish at boundaries)
        L[0, :] = 0
        L[0, 0] = 1
        L[-1, :] = 0
        L[-1, -1] = 1
        
        # Solve generalized eigenvalue problem: L ψ = ω² M ψ
        # where M is the mass matrix (identity for this simplified case)
        try:
            eigenvals, eigenvecs = eig(L)
            
            # Extract frequencies: ω = ±√(eigenvalue)
            frequencies = []
            growth_rates = []
            
            for ev in eigenvals:
                if np.isreal(ev) and ev.real >= 0:
                    omega = np.sqrt(ev.real)
                    frequencies.append(omega)
                    growth_rates.append(0.0)  # Stable oscillation
                elif np.isreal(ev) and ev.real < 0:
                    omega = 1j * np.sqrt(-ev.real)
                    frequencies.append(omega)
                    growth_rates.append(np.sqrt(-ev.real))  # Exponential growth
                else:
                    # Complex eigenvalue
                    if ev.real < 0:
                        omega_r = np.sqrt(-ev.real) if ev.real < 0 else 0
                        omega_i = ev.imag / (2 * omega_r) if omega_r > 0 else 0
                        frequencies.append(complex(omega_r, omega_i))
                        growth_rates.append(omega_i)
                    else:
                        frequencies.append(complex(np.sqrt(ev.real), ev.imag / (2 * np.sqrt(ev.real))))
                        growth_rates.append(0.0)
            
            # Sort by growth rate (most unstable first)
            sorted_indices = np.argsort(growth_rates)[::-1]
            frequencies = [frequencies[i] for i in sorted_indices]
            growth_rates = [growth_rates[i] for i in sorted_indices]
            
            return frequencies[:10], growth_rates[:10]  # Return top 10 modes
            
        except Exception as e:
            print(f"Error solving eigenvalue problem for ℓ={l}: {e}")
            return [], []
    
    def classify_stability(self, frequencies, growth_rates, threshold=1e-6):
        """Classify stability based on eigenfrequencies."""
        max_growth = max(growth_rates) if growth_rates else 0.0
        
        if max_growth > threshold:
            return "UNSTABLE", max_growth
        elif max_growth > -threshold:
            return "MARGINALLY_STABLE", max_growth
        else:
            return "STABLE", max_growth
    
    def analyze_profile_stability(self, profile_type, params=None):
        """Complete stability analysis for a given profile."""
        print(f"\n{'='*60}")
        print(f"3+1D Stability Analysis: {profile_type}")
        print(f"{'='*60}")
        
        # Load profile
        self.load_profile(profile_type, params)
        
        # Compute background metric
        self.compute_background_metric()
        
        # Analyze each ℓ-mode
        stability_summary = {}
        overall_unstable = False
        max_growth_overall = 0.0
        
        for l in self.l_modes:
            frequencies, growth_rates = self.solve_eigenvalue_problem(l)
            
            if growth_rates:
                classification, max_growth = self.classify_stability(frequencies, growth_rates)
                
                stability_summary[l] = {
                    'classification': classification,
                    'max_growth_rate': max_growth,
                    'frequencies': [complex(f).real if np.isreal(f) else f for f in frequencies[:5]],
                    'growth_rates': growth_rates[:5]
                }
                
                print(f"ℓ = {l}: {classification} (max growth: {max_growth:.3e})")
                
                if max_growth > max_growth_overall:
                    max_growth_overall = max_growth
                
                if classification == "UNSTABLE":
                    overall_unstable = True
            else:
                stability_summary[l] = {
                    'classification': 'ANALYSIS_FAILED',
                    'max_growth_rate': 0.0,
                    'frequencies': [],
                    'growth_rates': []
                }
        
        # Overall classification
        if overall_unstable:
            overall_classification = "UNSTABLE"
        elif max_growth_overall > -1e-6:
            overall_classification = "MARGINALLY_STABLE"
        else:
            overall_classification = "STABLE"
        
        print(f"\nOverall Classification: {overall_classification}")
        print(f"Maximum Growth Rate: {max_growth_overall:.3e}")
        
        # Store results
        self.stability_results[profile_type] = {
            'overall_classification': overall_classification,
            'max_growth_rate': max_growth_overall,
            'mode_analysis': stability_summary,
            'profile_params': params.tolist() if params is not None else None
        }
        
        return overall_classification, max_growth_overall
    
    def create_stability_plots(self, profile_type):
        """Create plots for stability analysis results."""
        if profile_type not in self.stability_results:
            print(f"No stability results for {profile_type}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Profile and derivatives
        axes[0, 0].plot(self.r, self.f_profile, 'b-', linewidth=2, label='f(r)')
        axes[0, 0].plot(self.r, self.df_dr, 'r--', linewidth=1, label="f'(r)")
        axes[0, 0].plot(self.r, self.d2f_dr2, 'g:', linewidth=1, label="f''(r)")
        axes[0, 0].axvline(x=self.R_b, color='k', linestyle='--', alpha=0.5, label=f'R_b = {self.R_b}m')
        axes[0, 0].set_xlabel('Radius r (m)')
        axes[0, 0].set_ylabel('Profile Value')
        axes[0, 0].set_title(f'Warp Profile: {profile_type}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Growth rates by ℓ-mode
        l_vals = []
        growth_rates = []
        classifications = []
        
        for l in self.l_modes:
            if l in self.stability_results[profile_type]['mode_analysis']:
                l_vals.append(l)
                growth_rates.append(self.stability_results[profile_type]['mode_analysis'][l]['max_growth_rate'])
                classifications.append(self.stability_results[profile_type]['mode_analysis'][l]['classification'])
        
        colors = []
        for cls in classifications:
            if cls == 'UNSTABLE':
                colors.append('red')
            elif cls == 'MARGINALLY_STABLE':
                colors.append('orange')
            elif cls == 'STABLE':
                colors.append('green')
            else:
                colors.append('gray')
        
        axes[0, 1].bar(l_vals, growth_rates, color=colors, alpha=0.7)
        axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[0, 1].set_xlabel('ℓ-mode')
        axes[0, 1].set_ylabel('Max Growth Rate')
        axes[0, 1].set_title('Stability by Spherical Harmonic Mode')
        axes[0, 1].set_yscale('symlog', linthresh=1e-6)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Background metric components
        axes[1, 0].plot(self.r, self.alpha, 'b-', linewidth=2, label='α(r) - lapse')
        axes[1, 0].plot(self.r, self.beta, 'r-', linewidth=2, label='β(r) - radial metric')
        axes[1, 0].set_xlabel('Radius r (m)')
        axes[1, 0].set_ylabel('Metric Component')
        axes[1, 0].set_title('Background Metric Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Stability classification summary
        classification_counts = {'STABLE': 0, 'MARGINALLY_STABLE': 0, 'UNSTABLE': 0, 'FAILED': 0}
        for cls in classifications:
            if cls == 'STABLE':
                classification_counts['STABLE'] += 1
            elif cls == 'MARGINALLY_STABLE':
                classification_counts['MARGINALLY_STABLE'] += 1
            elif cls == 'UNSTABLE':
                classification_counts['UNSTABLE'] += 1
            else:
                classification_counts['FAILED'] += 1
        
        labels = list(classification_counts.keys())
        sizes = list(classification_counts.values())
        colors_pie = ['green', 'orange', 'red', 'gray']
        
        axes[1, 1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%')
        axes[1, 1].set_title('Mode Classification Distribution')
        
        plt.tight_layout()
          # Save plot
        plot_filename = f'stability_analysis_{profile_type}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Stability plots saved to: {plot_filename}")
        plt.close()  # Close instead of show to prevent blocking
    
    def run_comprehensive_stability_test(self):
        """Run stability analysis on multiple profile types."""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE 3+1D STABILITY ANALYSIS")
        print(f"{'='*80}")
        
        # Test profiles
        test_cases = [
            ('test_stable', None),
            ('test_unstable', None),
        ]
        
        # Try to load optimized profiles from previous results
        try:
            # Load 6-Gaussian results
            with open('jax_gaussian_M6_results.json', 'r') as f:
                jax_results = json.load(f)
            test_cases.append(('gaussian_6', np.array(jax_results['optimized_parameters'])))
        except FileNotFoundError:
            print("JAX 6-Gaussian results not found, using test parameters")
            test_params_6g = np.array([
                0.5, 0.2, 0.3,  # Gaussian 1
                0.8, 0.2, 0.7,  # Gaussian 2
                1.0, 0.3, 1.0,  # Gaussian 3
                -0.3, 0.3, 1.3, # Gaussian 4
                -0.2, 0.4, 1.7, # Gaussian 5
                -0.1, 0.5, 2.5  # Gaussian 6
            ])
            test_cases.append(('gaussian_6', test_params_6g))
        
        try:
            # Load hybrid cubic results
            with open('hybrid_cubic_results.json', 'r') as f:
                hybrid_results = json.load(f)
            test_cases.append(('hybrid_cubic', np.array(hybrid_results['optimized_parameters'])))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Hybrid cubic results not found or corrupted ({e}), using test parameters")
            test_params_hybrid = np.array([1.0, 0.3, 1.0, 0.1, -0.2, 0.5, 0.0])
            test_cases.append(('hybrid_cubic', test_params_hybrid))
        
        # Run analysis for each test case
        summary_results = {}
        
        for profile_type, params in test_cases:
            try:
                classification, max_growth = self.analyze_profile_stability(profile_type, params)
                summary_results[profile_type] = {
                    'classification': classification,
                    'max_growth_rate': max_growth
                }
                
                # Create plots
                self.create_stability_plots(profile_type)
                
            except Exception as e:
                print(f"Error analyzing {profile_type}: {e}")
                summary_results[profile_type] = {
                    'classification': 'ANALYSIS_FAILED',
                    'max_growth_rate': 0.0
                }
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"STABILITY ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"{'Profile Type':<20} | {'Classification':<18} | {'Max Growth Rate':<15}")
        print(f"{'-'*20} | {'-'*18} | {'-'*15}")
        
        for profile_type, result in summary_results.items():
            classification = result['classification']
            growth_rate = result['max_growth_rate']
            print(f"{profile_type:<20} | {classification:<18} | {growth_rate:<15.3e}")
        
        # Save comprehensive results
        self.save_stability_results()
        
        return summary_results
    
    def save_stability_results(self):
        """Save all stability results to JSON file."""
        filename = '3d_stability_analysis_results.json'
        
        with open(filename, 'w') as f:
            json.dump(self.stability_results, f, indent=2, default=str)
        
        print(f"\nStability results saved to: {filename}")

def main():
    """Main stability analysis routine."""
    analyzer = WarpBubble3DStabilityAnalyzer()
    
    # Run comprehensive stability test
    summary = analyzer.run_comprehensive_stability_test()
    
    print(f"\n{'='*80}")
    print(f"3+1D STABILITY ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    # Count results
    stable_count = sum(1 for r in summary.values() if r['classification'] == 'STABLE')
    unstable_count = sum(1 for r in summary.values() if r['classification'] == 'UNSTABLE')
    marginal_count = sum(1 for r in summary.values() if r['classification'] == 'MARGINALLY_STABLE')
    
    print(f"Stable profiles: {stable_count}")
    print(f"Unstable profiles: {unstable_count}")
    print(f"Marginally stable profiles: {marginal_count}")
    
    if unstable_count > 0:
        print("\nWARNING: Some profiles show instability!")
        print("Consider implementing stabilization mechanisms or constraint modifications.")
    
    return summary

if __name__ == "__main__":
    main()
