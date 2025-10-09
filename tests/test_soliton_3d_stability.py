#!/usr/bin/env python3
"""
3+1D Stability Test for Enhanced Soliton Warp Bubble Profile

This script integrates the best soliton ansatz profile from enhanced_soliton_optimize.py
into the 3+1D evolution system to test dynamical stability.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
from typing import Tuple, Dict, Optional

# Add src to path
sys.path.insert(0, 'src')

# Import evolution system
from evolve_3plus1D_with_backreaction import MetricBackreactionEvolution

class SolitonProfile3D:
    """
    3D implementation of the soliton ansatz for initial conditions
    """
    
    def __init__(self, soliton_params: list, mu: float, R_ratio: float):
        """
        Initialize soliton profile with optimized parameters
        
        Args:
            soliton_params: [A1, r01, sig1, A2, r02, sig2] from optimization
            mu: Polymer parameter
            R_ratio: Geometric reduction factor R_ext/R_int
        """
        self.params = np.array(soliton_params)
        self.mu = mu
        self.R_ratio = R_ratio
        self.M_soliton = len(soliton_params) // 3
        
        # Physical constants
        self.hbar = 1.0545718e-34
        self.c = 299792458
        self.G = 6.67430e-11
        self.R = 1.0  # Characteristic length scale
        
        # Enhancement factors from optimization
        self.beta_backreaction = 1.9443254780147017
        self.sinc_polymer = np.sinc(mu / np.pi) if mu > 0 else 1.0
        self.geometric_reduction = R_ratio
        self.polymer_scale = mu
        
        print(f"🔧 Soliton Profile Initialized")
        print(f"   μ = {mu:.2e}")
        print(f"   R_ratio = {R_ratio:.2e}")
        print(f"   Enhancement factors: β={self.beta_backreaction:.3f}, sinc={self.sinc_polymer:.6f}")
        
    def f_soliton(self, r: np.ndarray) -> np.ndarray:
        """Calculate soliton ansatz f(r)"""
        r = np.atleast_1d(r)
        total = np.zeros_like(r)
        
        for i in range(self.M_soliton):
            Ai = self.params[3*i + 0]
            r0_i = self.params[3*i + 1]
            sig_i = max(self.params[3*i + 2], 1e-8)
            
            x = (r - r0_i) / sig_i
            
            # Robust sech^2 calculation
            mask = np.abs(x) < 20  # Prevent overflow
            sech2 = np.zeros_like(x)
            sech2[mask] = 1.0 / np.cosh(x[mask])**2
            
            total += Ai * sech2
        
        return np.clip(total, 0.0, 1.0)
    
    def f_soliton_prime(self, r: np.ndarray) -> np.ndarray:
        """Calculate derivative of soliton ansatz"""
        r = np.atleast_1d(r)
        deriv = np.zeros_like(r)
        
        for i in range(self.M_soliton):
            Ai = self.params[3*i + 0]
            r0_i = self.params[3*i + 1]
            sig_i = max(self.params[3*i + 2], 1e-8)
            
            x = (r - r0_i) / sig_i
            
            # Robust calculation
            mask = np.abs(x) < 20
            sech2 = np.zeros_like(x)
            tanh_x = np.zeros_like(x)
            
            sech2[mask] = 1.0 / np.cosh(x[mask])**2
            tanh_x[mask] = np.tanh(x[mask])
            
            deriv += Ai * (-2.0) * sech2 * tanh_x / sig_i
        
        return deriv
    
    def generate_initial_phi(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Generate initial φ field on 3D grid using soliton profile
        
        The warp factor relates to the scalar field via:
        φ(x,y,z) = α * f(r) where α is a coupling constant
        """
        R_grid = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Apply soliton profile
        f_vals = self.f_soliton(R_grid)
        
        # Convert to scalar field with appropriate coupling
        # Use dimensionally consistent coupling
        alpha = np.sqrt(self.hbar * self.c / self.G) * 1e-20  # Planck-scale coupling
        phi_initial = alpha * f_vals
        
        return phi_initial
    
    def generate_initial_pi(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Generate initial π field (time derivative of φ)
        Start with stationary profile: π = ∂φ/∂t = 0
        """
        return np.zeros_like(X)

class SolitonStabilityTester:
    """
    Test 3+1D stability of optimized soliton profiles
    """
    
    def __init__(self, results_file: str = 'enhanced_soliton_results.json'):
        """Load best soliton results"""
        try:
            with open(results_file, 'r') as f:
                self.results = json.load(f)
            
            self.best_result = self.results['best_result']
            print(f"✅ Loaded best soliton result:")
            print(f"   μ = {self.best_result['mu']:.2e}")
            print(f"   R_ratio = {self.best_result['R_ratio']:.2e}")
            print(f"   Energy = {self.best_result['energy_J']:.3e} J")
            
        except FileNotFoundError:
            print(f"❌ Could not find {results_file}")
            print("   Run enhanced_soliton_optimize.py first")
            sys.exit(1)
    
    def run_stability_test(self, 
                          evolution_time: float = 50.0,
                          grid_size: Tuple[int, int, int] = (32, 32, 32),
                          box_size: Tuple[float, float, float] = (8.0, 8.0, 8.0)):
        """
        Run 3+1D evolution to test stability
        
        Args:
            evolution_time: Total time to evolve (in units where c=1)
            grid_size: Grid resolution
            box_size: Physical size of simulation box
        """
        
        print(f"\n🚀 STARTING 3+1D SOLITON STABILITY TEST")
        print(f"=" * 60)
        print(f"Evolution time: {evolution_time}")
        print(f"Grid size: {grid_size}")
        print(f"Box size: {box_size}")
        
        # Initialize soliton profile
        soliton = SolitonProfile3D(
            self.best_result['parameters'],
            self.best_result['mu'],
            self.best_result['R_ratio']
        )
        
        # Initialize evolution system with optimized parameters
        dt = 0.005  # Stable time step
        evolution = MetricBackreactionEvolution(
            grid_size=grid_size,
            box_size=box_size,
            dt=dt,
            mu_polymer=self.best_result['mu'],
            R_bubble=2.5,
            include_backreaction=True
        )
        
        # Generate initial conditions
        print(f"\n📊 Generating initial conditions...")
        phi_initial = soliton.generate_initial_phi(evolution.X, evolution.Y, evolution.Z)
        pi_initial = soliton.generate_initial_pi(evolution.X, evolution.Y, evolution.Z)
        
        # Set initial state
        evolution.phi = phi_initial.copy()
        evolution.pi = pi_initial.copy()
          # Initialize metric perturbations to zero (match existing interface)
        evolution.h_tt = np.zeros_like(evolution.phi)
        evolution.h_rr = np.zeros_like(evolution.phi)
        
        # Analysis arrays
        n_steps = int(evolution_time / dt)
        times = np.linspace(0, evolution_time, n_steps + 1)
        
        # Diagnostic quantities
        total_energy = np.zeros(n_steps + 1)
        central_phi = np.zeros(n_steps + 1)
        max_phi = np.zeros(n_steps + 1)
        rms_phi = np.zeros(n_steps + 1)
        
        # Initial diagnostics
        total_energy[0] = self._calculate_total_energy(evolution)
        central_phi[0] = evolution.phi[evolution.nx//2, evolution.ny//2, evolution.nz//2]
        max_phi[0] = np.max(np.abs(evolution.phi))
        rms_phi[0] = np.sqrt(np.mean(evolution.phi**2))
        
        print(f"Initial diagnostics:")
        print(f"  Total energy: {total_energy[0]:.3e}")
        print(f"  Central φ: {central_phi[0]:.3e}")
        print(f"  Max |φ|: {max_phi[0]:.3e}")
        print(f"  RMS φ: {rms_phi[0]:.3e}")
        
        # Evolution loop
        print(f"\n⏱️  Starting evolution...")
        for step in range(n_steps):
            # Evolve one time step
            evolution.evolve_step()
            
            # Calculate diagnostics
            total_energy[step + 1] = self._calculate_total_energy(evolution)
            central_phi[step + 1] = evolution.phi[evolution.nx//2, evolution.ny//2, evolution.nz//2]
            max_phi[step + 1] = np.max(np.abs(evolution.phi))
            rms_phi[step + 1] = np.sqrt(np.mean(evolution.phi**2))
            
            # Progress report
            if (step + 1) % (n_steps // 10) == 0:
                progress = 100 * (step + 1) / n_steps
                print(f"  {progress:3.0f}% complete | t={times[step+1]:.1f} | Energy={total_energy[step+1]:.3e}")
        
        # Final analysis
        print(f"\n📈 STABILITY ANALYSIS RESULTS")
        print(f"=" * 40)
        
        # Energy conservation
        energy_drift = (total_energy[-1] - total_energy[0]) / abs(total_energy[0])
        print(f"Energy drift: {energy_drift:.1%}")
        
        # Field evolution
        phi_growth = max_phi[-1] / max_phi[0]
        print(f"Max field growth: {phi_growth:.2f}×")
        
        # Stability assessment
        if abs(energy_drift) < 0.1 and phi_growth < 2.0:
            stability = "✅ STABLE"
        elif abs(energy_drift) < 0.5 and phi_growth < 5.0:
            stability = "⚠️  MARGINALLY STABLE"
        else:
            stability = "❌ UNSTABLE"
        
        print(f"Stability assessment: {stability}")
        
        # Save results
        results = {
            'stability_assessment': stability,
            'evolution_time': evolution_time,
            'energy_drift_percent': energy_drift * 100,
            'field_growth_factor': phi_growth,
            'soliton_parameters': self.best_result,
            'times': times.tolist(),
            'total_energy': total_energy.tolist(),
            'central_phi': central_phi.tolist(),
            'max_phi': max_phi.tolist(),
            'rms_phi': rms_phi.tolist()
        }
        
        with open('soliton_stability_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        self._plot_stability_analysis(times, total_energy, central_phi, max_phi, rms_phi)
        
        return results
    
    def _calculate_total_energy(self, evolution) -> float:
        """Calculate total energy of the system"""
        # Kinetic energy density
        T_kinetic = 0.5 * evolution.pi**2
        
        # Gradient energy density
        phi_x = np.gradient(evolution.phi, evolution.dx, axis=0)
        phi_y = np.gradient(evolution.phi, evolution.dy, axis=1)
        phi_z = np.gradient(evolution.phi, evolution.dz, axis=2)
        T_gradient = 0.5 * (phi_x**2 + phi_y**2 + phi_z**2)
        
        # Total energy density
        rho_total = T_kinetic + T_gradient
        
        # Integrate over volume
        total_energy = np.sum(rho_total) * evolution.dx * evolution.dy * evolution.dz
        
        return total_energy
    
    def _plot_stability_analysis(self, times, total_energy, central_phi, max_phi, rms_phi):
        """Plot stability analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Energy evolution
        axes[0, 0].plot(times, total_energy, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Total Energy')
        axes[0, 0].set_title('Energy Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Central field value
        axes[0, 1].plot(times, central_phi, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('φ(0,0,0)')
        axes[0, 1].set_title('Central Field Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Maximum field
        axes[1, 0].plot(times, max_phi, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('max|φ|')
        axes[1, 0].set_title('Maximum Field Magnitude')
        axes[1, 0].grid(True, alpha=0.3)
        
        # RMS field
        axes[1, 1].plot(times, rms_phi, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('RMS φ')
        axes[1, 1].set_title('RMS Field Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('soliton_stability_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Stability plots saved to 'soliton_stability_analysis.png'")
        plt.close()  # Close instead of show to prevent blocking

def main():
    """Main execution"""
    print("🧪 SOLITON WARP BUBBLE 3+1D STABILITY TEST")
    print("=" * 60)
    
    # Initialize tester
    tester = SolitonStabilityTester()
    
    # Run stability test with moderate resolution
    results = tester.run_stability_test(
        evolution_time=20.0,  # Reasonable evolution time
        grid_size=(24, 24, 24),  # Moderate resolution
        box_size=(6.0, 6.0, 6.0)  # Adequate box size
    )
    
    print(f"\n🏁 Stability test complete!")
    print(f"Results saved to 'soliton_stability_results.json'")

if __name__ == "__main__":
    main()
