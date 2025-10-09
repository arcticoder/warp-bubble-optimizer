#!/usr/bin/env python3
"""
Advanced Shape Function Optimization with Quantum Inequality Constraints
=======================================================================

This implements the enhanced metric ansatz optimization described in the roadmap,
with JAX acceleration for both energy minimization and quantum inequality enforcement.
"""

import jax.numpy as jnp
from jax import jit, grad, vmap
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
import time

try:
    import jax
    from jax.lib import xla_bridge
    JAX_AVAILABLE = True
    print(f"JAX devices: {jax.devices()}")
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available, using NumPy fallback")

class AdvancedWarpShapeOptimizer:
    """Enhanced warp bubble shape optimizer with QI constraints."""
    
    def __init__(self, R_max: float = 10.0, N_points: int = 1000):
        self.R_max = R_max
        self.N_points = N_points
        self.rs = jnp.linspace(0.01, R_max, N_points)  # Avoid r=0 singularity
        self.dr = self.rs[1] - self.rs[0]
        
        # Physical constants
        self.c = 2.998e8  # m/s
        self.G = 6.674e-11  # m³/kg/s²
        
        # Quantum inequality parameter
        self.C_over_tau0 = 1e-3  # QI constant

    @jit
    def gaussian_ansatz(self, r: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Multi-Gaussian ansatz: f(r) = Σ A_i exp(-(r-r_i)²/σ_i²)"""
        # theta = [A1, r1, σ1, A2, r2, σ2, ...]
        n_gaussians = len(theta) // 3
        f = jnp.zeros_like(r)
        
        for i in range(n_gaussians):
            A = theta[3*i]
            r_center = theta[3*i + 1]
            sigma = theta[3*i + 2]
            f += A * jnp.exp(-((r - r_center)/sigma)**2)
        
        return f

    @jit
    def polynomial_ansatz(self, r: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Polynomial ansatz: f(r) = Σ a_i r^i with proper boundary conditions"""
        # Ensure f(∞) → 0 by using exponential envelope
        poly = jnp.polyval(theta[:-1], r)  # All coefficients except last
        envelope = jnp.exp(-r / theta[-1])  # Exponential decay with scale θ[-1]
        return poly * envelope

    @jit
    def f_r(self, r: jnp.ndarray, theta: jnp.ndarray, ansatz_type: str = "gaussian") -> jnp.ndarray:
        """Shape function with multiple ansatz options."""
        if ansatz_type == "gaussian":
            return self.gaussian_ansatz(r, theta)
        elif ansatz_type == "polynomial":
            return self.polynomial_ansatz(r, theta)
        else:
            # Default: simple Gaussian
            A, sigma = theta[:2]
            return A * jnp.exp(-(r/sigma)**2)

    @jit
    def stress_energy_tensor_00(self, r: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        """T_00 component (energy density)."""
        df_dr = jnp.gradient(f, self.dr)
        d2f_dr2 = jnp.gradient(df_dr, self.dr)
        
        # For Alcubierre metric: T_00 = -(c²/8πG) * [complicated expression]
        # Simplified version focusing on main terms
        T_00 = -(self.c**2 / (8 * jnp.pi * self.G)) * (
            df_dr**2 / r**2 +
            f * d2f_dr2 / r +
            2 * f * df_dr / r**2
        )
        return T_00

    @jit
    def neg_energy_integral(self, theta: jnp.ndarray, ansatz_type: str = "gaussian") -> jnp.ndarray:
        """Compute total negative energy integral."""
        f_profile = self.f_r(self.rs, theta, ansatz_type)
        T_00 = self.stress_energy_tensor_00(self.rs, f_profile)
        
        # Extract negative energy density only
        negative_T_00 = jnp.where(T_00 < 0, -T_00, 0.0)
        
        # Volume integral: E = 4π ∫ T_00 r² dr
        integrand = negative_T_00 * self.rs**2
        total_energy = 4 * jnp.pi * jnp.trapz(integrand, dx=self.dr)
        
        return total_energy

    @jit
    def qi_penalty(self, theta: jnp.ndarray, ansatz_type: str = "gaussian") -> jnp.ndarray:
        """Quantum inequality penalty function."""
        neg_energy = self.neg_energy_integral(theta, ansatz_type)
        
        # QI violation: if |E_neg| > C/τ₀⁴
        violations = jnp.maximum(0.0, neg_energy - self.C_over_tau0)
        return violations**2

    @jit
    def smoothness_penalty(self, theta: jnp.ndarray, ansatz_type: str = "gaussian") -> jnp.ndarray:
        """Penalty for non-smooth profiles."""
        f_profile = self.f_r(self.rs, theta, ansatz_type)
        
        # First and second derivative penalties
        df_dr = jnp.gradient(f_profile, self.dr)
        d2f_dr2 = jnp.gradient(df_dr, self.dr)
        
        # Penalize large derivatives
        smoothness = jnp.sum(df_dr**2) + 0.1 * jnp.sum(d2f_dr2**2)
        
        # Penalize oscillations (sign changes in derivative)
        sign_changes = jnp.sum(jnp.abs(jnp.diff(jnp.sign(df_dr))))
        
        return smoothness + 1000 * sign_changes

    @jit
    def boundary_penalty(self, theta: jnp.ndarray, ansatz_type: str = "gaussian") -> jnp.ndarray:
        """Boundary condition penalties."""
        f_profile = self.f_r(self.rs, theta, ansatz_type)
        penalty = 0.0
        
        # f(∞) → 0
        penalty += 1e6 * f_profile[-1]**2
        
        # f should be reasonable magnitude
        f_max = jnp.max(jnp.abs(f_profile))
        penalty += jnp.maximum(0.0, f_max - 10.0)**2  # Limit max amplitude
        
        return penalty

    @jit
    def total_objective(self, theta: jnp.ndarray, lambda_qi: float = 1e3, 
                       lambda_smooth: float = 1e-3, lambda_bc: float = 1.0,
                       ansatz_type: str = "gaussian") -> jnp.ndarray:
        """Complete objective function."""
        energy = self.neg_energy_integral(theta, ansatz_type)
        qi_pen = self.qi_penalty(theta, ansatz_type)
        smooth_pen = self.smoothness_penalty(theta, ansatz_type)
        bc_pen = self.boundary_penalty(theta, ansatz_type)
        
        return energy + lambda_qi * qi_pen + lambda_smooth * smooth_pen + lambda_bc * bc_pen

def run_comprehensive_optimization():
    """Run comprehensive optimization with multiple ansätze."""
    print("🚀 COMPREHENSIVE WARP SHAPE OPTIMIZATION")
    print("=" * 60)
    
    if not JAX_AVAILABLE:
        print("⚠️ JAX not available - using NumPy fallback")
        return
    
    optimizer = AdvancedWarpShapeOptimizer(R_max=15.0, N_points=1000)
    
    # Test different ansätze
    test_cases = [
        {
            'name': 'Single Gaussian',
            'theta_init': jnp.array([1.0, 2.0]),  # [A, σ]
            'ansatz_type': 'gaussian'
        },
        {
            'name': 'Double Gaussian',
            'theta_init': jnp.array([1.5, 3.0, 1.0, 0.8, 6.0, 2.0]),  # [A1,r1,σ1, A2,r2,σ2]
            'ansatz_type': 'gaussian'
        },
        {
            'name': 'Polynomial + Envelope',
            'theta_init': jnp.array([0.0, 1.0, -0.1, 0.01, 5.0]),  # [a0,a1,a2,a3, decay_scale]
            'ansatz_type': 'polynomial'
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print("-" * 40)
        
        # Create specialized objective for this ansatz
        @jit
        def objective_func(theta):
            return optimizer.total_objective(theta, ansatz_type=test_case['ansatz_type'])
        
        @jit 
        def grad_func(theta):
            return grad(objective_func)(theta)
        
        # Gradient descent
        theta = test_case['theta_init']
        lr = 1e-3
        history = []
        
        start_time = time.time()
        
        for i in range(300):
            energy = objective_func(theta)
            g = grad_func(theta)
            grad_norm = jnp.linalg.norm(g)
            
            history.append({'iter': i, 'energy': float(energy), 'grad_norm': float(grad_norm)})
            
            # Update
            theta = theta - lr * g
            
            if i % 50 == 0:
                print(f"  Iter {i:3d}: E = {energy:.6e}, |grad| = {grad_norm:.2e}")
            
            if grad_norm < 1e-6:
                print(f"  Converged at iteration {i}")
                break
        
        opt_time = time.time() - start_time
        
        # Analyze final solution
        final_energy = optimizer.neg_energy_integral(theta, test_case['ansatz_type'])
        qi_pen = optimizer.qi_penalty(theta, test_case['ansatz_type'])
        
        results[test_case['name']] = {
            'theta_opt': theta,
            'final_energy': float(final_energy),
            'qi_penalty': float(qi_pen),
            'optimization_time': opt_time,
            'iterations': i + 1,
            'history': history,
            'ansatz_type': test_case['ansatz_type']
        }
        
        print(f"  Final energy: {final_energy:.6e} J")
        print(f"  QI penalty: {qi_pen:.6e}")
        print(f"  Time: {opt_time:.2f}s")
        print(f"  Optimal θ: {theta}")
    
    # Visualization comparison
    create_comparison_plot(optimizer, results)
    
    return results

def create_comparison_plot(optimizer, results):
    """Create comparison visualization of different ansätze."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for idx, (name, result) in enumerate(results.items()):
        if idx >= len(axes):
            break
            
        # Plot shape function
        ax = axes[idx]
        f_profile = optimizer.f_r(optimizer.rs, result['theta_opt'], result['ansatz_type'])
        ax.plot(optimizer.rs, f_profile, color=colors[idx], linewidth=2, label=f'{name}')
        ax.set_xlabel('Radius r (m)')
        ax.set_ylabel('Shape function f(r)')
        ax.set_title(f'{name}\nE = {result["final_energy"]:.2e} J')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot stress-energy tensor
        if idx + 3 < len(axes):
            ax2 = axes[idx + 3]
            T_00 = optimizer.stress_energy_tensor_00(optimizer.rs, f_profile)
            ax2.plot(optimizer.rs, T_00, color=colors[idx], linewidth=2)
            ax2.set_xlabel('Radius r (m)')
            ax2.set_ylabel('T₀₀ (J/m³)')
            ax2.set_title(f'{name} - Energy Density')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
            
            # Highlight negative regions
            negative_mask = np.array(T_00) < 0
            if np.any(negative_mask):
                ax2.fill_between(optimizer.rs, T_00, 0, where=negative_mask, 
                               alpha=0.3, color=colors[idx])
    
    plt.tight_layout()
    plt.savefig('ansatz_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Convergence comparison
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        energies = [h['energy'] for h in result['history']]
        plt.semilogy(energies, label=f'{name} (final: {result["final_energy"]:.2e})')
    
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function')
    plt.title('Optimization Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results = run_comprehensive_optimization()
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Final Energy: {result['final_energy']:.6e} J")
        print(f"  QI Penalty: {result['qi_penalty']:.6e}")
        print(f"  Optimization Time: {result['optimization_time']:.2f}s")
        print(f"  Iterations: {result['iterations']}")
        print()
    
    print("✅ Comprehensive optimization complete!")
    print("📊 Visualizations saved: 'ansatz_comparison.png', 'convergence_comparison.png'")
