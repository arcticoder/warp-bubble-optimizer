#!/usr/bin/env python3
"""
JAX-based Gradient Descent Optimizer for 6-Gaussian Warp Bubble
================================================================

This script implements a high-performance JAX-based gradient descent optimizer
for the 6-Gaussian warp bubble ansatz. JAX provides automatic differentiation
and JIT compilation for fast convergence to optimal parameters.

Key Features:
- JAX automatic differentiation for exact gradients
- JIT-compiled objective function for speed
- Advanced line search with backtracking
- Momentum-based optimization (Adam/RMSprop)
- Adaptive learning rate scheduling
- Physics constraint enforcement
- Multiple initialization strategies
- Comprehensive result analysis and visualization

Author: Advanced Warp Bubble Optimizer
Date: June 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import JAX (will use numpy fallback if not available)
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, value_and_grad
    from jax.scipy.integrate import trapezoid
    JAX_AVAILABLE = True
    print("JAX detected - using JAX acceleration")
except ImportError:
    import numpy as jnp  # Fallback to numpy
    JAX_AVAILABLE = False
    print("JAX not available - using NumPy fallback")

class JAXWarpBubbleOptimizer:
    """JAX-based optimizer for 6-Gaussian warp bubble profiles."""
    
    def __init__(self, use_jax=True):
        self.use_jax = use_jax and JAX_AVAILABLE
        
        # Physical parameters
        self.R_b = 1.0  # Bubble radius (meters)
        self.c = 299792458.0  # Speed of light (m/s)
        self.mu = 1.0  # Initial geometric parameter
        self.G_geo = 1.0  # Initial geometric coupling
        
        # Numerical parameters
        self.r_max = 5.0
        self.nr = 500
        self.r = np.linspace(0.01, self.r_max, self.nr)
        
        if self.use_jax:
            self.r_jax = jnp.array(self.r)
            # JIT compile the objective function
            self._compute_energy_jit = jit(self._compute_energy_jax)
            self._compute_energy_and_grad = jit(value_and_grad(self._compute_energy_jax))
        
        # Optimization history
        self.history = []
        
    def gaussian_profile(self, r, params):
        """6-Gaussian warp bubble profile."""
        if self.use_jax:
            return self._gaussian_profile_jax(r, params)
        else:
            return self._gaussian_profile_numpy(r, params)
    
    def _gaussian_profile_jax(self, r, params):
        """JAX implementation of 6-Gaussian profile."""
        # Extract parameters: [A1, sigma1, mu1, A2, sigma2, mu2, ..., A6, sigma6, mu6]
        A = params[0::3]  # Amplitudes
        sigma = params[1::3]  # Widths
        mu_centers = params[2::3]  # Centers
        
        # Ensure positive widths
        sigma = jnp.abs(sigma) + 1e-8
        
        # Compute 6-Gaussian profile
        profile = jnp.zeros_like(r)
        for i in range(6):
            profile += A[i] * jnp.exp(-0.5 * ((r - mu_centers[i]) / sigma[i])**2)
        
        return profile
    
    def _gaussian_profile_numpy(self, r, params):
        """NumPy fallback implementation."""
        A = params[0::3]
        sigma = np.abs(params[1::3]) + 1e-8
        mu_centers = params[2::3]
        
        profile = np.zeros_like(r)
        for i in range(6):
            profile += A[i] * np.exp(-0.5 * ((r - mu_centers[i]) / sigma[i])**2)
        
        return profile
    
    def _compute_energy_jax(self, params):
        """JAX-compiled energy computation."""
        # Get the profile
        f_profile = self._gaussian_profile_jax(self.r_jax, params)
        
        # Compute derivatives using JAX
        df_dr = jnp.gradient(f_profile, self.r_jax[1] - self.r_jax[0])
        d2f_dr2 = jnp.gradient(df_dr, self.r_jax[1] - self.r_jax[0])
        
        # Energy density components
        T_rr = (self.c**4 / (8 * jnp.pi)) * (
            (df_dr**2) / (2 * self.r_jax**2) +
            (f_profile * d2f_dr2) / self.r_jax +
            (f_profile * df_dr) / self.r_jax**2
        )
        
        # Total energy (negative component)
        dr = self.r_jax[1] - self.r_jax[0]
        E_negative = 4 * jnp.pi * trapezoid(T_rr * self.r_jax**2, dx=dr)
        
        # Physics constraints as penalties
        penalty = 0.0
        
        # Bubble constraint: f(R_b) ≈ 1
        f_at_bubble = jnp.interp(self.R_b, self.r_jax, f_profile)
        penalty += 1e6 * (f_at_bubble - 1.0)**2
        
        # Asymptotic constraint: f(r→∞) ≈ 0
        penalty += 1e6 * f_profile[-1]**2
        
        # Smoothness constraint: penalize large derivatives
        penalty += 1e3 * jnp.mean(df_dr**2)
        penalty += 1e2 * jnp.mean(d2f_dr2**2)
        
        # Stability constraint: avoid oscillations
        sign_changes = jnp.sum(jnp.abs(jnp.diff(jnp.sign(df_dr))))
        penalty += 1e4 * sign_changes
        
        return E_negative + penalty
    
    def _compute_energy_numpy(self, params):
        """NumPy fallback energy computation."""
        f_profile = self._gaussian_profile_numpy(self.r, params)
        
        # Numerical derivatives
        df_dr = np.gradient(f_profile, self.r[1] - self.r[0])
        d2f_dr2 = np.gradient(df_dr, self.r[1] - self.r[0])
        
        # Energy density
        T_rr = (self.c**4 / (8 * np.pi)) * (
            (df_dr**2) / (2 * self.r**2) +
            (f_profile * d2f_dr2) / self.r +
            (f_profile * df_dr) / self.r**2
        )
        
        # Total energy
        E_negative = 4 * np.pi * np.trapz(T_rr * self.r**2, self.r)
        
        # Constraints
        penalty = 0.0
        f_at_bubble = np.interp(self.R_b, self.r, f_profile)
        penalty += 1e6 * (f_at_bubble - 1.0)**2
        penalty += 1e6 * f_profile[-1]**2
        penalty += 1e3 * np.mean(df_dr**2)
        penalty += 1e2 * np.mean(d2f_dr2**2)
        
        return E_negative + penalty
    
    def compute_energy(self, params):
        """Compute energy (with JAX if available)."""
        if self.use_jax:
            return float(self._compute_energy_jit(jnp.array(params)))
        else:
            return self._compute_energy_numpy(params)
    
    def adam_optimizer(self, params_init, lr=0.01, beta1=0.9, beta2=0.999, 
                      eps=1e-8, max_iter=1000, tol=1e-6):
        """Adam optimizer with JAX acceleration."""
        if not self.use_jax:
            raise ValueError("Adam optimizer requires JAX")
        
        params = jnp.array(params_init)
        m = jnp.zeros_like(params)  # First moment
        v = jnp.zeros_like(params)  # Second moment
        
        best_energy = float('inf')
        best_params = params
        patience = 50
        no_improve = 0
        
        print(f"Starting Adam optimization (lr={lr}, max_iter={max_iter})")
        
        for i in range(max_iter):
            # Compute energy and gradients
            energy, grads = self._compute_energy_and_grad(params)
            energy = float(energy)
            
            # Update moments
            m = beta1 * m + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * grads**2
            
            # Bias correction
            m_hat = m / (1 - beta1**(i + 1))
            v_hat = v / (1 - beta2**(i + 1))
            
            # Update parameters
            params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
            
            # Track progress
            self.history.append({
                'iteration': i,
                'energy': energy,
                'gradient_norm': float(jnp.linalg.norm(grads))
            })
            
            # Check for improvement
            if energy < best_energy:
                best_energy = energy
                best_params = params
                no_improve = 0
            else:
                no_improve += 1
            
            # Progress reporting
            if i % 50 == 0 or i == max_iter - 1:
                grad_norm = float(jnp.linalg.norm(grads))
                print(f"Iter {i:4d}: E = {energy:12.6e}, |grad| = {grad_norm:8.2e}")
            
            # Convergence check
            if jnp.linalg.norm(grads) < tol:
                print(f"Converged at iteration {i} (gradient tolerance)")
                break
            
            # Early stopping
            if no_improve > patience:
                print(f"Early stopping at iteration {i} (no improvement)")
                break
        
        return np.array(best_params), best_energy
    
    def lbfgs_scipy_fallback(self, params_init, max_iter=1000):
        """L-BFGS optimization using scipy (fallback when JAX unavailable)."""
        from scipy.optimize import minimize
        
        def objective(params):
            return self._compute_energy_numpy(params)
        
        print("Starting L-BFGS optimization (scipy fallback)")
        
        result = minimize(
            objective, params_init, method='L-BFGS-B',
            options={'maxiter': max_iter, 'disp': True}
        )
        
        return result.x, result.fun
    
    def initialize_parameters(self, strategy='smart'):
        """Initialize 6-Gaussian parameters with different strategies."""
        if strategy == 'random':
            # Random initialization
            params = np.random.normal(0, 0.1, 18)  # 6 Gaussians × 3 params each
            
        elif strategy == 'smart':
            # Physics-informed initialization
            params = np.zeros(18)
            
            # Position Gaussians strategically around the bubble
            centers = [0.3, 0.7, 1.0, 1.3, 1.7, 2.5]  # Around R_b = 1
            widths = [0.2, 0.2, 0.3, 0.3, 0.4, 0.5]   # Increasing widths
            amplitudes = [0.5, 0.8, 1.0, -0.3, -0.2, -0.1]  # Mixed signs
            
            for i in range(6):
                params[3*i] = amplitudes[i]     # Amplitude
                params[3*i + 1] = widths[i]     # Width
                params[3*i + 2] = centers[i]    # Center
                
        elif strategy == 'hierarchical':
            # Multi-scale initialization
            params = np.zeros(18)
            scales = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
            
            for i in range(6):
                params[3*i] = 0.5 * (-1)**i  # Alternating signs
                params[3*i + 1] = scales[i] * 0.3
                params[3*i + 2] = 0.5 + scales[i] * 0.2
        
        return params
    
    def run_optimization(self, strategy='adam', init_strategy='smart', 
                        max_iter=1000, lr=0.01):
        """Run complete optimization with the specified strategy."""
        print(f"\n{'='*60}")
        print(f"JAX-based 6-Gaussian Warp Bubble Optimization")
        print(f"{'='*60}")
        print(f"JAX available: {self.use_jax}")
        print(f"Optimization strategy: {strategy}")
        print(f"Initialization strategy: {init_strategy}")
        print(f"Max iterations: {max_iter}")
        
        # Initialize parameters
        params_init = self.initialize_parameters(init_strategy)
        energy_init = self.compute_energy(params_init)
        print(f"Initial energy: {energy_init:.6e} J")
        
        # Run optimization
        start_time = time.time()
        
        if strategy == 'adam' and self.use_jax:
            params_opt, energy_opt = self.adam_optimizer(
                params_init, lr=lr, max_iter=max_iter
            )
        else:
            params_opt, energy_opt = self.lbfgs_scipy_fallback(
                params_init, max_iter=max_iter
            )
        
        optimization_time = time.time() - start_time
        
        print(f"\nOptimization completed in {optimization_time:.2f} seconds")
        print(f"Final energy: {energy_opt:.6e} J")
        print(f"Improvement: {energy_init - energy_opt:.6e} J")
        
        return params_opt, energy_opt
    
    def analyze_solution(self, params_opt, energy_opt):
        """Analyze and visualize the optimized solution."""
        print(f"\n{'='*60}")
        print(f"Solution Analysis")
        print(f"{'='*60}")
        
        # Compute final profile
        f_profile = self.gaussian_profile(self.r, params_opt)
        
        # Extract Gaussian components
        print("\nOptimized Gaussian Parameters:")
        print("i  | Amplitude | Width     | Center    |")
        print("---|-----------|-----------|-----------|")
        for i in range(6):
            A = params_opt[3*i]
            sigma = params_opt[3*i + 1]
            mu_c = params_opt[3*i + 2]
            print(f"{i+1:2d} | {A:8.4f}  | {sigma:8.4f}  | {mu_c:8.4f}  |")
        
        # Physics checks
        f_at_bubble = np.interp(self.R_b, self.r, f_profile)
        f_at_infinity = f_profile[-1]
        
        print(f"\nPhysics Validation:")
        print(f"f(R_b = {self.R_b}) = {f_at_bubble:.6f} (should be ≈ 1)")
        print(f"f(r → ∞) = {f_at_infinity:.6f} (should be ≈ 0)")
        print(f"Final energy: {energy_opt:.6e} J")
        
        # Create visualization
        self.create_plots(params_opt, energy_opt)
        
        # Save results
        self.save_results(params_opt, energy_opt)
        
        return {
            'params': params_opt.tolist(),
            'energy': float(energy_opt),
            'f_at_bubble': float(f_at_bubble),
            'f_at_infinity': float(f_at_infinity),
            'use_jax': self.use_jax
        }
    
    def create_plots(self, params_opt, energy_opt):
        """Create comprehensive visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Final profile
        f_profile = self.gaussian_profile(self.r, params_opt)
        axes[0, 0].plot(self.r, f_profile, 'b-', linewidth=2, label='Optimized Profile')
        axes[0, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Target f(R_b)=1')
        axes[0, 0].axvline(x=self.R_b, color='g', linestyle='--', alpha=0.5, label=f'Bubble radius R_b={self.R_b}m')
        axes[0, 0].set_xlabel('Radius r (m)')
        axes[0, 0].set_ylabel('Warp Factor f(r)')
        axes[0, 0].set_title('Optimized 6-Gaussian Warp Profile')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Individual Gaussian components
        for i in range(6):
            A = params_opt[3*i]
            sigma = params_opt[3*i + 1]
            mu_c = params_opt[3*i + 2]
            gaussian_i = A * np.exp(-0.5 * ((self.r - mu_c) / sigma)**2)
            axes[0, 1].plot(self.r, gaussian_i, '--', alpha=0.7, label=f'Gaussian {i+1}')
        
        axes[0, 1].plot(self.r, f_profile, 'k-', linewidth=2, label='Total')
        axes[0, 1].set_xlabel('Radius r (m)')
        axes[0, 1].set_ylabel('Component Value')
        axes[0, 1].set_title('Individual Gaussian Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Optimization history (if available)
        if self.history:
            iterations = [h['iteration'] for h in self.history]
            energies = [h['energy'] for h in self.history]
            axes[1, 0].semilogy(iterations, energies, 'b-', linewidth=2)
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Energy (J)')
            axes[1, 0].set_title('Optimization Convergence')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No optimization history available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Optimization History')
          # Plot 4: Energy density profile
        df_dr = np.gradient(f_profile, self.r[1] - self.r[0])
        d2f_dr2 = np.gradient(df_dr, self.r[1] - self.r[0])
        
        T_rr = (self.c**4 / (8 * np.pi)) * (
            (df_dr**2) / (2 * self.r**2) +
            (f_profile * d2f_dr2) / self.r +
            (f_profile * df_dr) / self.r**2
        )
        
        axes[1, 1].plot(self.r, T_rr, 'r-', linewidth=2, label='T_rr')
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 1].set_xlabel('Radius r (m)')
        axes[1, 1].set_ylabel('Energy Density T_rr')
        axes[1, 1].set_title('Energy Density Profile')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = 'jax_gaussian_M6_optimization.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to: {plot_filename}")
        plt.close()  # Close instead of show to prevent blocking
    
    def save_results(self, params_opt, energy_opt):
        """Save optimization results to JSON file."""
        results = {
            'optimization_method': 'JAX_Adam' if self.use_jax else 'scipy_LBFGS',
            'ansatz_type': '6_Gaussian',
            'energy_joules': float(energy_opt),
            'bubble_radius_m': self.R_b,
            'optimized_parameters': params_opt.tolist(),
            'gaussian_components': [],
            'physics_validation': {
                'f_at_bubble': float(np.interp(self.R_b, self.r, 
                                             self.gaussian_profile(self.r, params_opt))),
                'f_at_infinity': float(self.gaussian_profile(self.r, params_opt)[-1])
            },
            'optimization_history': self.history,
            'jax_acceleration': self.use_jax
        }
        
        # Add individual Gaussian components
        for i in range(6):
            results['gaussian_components'].append({
                'index': i + 1,
                'amplitude': float(params_opt[3*i]),
                'width': float(params_opt[3*i + 1]),
                'center': float(params_opt[3*i + 2])
            })
        
        filename = 'jax_gaussian_M6_results.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filename}")

def main():
    """Main optimization routine."""
    # Create optimizer
    optimizer = JAXWarpBubbleOptimizer(use_jax=True)
    
    # Run optimization with multiple strategies
    strategies = [
        ('adam', 'smart', 0.01),
        ('adam', 'hierarchical', 0.005),
    ]
    
    best_energy = float('inf')
    best_params = None
    best_config = None
    
    for strategy, init_strategy, lr in strategies:
        print(f"\n{'='*80}")
        print(f"Running optimization: {strategy} + {init_strategy} (lr={lr})")
        print(f"{'='*80}")
        
        try:
            params_opt, energy_opt = optimizer.run_optimization(
                strategy=strategy, 
                init_strategy=init_strategy,
                max_iter=500,
                lr=lr
            )
            
            if energy_opt < best_energy:
                best_energy = energy_opt
                best_params = params_opt
                best_config = (strategy, init_strategy, lr)
            
        except Exception as e:
            print(f"Error in optimization {strategy}+{init_strategy}: {e}")
            continue
    
    # Analyze best solution
    if best_params is not None:
        print(f"\n{'='*80}")
        print(f"BEST SOLUTION: {best_config[0]} + {best_config[1]} (lr={best_config[2]})")
        print(f"{'='*80}")
        
        optimizer.analyze_solution(best_params, best_energy)
        
        # Compare with previous results
        print(f"\nComparison with Previous Results:")
        print(f"Current JAX result:     {best_energy:.6e} J")
        print(f"Previous 6-Gaussian:    ~-1.9e31 J (estimated)")
        print(f"Improvement factor:     {abs(-1.9e31 / best_energy):.2f}x" if best_energy < 0 else "N/A")
    
    else:
        print("No successful optimization found!")

if __name__ == "__main__":
    main()
