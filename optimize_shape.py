#!/usr/bin/env python3
"""
Shape Function Optimization for Warp Bubbles
===========================================

This module implements JAX-accelerated optimization of metric ansatz
functions to minimize negative energy requirements while respecting
physical constraints.
"""

import numpy as np
from typing import Tuple, Callable, Dict, Optional
import logging

# JAX imports with fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    JAX_AVAILABLE = True
except ImportError:
    # Fallback to NumPy
    jnp = np
    JAX_AVAILABLE = False
    
    def jit(func):
        return func
    def grad(func):
        def grad_func(x, *args, **kwargs):
            # Simple finite difference approximation
            eps = 1e-8
            x = np.asarray(x)
            grads = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                grads[i] = (func(x_plus, *args, **kwargs) - func(x_minus, *args, **kwargs)) / (2 * eps)
            return grads
        return grad_func
    def vmap(func, *args, **kwargs):
        return func

logger = logging.getLogger(__name__)

# Import ProgressTracker for enhanced progress monitoring
try:
    from progress_tracker import ProgressTracker
    PROGRESS_TRACKER_AVAILABLE = True
except ImportError:
    PROGRESS_TRACKER_AVAILABLE = False
    logger.warning("ProgressTracker not available. Using basic progress reporting.")

class WarpShapeOptimizer:
    """Optimizes warp bubble shape functions using JAX acceleration."""
    
    def __init__(self, ansatz_type: str = "gaussian", R_max: float = 10.0):
        """Initialize the shape optimizer.
        
        Args:
            ansatz_type: Type of ansatz ("gaussian", "polynomial", "hybrid")
            R_max: Maximum radius for integration
        """
        self.ansatz_type = ansatz_type
        self.R_max = R_max
        self.c = 299792458.0  # Speed of light
        
        # Set up the ansatz function
        if ansatz_type == "gaussian":
            self.f_r = self._gaussian_ansatz
            self.param_names = ["A", "sigma"]
            self.default_params = jnp.array([1.0, 2.0])
            self.param_bounds = [(0.1, 5.0), (0.5, 5.0)]
        elif ansatz_type == "polynomial":
            self.f_r = self._polynomial_ansatz
            self.param_names = ["a", "b", "c"]
            self.default_params = jnp.array([1.0, 0.5, 0.1])
            self.param_bounds = [(0.1, 2.0), (0.1, 1.0), (0.01, 0.5)]
        elif ansatz_type == "hybrid":
            self.f_r = self._hybrid_ansatz
            self.param_names = ["A_gauss", "sigma_gauss", "a_poly", "b_poly"]
            self.default_params = jnp.array([1.0, 2.0, 0.5, 0.2])
            self.param_bounds = [(0.1, 3.0), (0.5, 4.0), (0.1, 1.0), (0.1, 0.5)]
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")
            
        # JIT compile the functions if JAX is available
        if JAX_AVAILABLE:
            self.f_r = jit(self.f_r)
            self.neg_energy_integral = jit(self._neg_energy_integral)
            self.grad_objective = jit(grad(self._neg_energy_integral))
        else:
            self.neg_energy_integral = self._neg_energy_integral
            self.grad_objective = grad(self._neg_energy_integral)
    
    @staticmethod
    def _gaussian_ansatz(r: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Gaussian warp function: f(r) = 1 - A * exp(-(r/σ)²)"""
        A, sigma = theta
        return 1.0 - A * jnp.exp(-(r/sigma)**2)
    
    @staticmethod
    def _polynomial_ansatz(r: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Polynomial warp function: f(r) = 1 - a*r² - b*r⁴ - c*r⁶"""
        a, b, c = theta
        r_norm = r / 10.0  # Normalize to avoid numerical issues
        return 1.0 - a*r_norm**2 - b*r_norm**4 - c*r_norm**6
    
    @staticmethod
    def _hybrid_ansatz(r: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Hybrid warp function combining Gaussian and polynomial terms"""
        A_gauss, sigma_gauss, a_poly, b_poly = theta
        r_norm = r / 10.0
        gaussian_part = A_gauss * jnp.exp(-(r/sigma_gauss)**2)
        poly_part = a_poly * r_norm**2 + b_poly * r_norm**4
        return 1.0 - gaussian_part - poly_part
    
    def _neg_energy_integral(self, theta: jnp.ndarray) -> float:
        """Compute the negative energy integral for given parameters.
        
        This is a simplified model where the stress-energy tensor
        is approximated based on metric derivatives.
        """
        # Create radial grid
        rs = jnp.linspace(0.01, self.R_max, 1000)  # Avoid r=0 singularity
        
        # Compute metric function
        f_vals = self.f_r(rs, theta)
        
        # Compute derivatives (simplified stress-energy)
        df_dr = jnp.gradient(f_vals, rs[1] - rs[0])
        d2f_dr2 = jnp.gradient(df_dr, rs[1] - rs[0])
        
        # Approximate T_00 component (negative energy density)
        # This is a simplified model - real calculation would involve
        # full Einstein tensor computation
        T_00 = -(df_dr**2 + rs * df_dr * d2f_dr2) / (8 * jnp.pi)
        
        # Only count negative energy contributions
        neg_T_00 = jnp.where(T_00 < 0, -T_00, 0.0)
        
        # Integrate over volume (4πr²dr)
        integrand = neg_T_00 * rs**2
        total_neg_energy = 4 * jnp.pi * jnp.trapz(integrand, rs)
        
        return total_neg_energy
    
    def optimize(self, 
                 max_iter: int = 200, 
                 learning_rate: float = 1e-2,
                 initial_params: Optional[jnp.ndarray] = None) -> Dict:
        """Optimize the shape parameters to minimize negative energy.
        
        Args:
            max_iter: Maximum number of optimization iterations
            learning_rate: Gradient descent learning rate
            initial_params: Starting parameters (uses defaults if None)
            
        Returns:
            Dictionary with optimization results
        """
        # Initialize parameters
        if initial_params is None:
            theta = self.default_params.copy()
        else:
            theta = jnp.array(initial_params)
        
        # Store optimization history
        history = {
            'energy': [],
            'params': [],
            'gradients': []
        }
          logger.info(f"Starting optimization with {self.ansatz_type} ansatz")
        logger.info(f"Initial params: {theta}")
        
        # Initialize progress tracker
        if PROGRESS_TRACKER_AVAILABLE:
            progress = ProgressTracker(max_iter, f"Optimizing {self.ansatz_type} warp shape")
            progress.start({
                'ansatz_type': self.ansatz_type,
                'initial_params': str(theta),
                'learning_rate': learning_rate,
                'max_iterations': max_iter
            })
        
        # Optimization loop
        for i in range(max_iter):
            # Compute energy and gradient
            energy = self.neg_energy_integral(theta)
            grad_energy = self.grad_objective(theta)
            
            # Store history
            history['energy'].append(float(energy))
            history['params'].append(np.array(theta))
            history['gradients'].append(np.array(grad_energy))
            
            # Gradient descent update
            theta = theta - learning_rate * grad_energy
            
            # Apply parameter bounds
            for j, (lower, upper) in enumerate(self.param_bounds):
                theta = theta.at[j].set(jnp.clip(theta[j], lower, upper)) if JAX_AVAILABLE else \
                        jnp.clip(theta, lower, upper)
            
            # Progress reporting with enhanced tracking
            if PROGRESS_TRACKER_AVAILABLE and i % 20 == 0:
                grad_norm = jnp.linalg.norm(grad_energy)
                step_data = {
                    'energy': f"{energy:.6e}",
                    'gradient_norm': f"{grad_norm:.3e}",
                    'parameters': str(theta),
                    'convergence_rate': f"{grad_norm/energy:.2e}" if energy != 0 else "N/A"
                }
                progress.update(f"Iteration {i}", i, step_data)
            elif i % 20 == 0:
                grad_norm = jnp.linalg.norm(grad_energy)
                logger.info(f"Step {i:3d}: E = {energy:.6e}, |grad| = {grad_norm:.3e}, θ = {theta}")
        
        # Complete progress tracking
        if PROGRESS_TRACKER_AVAILABLE:
            final_energy = self.neg_energy_integral(theta)
            final_grad_norm = jnp.linalg.norm(self.grad_objective(theta))
            progress.complete({
                'final_energy': f"{final_energy:.6e}",
                'final_gradient_norm': f"{final_grad_norm:.3e}",
                'optimization_success': final_grad_norm < 1e-6,
                'iterations_completed': max_iter
            })
        
        # Final results
        final_energy = self.neg_energy_integral(theta)
        
        results = {
            'optimal_params': np.array(theta),
            'final_energy': float(final_energy),
            'param_names': self.param_names,
            'ansatz_type': self.ansatz_type,
            'history': history,
            'converged': len(history['energy']) > 10 and 
                        abs(history['energy'][-1] - history['energy'][-10]) < 1e-8
        }
        
        logger.info(f"Optimization complete!")
        logger.info(f"Final energy: {final_energy:.6e}")
        logger.info(f"Optimal parameters: {dict(zip(self.param_names, theta))}")
        
        return results
    
    def evaluate_shape(self, theta: jnp.ndarray, r_points: Optional[jnp.ndarray] = None) -> Dict:
        """Evaluate the optimized shape function at given points.
        
        Args:
            theta: Shape parameters
            r_points: Radial points to evaluate (uses default grid if None)
            
        Returns:
            Dictionary with shape evaluation results
        """
        if r_points is None:
            r_points = jnp.linspace(0, self.R_max, 200)
        
        # Evaluate shape function
        f_vals = self.f_r(r_points, theta)
        
        # Compute derivatives
        dr = r_points[1] - r_points[0] if len(r_points) > 1 else 0.01
        df_dr = jnp.gradient(f_vals, dr)
        
        # Energy density (simplified)
        T_00 = -(df_dr**2) / (8 * jnp.pi)
        
        return {
            'r': np.array(r_points),
            'f': np.array(f_vals),
            'df_dr': np.array(df_dr),
            'energy_density': np.array(T_00),
            'total_energy': float(self.neg_energy_integral(theta))
        }

def demo_shape_optimization():
    """Demonstrate shape optimization for different ansätze."""
    print("=" * 60)
    print("WARP SHAPE OPTIMIZATION DEMO")
    print("=" * 60)
    
    if not JAX_AVAILABLE:
        print("⚠️  JAX not available - using NumPy fallback")
    
    ansatz_types = ["gaussian", "polynomial", "hybrid"]
    results = {}
    
    for ansatz in ansatz_types:
        print(f"\n🔍 Optimizing {ansatz} ansatz...")
        
        optimizer = WarpShapeOptimizer(ansatz_type=ansatz)
        result = optimizer.optimize(max_iter=100, learning_rate=1e-3)
        results[ansatz] = result
        
        print(f"✅ {ansatz.capitalize()} optimization complete:")
        print(f"   Final energy: {result['final_energy']:.6e}")
        print(f"   Converged: {result['converged']}")
    
    # Compare results
    print(f"\n📊 COMPARISON:")
    for ansatz, result in results.items():
        print(f"   {ansatz:12s}: E = {result['final_energy']:.6e}")
    
    # Find best ansatz
    best_ansatz = min(results.keys(), key=lambda k: results[k]['final_energy'])
    print(f"\n🏆 Best ansatz: {best_ansatz}")
    
    return results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run demonstration
    demo_shape_optimization()
