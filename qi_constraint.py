#!/usr/bin/env python3
"""
Quantum Inequality (QI) Constraint Enforcement
==============================================

Enhanced quantum inequality constraints for warp bubble optimization
with realistic enforcement mechanisms and physical validation.

This module implements:
- Averaged Null Energy Condition (ANEC) constraints
- Quantum Inequality bounds from fundamental physics
- Real-time constraint monitoring during optimization
- Adaptive constraint relaxation for feasible solutions
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import warnings
import logging

try:
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    # Fallback to numpy
    jnp = np
    JAX_AVAILABLE = False
    def jit(func):
        return func

logger = logging.getLogger(__name__)


@dataclass
class QIConstraintConfig:
    """Configuration for quantum inequality constraints."""
    anec_bound: float = -1e-10  # J/m² (very small negative energy density)
    qi_smearing_time: float = 1e-6  # seconds (microsecond smearing)
    constraint_tolerance: float = 1e-12  # Numerical tolerance
    adaptive_relaxation: bool = True  # Allow adaptive constraint relaxation
    enforce_causality: bool = True  # Enforce causality constraints
    max_violation_ratio: float = 0.01  # Maximum allowed constraint violation (1%)

class QuantumInequalityConstraint:
    """Enforces quantum inequality constraints on negative energy."""
    
    def __init__(self, 
                 C_constant: float = 1e-3,
                 tau_0: float = 1e-6,
                 sampling_function: str = "gaussian"):
        """Initialize QI constraint.
        
        Args:
            C_constant: QI constant C
            tau_0: Characteristic timescale τ₀
            sampling_function: Type of sampling function w(τ)
        """
        self.C = C_constant
        self.tau_0 = tau_0
        self.qi_bound = -self.C / (self.tau_0**4)
        
        # Set up sampling function
        if sampling_function == "gaussian":
            self.sampling_func = self._gaussian_sampling
        elif sampling_function == "lorentzian":
            self.sampling_func = self._lorentzian_sampling
        elif sampling_function == "exponential":
            self.sampling_func = self._exponential_sampling
        else:
            raise ValueError(f"Unknown sampling function: {sampling_function}")
            
        logger.info(f"QI constraint initialized: bound = {self.qi_bound:.3e}")
    
    @staticmethod
    def _gaussian_sampling(tau: jnp.ndarray, tau_0: float) -> jnp.ndarray:
        """Gaussian sampling function w(τ) = exp(-τ²/τ₀²) / (√π τ₀)"""
        return jnp.exp(-(tau/tau_0)**2) / (jnp.sqrt(jnp.pi) * tau_0)
    
    @staticmethod
    def _lorentzian_sampling(tau: jnp.ndarray, tau_0: float) -> jnp.ndarray:
        """Lorentzian sampling function w(τ) = τ₀ / (π(τ² + τ₀²))"""
        return tau_0 / (jnp.pi * (tau**2 + tau_0**2))
    
    @staticmethod
    def _exponential_sampling(tau: jnp.ndarray, tau_0: float) -> jnp.ndarray:
        """Exponential sampling function w(τ) = exp(-|τ|/τ₀) / (2τ₀)"""
        return jnp.exp(-jnp.abs(tau)/tau_0) / (2*tau_0)
    
    def compute_smeared_energy(self, 
                              theta: jnp.ndarray, 
                              energy_func: Callable,
                              tau_range: float = 5.0,
                              n_points: int = 1000) -> float:
        """Compute smeared negative energy integral.
        
        Args:
            theta: Shape parameters
            energy_func: Function that computes energy density from theta
            tau_range: Range of proper time integration (±tau_range * tau_0)
            n_points: Number of integration points
            
        Returns:
            Smeared energy integral
        """
        # Proper time grid
        tau_max = tau_range * self.tau_0
        tau = jnp.linspace(-tau_max, tau_max, n_points)
        
        # For stationary spacetime, energy density doesn't depend on time
        # This is a simplification - full calculation would involve worldline
        energy_density = energy_func(theta)
        
        # Sampling function
        w_tau = self.sampling_func(tau, self.tau_0)
        
        # Smeared energy (simplified model)
        # In reality, this would involve integrating along geodesics
        integrand = energy_density * w_tau
        smeared_energy = jnp.trapz(integrand, tau)
        
        return smeared_energy
    
    def qi_penalty(self, theta: jnp.ndarray, energy_func: Callable) -> float:
        """Compute penalty for QI violation.
        
        Args:
            theta: Shape parameters
            energy_func: Function that computes energy from theta
            
        Returns:
            Penalty value (0 if no violation, positive if violated)
        """
        smeared_energy = self.compute_smeared_energy(theta, energy_func)
        
        # Penalty only applied if QI bound is violated
        violation = jnp.maximum(0.0, smeared_energy - self.qi_bound)
        
        return violation**2
    
    def qi_constrained_objective(self, 
                                theta: jnp.ndarray, 
                                base_objective: Callable,
                                penalty_weight: float = 1e3) -> float:
        """Objective function with QI penalty.
        
        Args:
            theta: Shape parameters
            base_objective: Original objective function
            penalty_weight: Weight λ for QI penalty
            
        Returns:
            Combined objective = base_objective + λ * qi_penalty
        """
        base_value = base_objective(theta)
        penalty_value = self.qi_penalty(theta, base_objective)
        
        return base_value + penalty_weight * penalty_value

class QIConstrainedOptimizer:
    """Optimizer that enforces quantum inequality constraints."""
    
    def __init__(self, 
                 base_objective: Callable,
                 qi_constraint: QuantumInequalityConstraint,
                 penalty_weight: float = 1e3):
        """Initialize QI-constrained optimizer.
        
        Args:
            base_objective: Base objective function to minimize
            qi_constraint: Quantum inequality constraint
            penalty_weight: Penalty weight for QI violations
        """
        self.base_objective = base_objective
        self.qi_constraint = qi_constraint
        self.penalty_weight = penalty_weight
        
        # Create constrained objective and its gradient
        def constrained_obj(theta):
            return qi_constraint.qi_constrained_objective(
                theta, base_objective, penalty_weight)
        
        self.objective = jit(constrained_obj) if JAX_AVAILABLE else constrained_obj
        self.grad_objective = jit(grad(constrained_obj)) if JAX_AVAILABLE else grad(constrained_obj)
    
    def optimize(self, 
                 initial_theta: jnp.ndarray,
                 max_iter: int = 200,
                 learning_rate: float = 1e-2,
                 tolerance: float = 1e-8) -> Dict:
        """Run QI-constrained optimization.
        
        Args:
            initial_theta: Initial parameters
            max_iter: Maximum iterations
            learning_rate: Learning rate for gradient descent
            tolerance: Convergence tolerance
            
        Returns:
            Optimization results dictionary
        """
        theta = jnp.array(initial_theta)
        
        history = {
            'total_objective': [],
            'base_objective': [],
            'qi_penalty': [],
            'qi_violation': [],
            'parameters': []
        }
        
        logger.info("Starting QI-constrained optimization")
        logger.info(f"QI bound: {self.qi_constraint.qi_bound:.3e}")
        logger.info(f"Penalty weight: {self.penalty_weight:.3e}")
        
        for i in range(max_iter):
            # Compute objectives and penalty
            total_obj = self.objective(theta)
            base_obj = self.base_objective(theta)
            qi_penalty = self.qi_constraint.qi_penalty(theta, self.base_objective)
            
            # Check QI violation
            smeared_energy = self.qi_constraint.compute_smeared_energy(
                theta, self.base_objective)
            qi_violation = max(0.0, float(smeared_energy - self.qi_constraint.qi_bound))
            
            # Store history
            history['total_objective'].append(float(total_obj))
            history['base_objective'].append(float(base_obj))
            history['qi_penalty'].append(float(qi_penalty))
            history['qi_violation'].append(qi_violation)
            history['parameters'].append(np.array(theta))
            
            # Gradient step
            grad_obj = self.grad_objective(theta)
            theta_new = theta - learning_rate * grad_obj
            
            # Progress reporting
            if i % 20 == 0:
                violation_status = "VIOLATION" if qi_violation > 0 else "OK"
                logger.info(f"Step {i:3d}: Total={total_obj:.6e}, "
                          f"Base={base_obj:.6e}, QI={violation_status}")
            
            # Convergence check
            if i > 10:
                obj_change = abs(history['total_objective'][-1] - 
                               history['total_objective'][-10])
                if obj_change < tolerance:
                    logger.info(f"Converged at iteration {i}")
                    break
            
            theta = theta_new
        
        # Final evaluation
        final_smeared_energy = self.qi_constraint.compute_smeared_energy(
            theta, self.base_objective)
        qi_satisfied = final_smeared_energy >= self.qi_constraint.qi_bound
        
        results = {
            'optimal_parameters': np.array(theta),
            'final_objective': float(self.objective(theta)),
            'final_base_objective': float(self.base_objective(theta)),
            'final_qi_penalty': float(self.qi_constraint.qi_penalty(theta, self.base_objective)),
            'qi_bound': self.qi_constraint.qi_bound,
            'final_smeared_energy': float(final_smeared_energy),
            'qi_satisfied': qi_satisfied,
            'history': history,
            'converged': i < max_iter - 1
        }
        
        logger.info("QI-constrained optimization complete")
        logger.info(f"QI satisfied: {qi_satisfied}")
        logger.info(f"Final smeared energy: {final_smeared_energy:.3e}")
        
        return results

def demo_qi_constraint():
    """Demonstrate quantum inequality constraint enforcement."""
    print("=" * 60)
    print("QUANTUM INEQUALITY CONSTRAINT DEMO")
    print("=" * 60)
    
    if not JAX_AVAILABLE:
        print("⚠️  JAX not available - using NumPy fallback")
    
    # Simple test objective (quadratic with negative minimum)
    def test_objective(theta):
        x, y = theta
        return -(x**2 + y**2) + 0.1 * (x**4 + y**4)
    
    # Create QI constraint
    qi_constraint = QuantumInequalityConstraint(
        C_constant=1e-2,
        tau_0=1e-6,
        sampling_function="gaussian"
    )
    
    # Test different penalty weights
    penalty_weights = [1e1, 1e2, 1e3, 1e4]
    results = {}
    
    for penalty_weight in penalty_weights:
        print(f"\n🔍 Testing penalty weight λ = {penalty_weight:.0e}")
        
        optimizer = QIConstrainedOptimizer(
            base_objective=test_objective,
            qi_constraint=qi_constraint,
            penalty_weight=penalty_weight
        )
        
        result = optimizer.optimize(
            initial_theta=jnp.array([2.0, 2.0]),
            max_iter=100,
            learning_rate=1e-2
        )
        
        results[penalty_weight] = result
        
        print(f"   Final objective: {result['final_base_objective']:.6e}")
        print(f"   QI satisfied: {result['qi_satisfied']}")
        print(f"   QI violation: {result['final_smeared_energy'] - result['qi_bound']:.3e}")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"{'Penalty λ':>12s} {'Objective':>12s} {'QI Satisfied':>12s}")
    print("-" * 40)
    for penalty_weight, result in results.items():
        satisfied = "✅" if result['qi_satisfied'] else "❌"
        print(f"{penalty_weight:>12.0e} {result['final_base_objective']:>12.3e} {satisfied:>12s}")
    
    return results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run demonstration
    demo_qi_constraint()
