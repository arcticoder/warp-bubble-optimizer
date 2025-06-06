#!/usr/bin/env python3
"""
Variational Optimization Utilities for Warp Bubble Metric Ansatzes

This module provides tools for variational optimization of metric ansatzes,
including gradient-based methods, constraint handling, and energy minimization.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.integrate import quad, dblquad
from typing import Dict, List, Tuple, Callable, Optional, Any
import warnings

class MetricAnsatzOptimizer:
    """
    Variational optimizer for warp bubble metric ansatzes.
    
    Supports various optimization strategies:
    - Gradient descent with energy constraints
    - Differential evolution for global optimization
    - Basin hopping for escaping local minima
    - Constrained optimization with stability requirements
    """
    
    def __init__(self, 
                 metric_ansatz: Callable,
                 energy_functional: Callable,
                 constraints: Optional[List[Dict]] = None):
        """
        Initialize the optimizer.
        
        Args:
            metric_ansatz: Function that takes parameters and returns metric
            energy_functional: Function that computes total energy given metric
            constraints: List of constraint dictionaries for scipy.optimize
        """
        self.metric_ansatz = metric_ansatz
        self.energy_functional = energy_functional
        self.constraints = constraints or []
        
        # Optimization history
        self.optimization_history = []
        self.best_params = None
        self.best_energy = np.inf
        
    def objective_function(self, params: np.ndarray) -> float:
        """
        Objective function for optimization (energy to minimize).
        
        Args:
            params: Parameters defining the metric ansatz
            
        Returns:
            Total energy (to be minimized)
        """
        try:
            # Generate metric from ansatz
            metric = self.metric_ansatz(params)
            
            # Compute energy
            energy = self.energy_functional(metric)
            
            # Store in history
            self.optimization_history.append({
                'params': params.copy(),
                'energy': energy,
                'metric': metric
            })
            
            # Update best if improved
            if energy < self.best_energy:
                self.best_energy = energy
                self.best_params = params.copy()
                
            return energy
            
        except Exception as e:
            warnings.warn(f"Error in objective function: {e}")
            return 1e10  # Large penalty for invalid parameters
    
    def gradient_descent_optimize(self, 
                                 initial_params: np.ndarray,
                                 method: str = 'L-BFGS-B',
                                 bounds: Optional[List[Tuple]] = None,
                                 options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Gradient-based optimization.
        
        Args:
            initial_params: Starting parameters
            method: Optimization method ('L-BFGS-B', 'SLSQP', etc.)
            bounds: Parameter bounds
            options: Optimization options
            
        Returns:
            Optimization result dictionary
        """
        if options is None:
            options = {'maxiter': 1000, 'ftol': 1e-9}
            
        result = minimize(
            fun=self.objective_function,
            x0=initial_params,
            method=method,
            bounds=bounds,
            constraints=self.constraints,
            options=options
        )
        
        return {
            'success': result.success,
            'params': result.x,
            'energy': result.fun,
            'message': result.message,
            'nit': result.nit,
            'nfev': result.nfev
        }
    
    def global_optimize(self, 
                       param_bounds: List[Tuple],
                       method: str = 'differential_evolution',
                       **kwargs) -> Dict[str, Any]:
        """
        Global optimization using stochastic methods.
        
        Args:
            param_bounds: Bounds for each parameter
            method: 'differential_evolution' or 'basinhopping'
            **kwargs: Additional arguments for the optimizer
            
        Returns:
            Optimization result dictionary
        """
        if method == 'differential_evolution':
            result = differential_evolution(
                func=self.objective_function,
                bounds=param_bounds,
                **kwargs
            )
        elif method == 'basinhopping':
            # Need initial point for basin hopping
            x0 = np.array([0.5 * (b[0] + b[1]) for b in param_bounds])
            result = basinhopping(
                func=self.objective_function,
                x0=x0,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown global optimization method: {method}")
            
        return {
            'success': result.success,
            'params': result.x,
            'energy': result.fun,
            'message': getattr(result, 'message', 'No message'),
            'nit': getattr(result, 'nit', 0),
            'nfev': getattr(result, 'nfev', 0)
        }
    
    def multi_start_optimize(self,
                           param_bounds: List[Tuple],
                           n_starts: int = 10,
                           method: str = 'L-BFGS-B') -> Dict[str, Any]:
        """
        Multi-start optimization to find global minimum.
        
        Args:
            param_bounds: Parameter bounds
            n_starts: Number of random starting points
            method: Local optimization method
            
        Returns:
            Best optimization result
        """
        best_result = None
        best_energy = np.inf
        
        for i in range(n_starts):
            # Random starting point
            x0 = np.array([
                np.random.uniform(b[0], b[1]) for b in param_bounds
            ])
            
            # Local optimization
            result = self.gradient_descent_optimize(
                initial_params=x0,
                method=method,
                bounds=param_bounds
            )
            
            # Check if this is the best so far
            if result['energy'] < best_energy:
                best_energy = result['energy']
                best_result = result
                best_result['start_number'] = i
        
        return best_result

class EnergyConstraintHandler:
    """
    Handle energy and stability constraints for metric optimization.
    """
    
    def __init__(self, 
                 max_negative_energy: float = -1e-6,
                 stability_threshold: float = 1e-3):
        """
        Initialize constraint handler.
        
        Args:
            max_negative_energy: Maximum allowed negative energy
            stability_threshold: Minimum stability margin
        """
        self.max_negative_energy = max_negative_energy
        self.stability_threshold = stability_threshold
    
    def energy_constraint(self, params: np.ndarray, 
                         energy_func: Callable) -> float:
        """
        Energy constraint: negative energy should be minimized.
        
        Returns:
            Constraint value (>= 0 for feasible)
        """
        energy = energy_func(params)
        return -(energy - self.max_negative_energy)  # Constraint: energy >= max_negative_energy
    
    def stability_constraint(self, params: np.ndarray,
                           stability_func: Callable) -> float:
        """
        Stability constraint: ensure sufficient stability margin.
        
        Returns:
            Constraint value (>= 0 for feasible)
        """
        stability = stability_func(params)
        return stability - self.stability_threshold
    
    def generate_constraints(self, 
                           energy_func: Callable,
                           stability_func: Optional[Callable] = None) -> List[Dict]:
        """
        Generate constraint dictionaries for scipy.optimize.
        
        Args:
            energy_func: Function to compute energy given parameters
            stability_func: Optional function to compute stability
            
        Returns:
            List of constraint dictionaries
        """
        constraints = []
        
        # Energy constraint
        constraints.append({
            'type': 'ineq',
            'fun': lambda params: self.energy_constraint(params, energy_func)
        })
        
        # Stability constraint (if provided)
        if stability_func is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda params: self.stability_constraint(params, stability_func)
            })
        
        return constraints

class VariationalCalculus:
    """
    Tools for variational calculus in metric optimization.
    """
    
    @staticmethod
    def numerical_gradient(func: Callable, 
                          params: np.ndarray, 
                          h: float = 1e-8) -> np.ndarray:
        """
        Compute numerical gradient using central differences.
        
        Args:
            func: Function to differentiate
            params: Point at which to evaluate gradient
            h: Step size for finite differences
            
        Returns:
            Gradient vector
        """
        grad = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            
            params_plus[i] += h
            params_minus[i] -= h
            
            grad[i] = (func(params_plus) - func(params_minus)) / (2 * h)
            
        return grad
    
    @staticmethod
    def numerical_hessian(func: Callable,
                         params: np.ndarray,
                         h: float = 1e-6) -> np.ndarray:
        """
        Compute numerical Hessian matrix.
        
        Args:
            func: Function to differentiate
            params: Point at which to evaluate Hessian
            h: Step size for finite differences
            
        Returns:
            Hessian matrix
        """
        n = len(params)
        hessian = np.zeros((n, n))
        
        f0 = func(params)
        
        for i in range(n):
            for j in range(n):
                params_pp = params.copy()
                params_pm = params.copy()
                params_mp = params.copy()
                params_mm = params.copy()
                
                params_pp[i] += h
                params_pp[j] += h
                
                params_pm[i] += h
                params_pm[j] -= h
                
                params_mp[i] -= h
                params_mp[j] += h
                
                params_mm[i] -= h
                params_mm[j] -= h
                
                hessian[i, j] = (func(params_pp) - func(params_pm) - 
                               func(params_mp) + func(params_mm)) / (4 * h * h)
        
        return hessian

def create_ansatz_optimizer(ansatz_type: str = 'polynomial') -> Callable:
    """
    Factory function to create specific metric ansatz optimizers.
    
    Args:
        ansatz_type: Type of ansatz ('polynomial', 'exponential', 'soliton')
        
    Returns:
        Optimizer function for the specified ansatz type
    """
    if ansatz_type == 'polynomial':
        def polynomial_ansatz(params):
            """Polynomial ansatz: f(r) = sum(a_i * r^i)"""
            def metric_func(r):
                return sum(params[i] * r**i for i in range(len(params)))
            return metric_func
        return polynomial_ansatz
        
    elif ansatz_type == 'exponential':
        def exponential_ansatz(params):
            """Exponential ansatz: f(r) = exp(sum(a_i * r^i))"""
            def metric_func(r):
                exponent = sum(params[i] * r**i for i in range(len(params)))
                return np.exp(exponent)
            return metric_func
        return exponential_ansatz
        
    elif ansatz_type == 'soliton':
        def soliton_ansatz(params):
            """Soliton-like ansatz: f(r) = tanh(a*r + b) + c"""
            def metric_func(r):
                a, b, c = params[:3]
                return np.tanh(a * r + b) + c
            return metric_func
        return soliton_ansatz
        
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")
