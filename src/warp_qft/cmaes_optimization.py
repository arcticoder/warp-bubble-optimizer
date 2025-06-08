#!/usr/bin/env python3
"""
CMA-ES Optimization Module for Warp Bubble Pipeline

This module provides CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
optimization for warp bubble ansatz parameters, integrated with the complete
simulation pipeline.

Features:
- Parameter optimization for metric ansatz control points
- Multi-objective optimization (energy minimization + stability maximization)
- Integration with existing CMA-ES implementations
- Parallel evaluation support
- Detailed optimization diagnostics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
import time
import json
from dataclasses import dataclass
from pathlib import Path

# Try to import CMA-ES with graceful fallback
HAS_CMA = False
try:
    import cma
    HAS_CMA = True
    logging.info("✅ CMA-ES available for optimization")
except ImportError:
    logging.warning("⚠️  CMA-ES not available. Install with: pip install cma")
    logging.warning("   Falling back to scipy optimization")

# Fallback imports
from scipy.optimize import minimize, differential_evolution

# Import warp bubble components
from .integrated_warp_solver import WarpBubbleSolver, WarpSimulationResult

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Results from CMA-ES optimization."""
    success: bool
    best_parameters: Dict[str, float]
    best_score: float
    best_energy: float
    best_stability: float
    generations: int
    function_evaluations: int
    execution_time: float
    convergence_history: List[float]
    parameter_history: List[Dict[str, float]]
    final_simulation: Optional[WarpSimulationResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'success': self.success,
            'best_score': self.best_score,
            'best_energy': self.best_energy,
            'best_stability': self.best_stability,
            'generations': self.generations,
            'function_evaluations': self.function_evaluations,
            'execution_time': self.execution_time,
            'best_parameters': self.best_parameters
        }


class CMAESOptimizer:
    """
    CMA-ES optimizer for warp bubble ansatz parameters.
    
    Optimizes metric ansatz control points to minimize energy requirements
    while maintaining bubble stability.
    """
    
    def __init__(self, solver: WarpBubbleSolver, param_names: List[str], 
                 bounds: List[Tuple[float, float]], 
                 fixed_radius: float = 10.0, fixed_speed: float = 5000.0):
        """
        Initialize CMA-ES optimizer.
        
        Args:
            solver: WarpBubbleSolver instance to optimize
            param_names: List of parameter names to optimize
            bounds: List of (min, max) bounds for each parameter
            fixed_radius: Fixed bubble radius for optimization
            fixed_speed: Fixed bubble speed for optimization
        """
        self.solver = solver
        self.param_names = param_names
        self.bounds = bounds
        self.fixed_radius = fixed_radius
        self.fixed_speed = fixed_speed
        
        # Validation
        if len(param_names) != len(bounds):
            raise ValueError("Number of parameter names must match number of bounds")
        
        # Optimization state
        self.evaluation_count = 0
        self.convergence_history = []
        self.parameter_history = []
        self.best_result = None
        
        # Objective function weights
        self.energy_weight = 1.0
        self.stability_weight = 0.5
        self.penalty_weight = 10.0
        
    def _params_to_dict(self, param_vector: np.ndarray) -> Dict[str, float]:
        """Convert parameter vector to dictionary."""
        return {name: param_vector[i] for i, name in enumerate(self.param_names)}
    
    def _dict_to_params(self, param_dict: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to vector."""
        return np.array([param_dict[name] for name in self.param_names])
    
    def _apply_bounds(self, param_vector: np.ndarray) -> np.ndarray:
        """Apply parameter bounds with clipping."""
        bounded = np.copy(param_vector)
        for i, (min_val, max_val) in enumerate(self.bounds):
            bounded[i] = np.clip(bounded[i], min_val, max_val)
        return bounded
    
    def _objective_function(self, param_vector: np.ndarray) -> float:
        """
        Objective function for optimization.
        
        Minimizes: energy_weight * |energy| - stability_weight * stability + penalties
        
        Args:
            param_vector: Parameter values to evaluate
            
        Returns:
            Objective value (lower is better)
        """
        self.evaluation_count += 1
        
        # Apply bounds
        bounded_params = self._apply_bounds(param_vector)
        param_dict = self._params_to_dict(bounded_params)
        
        try:
            # Set parameters in solver
            self.solver.set_ansatz_parameters(param_dict)
            
            # Run simulation
            result = self.solver.simulate(
                radius=self.fixed_radius,
                speed=self.fixed_speed,
                detailed_analysis=False
            )
            
            if not result.success:
                logger.warning(f"Simulation failed for params: {param_dict}")
                return 1e6  # Large penalty for failed simulations
            
            # Compute objective components
            energy_term = self.energy_weight * abs(result.energy_total)
            stability_term = -self.stability_weight * result.stability  # Maximize stability
            
            # Add penalty for extreme parameter values
            penalty = 0.0
            for i, (param_val, (min_val, max_val)) in enumerate(zip(bounded_params, self.bounds)):
                if param_val <= min_val + 1e-6 or param_val >= max_val - 1e-6:
                    penalty += self.penalty_weight
            
            objective = energy_term + stability_term + penalty
            
            # Store history
            self.convergence_history.append(objective)
            self.parameter_history.append(param_dict.copy())
            
            # Track best result
            if self.best_result is None or objective < self.best_result['objective']:
                self.best_result = {
                    'objective': objective,
                    'parameters': param_dict.copy(),
                    'simulation': result,
                    'evaluation': self.evaluation_count
                }
            
            if self.evaluation_count % 10 == 0:
                logger.info(f"Eval {self.evaluation_count}: obj={objective:.3e}, "
                           f"E={result.energy_total:.2e}, stab={result.stability:.3f}")
            
            return objective
            
        except Exception as e:
            logger.error(f"Objective evaluation failed: {e}")
            return 1e6  # Large penalty for errors
    
    def optimize(self, generations: int = 50, pop_size: int = 20, 
                 sigma0: float = 0.3, seed: Optional[int] = None) -> OptimizationResult:
        """
        Run CMA-ES optimization.
        
        Args:
            generations: Maximum number of generations
            pop_size: Population size
            sigma0: Initial step size
            seed: Random seed for reproducibility
            
        Returns:
            Optimization results
        """
        start_time = time.time()
        
        # Reset state
        self.evaluation_count = 0
        self.convergence_history = []
        self.parameter_history = []
        self.best_result = None
        
        # Initial guess (center of bounds)
        x0 = np.array([(bounds[0] + bounds[1]) / 2 for bounds in self.bounds])
        
        logger.info(f"Starting CMA-ES optimization with {len(self.param_names)} parameters")
        logger.info(f"Generations: {generations}, Population: {pop_size}")
        
        if HAS_CMA:
            return self._optimize_cma(x0, generations, pop_size, sigma0, seed)
        else:
            return self._optimize_scipy(x0, generations * pop_size)
    
    def _optimize_cma(self, x0: np.ndarray, generations: int, pop_size: int, 
                     sigma0: float, seed: Optional[int]) -> OptimizationResult:
        """Run optimization using CMA-ES library."""
        try:
            # Configure CMA-ES
            options = {
                'popsize': pop_size,
                'maxiter': generations,
                'bounds': self.bounds,
                'seed': seed,
                'verbose': -1  # Suppress CMA output
            }
            
            # Run CMA-ES
            es = cma.CMAEvolutionStrategy(x0, sigma0, options)
            
            while not es.stop():
                solutions = es.ask()
                fitness_values = [self._objective_function(x) for x in solutions]
                es.tell(solutions, fitness_values)
                
                if es.countiter % 10 == 0:
                    logger.info(f"Generation {es.countiter}: best={min(fitness_values):.3e}")
            
            # Extract results
            best_params_vec = es.result.xbest
            best_score = es.result.fbest
            
            success = True
            
        except Exception as e:
            logger.error(f"CMA-ES optimization failed: {e}")
            success = False
            best_params_vec = x0
            best_score = float('inf')
        
        return self._create_result(best_params_vec, best_score, success, start_time)
    
    def _optimize_scipy(self, x0: np.ndarray, max_evaluations: int) -> OptimizationResult:
        """Fallback optimization using scipy."""
        logger.info("Using scipy differential evolution as CMA-ES fallback")
        
        try:
            result = differential_evolution(
                self._objective_function,
                bounds=self.bounds,
                maxiter=max_evaluations // len(x0),
                seed=42,
                disp=False
            )
            
            success = result.success
            best_params_vec = result.x
            best_score = result.fun
            
        except Exception as e:
            logger.error(f"Scipy optimization failed: {e}")
            success = False
            best_params_vec = x0
            best_score = float('inf')
        
        return self._create_result(best_params_vec, best_score, success, time.time())
    
    def _create_result(self, best_params_vec: np.ndarray, best_score: float, 
                      success: bool, start_time: float) -> OptimizationResult:
        """Create optimization result object."""
        execution_time = time.time() - start_time
        
        # Convert best parameters
        best_params_dict = self._params_to_dict(best_params_vec)
        
        # Run final simulation with best parameters
        final_simulation = None
        best_energy = float('inf')
        best_stability = 0.0
        
        if self.best_result is not None:
            final_simulation = self.best_result['simulation']
            best_energy = final_simulation.energy_total
            best_stability = final_simulation.stability
        
        result = OptimizationResult(
            success=success,
            best_parameters=best_params_dict,
            best_score=best_score,
            best_energy=best_energy,
            best_stability=best_stability,
            generations=len(self.convergence_history) // 20 if self.convergence_history else 0,
            function_evaluations=self.evaluation_count,
            execution_time=execution_time,
            convergence_history=self.convergence_history.copy(),
            parameter_history=self.parameter_history.copy(),
            final_simulation=final_simulation
        )
        
        logger.info(f"Optimization completed: score={best_score:.3e}, "
                   f"evaluations={self.evaluation_count}, time={execution_time:.1f}s")
        
        return result
    
    def save_results(self, result: OptimizationResult, filepath: str) -> None:
        """Save optimization results to JSON file."""
        results_data = result.to_dict()
        
        # Add convergence history
        results_data['convergence_history'] = result.convergence_history
        results_data['parameter_history'] = result.parameter_history
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")


def create_4d_optimizer(solver: WarpBubbleSolver, 
                       fixed_radius: float = 10.0, 
                       fixed_speed: float = 5000.0) -> CMAESOptimizer:
    """
    Create CMA-ES optimizer for 4D B-spline ansatz parameters.
    
    Args:
        solver: WarpBubbleSolver instance
        fixed_radius: Fixed bubble radius for optimization
        fixed_speed: Fixed bubble speed for optimization
        
    Returns:
        Configured CMAESOptimizer
    """
    param_names = ["cp1", "cp2", "cp3", "cp4", "cp5"]
    bounds = [(0.1, 5.0)] * 5  # Reasonable bounds for control points
    
    return CMAESOptimizer(solver, param_names, bounds, fixed_radius, fixed_speed)


def create_hybrid_optimizer(solver: WarpBubbleSolver,
                          fixed_radius: float = 10.0,
                          fixed_speed: float = 5000.0) -> CMAESOptimizer:
    """
    Create CMA-ES optimizer for Van den Broeck-Natário hybrid ansatz.
    
    Args:
        solver: WarpBubbleSolver instance  
        fixed_radius: Fixed bubble radius for optimization
        fixed_speed: Fixed bubble speed for optimization
        
    Returns:
        Configured CMAESOptimizer
    """
    param_names = ["R_int", "R_ext", "sigma", "v_bubble"]
    bounds = [
        (1e-36, 1e-30),  # R_int: Planck scale range
        (5.0, 50.0),     # R_ext: Bubble scale range  
        (0.1, 2.0),      # sigma: Transition width
        (0.01, 0.5)      # v_bubble: Sub-relativistic velocities
    ]
    
    return CMAESOptimizer(solver, param_names, bounds, fixed_radius, fixed_speed)


# Example usage and testing
if __name__ == "__main__":
    from .integrated_warp_solver import create_optimal_ghost_solver
    
    # Create solver and optimizer
    solver = create_optimal_ghost_solver()
    optimizer = create_4d_optimizer(solver)
    
    # Run optimization
    print("Running test optimization...")
    result = optimizer.optimize(generations=20, pop_size=15)
    
    print(f"Optimization completed:")
    print(f"Success: {result.success}")
    print(f"Best score: {result.best_score:.3e}")
    print(f"Best energy: {result.best_energy:.2e} J")
    print(f"Best stability: {result.best_stability:.3f}")
    print(f"Best parameters: {result.best_parameters}")
    
    # Save results
    optimizer.save_results(result, "test_optimization_results.json")
