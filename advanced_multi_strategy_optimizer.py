#!/usr/bin/env python3
"""
Advanced Multi-Strategy Optimizer for Warp Bubble Design
=======================================================

This module implements multiple optimization strategies for warp bubble design,
including genetic algorithms, Bayesian optimization, and differential evolution.

PLATINUM-ROAD INTEGRATION:
- 2D parameter space sweep over (μ_g, b) for comprehensive optimization
- Automated yield and critical field ratio calculations
- Integration with enhanced uncertainty quantification
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

# PLATINUM-ROAD INTEGRATION: Import parameter sweep
try:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from parameter_space_sweep import integrate_parameter_sweep_into_pipeline, WarpBubbleParameterSweep, ParameterSweepConfig
    PARAMETER_SWEEP_AVAILABLE = True
    print("✓ Parameter sweep module loaded successfully")
except ImportError as e:
    PARAMETER_SWEEP_AVAILABLE = False
    print(f"⚠ Warning: Parameter sweep module not available: {e}")

# ── ADVANCED IMPORTS ──────────────────────────────────────────────────────────
# Bayesian Optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.acquisition import gaussian_ei
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
    HAS_SKOPT = True
    print("✅ Scikit-optimize detected - Bayesian optimization enabled")
except ImportError:
    HAS_SKOPT = False
    print("⚠️  Install scikit-optimize: pip install scikit-optimize")

# Multi-objective optimization
try:
    from deap import base, creator, tools, algorithms
    import random
    HAS_DEAP = True
    print("✅ DEAP detected - Multi-objective optimization enabled")
except ImportError:
    HAS_DEAP = False
    print("⚠️  Install DEAP: pip install deap")

# CMA-ES for high-dimensional search
try:
    import cma
    HAS_CMA = True
    print("✅ CMA-ES detected - High-dimensional global search enabled")
except ImportError:
    HAS_CMA = False
    print("⚠️  Install CMA-ES: pip install cma")

# JAX for gradient-enhanced optimization
try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.optimize import minimize as jax_minimize
    from jax import grad, jit, value_and_grad, vmap
    HAS_JAX = True
    print("✅ JAX detected - Gradient-enhanced optimization enabled")
except ImportError:
    HAS_JAX = False
    print("⚠️  Install JAX: pip install jax jaxlib")

# ── PHYSICAL CONSTANTS ────────────────────────────────────────────────────────
beta_back = 1.9443254780147017
hbar = 1.0545718e-34
c = 299792458
G = 6.67430e-11
tau = 1e-9
v = 1.0
R = 1.0

# Optimal base parameters from previous studies
mu0_optimal = 5.2e-6
G_geo_optimal = 2.5e-5

# Conversion factors
c4_8piG = c**4 / (8.0 * np.pi * G)

# ── LQG BOUND ENFORCEMENT ──────────────────────────────────────────────────────
# Import LQG-modified quantum inequality bound enforcement
try:
    import sys
    sys.path.append('src')
    from warp_qft.stability import enforce_lqg_bound, lqg_modified_bounds
    HAS_LQG_BOUNDS = True
    print("✅ LQG-modified quantum inequality bounds loaded")
except ImportError:
    HAS_LQG_BOUNDS = False
    print("⚠️  LQG bounds not available - using classical energy computation")
    def enforce_lqg_bound(energy, spatial_scale, flight_time, C_lqg=None):
        return energy  # Fallback: no enforcement

# ── ANSATZ CONFIGURATIONS ─────────────────────────────────────────────────────
class AnsatzConfig:
    """Configuration class for different ansatz types"""
    
    # Strategy 1: Mixed-basis (Gaussians + Fourier)
    MIXED_GAUSSIANS = 8
    MIXED_FOURIER_MODES = 4
    MIXED_PARAM_DIM = 2 + 3*MIXED_GAUSSIANS + MIXED_FOURIER_MODES  # 38 params
    
    # Strategy 4: High-dimensional pure Gaussian
    HIGH_DIM_GAUSSIANS = 16
    HIGH_DIM_PARAM_DIM = 2 + 3*HIGH_DIM_GAUSSIANS  # 50 params
    
    # Multi-objective optimization targets
    MAX_LAMBDA_THRESHOLD = -1e-8  # Stability requirement
    
    # Grid configuration
    N_GRID = 1200  # Higher resolution for accuracy
    
    @classmethod
    def get_grid(cls):
        r = np.linspace(0.0, R, cls.N_GRID)
        dr = r[1] - r[0]
        vol_weights = 4.0 * np.pi * r**2
        return r, dr, vol_weights

print(f"🎯 Mixed-basis ansatz: {AnsatzConfig.MIXED_PARAM_DIM} parameters")
print(f"🎯 High-dimensional ansatz: {AnsatzConfig.HIGH_DIM_PARAM_DIM} parameters")

# ── STRATEGY 1: MIXED-BASIS ANSATZ ────────────────────────────────────────────
def f_mixed_basis_numpy(r, params):
    """
    Mixed-basis ansatz: Gaussians + Fourier modes
    
    f(r) = Σᵢ Aᵢ exp(-0.5((r-rᵢ)/σᵢ)²) + Σₖ Bₖ cos(πkr/R)
    
    Args:
        r: Radial coordinate
        params: [mu, G_geo] + 8×[A,r,σ] + 4×[B]
    """
    r = np.atleast_1d(r)
    result = np.zeros_like(r)
    
    # Gaussian components (parameters 2 to 2+24-1 = 25)
    for i in range(AnsatzConfig.MIXED_GAUSSIANS):
        idx = 2 + 3*i
        A_i = params[idx]
        r0_i = params[idx + 1] 
        sigma_i = max(params[idx + 2], 1e-6)
        
        gaussian = A_i * np.exp(-0.5 * ((r - r0_i) / sigma_i)**2)
        result += gaussian
    
    # Fourier components (parameters 26 to 29)
    fourier_start = 2 + 3*AnsatzConfig.MIXED_GAUSSIANS
    for k in range(AnsatzConfig.MIXED_FOURIER_MODES):
        B_k = params[fourier_start + k]
        fourier_mode = B_k * np.cos(np.pi * (k + 1) * r / R)
        result += fourier_mode
    
    return result

def f_mixed_basis_prime_numpy(r, params):
    """Derivative of mixed-basis ansatz"""
    r = np.atleast_1d(r)
    result = np.zeros_like(r)
    
    # Gaussian derivatives
    for i in range(AnsatzConfig.MIXED_GAUSSIANS):
        idx = 2 + 3*i
        A_i = params[idx]
        r0_i = params[idx + 1]
        sigma_i = max(params[idx + 2], 1e-6)
        
        gaussian = A_i * np.exp(-0.5 * ((r - r0_i) / sigma_i)**2)
        gaussian_prime = -gaussian * (r - r0_i) / (sigma_i**2)
        result += gaussian_prime
    
    # Fourier derivatives
    fourier_start = 2 + 3*AnsatzConfig.MIXED_GAUSSIANS
    for k in range(AnsatzConfig.MIXED_FOURIER_MODES):
        B_k = params[fourier_start + k]
        fourier_prime = -B_k * (np.pi * (k + 1) / R) * np.sin(np.pi * (k + 1) * r / R)
        result += fourier_prime
    
    return result

if HAS_JAX:
    @jax.jit
    def f_mixed_basis_jax(r, params):
        """JAX version of mixed-basis ansatz"""
        r = jnp.atleast_1d(r)
        result = jnp.zeros_like(r)
        
        # Gaussian components
        for i in range(AnsatzConfig.MIXED_GAUSSIANS):
            idx = 2 + 3*i
            A_i = params[idx]
            r0_i = params[idx + 1]
            sigma_i = jnp.maximum(params[idx + 2], 1e-6)
            
            gaussian = A_i * jnp.exp(-0.5 * ((r - r0_i) / sigma_i)**2)
            result += gaussian
        
        # Fourier components
        fourier_start = 2 + 3*AnsatzConfig.MIXED_GAUSSIANS
        for k in range(AnsatzConfig.MIXED_FOURIER_MODES):
            B_k = params[fourier_start + k]
            fourier_mode = B_k * jnp.cos(jnp.pi * (k + 1) * r / R)
            result += fourier_mode
        
        return result

# ── ENERGY COMPUTATION ────────────────────────────────────────────────────────
def compute_energy_mixed_basis_numpy(params, use_qi_enhancement=True):
    """
    Compute negative energy for mixed-basis ansatz with LQG-modified bound enforcement
    
    Returns:
        E₋ in Joules (more negative = better), respecting LQG bound E₋ ≥ -C_LQG/T^4
    """
    try:
        mu, G_geo = params[0], params[1]
        r_grid, dr, vol_weights = AnsatzConfig.get_grid()
        
        # Compute profile and derivative
        f_vals = f_mixed_basis_numpy(r_grid, params)
        df_dr = f_mixed_basis_prime_numpy(r_grid, params)
        
        # Enhanced energy density with curvature terms
        T_rr = (v**2 * df_dr**2) / (2 * r_grid + 1e-12)
        
        # Add second derivative contributions
        d2f_dr2 = np.gradient(df_dr, dr)
        T_rr += (v**2 * f_vals * d2f_dr2) / (r_grid + 1e-12)
        T_rr += (v**2 * f_vals * df_dr) / (r_grid**2 + 1e-12)
        
        # Volume integration
        E_negative = c4_8piG * np.sum(T_rr * vol_weights) * dr
        
        # Quantum improvement enhancement
        if use_qi_enhancement and mu > 0 and G_geo > 0:
            f_at_R = f_vals[-1]
            QI = beta_back * abs(f_at_R) / tau
            delta_E_qi = -QI * mu * G_geo * c**2
            E_negative += delta_E_qi
        
        # ⭐ ENFORCE LQG-MODIFIED QUANTUM INEQUALITY BOUND ⭐
        # Target: Push E₋ as close as possible to -C_LQG/T^4 (stricter than Ford-Roman)
        if HAS_LQG_BOUNDS:
            E_negative = enforce_lqg_bound(E_negative, R, tau)
        
        return E_negative
        
    except Exception as e:
        return 1e50  # Large penalty for failed evaluations

if HAS_JAX:
    @jax.jit
    def compute_energy_mixed_basis_jax(params):
        """JAX-accelerated energy computation with LQG bound enforcement"""
        mu, G_geo = params[0], params[1]
        r_grid = jnp.linspace(0.0, R, AnsatzConfig.N_GRID)
        dr = r_grid[1] - r_grid[0]
        vol_weights = 4.0 * jnp.pi * r_grid**2
        
        # Profile and derivatives
        f_vals = f_mixed_basis_jax(r_grid, params)
        df_dr = jnp.gradient(f_vals, dr)
        d2f_dr2 = jnp.gradient(df_dr, dr)
        
        # Energy density
        T_rr = (v**2 * df_dr**2) / (2 * r_grid + 1e-12)
        T_rr += (v**2 * f_vals * d2f_dr2) / (r_grid + 1e-12)
        T_rr += (v**2 * f_vals * df_dr) / (r_grid**2 + 1e-12)
        
        # Integration
        E_negative = c4_8piG * jnp.sum(T_rr * vol_weights) * dr
        
        # QI enhancement
        f_at_R = f_vals[-1]
        QI = beta_back * jnp.abs(f_at_R) / tau
        delta_E_qi = -QI * mu * G_geo * c**2
        E_negative += delta_E_qi
        
        # Note: LQG bound enforcement in JAX done in wrapper function
        return E_negative
        
    def compute_energy_mixed_basis_jax_with_lqg(params):
        """JAX energy computation with LQG bound enforcement"""
        E_negative = compute_energy_mixed_basis_jax(params)
        if HAS_LQG_BOUNDS:
            E_negative = enforce_lqg_bound(float(E_negative), R, tau)
        return E_negative

# ── CONSTRAINT FUNCTIONS ──────────────────────────────────────────────────────
def compute_boundary_penalty(params, ansatz_type='mixed'):
    """Enhanced boundary condition penalties"""
    penalty = 0.0
    
    # f(0) ≈ 1 constraint
    if ansatz_type == 'mixed':
        f_0 = f_mixed_basis_numpy(0.0, params)
    else:
        f_0 = f_high_dim_gaussian_numpy(0.0, params)
    
    penalty += 1e6 * (f_0 - 1.0)**2
    
    # f(R) ≈ 0 constraint
    if ansatz_type == 'mixed':
        f_R = f_mixed_basis_numpy(R, params)
    else:
        f_R = f_high_dim_gaussian_numpy(R, params)
    
    penalty += 1e6 * f_R**2
    
    return penalty

def compute_stability_penalty(params, ansatz_type='mixed'):
    """
    Estimate stability penalty (simplified Lyapunov analysis)
    In full implementation, this would solve the linearized Einstein equations
    """
    # Placeholder: penalize high curvature as proxy for instability
    r_test = np.linspace(0.1*R, 0.9*R, 50)
    
    if ansatz_type == 'mixed':
        df_dr = f_mixed_basis_prime_numpy(r_test, params)
    else:
        df_dr = f_high_dim_gaussian_prime_numpy(r_test, params)
    
    # Penalty for excessive gradients (instability indicator)
    max_gradient = np.max(np.abs(df_dr))
    if max_gradient > 10.0:  # Threshold for stability
        return 1e8 * (max_gradient - 10.0)**2
    
    return 0.0

# ── STRATEGY 2: SURROGATE-ASSISTED BAYESIAN OPTIMIZATION ─────────────────────
class SurrogateAssistedOptimizer:
    """
    Enhanced multi-strategy optimizer with surrogate modeling.
    
    PLATINUM-ROAD INTEGRATION: Now includes 2D parameter space sweep capabilities.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # PLATINUM-ROAD: Initialize parameter sweep capability
        self.parameter_sweep_integrated = False
        if PARAMETER_SWEEP_AVAILABLE:
            self.sweep_config = ParameterSweepConfig(
                mu_g_min=0.1, mu_g_max=0.6, mu_g_points=25,
                b_min=0.0, b_max=10.0, b_points=20,
                n_cores=4
            )
            self.parameter_sweep = WarpBubbleParameterSweep(self.sweep_config)
            self.parameter_sweep_integrated = True
            print("✓ Parameter sweep integrated into optimizer")
        else:
            print("⚠ Parameter sweep not available")
        
        # Existing initialization code...
        self.ansatz_type = 'mixed'
        self.evaluated_points = []
        self.evaluated_energies = []
        self.gp_model = None
        self.current_best_energy = np.inf
        self.current_best_params = None
        
        if self.ansatz_type == 'mixed':
            self.param_dim = AnsatzConfig.MIXED_PARAM_DIM
            self.energy_func = compute_energy_mixed_basis_numpy
        else:
            self.param_dim = AnsatzConfig.HIGH_DIM_PARAM_DIM
            self.energy_func = compute_energy_high_dim_numpy
    
    def create_search_space(self):
        """Define search space for Bayesian optimization"""
        space = [
            Real(1e-7, 1e-4, name='mu', prior='log-uniform'),
            Real(1e-7, 1e-4, name='G_geo', prior='log-uniform'),
        ]
        
        if self.ansatz_type == 'mixed':
            # Gaussian parameters
            for i in range(AnsatzConfig.MIXED_GAUSSIANS):
                space.extend([
                    Real(0.0, 2.0, name=f'A_{i}'),
                    Real(0.0, R, name=f'r_{i}'),
                    Real(0.01*R, 0.5*R, name=f'sigma_{i}', prior='log-uniform'),
                ])
            # Fourier parameters
            for k in range(AnsatzConfig.MIXED_FOURIER_MODES):
                space.append(Real(-1.0, 1.0, name=f'B_{k}'))
        else:
            # High-dimensional Gaussian parameters
            for i in range(AnsatzConfig.HIGH_DIM_GAUSSIANS):
                space.extend([
                    Real(0.0, 2.0, name=f'A_{i}'),
                    Real(0.0, R, name=f'r_{i}'),
                    Real(0.01*R, 0.5*R, name=f'sigma_{i}', prior='log-uniform'),
                ])
        
        return space
    
    def objective_with_penalties(self, params):
        """Objective function with all constraints"""
        params = np.array(params)
        
        # Core energy computation
        energy = self.energy_func(params)
        
        # Add penalties
        boundary_penalty = compute_boundary_penalty(params, self.ansatz_type)
        stability_penalty = compute_stability_penalty(params, self.ansatz_type)
        
        total_objective = energy + boundary_penalty + stability_penalty
        
        # Track best solution
        if total_objective < self.current_best_energy:
            self.current_best_energy = total_objective
            self.current_best_params = params.copy()
        
        return total_objective
    
    def run_bayesian_optimization(self, n_calls=200, n_initial=50, verbose=True):
        """Run Gaussian Process Bayesian optimization"""
        if not HAS_SKOPT:
            print("❌ Scikit-optimize not available for Bayesian optimization")
            return None
        
        space = self.create_search_space()
        
        if verbose:
            print(f"\n🧠 Starting Bayesian Optimization ({self.ansatz_type})")
            print(f"   Dimension: {self.param_dim}")
            print(f"   Total calls: {n_calls}")
            print(f"   Initial random: {n_initial}")
        
        start_time = time.time()
        
        # Enhanced GP with composite kernel
        kernel = (Matern(length_scale=1.0, nu=2.5) + 
                 RBF(length_scale=1.0) + 
                 WhiteKernel(noise_level=1e-5))
        
        result = gp_minimize(
            func=self.objective_with_penalties,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=n_initial,
            acq_func='EI',  # Expected Improvement
            n_jobs=-1,  # Parallel initial evaluations
            random_state=42
        )
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"✅ Bayesian optimization completed in {elapsed:.1f}s")
            print(f"   Best energy: {result.fun:.6e} J")
            print(f"   Function evaluations: {len(result.func_vals)}")
        
        return {
            'params': np.array(result.x),
            'energy': result.fun,
            'time': elapsed,
            'evaluations': len(result.func_vals),
            'method': f'Bayesian-{self.ansatz_type}'
        }

# ── STRATEGY 3: MULTI-OBJECTIVE OPTIMIZATION ─────────────────────────────────
def setup_nsga2_for_warp_bubble():
    """Setup NSGA-II for multi-objective optimization"""
    if not HAS_DEAP:
        print("❌ DEAP not available for multi-objective optimization")
        return None
    
    # Create fitness and individual classes
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # Minimize both
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    
    # Parameter bounds for mixed-basis ansatz
    param_bounds = []
    param_bounds.extend([(1e-7, 1e-4), (1e-7, 1e-4)])  # mu, G_geo
    
    for i in range(AnsatzConfig.MIXED_GAUSSIANS):
        param_bounds.extend([
            (0.0, 2.0),      # Amplitude
            (0.0, R),        # Position
            (0.01*R, 0.5*R)  # Width
        ])
    
    for k in range(AnsatzConfig.MIXED_FOURIER_MODES):
        param_bounds.append((-1.0, 1.0))  # Fourier coefficient
    
    # Registration functions
    def create_individual():
        return [np.random.uniform(low, high) for low, high in param_bounds]
    
    def evaluate_multi_objective(individual):
        """Evaluate both energy and stability"""
        params = np.array(individual)
        
        # Objective 1: Energy (more negative = better)
        energy = compute_energy_mixed_basis_numpy(params)
        energy_penalty = compute_boundary_penalty(params, 'mixed')
        total_energy = energy + energy_penalty
        
        # Objective 2: Stability (estimate maximum Lyapunov exponent)
        stability_measure = compute_stability_penalty(params, 'mixed')
        
        return total_energy, stability_measure
    
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_multi_objective)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    
    return toolbox

def run_nsga2_optimization(population_size=100, generations=50, verbose=True):
    """Run NSGA-II multi-objective optimization"""
    toolbox = setup_nsga2_for_warp_bubble()
    if toolbox is None:
        return None
    
    if verbose:
        print(f"\n🎯 Starting NSGA-II Multi-Objective Optimization")
        print(f"   Population: {population_size}")
        print(f"   Generations: {generations}")
    
    start_time = time.time()
    
    # Initialize population
    population = toolbox.population(n=population_size)
    
    # Evaluate initial population
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Evolution loop
    for gen in range(generations):
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.7:  # Crossover probability
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if np.random.random() < 0.2:  # Mutation probability
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Select next generation
        population = toolbox.select(population + offspring, population_size)
        
        if verbose and gen % 10 == 0:
            print(f"   Generation {gen}: Pareto front size = {len(population)}")
    
    elapsed = time.time() - start_time
    
    # Extract Pareto front
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    
    if verbose:
        print(f"✅ NSGA-II completed in {elapsed:.1f}s")
        print(f"   Pareto front size: {len(pareto_front)}")
    
    # Find solution with best energy among stable solutions
    best_solution = None
    best_energy = np.inf
    
    for individual in pareto_front:
        energy, stability = individual.fitness.values
        if stability < 1e6:  # Stability threshold
            if energy < best_energy:
                best_energy = energy
                best_solution = np.array(individual)
    
    if best_solution is not None:
        return {
            'params': best_solution,
            'energy': best_energy,
            'time': elapsed,
            'pareto_size': len(pareto_front),
            'method': 'NSGA-II'
        }
    else:
        print("⚠️  No stable solution found in Pareto front")
        return None

# ── STRATEGY 4: HIGH-DIMENSIONAL GLOBAL SEARCH ───────────────────────────────
def f_high_dim_gaussian_numpy(r, params):
    """High-dimensional pure Gaussian ansatz (16 Gaussians)"""
    r = np.atleast_1d(r)
    result = np.zeros_like(r)
    
    for i in range(AnsatzConfig.HIGH_DIM_GAUSSIANS):
        idx = 2 + 3*i
        A_i = params[idx]
        r0_i = params[idx + 1]
        sigma_i = max(params[idx + 2], 1e-6)
        
        gaussian = A_i * np.exp(-0.5 * ((r - r0_i) / sigma_i)**2)
        result += gaussian
    
    return result

def f_high_dim_gaussian_prime_numpy(r, params):
    """Derivative of high-dimensional Gaussian ansatz"""
    r = np.atleast_1d(r)
    result = np.zeros_like(r)
    
    for i in range(AnsatzConfig.HIGH_DIM_GAUSSIANS):
        idx = 2 + 3*i
        A_i = params[idx]
        r0_i = params[idx + 1]
        sigma_i = max(params[idx + 2], 1e-6)
        
        gaussian = A_i * np.exp(-0.5 * ((r - r0_i) / sigma_i)**2)
        gaussian_prime = -gaussian * (r - r0_i) / (sigma_i**2)
        result += gaussian_prime
    
    return result

def compute_energy_high_dim_numpy(params):
    """Energy computation for high-dimensional ansatz with LQG bound enforcement"""
    try:
        mu, G_geo = params[0], params[1]
        r_grid, dr, vol_weights = AnsatzConfig.get_grid()
        
        f_vals = f_high_dim_gaussian_numpy(r_grid, params)
        df_dr = f_high_dim_gaussian_prime_numpy(r_grid, params)
        
        # Enhanced energy density
        T_rr = (v**2 * df_dr**2) / (2 * r_grid + 1e-12)
        d2f_dr2 = np.gradient(df_dr, dr)
        T_rr += (v**2 * f_vals * d2f_dr2) / (r_grid + 1e-12)
        T_rr += (v**2 * f_vals * df_dr) / (r_grid**2 + 1e-12)
        
        E_negative = c4_8piG * np.sum(T_rr * vol_weights) * dr
        
        # QI enhancement
        if mu > 0 and G_geo > 0:
            f_at_R = f_vals[-1]
            QI = beta_back * abs(f_at_R) / tau
            delta_E_qi = -QI * mu * G_geo * c**2
            E_negative += delta_E_qi
        
        # ⭐ ENFORCE LQG-MODIFIED QUANTUM INEQUALITY BOUND ⭐
        if HAS_LQG_BOUNDS:
            E_negative = enforce_lqg_bound(E_negative, R, tau)
        
        return E_negative
        
    except Exception as e:
        return 1e50
        
        return E_negative
        
    except Exception:
        return 1e50

def run_high_dimensional_cma_es(max_evals=10000, popsize=256, verbose=True):
    """Massive CMA-ES search in 50-dimensional space"""
    if not HAS_CMA:
        print("❌ CMA-ES not available for high-dimensional search")
        return None
    
    param_dim = AnsatzConfig.HIGH_DIM_PARAM_DIM
    
    if verbose:
        print(f"\n🌍 Starting High-Dimensional CMA-ES Search")
        print(f"   Dimension: {param_dim}")
        print(f"   Population: {popsize}")
        print(f"   Max evaluations: {max_evals}")
    
    # Initial guess
    x0 = np.zeros(param_dim)
    x0[0] = mu0_optimal      # mu
    x0[1] = G_geo_optimal    # G_geo
    
    # Initialize Gaussians with smart spacing
    for i in range(AnsatzConfig.HIGH_DIM_GAUSSIANS):
        base_idx = 2 + 3*i
        x0[base_idx] = 0.5 / (i + 1)  # Decreasing amplitudes
        x0[base_idx + 1] = (i + 0.5) * R / AnsatzConfig.HIGH_DIM_GAUSSIANS  # Spread positions
        x0[base_idx + 2] = 0.1 * R    # Moderate widths
    
    # Bounds
    bounds_list = []
    bounds_list.extend([(1e-8, 1e-3), (1e-8, 1e-2)])  # mu, G_geo
    for i in range(AnsatzConfig.HIGH_DIM_GAUSSIANS):
        bounds_list.extend([(0.0, 3.0), (0.0, R), (0.01*R, 0.5*R)])
    
    bounds_array = np.array(bounds_list)
    cma_bounds = [bounds_array[:, 0].tolist(), bounds_array[:, 1].tolist()]
    
    # Objective with penalties
    def objective_with_penalties(params):
        energy = compute_energy_high_dim_numpy(params)
        boundary_penalty = compute_boundary_penalty(params, 'high_dim')
        return energy + boundary_penalty
    
    # CMA-ES options with restart strategy
    opts = {
        'bounds': cma_bounds,
        'popsize': popsize,
        'maxiter': max_evals // popsize,
        'tolfun': 1e-15,
        'tolx': 1e-15,
        'verb_log': 0 if not verbose else 1,
        'CMA_mirrormethod': 2,  # Mirrored sampling
        'CMA_active': True,     # Active CMA
    }
    
    start_time = time.time()
    
    # Run with restarts
    best_energy = float('inf')
    best_params = None
    total_evals = 0
    
    for restart in range(3):  # Multiple restarts
        if verbose:
            print(f"   Restart {restart + 1}/3")
        
        es = cma.CMAEvolutionStrategy(x0.tolist(), 0.3, opts)
        
        while not es.stop() and total_evals < max_evals:
            solutions = es.ask()
            energies = []
            
            for sol in solutions:
                energy = objective_with_penalties(np.array(sol))
                energies.append(energy)
                total_evals += 1
                
                if energy < best_energy:
                    best_energy = energy
                    best_params = np.array(sol)
                    if verbose and total_evals % 500 == 0:
                        print(f"      Eval {total_evals}: E = {energy:.6e} J")
            
            es.tell(solutions, energies)
        
        # Perturb for next restart
        if restart < 2:
            x0 = best_params + 0.1 * np.random.randn(param_dim)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"✅ High-dimensional CMA-ES completed in {elapsed:.1f}s")
        print(f"   Best energy: {best_energy:.6e} J") 
        print(f"   Total evaluations: {total_evals}")
    
    return {
        'params': best_params,
        'energy': best_energy,
        'evaluations': total_evals,
        'time': elapsed,
        'method': 'High-Dim-CMA-ES'
    }

# ── STRATEGY 5: GRADIENT-ENHANCED LOCAL DESCENT ──────────────────────────────
def run_gradient_enhanced_refinement(theta_init, ansatz_type='mixed', max_iter=500, verbose=True):
    """JAX-accelerated gradient-based local refinement"""
    if not HAS_JAX:
        print("❌ JAX not available for gradient-enhanced optimization")
        return {'params': theta_init, 'energy': compute_energy_mixed_basis_numpy(theta_init), 
                'method': 'no_refinement'}
    
    if verbose:
        print(f"\n⚡ Starting Gradient-Enhanced Refinement ({ansatz_type})")    # Choose appropriate energy function
    if ansatz_type == 'mixed':
        param_dim = AnsatzConfig.MIXED_PARAM_DIM
    else:
        # Would implement JAX version for high-dim if needed
        print("   High-dim JAX not implemented, using NumPy")
        return {'params': theta_init, 'energy': compute_energy_high_dim_numpy(theta_init)}
      # Progressive boundary penalty refinement
    def create_progressive_objective(penalty_weight):
        @jax.jit
        def objective(params):
            energy = compute_energy_mixed_basis_jax(params)
            
            # Apply LQG bound enforcement (outside of JIT)
            # Note: LQG enforcement done in wrapper function
            
            # Boundary penalties
            f_0 = f_mixed_basis_jax(jnp.array([0.0]), params)[0]
            f_R = f_mixed_basis_jax(jnp.array([R]), params)[0]
            
            boundary_penalty = penalty_weight * ((f_0 - 1.0)**2 + f_R**2)
            
            return energy + boundary_penalty
        
        # Wrapper for LQG enforcement
        def objective_with_lqg(params):
            raw_result = objective(params)
            if HAS_LQG_BOUNDS:
                # Apply LQG bound to raw energy component
                return enforce_lqg_bound(float(raw_result), R, tau)
            return float(raw_result)
        
        return objective_with_lqg
        
        return objective
    
    theta_jax = jnp.array(theta_init)
    current_best = theta_jax
      # Progressive refinement with increasing boundary enforcement
    penalty_weights = [1e3, 1e4, 1e5, 1e6]
    
    for i, weight in enumerate(penalty_weights):
        objective = create_progressive_objective(weight)
        
        try:
            result = jax_minimize(
                objective,
                current_best,
                method='BFGS',
                options={'maxiter': max_iter // len(penalty_weights), 'gtol': 1e-10}
            )
            
            if hasattr(result, 'x'):
                current_best = result.x
                
            if verbose:
                energy_val = float(compute_energy_mixed_basis_jax_with_lqg(current_best))
                print(f"   Stage {i+1}: E = {energy_val:.6e} J (penalty weight: {weight})")
                
        except Exception as e:
            if verbose:
                print(f"   Stage {i+1} failed: {e}")
            continue
    
    final_energy = float(compute_energy_mixed_basis_jax_with_lqg(current_best))
    
    if verbose:
        print(f"   Final refined energy: {final_energy:.6e} J")
    
    return {
        'params': np.array(current_best),
        'energy': final_energy,
        'method': f'Gradient-Enhanced-{ansatz_type}',
        'success': True
    }

# ── STRATEGY 6: PARALLEL EVALUATION & VECTORIZATION ──────────────────────────
def parallel_energy_evaluation(param_batch, ansatz_type='mixed'):
    """Vectorized energy evaluation for entire parameter batches"""
    if ansatz_type == 'mixed':
        return [compute_energy_mixed_basis_numpy(params) for params in param_batch]
    else:
        return [compute_energy_high_dim_numpy(params) for params in param_batch]

def run_parallel_cma_es_batch(max_evals=5000, ansatz_type='mixed', n_workers=None, verbose=True):
    """CMA-ES with parallel batch evaluation"""
    if not HAS_CMA:
        return None
    
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)
    
    if ansatz_type == 'mixed':
        param_dim = AnsatzConfig.MIXED_PARAM_DIM
        energy_func = compute_energy_mixed_basis_numpy
    else:
        param_dim = AnsatzConfig.HIGH_DIM_PARAM_DIM  
        energy_func = compute_energy_high_dim_numpy
    
    if verbose:
        print(f"\n🚀 Parallel CMA-ES ({ansatz_type})")
        print(f"   Workers: {n_workers}")
        print(f"   Dimension: {param_dim}")
    
    # Setup similar to previous CMA-ES implementations
    x0 = np.zeros(param_dim)
    x0[0] = mu0_optimal
    x0[1] = G_geo_optimal
    
    # Initialize remaining parameters based on ansatz type
    if ansatz_type == 'mixed':
        for i in range(AnsatzConfig.MIXED_GAUSSIANS):
            base_idx = 2 + 3*i
            x0[base_idx] = 0.5 / (i + 1)
            x0[base_idx + 1] = (i + 0.5) * R / AnsatzConfig.MIXED_GAUSSIANS
            x0[base_idx + 2] = 0.1 * R
        # Fourier modes start small
        fourier_start = 2 + 3*AnsatzConfig.MIXED_GAUSSIANS
        for k in range(AnsatzConfig.MIXED_FOURIER_MODES):
            x0[fourier_start + k] = 0.1 * np.random.randn()
    else:
        for i in range(AnsatzConfig.HIGH_DIM_GAUSSIANS):
            base_idx = 2 + 3*i
            x0[base_idx] = 0.5 / (i + 1)
            x0[base_idx + 1] = (i + 0.5) * R / AnsatzConfig.HIGH_DIM_GAUSSIANS
            x0[base_idx + 2] = 0.1 * R
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel evaluation
    def objective_batch(solutions_list):
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            param_arrays = [np.array(sol) for sol in solutions_list]
            energies = list(executor.map(energy_func, param_arrays))
        return energies
    
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.25, {'popsize': 50, 'maxiter': max_evals // 50})
    
    best_energy = float('inf')
    best_params = None
    eval_count = 0
    
    while not es.stop() and eval_count < max_evals:
        solutions = es.ask()
        
        # Parallel batch evaluation
        energies = objective_batch(solutions)
        eval_count += len(energies)
        
        # Track best
        for sol, energy in zip(solutions, energies):
            if energy < best_energy:
                best_energy = energy
                best_params = np.array(sol)
        
        es.tell(solutions, energies)
        
        if verbose and eval_count % 250 == 0:
            print(f"   Eval {eval_count}: Best E = {best_energy:.6e} J")
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"✅ Parallel CMA-ES completed in {elapsed:.1f}s")
        print(f"   Best energy: {best_energy:.6e} J")
    
    return {
        'params': best_params,
        'energy': best_energy,
        'evaluations': eval_count,
        'time': elapsed,
        'method': f'Parallel-CMA-{ansatz_type}'
    }

# ── MAIN OPTIMIZATION ORCHESTRATOR ────────────────────────────────────────────
def run_comprehensive_multi_strategy_optimization(strategies='all', verbose=True):
    """
    Run comprehensive optimization using all six strategies
    
    Args:
        strategies: 'all' or list of strategy names
        verbose: Progress output
    
    Returns:
        Dictionary with results from all strategies
    """
    if verbose:
        print("\n" + "="*80)
        print("🌟 COMPREHENSIVE MULTI-STRATEGY WARP BUBBLE OPTIMIZATION 🌟")
        print("="*80)
    
    results = {}
    start_time = time.time()
    
    # Strategy 1: Mixed-basis Bayesian optimization
    if strategies == 'all' or 'mixed_bayesian' in strategies:
        try:
            optimizer = SurrogateAssistedOptimizer(ansatz_type='mixed')
            result = optimizer.run_bayesian_optimization(n_calls=150, verbose=verbose)
            if result:
                results['mixed_bayesian'] = result
        except Exception as e:
            if verbose:
                print(f"❌ Mixed-basis Bayesian failed: {e}")
    
    # Strategy 2: Multi-objective NSGA-II
    if strategies == 'all' or 'nsga2' in strategies:
        try:
            result = run_nsga2_optimization(population_size=80, generations=40, verbose=verbose)
            if result:
                results['nsga2'] = result
        except Exception as e:
            if verbose:
                print(f"❌ NSGA-II failed: {e}")
    
    # Strategy 3: High-dimensional CMA-ES
    if strategies == 'all' or 'high_dim_cma' in strategies:
        try:
            result = run_high_dimensional_cma_es(max_evals=8000, verbose=verbose)
            if result:
                results['high_dim_cma'] = result
        except Exception as e:
            if verbose:
                print(f"❌ High-dimensional CMA-ES failed: {e}")
    
    # Strategy 4: Parallel mixed-basis CMA-ES
    if strategies == 'all' or 'parallel_mixed' in strategies:
        try:
            result = run_parallel_cma_es_batch(max_evals=4000, ansatz_type='mixed', verbose=verbose)
            if result:
                results['parallel_mixed'] = result
        except Exception as e:
            if verbose:
                print(f"❌ Parallel mixed CMA-ES failed: {e}")
    
    # Strategy 5: Gradient refinement of best candidates
    if len(results) > 0:
        best_result = min(results.values(), key=lambda x: x.get('energy', np.inf))
        
        if 'mixed' in best_result.get('method', '').lower():
            ansatz_type = 'mixed'
        else:
            ansatz_type = 'high_dim'
        
        try:
            refined_result = run_gradient_enhanced_refinement(
                best_result['params'], ansatz_type=ansatz_type, verbose=verbose
            )
            results['gradient_refined'] = refined_result
        except Exception as e:
            if verbose:
                print(f"❌ Gradient refinement failed: {e}")
    
    total_time = time.time() - start_time
    
    if verbose:
        print("\n" + "="*80)
        print("📊 OPTIMIZATION SUMMARY")
        print("="*80)
        
        for name, result in results.items():
            energy = result.get('energy', np.nan)
            method = result.get('method', name)
            eval_time = result.get('time', 0)
            
            print(f"{method:25s}: E = {energy:.6e} J ({eval_time:.1f}s)")
        
        if results:
            best_name = min(results.keys(), key=lambda k: results[k].get('energy', np.inf))
            best_energy = results[best_name]['energy']
            print(f"\n🏆 BEST RESULT: {best_name}")
            print(f"   Energy: {best_energy:.6e} J")
            print(f"   Total time: {total_time:.1f}s")
        
        print("="*80)
    
    return results

# ── ANALYSIS AND VISUALIZATION ────────────────────────────────────────────────
def analyze_optimization_results(results, save_plots=True):
    """Comprehensive analysis of optimization results"""
    if not results:
        print("No results to analyze")
        return
    
    # Extract energies and methods
    methods = []
    energies = []
    times = []
    
    for name, result in results.items():
        methods.append(result.get('method', name))
        energies.append(result.get('energy', np.nan))
        times.append(result.get('time', 0))
    
    # Performance comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Energy comparison
    bars1 = ax1.bar(range(len(methods)), energies, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Optimization Method')
    ax1.set_ylabel('Energy E₋ (J)')
    ax1.set_title('Energy Comparison Across Methods')
    ax1.set_yscale('log')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add energy values on bars
    for bar, energy in zip(bars1, energies):
        if not np.isnan(energy):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{energy:.2e}', ha='center', va='bottom', fontsize=8)
    
    # Time comparison
    bars2 = ax2.bar(range(len(methods)), times, color='lightcoral', edgecolor='darkred')
    ax2.set_xlabel('Optimization Method')
    ax2.set_ylabel('Computation Time (s)')
    ax2.set_title('Computational Cost Comparison')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('multi_strategy_optimization_comparison.png', dpi=300, bbox_inches='tight')
        print("📊 Saved optimization comparison plot")
    
    plt.show()
    
    # Best profile visualization
    best_name = min(results.keys(), key=lambda k: results[k].get('energy', np.inf))
    best_params = results[best_name]['params']
    
    if 'mixed' in results[best_name].get('method', '').lower():
        plot_mixed_basis_profile(best_params, title=f'Best Profile: {best_name}')
    else:
        plot_high_dim_gaussian_profile(best_params, title=f'Best Profile: {best_name}')

def plot_mixed_basis_profile(params, title='Mixed-Basis Profile'):
    """Plot the mixed-basis ansatz profile"""
    r_plot = np.linspace(0, R, 1000)
    f_vals = f_mixed_basis_numpy(r_plot, params)
    
    plt.figure(figsize=(12, 8))
    
    # Main profile
    plt.subplot(2, 2, 1)
    plt.plot(r_plot, f_vals, 'b-', linewidth=2, label='f(r)')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('r (m)')
    plt.ylabel('f(r)')
    plt.title(f'{title} - Shape Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Derivative
    plt.subplot(2, 2, 2)
    df_dr = f_mixed_basis_prime_numpy(r_plot, params)
    plt.plot(r_plot, df_dr, 'r-', linewidth=2, label="f'(r)")
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('r (m)')
    plt.ylabel("f'(r)")
    plt.title('Derivative')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Component breakdown
    plt.subplot(2, 2, 3)
    
    # Gaussian components
    gauss_total = np.zeros_like(r_plot)
    for i in range(AnsatzConfig.MIXED_GAUSSIANS):
        idx = 2 + 3*i
        A_i = params[idx]
        r0_i = params[idx + 1]
        sigma_i = max(params[idx + 2], 1e-6)
        gauss_i = A_i * np.exp(-0.5 * ((r_plot - r0_i) / sigma_i)**2)
        gauss_total += gauss_i
        if i < 3:  # Plot first 3 for clarity
            plt.plot(r_plot, gauss_i, '--', alpha=0.7, label=f'G{i+1}')
    
    plt.plot(r_plot, gauss_total, 'g-', linewidth=2, label='Gaussians')
    
    # Fourier components
    fourier_total = np.zeros_like(r_plot)
    fourier_start = 2 + 3*AnsatzConfig.MIXED_GAUSSIANS
    for k in range(AnsatzConfig.MIXED_FOURIER_MODES):
        B_k = params[fourier_start + k]
        fourier_k = B_k * np.cos(np.pi * (k + 1) * r_plot / R)
        fourier_total += fourier_k
    
    plt.plot(r_plot, fourier_total, 'm-', linewidth=2, label='Fourier')
    plt.xlabel('r (m)')
    plt.ylabel('Component value')
    plt.title('Component Breakdown')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Parameters display
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    mu, G_geo = params[0], params[1]
    energy = compute_energy_mixed_basis_numpy(params)
    
    param_text = f"""
    Physical Parameters:
    μ = {mu:.3e}
    G_geo = {G_geo:.3e}
    
    Energy:
    E₋ = {energy:.6e} J
    
    Gaussians: {AnsatzConfig.MIXED_GAUSSIANS}
    Fourier modes: {AnsatzConfig.MIXED_FOURIER_MODES}
    
    Boundary conditions:
    f(0) = {f_mixed_basis_numpy(0.0, params):.6f}
    f(R) = {f_mixed_basis_numpy(R, params):.6f}
    """
    
    plt.text(0.1, 0.9, param_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'mixed_basis_profile_{int(time.time())}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Saved mixed-basis profile plot")
    plt.show()

def plot_high_dim_gaussian_profile(params, title='High-Dimensional Gaussian Profile'):
    """Plot the high-dimensional Gaussian ansatz profile"""
    # Similar implementation to plot_mixed_basis_profile but for pure Gaussians
    r_plot = np.linspace(0, R, 1000)
    f_vals = f_high_dim_gaussian_numpy(r_plot, params)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(r_plot, f_vals, 'b-', linewidth=2)
    plt.xlabel('r (m)')
    plt.ylabel('f(r)')
    plt.title(f'{title} - Shape Function')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    df_dr = f_high_dim_gaussian_prime_numpy(r_plot, params)
    plt.plot(r_plot, df_dr, 'r-', linewidth=2)
    plt.xlabel('r (m)')
    plt.ylabel("f'(r)")
    plt.title('Derivative')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'high_dim_gaussian_profile_{int(time.time())}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Saved high-dimensional Gaussian profile plot")
    plt.show()

# ── MAIN EXECUTION ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"""
{'='*80}
🌟 ADVANCED MULTI-STRATEGY WARP BUBBLE OPTIMIZER 🌟
{'='*80}

Available strategies:
1. 🌊 Mixed-basis ansatz (Gaussians + Fourier modes)
2. 🧠 Surrogate-assisted Bayesian optimization
3. 🎯 Multi-objective search (energy vs. stability)  
4. 🌍 High-dimensional global search (16 Gaussians)
5. ⚡ Gradient-enhanced local descent
6. 🚀 Parallel evaluation & vectorization

Target: Push E₋ beyond -1×10³³ J
{'='*80}
    """)
    
    # Run comprehensive optimization
    results = run_comprehensive_multi_strategy_optimization(
        strategies=['mixed_bayesian', 'parallel_mixed', 'high_dim_cma'],
        verbose=True
    )
    
    # Analyze and visualize results
    if results:
        analyze_optimization_results(results, save_plots=True)
        
        # Save results
        timestamp = int(time.time())
        results_file = f'multi_strategy_results_{timestamp}.json'
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for name, result in results.items():
            json_result = result.copy()
            if 'params' in json_result and hasattr(json_result['params'], 'tolist'):
                json_result['params'] = json_result['params'].tolist()
            json_results[name] = json_result
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to {results_file}")
    else:
        print("\n❌ No successful optimizations completed")
