#!/usr/bin/env python3
"""
JAX-Accelerated 4D Warp Bubble Optimizer

This script implements the complete time-dependent warp bubble optimization
with JAX acceleration, exploiting quantum inequality T^-4 scaling for 
vanishingly small exotic energy requirements.

Key Features:
- 4D spacetime optimization f(r,t)
- Gravity compensation a_warp(t) ≥ g
- LQG-corrected quantum inequality bounds
- Volume scaling and temporal smearing
- Real-time visualization and analysis

Usage:
    python jax_4d_optimizer.py --volume 5.0 --duration 21 --output results/
    
Author: Advanced Warp Bubble Research Team
Date: June 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# ProgressTracker import with fallback
try:
    from progress_tracker import ProgressTracker
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False
    class ProgressTracker:
        def __init__(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def set_stage(self, *args, **kwargs): pass
        def log_metric(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

class DummyContext:
    """Dummy context manager for fallback."""
    def __enter__(self): return self
    def __exit__(self, *args): pass

# Try to import JAX, fallback to NumPy if not available
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    from jax.scipy.optimize import minimize
    JAX_AVAILABLE = True
    print("JAX acceleration enabled")
except ImportError:
    print("JAX not available, falling back to NumPy (slower)")
    import numpy as jnp
    JAX_AVAILABLE = False
    
    # Mock JAX decorators for NumPy fallback
    def jit(func):
        return func
    
    def grad(func):
        # Simple finite difference gradient
        def gradient_func(x, *args, **kwargs):
            eps = 1e-8
            grad_result = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                grad_result[i] = (func(x_plus, *args, **kwargs) - func(x_minus, *args, **kwargs)) / (2 * eps)
            return grad_result
        return gradient_func

# Physical constants
C_LIGHT = 2.998e8  # m/s
G_NEWTON = 6.674e-11  # m³/kg/s²
HBAR = 1.055e-34  # J·s
G_EARTH = 9.81  # m/s²

class JAX4DWarpOptimizer:
    """
    JAX-accelerated 4D warp bubble optimizer with quantum inequality exploitation.
    """
    
    def __init__(self, bubble_volume: float = 5.0, flight_duration_days: float = 21,
                 C_LQG: float = 1e-3, grid_resolution: int = 128):
        """
        Initialize the 4D warp optimizer.
        
        Args:
            bubble_volume: Bubble volume in m³
            flight_duration_days: Flight duration in days
            C_LQG: LQG quantum inequality constant (J·s⁴)
            grid_resolution: Spatial and temporal grid resolution
        """
        self.V = bubble_volume
        self.T_total = flight_duration_days * 86400  # Convert to seconds
        self.C_LQG = C_LQG
        self.N = grid_resolution
        
        # Derived parameters
        self.R = (3 * self.V / (4 * np.pi))**(1/3)  # Spherical bubble radius
        self.qi_bound = self.V * self.C_LQG / (self.T_total**4)
        
        # Grid setup
        self.r_grid = jnp.linspace(0, self.R, self.N)
        self.t_grid = jnp.linspace(0, self.T_total, self.N)
        self.dr = self.R / (self.N - 1) if self.N > 1 else self.R
        self.dt = self.T_total / (self.N - 1) if self.N > 1 else self.T_total
        
        # Control point configuration
        self.M_spatial = 24   # Spatial control points for f(r)
        self.M_temporal = 16  # Temporal control points for a(t)
        
        # Penalty weights
        self.alpha_gravity = 1e4
        self.alpha_qi = 1e6
        self.alpha_stability = 1e3
        
        print(f"JAX 4D Warp Optimizer Initialized:")
        print(f"  Bubble: {self.V:.1f} m³ (R = {self.R:.3f} m)")
        print(f"  Flight: {flight_duration_days:.1f} days ({self.T_total:.0e} s)")
        print(f"  QI bound: {self.qi_bound:.2e} J")
        print(f"  Grid: {self.N}×{self.N} = {self.N**2:,} spacetime points")
    
    @jit
    def spatial_interpolation(self, r: jnp.ndarray, control_points: jnp.ndarray) -> jnp.ndarray:
        """Interpolate spatial control points to get f(r)."""
        r_cp_grid = jnp.linspace(0, self.R, len(control_points))
        return jnp.interp(r, r_cp_grid, control_points)
    
    @jit
    def temporal_interpolation(self, t: jnp.ndarray, control_points: jnp.ndarray) -> jnp.ndarray:
        """Interpolate temporal control points to get a(t)."""
        t_cp_grid = jnp.linspace(0, self.T_total, len(control_points))
        return jnp.interp(t, t_cp_grid, control_points)
    
    @jit
    def time_envelope(self, t: jnp.ndarray) -> jnp.ndarray:
        """Smooth temporal envelope for bubble formation/dissolution."""
        ramp_duration = self.T_total * 0.15  # 15% of flight time for ramps
        
        # Smooth ramp-on
        ramp_on = jnp.where(
            t < ramp_duration,
            0.5 * (1 + jnp.tanh((t - ramp_duration/2) / (ramp_duration/10))),
            1.0
        )
        
        # Smooth ramp-off
        ramp_off = jnp.where(
            t > self.T_total - ramp_duration,
            0.5 * (1 + jnp.tanh((self.T_total - ramp_duration/2 - t) / (ramp_duration/10))),
            1.0
        )
        
        return ramp_on * ramp_off
    
    @jit
    def spacetime_ansatz(self, theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute 4D spacetime ansatz f(r,t) and a(t).
        
        Args:
            theta: Parameter vector [mu, G_geo, spatial_cps, temporal_cps]
            
        Returns:
            (f_rt_grid, a_t_grid): Shape function and acceleration on full grid
        """
        # Extract parameters
        mu = theta[0]
        G_geo = theta[1]
        spatial_cps = theta[2:2+self.M_spatial]
        temporal_cps = theta[2+self.M_spatial:2+self.M_spatial+self.M_temporal]
        
        # Interpolate to full grids
        f_r = self.spatial_interpolation(self.r_grid, spatial_cps)
        a_t = self.temporal_interpolation(self.t_grid, temporal_cps)
        
        # Time envelope modulation
        envelope_t = self.time_envelope(self.t_grid)
        
        # Create 4D grid: f(r,t) = f(r) * envelope(t)
        f_rt_grid = jnp.outer(f_r, envelope_t)
        a_t_grid = a_t
        
        return f_rt_grid, a_t_grid
    
    @jit
    def energy_density_4d(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Compute 4D energy density T₀₀(r,t) with LQG corrections.
        """
        mu = theta[0]
        G_geo = theta[1]
        
        # Get spacetime ansatz
        f_rt_grid, a_t_grid = self.spacetime_ansatz(theta)
        
        # Velocity buildup: v(t) ≈ a(t) * t (simplified)
        # In full theory, this would be v(t) = ∫₀ᵗ a(t') dt'
        R_mesh, T_mesh = jnp.meshgrid(self.r_grid, self.t_grid, indexing='ij')
        v_t_grid = a_t_grid * self.t_grid / C_LIGHT
        V_mesh = jnp.outer(jnp.ones_like(self.r_grid), v_t_grid)
        
        # Classical energy density
        T00_classical = f_rt_grid**2 * V_mesh**2
        
        # LQG correction factor
        sinc_argument = jnp.pi * mu * f_rt_grid
        sinc_factor = jnp.sinc(sinc_argument)
        lqg_correction = 1 + G_geo * sinc_factor
        
        # Total energy density (negative for exotic energy)
        T00_total = -T00_classical * lqg_correction
        
        return T00_total
    
    @jit  
    def total_exotic_energy(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Integrate total exotic energy over 4D spacetime.
        
        E = ∫₀ᵀ ∫₀ᴿ T₀₀(r,t) * 4πr² dr dt
        """
        T00_grid = self.energy_density_4d(theta)
        
        # 4D volume element integration
        total_energy = 0.0
        for i in range(self.N):
            r = self.r_grid[i]
            for j in range(self.N):
                volume_element = 4 * jnp.pi * r**2 * self.dr * self.dt
                total_energy += T00_grid[i, j] * volume_element
        
        return total_energy
    
    @jit
    def gravity_penalty(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Penalty for insufficient gravity compensation.
        """
        _, a_t_grid = self.spacetime_ansatz(theta)
        
        # Penalty where a(t) < g_earth
        gravity_deficit = jnp.maximum(G_EARTH - a_t_grid, 0.0)
        penalty = jnp.sum(gravity_deficit**2) * self.dt
        
        return penalty
    
    @jit
    def quantum_inequality_penalty(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Penalty for violating quantum inequality bounds.
        """
        E_total = self.total_exotic_energy(theta)
        
        # Violation if |E_total| < qi_bound
        violation = jnp.maximum(self.qi_bound - jnp.abs(E_total), 0.0)
        return violation**2
    
    @jit
    def stability_penalty(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Stability penalty (placeholder for full eigenvalue analysis).
        """
        # Simple penalty for extreme parameter values
        mu, G_geo = theta[0], theta[1]
        
        # Penalize mu too large (unstable polymer regime)
        mu_penalty = jnp.maximum(mu - 1e-3, 0.0)**2 * 1e6
        
        # Penalize G_geo too large (breakdown of perturbation theory)
        G_penalty = jnp.maximum(G_geo - 1e-2, 0.0)**2 * 1e6
        
        return mu_penalty + G_penalty
    
    @jit
    def objective_function(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Combined objective function for optimization.
        """
        # Primary objective: minimize |exotic energy|
        E_exotic = jnp.abs(self.total_exotic_energy(theta))
        
        # Constraint penalties
        gravity_pen = self.alpha_gravity * self.gravity_penalty(theta)
        qi_pen = self.alpha_qi * self.quantum_inequality_penalty(theta)
        stability_pen = self.alpha_stability * self.stability_penalty(theta)
        
        total_objective = E_exotic + gravity_pen + qi_pen + stability_pen
        
        return total_objective
    
    def initialize_parameters(self) -> jnp.ndarray:
        """
        Initialize parameter vector with physically reasonable values.
        """
        # LQG parameters
        mu_init = 5e-5
        G_geo_init = 1e-4
        
        # Spatial control points: Gaussian-like profile
        r_normalized = np.linspace(0, 1, self.M_spatial)
        spatial_cps_init = np.exp(-3 * r_normalized**2)
        
        # Temporal control points: smooth acceleration profile
        temporal_cps_init = np.full(self.M_temporal, G_EARTH * 1.3)  # 30% above gravity
        
        # Add some variation for optimization
        temporal_cps_init[:4] *= 1.5   # Higher initial acceleration
        temporal_cps_init[-4:] *= 0.8  # Lower final acceleration
        
        theta_init = jnp.concatenate([
            jnp.array([mu_init, G_geo_init]),
            jnp.array(spatial_cps_init),
            jnp.array(temporal_cps_init)        ])
        
        return theta_init
    
    def optimize(self, max_iterations: int = 1000, tolerance: float = 1e-10) -> Dict:
        """
        Run the 4D warp bubble optimization.
        """
        print(f"\nStarting 4D JAX optimization...")
        print(f"Target: Drive exotic energy to QI bound {self.qi_bound:.2e} J")
        
        # Initialize parameters
        theta_init = self.initialize_parameters()
        print(f"Parameter vector dimension: {len(theta_init)}")
        
        # Initial evaluation
        initial_objective = float(self.objective_function(theta_init))
        initial_energy = float(self.total_exotic_energy(theta_init))
        
        print(f"Initial objective: {initial_objective:.2e}")
        print(f"Initial exotic energy: {initial_energy:.2e} J")
        
        start_time = time.time()
        
        # Initialize progress tracking
        progress = None
        if PROGRESS_AVAILABLE:
            try:
                progress = ProgressTracker(
                    total_iterations=max_iterations,
                    description="4D Warp Bubble Optimization",
                    log_level=logging.INFO if 'logging' in globals() else None
                )
                progress.set_stage("parameter_optimization")
                progress.log_metric("initial_exotic_energy", initial_energy)
                progress.log_metric("qi_bound", self.qi_bound)
            except Exception as e:
                print(f"Failed to initialize ProgressTracker: {e}")
                progress = None
        
        with progress if progress else DummyContext():
            if JAX_AVAILABLE:
                # Use JAX optimization
                try:
                    result = minimize(
                        self.objective_function,
                        theta_init,
                        method='BFGS',
                        options={
                            'maxiter': max_iterations,
                            'gtol': tolerance
                        }
                    )
                    theta_opt = result.x
                    success = result.success
                    iterations = result.nit
                    
                    if progress:
                        try:
                            progress.update(iterations)
                        except Exception as e:
                            print(f"Progress update failed: {e}")
                    
                except Exception as e:
                    print(f"JAX optimization failed: {e}")
                    # Fallback to simple gradient descent
                    theta_opt, success, iterations = self._fallback_optimization(theta_init, max_iterations, progress)
            else:
                # NumPy fallback optimization
                theta_opt, success, iterations = self._fallback_optimization(theta_init, max_iterations, progress)
        
        optimization_time = time.time() - start_time
        
        # Compute final metrics
        final_objective = float(self.objective_function(theta_opt))
        final_energy = float(self.total_exotic_energy(theta_opt))
        energy_ratio = abs(final_energy) / self.qi_bound
        
        # Analyze constraints
        gravity_check = self._check_gravity_compensation(theta_opt)
        qi_check = abs(final_energy) >= self.qi_bound * 0.9  # Within 10% of bound
        
        results = {
            'success': success,
            'iterations': iterations,
            'optimization_time': optimization_time,
            'theta_optimal': theta_opt,
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_ratio': energy_ratio,
            'qi_bound': self.qi_bound,
            'gravity_compensated': gravity_check,
            'qi_satisfied': qi_check,
            'is_feasible': success and gravity_check and qi_check
        }
        
        # Print results
        print(f"\nOptimization Results:")
        print(f"  Success: {success}")
        print(f"  Iterations: {iterations}")
        print(f"  Time: {optimization_time:.1f} s")
        print(f"  Final energy: {final_energy:.2e} J")
        print(f"  Energy/QI ratio: {energy_ratio:.3f}")
        print(f"  Gravity compensated: {gravity_check}")
        print(f"  QI bound satisfied: {qi_check}")
        print(f"  Overall feasible: {results['is_feasible']}")        
        if energy_ratio < 1.1:
            print("  🎉 Near-optimal solution found!")
        
        return results
    
    def _fallback_optimization(self, theta_init: jnp.ndarray, max_iter: int, progress=None) -> Tuple:
        """Simple gradient descent fallback when JAX optimization fails."""
        theta = theta_init.copy()
        learning_rate = 1e-6
        
        grad_func = grad(self.objective_function)
        
        for i in range(max_iter):
            if i % 100 == 0:
                obj_val = self.objective_function(theta)
                print(f"  Iteration {i}: objective = {obj_val:.2e}")
                
                # Update progress
                if progress:
                    try:
                        progress.update(100 if i > 0 else 1)
                        progress.log_metric("objective_value", float(obj_val))
                    except Exception as e:
                        print(f"Progress update failed: {e}")
            
            gradient = grad_func(theta)
            theta = theta - learning_rate * gradient
            
            # Simple convergence check
            if i > 10 and jnp.linalg.norm(gradient) < 1e-8:
                if progress:
                    try:
                        progress.update(max_iter - i)  # Complete remaining iterations
                    except Exception:
                        pass
                return theta, True, i
        
        return theta, False, max_iter
    
    def _check_gravity_compensation(self, theta: jnp.ndarray) -> bool:
        """Check if acceleration profile provides gravity compensation."""
        _, a_t_grid = self.spacetime_ansatz(theta)
        return bool(jnp.all(a_t_grid >= G_EARTH))
    
    def visualize_solution(self, theta_opt: jnp.ndarray, save_dir: Optional[str] = None):
        """
        Create comprehensive visualization of the optimized solution.
        """
        print("\nGenerating solution visualization...")
        
        # Extract solution
        mu, G_geo = theta_opt[0], theta_opt[1] 
        f_rt_grid, a_t_grid = self.spacetime_ansatz(theta_opt)
        T00_grid = self.energy_density_4d(theta_opt)
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Spacetime shape function f(r,t)
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(f_rt_grid.T, aspect='auto', origin='lower',
                        extent=[0, self.R, 0, self.T_total/86400],
                        cmap='viridis')
        ax1.set_xlabel('Radius (m)')
        ax1.set_ylabel('Time (days)')
        ax1.set_title('Shape Function f(r,t)')
        plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Energy density T₀₀(r,t)
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(T00_grid.T, aspect='auto', origin='lower',
                        extent=[0, self.R, 0, self.T_total/86400],
                        cmap='RdBu_r')
        ax2.set_xlabel('Radius (m)')
        ax2.set_ylabel('Time (days)')
        ax2.set_title('Energy Density T₀₀(r,t)')
        plt.colorbar(im2, ax=ax2, label='J/m³')
        
        # Plot 3: Temporal acceleration profile
        ax3 = plt.subplot(2, 3, 3)
        t_days = self.t_grid / 86400
        ax3.plot(t_days, a_t_grid, 'b-', linewidth=2, label='Warp acceleration')
        ax3.axhline(G_EARTH, color='r', linestyle='--', alpha=0.7, label='Earth gravity')
        ax3.set_xlabel('Time (days)')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.set_title('Temporal Acceleration Profile')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Spatial shape at key times
        ax4 = plt.subplot(2, 3, 4)
        time_slices = [0.1, 0.3, 0.5, 0.7, 0.9]  # Fractions of total time
        for frac in time_slices:
            t_idx = int(frac * (self.N - 1))
            f_slice = f_rt_grid[:, t_idx]
            ax4.plot(self.r_grid, f_slice, label=f't = {frac*self.T_total/86400:.1f} days')
        ax4.set_xlabel('Radius (m)')
        ax4.set_ylabel('Shape function f(r)')
        ax4.set_title('Spatial Profiles at Different Times')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Energy evolution over time
        ax5 = plt.subplot(2, 3, 5)
        energy_vs_time = []
        for j in range(self.N):
            # Integrate over radius at each time
            T00_slice = T00_grid[:, j]
            volume_elements = 4 * np.pi * self.r_grid**2 * self.dr
            energy_at_t = np.sum(T00_slice * volume_elements)
            energy_vs_time.append(energy_at_t)
        
        ax5.plot(t_days, energy_vs_time, 'g-', linewidth=2)
        ax5.axhline(-self.qi_bound, color='r', linestyle='--', alpha=0.7, 
                   label=f'QI bound ({-self.qi_bound:.2e} J)')
        ax5.set_xlabel('Time (days)')
        ax5.set_ylabel('Instantaneous Energy (J)')
        ax5.set_title('Energy Evolution Over Time')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Parameter summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Compute key metrics
        total_energy = float(self.total_exotic_energy(theta_opt))
        energy_ratio = abs(total_energy) / self.qi_bound
        
        summary_text = f"""Optimization Summary

Bubble Configuration:
• Volume: {self.V:.1f} m³
• Radius: {self.R:.3f} m  
• Flight time: {self.T_total/86400:.1f} days

Results:
• Total exotic energy: {total_energy:.2e} J
• QI bound: {self.qi_bound:.2e} J
• Energy ratio: {energy_ratio:.3f}

LQG Parameters:
• μ (polymer): {mu:.2e}
• G_geo (coupling): {G_geo:.2e}

Performance:
• Grid points: {self.N**2:,}
• JAX acceleration: {JAX_AVAILABLE}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save if requested
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
            plt.savefig(save_path / 'jax_4d_optimization_results.png', 
                       dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path / 'jax_4d_optimization_results.png'}")
        
        plt.close('all')
    
    def save_results(self, results: Dict, save_dir: str):
        """Save optimization results to files."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save parameters
        np.save(save_path / 'optimal_parameters.npy', results['theta_optimal'])
        
        # Save metadata
        metadata = {
            'bubble_volume': self.V,
            'flight_duration_days': self.T_total / 86400,
            'qi_bound': self.qi_bound,
            'grid_resolution': self.N,
            'success': bool(results['success']),
            'final_energy': float(results['final_energy']),
            'energy_ratio': float(results['energy_ratio']),
            'optimization_time': results['optimization_time']
        }
        
        with open(save_path / 'optimization_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Results saved to {save_path}")


def run_parameter_study():
    """
    Run a comprehensive parameter study showing T⁻⁴ scaling.
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE T⁻⁴ SCALING PARAMETER STUDY")
    print("="*60)
    
    # Study configurations
    volumes = [1.0, 5.0, 10.0]  # Different bubble sizes
    durations = [7, 14, 21, 30, 60]  # Different flight times (days)
    
    results = []
    
    for V in volumes:
        print(f"\nBubble Volume: {V:.1f} m³")
        print("-" * 40)
        
        for T_days in durations:
            print(f"  Optimizing {T_days}-day flight...")
            
            # Create optimizer
            optimizer = JAX4DWarpOptimizer(
                bubble_volume=V,
                flight_duration_days=T_days,
                grid_resolution=64  # Reduced for speed
            )
            
            # Quick optimization
            result = optimizer.optimize(max_iterations=200)
            
            if result['success']:
                results.append({
                    'volume': V,
                    'duration_days': T_days,
                    'duration_seconds': T_days * 86400,
                    'qi_bound': result['qi_bound'],
                    'final_energy': result['final_energy'],
                    'energy_ratio': result['energy_ratio'],
                    'feasible': result['is_feasible']
                })
                
                print(f"    Energy: {result['final_energy']:.2e} J")
                print(f"    QI bound: {result['qi_bound']:.2e} J")
                print(f"    Ratio: {result['energy_ratio']:.3f}")
                print(f"    Feasible: {result['is_feasible']}")
    
    # Visualize scaling
    plt.figure(figsize=(12, 8))
    
    # Plot QI bounds vs time for different volumes
    for V in volumes:
        V_results = [r for r in results if r['volume'] == V and r['feasible']]
        if V_results:
            durations_plot = [r['duration_days'] for r in V_results]
            qi_bounds = [r['qi_bound'] for r in V_results]
            energies = [abs(r['final_energy']) for r in V_results]
            
            plt.loglog(durations_plot, qi_bounds, 'o-', 
                      label=f'QI bound, V = {V:.1f} m³', alpha=0.7)
            plt.loglog(durations_plot, energies, 's-', 
                      label=f'Optimized energy, V = {V:.1f} m³')
    
    # Theoretical T⁻⁴ line
    T_theory = np.logspace(0.5, 2, 50)
    QI_theory = 1e-3 / (T_theory * 86400)**4  # C_LQG = 1e-3
    plt.loglog(T_theory, QI_theory, 'k--', alpha=0.5, linewidth=2, 
              label='T⁻⁴ theoretical scaling')
    
    plt.xlabel('Flight Duration (days)')
    plt.ylabel('Energy (J)')
    plt.title('Quantum Inequality T⁻⁴ Scaling Verification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('parameter_study_scaling.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    
    return results


def main():
    """Main optimization script with command-line interface."""
    parser = argparse.ArgumentParser(description='JAX 4D Warp Bubble Optimizer')
    parser.add_argument('--volume', type=float, default=5.0, 
                       help='Bubble volume in m³ (default: 5.0)')
    parser.add_argument('--duration', type=float, default=21, 
                       help='Flight duration in days (default: 21)')
    parser.add_argument('--resolution', type=int, default=128,
                       help='Grid resolution (default: 128)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--study', action='store_true',
                       help='Run parameter study instead of single optimization')
    
    args = parser.parse_args()
    
    if args.study:
        # Run parameter study
        study_results = run_parameter_study()
        return
    
    # Single optimization run
    print("JAX-Accelerated 4D Warp Bubble Optimizer")
    print("=" * 50)
    
    # Create optimizer
    optimizer = JAX4DWarpOptimizer(
        bubble_volume=args.volume,
        flight_duration_days=args.duration,
        grid_resolution=args.resolution
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    if results['success']:
        # Visualize solution
        optimizer.visualize_solution(results['theta_optimal'], save_dir=args.output)
        
        # Save results
        optimizer.save_results(results, args.output)
        
        print(f"\n🎉 Optimization completed successfully!")
        print(f"Results saved to '{args.output}' directory")
        
        if results['is_feasible']:
            print("✅ Feasible warp bubble configuration found!")
        else:
            print("⚠️  Configuration may not satisfy all physical constraints")
    
    else:
        print("❌ Optimization failed to converge")


if __name__ == "__main__":
    main()
