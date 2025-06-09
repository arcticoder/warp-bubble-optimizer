"""
Time-Dependent Dynamic Bubble Simulation
========================================

This module implements dynamic warp bubble simulation with:
- Time-dependent bubble radius R(t) evolution
- Acceleration and deceleration phases
- Perturbation robustness testing
- Integration with existing time-dependent optimizers

Features:
- Smooth trajectory profiles with configurable ramping
- Pulse shaping for transient stability
- Real-time parameter adjustments
- Robustness analysis under perturbations
"""

# JAX for GPU acceleration
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, device_put, vmap, lax
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False

import numpy as np  # Keep for non-JAX operations
import time
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import logging
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Import existing components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from warp_qft.integrated_warp_solver import WarpBubbleSolver, WarpSimulationResult
    from warp_qft.enhancement_pathway import EnhancementPathway
except ImportError:
    # Fallback for development
    WarpBubbleSolver = None
    WarpSimulationResult = None
    EnhancementPathway = None

logger = logging.getLogger(__name__)

@dataclass 
class TrajectoryPoint:
    """Single point in warp bubble trajectory."""
    time: float
    radius: float  # m
    speed: float   # m/s
    energy: float  # J
    stability: float  # [0,1]
    acceleration: float  # m/s²

@dataclass
class DynamicSimulationResult:
    """Results from dynamic bubble simulation."""
    success: bool
    trajectory: List[TrajectoryPoint]
    total_energy: float  # Total energy requirement
    max_acceleration: float  # Maximum acceleration
    min_stability: float  # Minimum stability during trajectory
    perturbation_robustness: float  # [0,1] robustness score
    execution_time: float

@dataclass
class BubbleConfig:
    """Configuration for warp bubble simulation."""
    initial_radius: float = 10.0  # m
    target_radius: float = 20.0   # m
    max_speed: float = 1000.0     # m/s
    max_acceleration: float = 100.0  # m/s²
    stability_threshold: float = 0.7  # [0,1]
    energy_budget: float = 1e12   # J
    simulation_time: float = 100.0  # s
    time_step: float = 0.1        # s

class TrajectoryProfile:
    """
    Defines smooth trajectory profiles for warp bubble dynamics.
    """
    
    def __init__(self, profile_type: str = "smooth_ramp"):
        self.profile_type = profile_type
        
    def linear_ramp(self, t_max: float, R0: float, R1: float, v0: float = 0.0, v1: float = 1000.0):
        """Linear ramp profile."""
        self.t_max = t_max
        self.R0, self.R1 = R0, R1
        self.v0, self.v1 = v0, v1
        
        def R_func(t):
            return self.R0 + (self.R1 - self.R0) * (t / self.t_max)
            
        def v_func(t):
            return self.v0 + (self.v1 - self.v0) * (t / self.t_max)
            
        self.R_func = R_func
        self.v_func = v_func
        return self
    
    def smooth_tanh_ramp(self, t_max: float, R0: float, R1: float, 
                        v0: float = 0.0, v1: float = 1000.0,
                        transition_sharpness: float = 2.0):
        """Smooth hyperbolic tangent ramp."""
        self.t_max = t_max
        self.R0, self.R1 = R0, R1
        self.v0, self.v1 = v0, v1
        
        def R_func(t):
            s = np.tanh(transition_sharpness * (2*t/t_max - 1))
            return self.R0 + (self.R1 - self.R0) * (s + 1) / 2
            
        def v_func(t):
            s = np.tanh(transition_sharpness * (2*t/t_max - 1))
            return self.v0 + (self.v1 - self.v0) * (s + 1) / 2
            
        self.R_func = R_func
        self.v_func = v_func
        return self
    
    def acceleration_profile(self, t_max: float, 
                           accel_phase: float = 0.3,
                           cruise_phase: float = 0.4, 
                           decel_phase: float = 0.3,
                           R0: float = 5.0, R_cruise: float = 15.0, R1: float = 10.0,
                           v_max: float = 5000.0):
        """Three-phase acceleration profile: accel -> cruise -> decel."""
        self.t_max = t_max
        
        t_accel = t_max * accel_phase
        t_cruise = t_max * cruise_phase  
        t_decel = t_max * decel_phase
        
        def R_func(t):
            if t <= t_accel:
                # Acceleration phase: smooth growth
                s = np.sin(np.pi * t / (2 * t_accel))**2
                return R0 + (R_cruise - R0) * s
            elif t <= t_accel + t_cruise:
                # Cruise phase: constant radius
                return R_cruise
            else:
                # Deceleration phase: smooth shrinkage
                t_rel = (t - t_accel - t_cruise) / t_decel
                s = np.cos(np.pi * t_rel / 2)**2
                return R_cruise + (R1 - R_cruise) * (1 - s)
        
        def v_func(t):
            if t <= t_accel:
                # Smooth acceleration
                s = np.sin(np.pi * t / (2 * t_accel))**2
                return v_max * s
            elif t <= t_accel + t_cruise:
                # Constant cruise speed
                return v_max
            else:
                # Smooth deceleration
                t_rel = (t - t_accel - t_cruise) / t_decel
                s = np.cos(np.pi * t_rel / 2)**2
                return v_max * s
        
        self.R_func = R_func
        self.v_func = v_func
        return self
    
    def pulse_shaped_profile(self, t_max: float, 
                           pulse_duration: float = 2.0,
                           R_base: float = 8.0, R_pulse: float = 20.0,
                           v_base: float = 100.0, v_pulse: float = 8000.0):
        """Pulse-shaped profile for transient stability testing."""
        self.t_max = t_max
        
        def gaussian_pulse(t, t_center, width):
            return np.exp(-((t - t_center) / width)**2)
        
        def R_func(t):
            pulse = gaussian_pulse(t, t_max/2, pulse_duration/4)
            return R_base + (R_pulse - R_base) * pulse
            
        def v_func(t):
            pulse = gaussian_pulse(t, t_max/2, pulse_duration/4)
            return v_base + (v_pulse - v_base) * pulse
        
        self.R_func = R_func
        self.v_func = v_func
        return self
    
    def evaluate(self, t: float) -> Tuple[float, float]:
        """Evaluate radius and speed at time t."""
        if not hasattr(self, 'R_func'):
            raise ValueError("No trajectory profile defined. Call a profile method first.")
        return self.R_func(t), self.v_func(t)


class DynamicBubbleSimulator:
    """
    Simulates time-dependent warp bubble dynamics.
    """
    
    def __init__(self, 
                 warp_solver: Optional[Any] = None,
                 time_step: float = 0.1,
                 perturbation_amplitude: float = 0.01):
        """
        Initialize dynamic bubble simulator.
        
        Args:
            warp_solver: Warp bubble solver instance
            time_step: Simulation time step (s)
            perturbation_amplitude: Amplitude for robustness testing
        """
        if WarpBubbleSolver is not None and warp_solver is None:
            self.warp_solver = WarpBubbleSolver()
        else:
            self.warp_solver = warp_solver
        
        self.time_step = time_step
        self.perturbation_amplitude = perturbation_amplitude
        
    def simulate_trajectory(self, 
                          trajectory_profile: TrajectoryProfile,
                          duration: float,
                          include_perturbations: bool = False) -> DynamicSimulationResult:
        """
        Simulate warp bubble trajectory over time.
        
        Args:
            trajectory_profile: Trajectory profile to follow
            duration: Total simulation duration (s)
            include_perturbations: Whether to test perturbation robustness
            
        Returns:
            DynamicSimulationResult with complete trajectory data
        """
        start_time = time.time()
        
        try:
            # Time grid
            times = np.arange(0, duration + self.time_step, self.time_step)
            trajectory = []
            
            # Simulation loop
            for i, t in enumerate(times):
                # Get nominal trajectory point
                R_nominal, v_nominal = trajectory_profile.evaluate(t)
                
                # Add perturbations if requested
                if include_perturbations and i > 0:
                    R_perturbed = R_nominal * (1 + self.perturbation_amplitude * np.random.randn())
                    v_perturbed = v_nominal * (1 + 0.5 * self.perturbation_amplitude * np.random.randn())
                else:
                    R_perturbed, v_perturbed = R_nominal, v_nominal
                
                # Simulate bubble at this configuration
                if self.warp_solver is not None:
                    result = self.warp_solver.simulate(
                        radius=max(R_perturbed, 0.1),  # Ensure positive radius
                        speed=max(v_perturbed, 0.0),   # Ensure positive speed
                        detailed_analysis=False
                    )
                    energy = result.energy_total
                    stability = result.stability
                else:
                    # Fallback calculation
                    energy = self._estimate_energy(R_perturbed, v_perturbed)
                    stability = self._estimate_stability(R_perturbed, v_perturbed)
                
                # Compute acceleration
                if i > 0:
                    prev_v = trajectory[i-1].speed
                    acceleration = (v_perturbed - prev_v) / self.time_step
                else:
                    acceleration = 0.0
                
                # Store trajectory point
                point = TrajectoryPoint(
                    time=t,
                    radius=R_perturbed,
                    speed=v_perturbed,
                    energy=energy,
                    stability=stability,
                    acceleration=acceleration
                )
                trajectory.append(point)
            
            # Analyze results
            total_energy = sum(abs(p.energy) for p in trajectory)
            max_acceleration = max(abs(p.acceleration) for p in trajectory)
            min_stability = min(p.stability for p in trajectory)
            
            # Compute perturbation robustness
            if include_perturbations:
                perturbation_robustness = self._compute_robustness(trajectory)
            else:
                perturbation_robustness = 1.0  # No perturbations tested
            
            execution_time = time.time() - start_time
            
            return DynamicSimulationResult(
                success=min_stability > 0.5,  # Threshold for success
                trajectory=trajectory,
                total_energy=total_energy,
                max_acceleration=max_acceleration,
                min_stability=min_stability,
                perturbation_robustness=perturbation_robustness,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Dynamic simulation failed: {e}")
            execution_time = time.time() - start_time
            
            return DynamicSimulationResult(
                success=False,
                trajectory=[],
                total_energy=float('inf'),
                max_acceleration=float('inf'),
                min_stability=0.0,
                perturbation_robustness=0.0,
                execution_time=execution_time
            )
    
    def _estimate_energy(self, radius: float, speed: float) -> float:
        """Fallback energy estimation."""
        # Simplified energy scaling
        c = 299792458  # m/s
        beta = speed / c
        gamma = 1 / np.sqrt(1 - beta**2) if beta < 1 else 10.0  # Relativistic factor
        
        # Energy scales with bubble volume and relativistic factor
        volume = 4/3 * np.pi * radius**3
        energy_density = -1e20 * gamma  # Negative energy density estimate
        return energy_density * volume
    
    def _estimate_stability(self, radius: float, speed: float) -> float:
        """Fallback stability estimation."""
        # Stability decreases with high speed and small radius
        c = 299792458
        beta = speed / c
        
        # Empirical stability model
        stability = 0.9 * np.exp(-beta) * (radius / 10.0) / (1 + radius / 10.0)
        return max(0.0, min(1.0, stability))
    
    def _compute_robustness(self, trajectory: List[TrajectoryPoint]) -> float:
        """Compute perturbation robustness score."""
        if len(trajectory) < 2:
            return 0.0
        
        # Measure stability variance as robustness indicator
        stabilities = [p.stability for p in trajectory]
        stability_variance = np.var(stabilities)
        
        # Robustness is inversely related to variance
        # High variance = low robustness
        robustness = 1.0 / (1.0 + 10.0 * stability_variance)
        return robustness
    
    def plot_trajectory(self, result: DynamicSimulationResult, save_path: Optional[str] = None):
        """Plot trajectory results."""
        if not result.trajectory:
            logger.warning("No trajectory data to plot")
            return
        
        times = [p.time for p in result.trajectory]
        radii = [p.radius for p in result.trajectory]
        speeds = [p.speed for p in result.trajectory]
        energies = [p.energy for p in result.trajectory]
        stabilities = [p.stability for p in result.trajectory]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Radius vs time
        axes[0,0].plot(times, radii, 'b-', linewidth=2)
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Bubble Radius (m)')
        axes[0,0].set_title('Bubble Radius Evolution')
        axes[0,0].grid(True)
        
        # Speed vs time
        axes[0,1].plot(times, speeds, 'r-', linewidth=2)
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Bubble Speed (m/s)')
        axes[0,1].set_title('Bubble Speed Evolution')
        axes[0,1].grid(True)
        
        # Energy vs time
        axes[1,0].plot(times, [abs(e) for e in energies], 'g-', linewidth=2)
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('|Energy| (J)')
        axes[1,0].set_yscale('log')
        axes[1,0].set_title('Energy Requirement')
        axes[1,0].grid(True)
        
        # Stability vs time
        axes[1,1].plot(times, stabilities, 'm-', linewidth=2)
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Stability')
        axes[1,1].set_title('Bubble Stability')
        axes[1,1].set_ylim([0, 1])
        axes[1,1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trajectory plot saved to {save_path}")
        else:
            plt.show()
    
    def test_perturbation_robustness(self, 
                                   trajectory_profile: TrajectoryProfile,
                                   duration: float,
                                   n_trials: int = 10) -> Dict[str, float]:
        """
        Test robustness against various perturbation amplitudes.
        
        Returns statistical analysis of robustness.
        """
        perturbation_levels = np.logspace(-3, -1, 5)  # 0.1% to 10%
        robustness_scores = []
        
        for perturb_amp in perturbation_levels:
            self.perturbation_amplitude = perturb_amp
            trial_scores = []
            
            for trial in range(n_trials):
                result = self.simulate_trajectory(
                    trajectory_profile, duration, include_perturbations=True
                )
                trial_scores.append(result.perturbation_robustness)
            
            avg_score = np.mean(trial_scores)
            robustness_scores.append(avg_score)
        
        return {
            "perturbation_levels": perturbation_levels.tolist(),
            "robustness_scores": robustness_scores,
            "mean_robustness": np.mean(robustness_scores),
            "robustness_std": np.std(robustness_scores)
        }
    
    @staticmethod
    @jit
    def _jax_simulate_trajectory_step(state, profile_params, dt):
        """Single trajectory step using JAX for GPU acceleration.
        
        Args:
            state: Current state [time, radius, speed, energy, stability]
            profile_params: Parameters for trajectory profile
            dt: Time step
            
        Returns:
            next_state: Updated state after time step
        """
        t, R, v, E, stability = state
        
        # Evaluate trajectory profile at next time
        t_next = t + dt
        
        # Simple linear profile for demonstration
        R0, R1, v0, v1, t_max = profile_params
        alpha = jnp.clip(t_next / t_max, 0.0, 1.0)
        
        R_next = R0 + (R1 - R0) * alpha
        v_next = v0 + (v1 - v0) * alpha
        
        # Estimate energy (simplified)
        c = 299792458.0
        beta = v_next / c
        gamma = 1.0 / jnp.sqrt(jnp.maximum(1 - beta**2, 1e-10))
        
        volume = 4.0/3.0 * jnp.pi * R_next**3
        E_next = -1e20 * gamma * volume  # Negative energy scaling
        
        # Estimate stability (heuristic)
        stability_next = jnp.exp(-0.1 * jnp.abs(beta)) * jnp.exp(-0.01 * jnp.abs(R_next - 10.0))
        
        return jnp.array([t_next, R_next, v_next, E_next, stability_next])
    
    @staticmethod
    @jit 
    def _jax_simulate_full_trajectory(initial_state, profile_params, dt, n_steps):
        """Simulate full trajectory using JAX scan for GPU acceleration.
        
        Args:
            initial_state: [time, radius, speed, energy, stability] 
            profile_params: [R0, R1, v0, v1, t_max]
            dt: Time step
            n_steps: Number of simulation steps
            
        Returns:
            trajectory: (n_steps, 5) array of states over time
        """
        
        def step_fn(state, _):
            next_state = DynamicBubbleSimulator._jax_simulate_trajectory_step(
                state, profile_params, dt
            )
            return next_state, next_state
        
        # Use JAX scan for efficient unrolling
        final_state, trajectory = lax.scan(step_fn, initial_state, None, length=n_steps)
        
        return trajectory
    
    def simulate_trajectory_jax(self, 
                               trajectory_profile: TrajectoryProfile,
                               duration: float,
                               include_perturbations: bool = False,
                               use_gpu: bool = True) -> DynamicSimulationResult:
        """
        JAX-accelerated trajectory simulation for GPU computation.
        
        Args:
            trajectory_profile: Trajectory profile to follow
            duration: Total simulation duration (s)
            include_perturbations: Whether to test perturbation robustness
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            DynamicSimulationResult with complete trajectory data
        """
        start_time = time.time()
        
        if not JAX_AVAILABLE or not use_gpu:
            logger.warning("JAX not available or GPU disabled, falling back to CPU simulation")
            return self.simulate_trajectory(trajectory_profile, duration, include_perturbations)
        
        try:
            # Setup simulation parameters 
            n_steps = int(duration / self.time_step)
            dt = duration / n_steps
            
            # Get trajectory profile parameters
            if hasattr(trajectory_profile, 'R0'):
                profile_params = jnp.array([
                    trajectory_profile.R0, trajectory_profile.R1,
                    trajectory_profile.v0, trajectory_profile.v1,
                    trajectory_profile.t_max
                ])
            else:
                # Default parameters
                profile_params = jnp.array([10.0, 20.0, 0.0, 1000.0, duration])
            
            # Transfer to GPU
            profile_params = device_put(profile_params)
            
            # Initial state: [time, radius, speed, energy, stability]
            initial_state = jnp.array([0.0, profile_params[0], profile_params[2], 0.0, 1.0])
            initial_state = device_put(initial_state)
            
            # Run JAX-accelerated simulation
            trajectory_jax = self._jax_simulate_full_trajectory(
                initial_state, profile_params, dt, n_steps
            )
            
            # Convert back to CPU and create trajectory points
            trajectory_cpu = np.array(trajectory_jax)
            trajectory = []
            
            for i, state in enumerate(trajectory_cpu):
                t, R, v, E, stability = state
                
                # Compute acceleration
                if i > 0:
                    prev_v = trajectory_cpu[i-1, 2]
                    acceleration = (v - prev_v) / dt
                else:
                    acceleration = 0.0
                
                point = TrajectoryPoint(
                    time=float(t),
                    radius=float(R),
                    speed=float(v),
                    energy=float(E),
                    stability=float(stability),
                    acceleration=float(acceleration)
                )
                trajectory.append(point)
            
            # Analyze results
            total_energy = sum(abs(p.energy) for p in trajectory)
            max_acceleration = max(abs(p.acceleration) for p in trajectory)
            min_stability = min(p.stability for p in trajectory)
            
            # Perturbation robustness (simplified for GPU version)
            perturbation_robustness = min_stability if include_perturbations else 1.0
            
            execution_time = time.time() - start_time
            
            return DynamicSimulationResult(
                success=min_stability > 0.5,
                trajectory=trajectory,
                total_energy=total_energy,
                max_acceleration=max_acceleration,
                min_stability=min_stability,
                perturbation_robustness=perturbation_robustness,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"JAX trajectory simulation failed: {e}")
            # Fallback to CPU simulation
            return self.simulate_trajectory(trajectory_profile, duration, include_perturbations)


# JAX-accelerated simulation functions
if JAX_AVAILABLE:
    @jit
    def jax_simulate_step(R: float, v: float, t: float, params: jnp.ndarray) -> Tuple[float, float, float]:
        """
        JAX-accelerated single simulation step.
        
        Returns: (energy, stability, acceleration)
        """
        # Extract parameters
        c = params[0]  # Speed of light
        energy_scale = params[1]
        stability_factor = params[2]
        
        # Compute negative energy density (Alcubierre metric)
        energy_density = -energy_scale * (R**3) * (v/c)**2 / (1 + (v/c)**2)
        total_energy = 4 * jnp.pi * R**3 * energy_density / 3
        
        # Stability metric (higher is better)
        v_normalized = v / c
        stability = stability_factor * jnp.exp(-v_normalized**2) * jnp.exp(-R/100.0)
        stability = jnp.clip(stability, 0.0, 1.0)
        
        # Acceleration based on energy gradients
        dE_dv = -2 * energy_scale * R**3 * (v/c) / (c * (1 + (v/c)**2)**2)
        acceleration = jnp.abs(dE_dv / 1e9)  # Scale for realistic values
        
        return total_energy, stability, acceleration

    @jit
    def jax_trajectory_step(state: jnp.ndarray, t: float, params: jnp.ndarray) -> jnp.ndarray:
        """
        JAX-accelerated trajectory evolution step.
        
        state = [R, v, E, stability]
        """
        R, v, E, stability = state
        
        # Simulate dynamics
        energy, new_stability, acceleration = jax_simulate_step(R, v, t, params)
        
        # Simple dynamics: R and v evolve based on profiles
        dt = 0.1
        dR_dt = (20.0 - 10.0) / 100.0  # Linear ramp
        dv_dt = (1000.0 - 0.0) / 100.0  # Linear ramp
        
        new_R = R + dR_dt * dt
        new_v = v + dv_dt * dt
        
        return jnp.array([new_R, new_v, energy, new_stability])

    def simulate_trajectory_jax(R_profile_func, v_profile_func, t_max: float = 100.0, dt: float = 0.1) -> List[TrajectoryPoint]:
        """
        JAX-accelerated trajectory simulation.
        """
        # Create time array
        times = jnp.arange(0, t_max, dt)
        n_steps = len(times)
        
        # Evaluate profiles at all time points
        R_vals = jnp.array([R_profile_func(t) for t in times])
        v_vals = jnp.array([v_profile_func(t) for t in times])
        
        # Simulation parameters
        params = jnp.array([
            2.998e8,  # Speed of light
            1e30,     # Energy scale
            0.95      # Stability factor
        ])
        
        # Vectorized simulation over all time steps
        @vmap
        def simulate_all_steps(R, v, t):
            return jax_simulate_step(R, v, t, params)
        
        # Move data to device and compute
        R_device = device_put(R_vals)
        v_device = device_put(v_vals)
        times_device = device_put(times)
        
        energies, stabilities, accelerations = simulate_all_steps(R_device, v_device, times_device)
        
        # Convert back to TrajectoryPoint objects
        trajectory = []
        for i in range(n_steps):
            point = TrajectoryPoint(
                time=float(times[i]),
                radius=float(R_vals[i]),
                speed=float(v_vals[i]),
                energy=float(energies[i]),
                stability=float(stabilities[i]),
                acceleration=float(accelerations[i])
            )
            trajectory.append(point)
        
        return trajectory

else:
    # Fallback CPU implementation
    def simulate_trajectory_jax(R_profile_func, v_profile_func, t_max: float = 100.0, dt: float = 0.1) -> List[TrajectoryPoint]:
        """CPU fallback for trajectory simulation."""
        return simulate_trajectory(R_profile_func, v_profile_func, dt)


# Utility functions for backwards compatibility
def simulate_trajectory(R_profile, v_profile, dt: float = 0.5) -> List[Tuple]:
    """
    Utility function for backwards compatibility - simulates trajectory using profiles.
    Returns: List of (time, radius, velocity, energy, stability) tuples
    """
    simulator = DynamicBubbleSimulator(time_step=dt)
    
    # Create a trajectory profile from the input functions
    trajectory_profile = TrajectoryProfile()
    trajectory_profile.R_func = R_profile
    trajectory_profile.v_func = v_profile
    trajectory_profile.t_max = 10.0  # Default duration
    
    result = simulator.simulate_trajectory(trajectory_profile, duration=trajectory_profile.t_max)
    
    # Convert to expected format
    trajectory = []
    for point in result.trajectory:
        trajectory.append((point.time, point.radius, point.speed, point.energy, point.stability))
    
    return trajectory

class LinearRamp:
    """Simple linear ramp for backwards compatibility."""
    
    def __init__(self, duration: float, start_val: float, end_val: float):
        self.duration = duration
        self.start_val = start_val
        self.end_val = end_val
        
    def __call__(self, t: float) -> float:
        if t >= self.duration:
            return self.end_val
        return self.start_val + (self.end_val - self.start_val) * (t / self.duration)


# Example usage and testing
if __name__ == "__main__":
    # Create trajectory profiles
    print("=== Testing Trajectory Profiles ===")
    
    # Linear ramp
    linear_profile = TrajectoryProfile().linear_ramp(
        t_max=10.0, R0=5.0, R1=15.0, v0=0.0, v1=3000.0
    )
    
    # Smooth tanh ramp  
    smooth_profile = TrajectoryProfile().smooth_tanh_ramp(
        t_max=15.0, R0=8.0, R1=12.0, v0=100.0, v1=5000.0
    )
    
    # Acceleration profile
    accel_profile = TrajectoryProfile().acceleration_profile(
        t_max=20.0, R0=5.0, R_cruise=18.0, R1=10.0, v_max=8000.0
    )
    
    # Pulse profile for stability testing
    pulse_profile = TrajectoryProfile().pulse_shaped_profile(
        t_max=8.0, pulse_duration=2.0, R_base=8.0, R_pulse=25.0
    )
    
    # Test trajectory evaluation
    for i, profile in enumerate([linear_profile, smooth_profile, accel_profile, pulse_profile]):
        name = ["Linear", "Smooth", "Acceleration", "Pulse"][i]
        print(f"\n{name} Profile:")
        for t in [0, 2.5, 5.0, 7.5, 10.0]:
            if hasattr(profile, 'R_func'):
                R, v = profile.evaluate(t)
                print(f"  t={t:4.1f}s: R={R:6.1f}m, v={v:6.0f}m/s")
    
    # Test dynamic simulation
    print("\n=== Testing Dynamic Simulation ===")
    simulator = DynamicBubbleSimulator(time_step=0.5)
    
    result = simulator.simulate_trajectory(
        smooth_profile, duration=10.0, include_perturbations=False
    )
    
    print(f"Simulation success: {result.success}")
    print(f"Total energy: {result.total_energy:.2e} J")
    print(f"Max acceleration: {result.max_acceleration:.2e} m/s²")
    print(f"Min stability: {result.min_stability:.3f}")
    print(f"Execution time: {result.execution_time:.3f}s")
    
    # Test perturbation robustness
    print("\n=== Testing Perturbation Robustness ===")
    robustness_analysis = simulator.test_perturbation_robustness(
        pulse_profile, duration=5.0, n_trials=3  # Reduced for testing
    )
    
    print(f"Mean robustness: {robustness_analysis['mean_robustness']:.3f}")
    print(f"Robustness std: {robustness_analysis['robustness_std']:.3f}")
    
    # Plot example trajectory
    result = simulator.simulate_trajectory(accel_profile, duration=15.0)
    simulator.plot_trajectory(result, save_path="dynamic_trajectory_test.png")
