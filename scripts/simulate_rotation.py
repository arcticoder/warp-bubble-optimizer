#!/usr/bin/env python3
"""
Warp Bubble Rotation and Attitude Control Simulation
===================================================

This module implements rotation and fine-pointing capabilities for warp
bubbles, enabling precise attitude control and orientation maneuvers.
Includes quaternion-based rotation, angular velocity profiles, and
integrated translation+rotation for complex 6-DOF maneuvers.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass

# JAX imports with fallback
try:
    import jax.numpy as jnp  # type: ignore
    from jax import jit, vmap, grad  # type: ignore
    JAX_AVAILABLE = True
    print("🚀 JAX acceleration enabled for rotation simulation")
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False
    print("⚠️  JAX not available - using NumPy fallback")
    # Fallback decorators
    def jit(func): return func
    def vmap(func, in_axes=None): 
        def vectorized(*args):
            return np.array([func(*[arg[i] if isinstance(arg, np.ndarray) else arg 
                                  for arg in args]) for i in range(len(args[0]))])
        return vectorized
    def grad(func): 
        def grad_func(x, *args):
            h = 1e-8
            return (func(x + h, *args) - func(x - h, *args)) / (2 * h)
        return grad_func

# ProgressTracker import with fallback
try:
    from progress_tracker import ProgressTracker as _ProgressTracker  # type: ignore
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False
    class _ProgressTracker:
        def __init__(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def set_stage(self, *args, **kwargs): pass
        def complete(self, *args, **kwargs): pass
        def log_metric(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
ProgressTracker = _ProgressTracker

@dataclass
class Quaternion:
    """Quaternion for 3D rotations with operations."""
    
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Initialize quaternion with scalar and vector components."""
        self.w = w  # Scalar part
        self.x = x  # Vector parts
        self.y = y
        self.z = z
        self.q = jnp.array([self.w, self.x, self.y, self.z])
        self.q = self.q / jnp.linalg.norm(self.q)  # Normalize
        # Update components after normalization
        self.w, self.x, self.y, self.z = float(self.q[0]), float(self.q[1]), float(self.q[2]), float(self.q[3])
    
    @classmethod
    def from_axis_angle(cls, axis: jnp.ndarray, angle: float) -> 'Quaternion':
        """Create quaternion from axis-angle representation."""
        axis = axis / jnp.linalg.norm(axis)
        half_angle = angle / 2.0
        w = jnp.cos(half_angle)
        xyz = jnp.sin(half_angle) * axis
        return cls(float(w), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    
    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> 'Quaternion':
        """Create quaternion from Euler angles (ZYX convention)."""
        cr = jnp.cos(roll * 0.5)
        sr = jnp.sin(roll * 0.5)
        cp = jnp.cos(pitch * 0.5)
        sp = jnp.sin(pitch * 0.5)
        cy = jnp.cos(yaw * 0.5)
        sy = jnp.sin(yaw * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return cls(float(w), float(x), float(y), float(z))
    
    def conjugate(self) -> 'Quaternion':
        """Quaternion conjugate."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def multiply(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion multiplication."""
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(float(w), float(x), float(y), float(z))
    
    def to_rotation_matrix(self) -> jnp.ndarray:
        """Convert to 3x3 rotation matrix."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return jnp.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
    
    def to_euler(self) -> Tuple[float, float, float]:
        """Convert to Euler angles (roll, pitch, yaw)."""
        w, x, y, z = self.w, self.x, self.y, self.z
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = jnp.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if jnp.abs(sinp) >= 1:
            pitch = jnp.copysign(jnp.pi / 2, sinp)
        else:
            pitch = jnp.arcsin(sinp)
          # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = jnp.arctan2(siny_cosp, cosy_cosp)
        
        return float(roll), float(pitch), float(yaw)
    
    def angular_distance(self, other: 'Quaternion') -> float:
        """Calculate angular distance between two quaternions."""
        dot = jnp.abs(jnp.dot(self.q, other.q))
        return 2.0 * jnp.arccos(jnp.clip(dot, 0.0, 1.0))
    
    def slerp(self, other: 'Quaternion', t: float) -> 'Quaternion':
        """Spherical linear interpolation between quaternions."""
        dot = jnp.dot(self.q, other.q)
        
        # If dot product is negative, use -other to take shorter path
        if dot < 0.0:
            other = Quaternion(-other.w, -other.x, -other.y, -other.z)
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = self.q + t * (other.q - self.q)
            result = result / jnp.linalg.norm(result)
            return Quaternion(*result)
        
        # Calculate angle between quaternions
        theta_0 = jnp.arccos(jnp.clip(jnp.abs(dot), 0.0, 1.0))
        sin_theta_0 = jnp.sin(theta_0)
        theta = theta_0 * t
        sin_theta = jnp.sin(theta)
        
        s0 = jnp.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        result = s0 * self.q + s1 * other.q
        return Quaternion(*result)
    
    def __str__(self) -> str:
        return f"Q({self.w:.3f}, {self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

@dataclass
class RotationProfile:
    """Configuration for rotation maneuver."""
    target_orientation: Optional[Quaternion] = None     # Target orientation
    omega_max: float = 0.1                    # Maximum angular velocity (rad/s)
    t_up: float = 5.0                         # Ramp-up time
    t_hold: float = 10.0                      # Hold time (fine pointing)
    t_down: float = 5.0                       # Ramp-down time
    n_steps: int = 500                        # Time discretization
    control_mode: str = "smooth"              # "smooth", "bang_bang", "exponential"
    
    def __post_init__(self):
        if self.target_orientation is None:
            self.target_orientation = Quaternion.from_euler(0.0, 0.0, np.pi/4)  # 45° yaw

@dataclass  
class WarpBubbleRotational:
    """Warp bubble configuration for rotational maneuvers."""
    R_max: float = 100.0                      # Maximum radius
    n_r: int = 1000                          # Radial grid points
    shape_params: jnp.ndarray = None          # Shape parameters
    thickness: float = 1.0                    # Bubble wall thickness
    moment_of_inertia: float = 1e6            # Effective moment of inertia (kg⋅m²)
    rotational_coupling: float = 0.5          # Coupling between rotation and energy
    
    def __post_init__(self):
        if self.shape_params is None:
            self.shape_params = jnp.array([1.0, 2.0, 0.5])

def angular_velocity_profile(t: float, omega_max: float, t_up: float, 
                           t_hold: float, t_down: float, mode: str = "smooth") -> float:
    """
    Compute angular velocity magnitude at time t.
    
    Args:
        t: Time coordinate
        omega_max: Maximum angular velocity
        t_up: Ramp-up duration
        t_hold: Hold duration
        t_down: Ramp-down duration
        mode: Control profile type
        
    Returns:
        Angular velocity magnitude at time t
    """
    if t < 0:
        return 0.0
    elif t < t_up:
        if mode == "smooth":
            # Smooth polynomial ramp
            tau = t / t_up
            return omega_max * (3*tau**2 - 2*tau**3)
        elif mode == "bang_bang":
            return omega_max
        else:  # exponential
            return omega_max * (1.0 - jnp.exp(-5*t/t_up))
    elif t < t_up + t_hold:
        return omega_max
    elif t < t_up + t_hold + t_down:
        if mode == "smooth":
            tau = (t - t_up - t_hold) / t_down
            return omega_max * (1.0 - (3*tau**2 - 2*tau**3))
        elif mode == "bang_bang":
            return 0.0
        else:  # exponential
            return omega_max * jnp.exp(-5*(t - t_up - t_hold)/t_down)
    else:
        return 0.0

def compute_rotation_trajectory(profile: RotationProfile) -> Dict[str, Any]:
    """
    Compute quaternion trajectory for rotation maneuver.
    
    Args:
        profile: Rotation profile configuration
        
    Returns:
        Rotation trajectory data
    """
    t_total = profile.t_up + profile.t_hold + profile.t_down
    ts = jnp.linspace(0.0, t_total, profile.n_steps)
    dt = ts[1] - ts[0] if len(ts) > 1 else 1.0
    
    # Compute angular velocity magnitudes
    omega_mags = jnp.array([angular_velocity_profile(
        t, profile.omega_max, profile.t_up, profile.t_hold, profile.t_down, profile.control_mode
    ) for t in ts])
    
    # Compute rotation axis (fixed for this implementation)
    initial_q = Quaternion(1.0, 0.0, 0.0, 0.0)  # Identity
    target_q = profile.target_orientation if profile.target_orientation is not None else initial_q
    
    # Find rotation axis using quaternion logarithm
    relative_q = initial_q.conjugate().multiply(target_q)
    if relative_q.w < 0:
        relative_q = Quaternion(-relative_q.w, -relative_q.x, -relative_q.y, -relative_q.z)
    
    # Extract axis and total angle
    vec_norm = jnp.sqrt(relative_q.x**2 + relative_q.y**2 + relative_q.z**2)
    if vec_norm > 1e-6:
        rotation_axis = jnp.array([relative_q.x, relative_q.y, relative_q.z]) / vec_norm
        total_angle = 2.0 * jnp.arctan2(vec_norm, relative_q.w)
    else:
        rotation_axis = jnp.array([0.0, 0.0, 1.0])  # Default to z-axis
        total_angle = 0.0
    
    # Integrate angular velocity to get orientation trajectory
    orientations = []
    angles = []
    current_q = initial_q
    current_angle = 0.0
    
    for i, omega_mag in enumerate(omega_mags):
        if i > 0:
            # Integrate angular velocity
            d_angle = omega_mag * dt
            current_angle += d_angle
            
            # Clamp to total rotation
            if current_angle > total_angle:
                current_angle = total_angle
            
            # Create incremental rotation
            if total_angle > 1e-6:
                progress = current_angle / total_angle
                current_q = initial_q.slerp(target_q, progress)
        
        orientations.append(current_q)
        angles.append(current_angle)
    
    return {
        'time_grid': np.array(ts),
        'angular_velocities': np.array(omega_mags),
        'orientations': orientations,
        'rotation_angles': np.array(angles),
        'rotation_axis': rotation_axis,
        'total_angle': total_angle,
        'final_orientation': orientations[-1]
    }

@jit
def rotational_energy_density(r: jnp.ndarray, omega: float, warp_params: WarpBubbleRotational) -> jnp.ndarray:
    """
    Compute energy density for rotating warp bubble.
    
    Args:
        r: Radial coordinates
        omega: Angular velocity magnitude
        warp_params: Bubble configuration
        
    Returns:
        Energy density T00(r) for rotation
    """
    if omega == 0.0:
        return jnp.zeros_like(r)
    
    # Rotational energy scales as ω²
    base_density = -1e12 * omega**2 * warp_params.rotational_coupling
    
    # Spatial profile: enhanced at bubble wall
    sigma = warp_params.thickness
    r_center = warp_params.R_max * 0.85  # Slightly further out for rotation
    
    # Gaussian profile with rotational enhancement
    spatial_profile = jnp.exp(-0.5 * ((r - r_center) / sigma)**2)
    
    # Additional centrifugal term
    centrifugal_factor = 1.0 + 0.1 * (r / warp_params.R_max)**2 * omega**2
    
    return base_density * spatial_profile * centrifugal_factor

@jit
def compute_rotational_energy_integral(omega: float, warp_params: WarpBubbleRotational) -> float:
    """
    Compute total rotational energy requirement.
    
    Args:
        omega: Angular velocity magnitude
        warp_params: Bubble configuration
        
    Returns:
        Total rotational energy (J)
    """
    rs = jnp.linspace(0.0, warp_params.R_max, warp_params.n_r)
    densities = rotational_energy_density(rs, omega, warp_params)
    
    # Volume integral
    integrand = 4 * jnp.pi * rs**2 * jnp.abs(densities)
    # Use trapezoid integration to avoid deprecation warnings
    _trap = getattr(jnp, 'trapezoid', None)
    energy_field = (_trap or jnp.trapz)(integrand, rs)
    
    # Add classical rotational kinetic energy
    classical_energy = 0.5 * warp_params.moment_of_inertia * omega**2
    
    return float(energy_field) + float(classical_energy)

def simulate_rotation_maneuver(profile: RotationProfile, warp_params: WarpBubbleRotational,
                             enable_progress: bool = True) -> Dict[str, Any]:
    """
    Simulate complete rotational maneuver.
    
    Args:
        profile: Rotation profile configuration
        warp_params: Bubble parameters
        enable_progress: Enable progress tracking
        
    Returns:
        Simulation results
    """
    # Initialize progress tracking
    progress = None
    if PROGRESS_AVAILABLE and enable_progress:
        try:
            progress = ProgressTracker(
                total_steps=5,
                description="Rotation Simulation"
            )
            progress.set_stage("rotation_planning")
        except Exception as e:
            print(f"⚠️  ProgressTracker initialization failed: {e}")
    
    class DummyContext:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    with progress if progress else DummyContext():
        
        if progress: progress.update("Computing rotation trajectory", step_number=1)
        
        # Compute trajectory
        trajectory = compute_rotation_trajectory(profile)
        
        if progress: 
            progress.update("Computing rotational energies", step_number=2)
            progress.log_metric("total_rotation_angle", float(trajectory['total_angle']))
            progress.log_metric("max_angular_velocity", float(profile.omega_max))
        
        # Compute energy at each time step
        omega_mags = trajectory['angular_velocities']
        if JAX_AVAILABLE:
            energy_func = vmap(lambda omega: compute_rotational_energy_integral(omega, warp_params))
            E_ts = energy_func(omega_mags)
        else:
            E_ts = np.array([compute_rotational_energy_integral(omega, warp_params) for omega in omega_mags])
        
        if progress: progress.update("Analyzing rotation performance", step_number=3)

        # Performance metrics
        ts = trajectory['time_grid']
        _trap = getattr(jnp, 'trapezoid', None)
        E_total = (_trap or jnp.trapz)(E_ts, ts)
        E_peak = jnp.max(E_ts)

        # Rotation accuracy
        final_q = trajectory['final_orientation']
        target_q = profile.target_orientation if profile.target_orientation is not None else final_q

        # Quaternion distance as accuracy metric
        q_error = final_q.conjugate().multiply(target_q)
        rotation_error = 2.0 * jnp.arccos(jnp.clip(jnp.abs(q_error.w), 0.0, 1.0))
        rotation_accuracy = 1.0 - rotation_error / (2.0 * jnp.pi)

        if progress: progress.update("Computing attitude control metrics", step_number=4)

        # Angular acceleration analysis
        omega_dot = jnp.gradient(omega_mags, ts[1] - ts[0] if len(ts) > 1 else 1.0)
        max_angular_accel = jnp.max(jnp.abs(omega_dot))

        # Stability metrics
        omega_variance = jnp.var(omega_mags)
        control_smoothness = 1.0 / (1.0 + max_angular_accel)

        if progress:
            progress.update("Finalizing rotation results", step_number=5)
            progress.log_metric("total_energy", float(E_total))
            progress.log_metric("rotation_accuracy", float(rotation_accuracy))

        results = {
            'time_grid': np.array(ts),
            'angular_velocity_profile': np.array(omega_mags),
            'orientation_trajectory': trajectory['orientations'],
            'rotation_angles': trajectory['rotation_angles'],
            'energy_timeline': np.array(E_ts),
            'total_energy': float(E_total),
            'peak_energy': float(E_peak),
            'maneuver_duration': float(ts[-1]),
            'rotation_error': float(rotation_error),
            'rotation_accuracy': float(rotation_accuracy),
            'max_angular_accel': float(max_angular_accel),
            'control_smoothness': float(control_smoothness),
            'final_orientation': final_q,
            'target_orientation': target_q,
            'rotation_axis': trajectory['rotation_axis'],
            'total_rotation_angle': trajectory['total_angle'],
            'profile': profile,
            'warp_params': warp_params
        }

        if progress:
            progress.complete({
                'total_energy_MJ': float(E_total)/1e6,
                'accuracy_percent': rotation_accuracy*100,
                'rotation_degrees': np.degrees(trajectory['total_angle'])
            })

        return results

def combined_translation_rotation(translation_target: jnp.ndarray, 
                                rotation_target: Quaternion,
                                maneuver_time: float = 60.0,
                                n_steps: int = 1000) -> Dict[str, Any]:
    """
    Simulate combined translation and rotation maneuver.
    
    Args:
        translation_target: Target displacement vector (m)
        rotation_target: Target orientation quaternion
        maneuver_time: Total maneuver duration (s)
        n_steps: Time discretization steps
        
    Returns:
        Combined 6-DOF simulation results
    """
    print("🔄 Simulating combined 6-DOF maneuver...")
    
    # Time grid
    ts = jnp.linspace(0.0, maneuver_time, n_steps)
    dt = ts[1] - ts[0]
    
    # Translation parameters
    v_max = jnp.linalg.norm(translation_target) / (maneuver_time * 0.6)  # 60% duty cycle
    t_phase = maneuver_time / 3.0  # Equal phases
    
    # Rotation parameters
    initial_q = Quaternion(1.0, 0.0, 0.0, 0.0)
    relative_q = initial_q.conjugate().multiply(rotation_target)
    vec_norm = jnp.sqrt(relative_q.x**2 + relative_q.y**2 + relative_q.z**2)
    total_angle = 2.0 * jnp.arctan2(vec_norm, jnp.abs(relative_q.w))
    omega_max = total_angle / (maneuver_time * 0.6)
    
    # Compute trajectories
    positions = []
    orientations = []
    velocities = []
    angular_velocities = []
    combined_energies = []
    
    current_pos = jnp.zeros(3)
    current_q = initial_q
    
    for i, t in enumerate(ts):
        # Translation velocity
        if t < t_phase:
            v_scalar = v_max * (t / t_phase)
        elif t < 2 * t_phase:
            v_scalar = v_max
        elif t < 3 * t_phase:
            v_scalar = v_max * (1.0 - (t - 2*t_phase) / t_phase)
        else:
            v_scalar = 0.0
        
        direction = translation_target / jnp.linalg.norm(translation_target)
        velocity = v_scalar * direction
        
        # Angular velocity
        omega = angular_velocity_profile(t, omega_max, t_phase, t_phase, t_phase)
        
        # Update position and orientation
        if i > 0:
            current_pos += velocity * dt
            if omega > 0:
                progress = min(1.0, jnp.sum(angular_velocities) * dt / total_angle)
                current_q = initial_q.slerp(rotation_target, progress)
        
        # Energy calculation (simplified combined model)
        trans_energy = 1e15 * v_scalar**2  # Translation energy
        rot_energy = 1e12 * omega**2       # Rotational energy
        coupling_energy = 5e13 * v_scalar * omega  # Coupling term
        total_energy = trans_energy + rot_energy + coupling_energy
        
        positions.append(current_pos)
        orientations.append(current_q)
        velocities.append(velocity)
        angular_velocities.append(omega)
        combined_energies.append(total_energy)
    
    # Final analysis
    final_pos_error = jnp.linalg.norm(current_pos - translation_target)
    final_q_error = current_q.conjugate().multiply(rotation_target)
    final_rot_error = 2.0 * jnp.arccos(jnp.clip(jnp.abs(final_q_error.w), 0.0, 1.0))
    
    return {
        'time_grid': np.array(ts),
        'position_trajectory': np.array(positions),
        'orientation_trajectory': orientations,
        'velocity_profile': np.array(velocities),
        'angular_velocity_profile': np.array(angular_velocities),
        'energy_timeline': np.array(combined_energies),
        'total_energy': float(jnp.trapz(combined_energies, ts)),
        'final_position_error': float(final_pos_error),
        'final_rotation_error': float(final_rot_error),
        'translation_accuracy': float(1.0 - final_pos_error / jnp.linalg.norm(translation_target)),
        'rotation_accuracy': float(1.0 - final_rot_error / (2.0 * jnp.pi)),
        'target_translation': translation_target,
        'target_rotation': rotation_target
    }

def visualize_rotation_results(results: Dict[str, Any], save_plots: bool = True):
    """
    Create comprehensive visualization of rotation simulation.
    
    Args:
        results: Results from simulate_rotation_maneuver()
        save_plots: Whether to save plots
    """
    print("📊 Generating rotation simulation visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Warp Bubble Rotation and Attitude Control', fontsize=16, fontweight='bold')
    
    ts = results['time_grid']
    omegas = results['angular_velocity_profile']
    angles = results['rotation_angles']
    Es = results['energy_timeline']
    
    # Angular velocity profile
    ax1.plot(ts, omegas, linewidth=2, color='blue', label='Angular Velocity')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angular Velocity (rad/s)')
    ax1.set_title('Angular Velocity Profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Rotation angle progression
    ax2.plot(ts, np.degrees(angles), linewidth=2, color='green', label='Rotation Angle')
    target_angle = np.degrees(results['total_rotation_angle'])
    ax2.axhline(target_angle, color='red', linestyle='--', label=f'Target: {target_angle:.1f}°')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Rotation Angle (degrees)')
    ax2.set_title('Rotation Progress')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Energy timeline
    ax3.plot(ts, Es / 1e6, linewidth=2, color='red', label='Rotational Energy')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Energy Requirement (MJ)')
    ax3.set_title('Rotational Energy Timeline')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Mission summary
    ax4.axis('off')
    
    profile = results['profile']
    final_euler = results['final_orientation'].to_euler()
    target_euler = results['target_orientation'].to_euler()
    
    summary_text = f"""
ROTATION MISSION SUMMARY

Target Rotation: {np.degrees(target_euler[0]):.1f}°, {np.degrees(target_euler[1]):.1f}°, {np.degrees(target_euler[2]):.1f}°
Final Rotation: {np.degrees(final_euler[0]):.1f}°, {np.degrees(final_euler[1]):.1f}°, {np.degrees(final_euler[2]):.1f}°
Rotation Error: {np.degrees(results['rotation_error']):.2f}°
Accuracy: {results['rotation_accuracy']*100:.1f}%

Maximum ω: {profile.omega_max:.3f} rad/s
Total Angle: {np.degrees(results['total_rotation_angle']):.1f}°
Duration: {results['maneuver_duration']:.1f} s
Control Mode: {profile.control_mode}

Energy Requirements:
• Total Energy: {results['total_energy']/1e6:.2f} MJ
• Peak Energy: {results['peak_energy']/1e6:.2f} MJ
• Energy/Degree: {results['total_energy']/(np.degrees(results['total_rotation_angle'])*1e3):.1f} kJ/°

Performance:
• Accuracy: {results['rotation_accuracy']*100:.1f}%
• Smoothness: {results['control_smoothness']*100:.1f}%
• Max α: {results['max_angular_accel']:.3f} rad/s²
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plots:
        angle_deg = np.degrees(results['total_rotation_angle'])
        filename = f"rotation_maneuver_{angle_deg:.0f}deg.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"📁 Plot saved: {filename}")
    
    plt.show()

# Example usage and testing
if __name__ == "__main__":
    print("🔄 WARP BUBBLE ROTATION SIMULATION")
    print("=" * 60)
    
    # Example 1: Simple rotation maneuver
    print("\n1. 📊 Single Rotation Maneuver")
    target_orientation = Quaternion.from_euler(0.3, 0.2, 0.5)  # Roll, pitch, yaw
    
    profile = RotationProfile(
        target_orientation=target_orientation,
        omega_max=0.05,        # 0.05 rad/s max
        t_up=8.0,
        t_hold=15.0,
        t_down=8.0,
        n_steps=800,
        control_mode="smooth"
    )
    
    warp_params = WarpBubbleRotational(
        R_max=120.0,
        thickness=2.5,
        moment_of_inertia=5e6,
        rotational_coupling=0.3
    )
    
    rotation_results = simulate_rotation_maneuver(profile, warp_params)
    
    print(f"✅ Rotation simulation complete!")
    print(f"   Target: {target_orientation}")
    print(f"   Final: {rotation_results['final_orientation']}")
    print(f"   Error: {np.degrees(rotation_results['rotation_error']):.2f}°")
    print(f"   Accuracy: {rotation_results['rotation_accuracy']*100:.1f}%")
    print(f"   Total energy: {rotation_results['total_energy']/1e6:.2f} MJ")
    
    # Example 2: Combined 6-DOF maneuver
    print(f"\n2. 🚀 Combined Translation + Rotation")
    trans_target = jnp.array([500.0, 300.0, -100.0])  # 600m displacement
    rot_target = Quaternion.from_euler(0.1, -0.2, 0.4)
    
    combined_results = combined_translation_rotation(trans_target, rot_target, maneuver_time=45.0)
    
    print(f"✅ Combined maneuver complete!")
    print(f"   Translation accuracy: {combined_results['translation_accuracy']*100:.1f}%")
    print(f"   Rotation accuracy: {combined_results['rotation_accuracy']*100:.1f}%")
    print(f"   Total energy: {combined_results['total_energy']/1e9:.3f} GJ")
    
    # Example 3: Visualization
    print(f"\n3. 📊 Generating Visualizations")
    visualize_rotation_results(rotation_results, save_plots=True)
    
    print(f"\n🎯 Rotation simulation complete!")
    print(f"💡 Features demonstrated:")
    print(f"   • Quaternion-based attitude control")
    print(f"   • Angular velocity profile optimization")
    print(f"   • Combined 6-DOF maneuvers")
    print(f"   • Rotation energy density modeling")
    print(f"   • Fine-pointing accuracy analysis")
