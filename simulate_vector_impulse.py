#!/usr/bin/env python3
"""
Vectorized Translation Impulse Engine Simulation
===============================================

This module extends the impulse engine simulation to handle 3D vectorized
translation maneuvers with direction control, trajectory planning, and
multi-axis coordinate transformations.
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
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    JAX_AVAILABLE = True
    print("🚀 JAX acceleration enabled for vectorized impulse simulation")
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
    from progress_tracker import ProgressTracker
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False
    class ProgressTracker:
        def __init__(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def set_stage(self, *args, **kwargs): pass
        def complete(self, *args, **kwargs): pass
        def log_metric(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

@dataclass
class Vector3D:
    """3D vector with coordinate operations."""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Initialize 3D vector with components."""
        self.x = x
        self.y = y
        self.z = z
        self.vec = jnp.array([self.x, self.y, self.z])
    
    @property
    def magnitude(self) -> float:
        """Vector magnitude."""
        return float(jnp.linalg.norm(self.vec))
    
    @property
    def unit(self) -> 'Vector3D':
        """Unit vector in same direction."""
        mag = self.magnitude
        if mag < 1e-12:
            return Vector3D(1.0, 0.0, 0.0)  # Default direction
        return Vector3D(*(self.vec / mag))
    
    def dot(self, other: 'Vector3D') -> float:
        """Dot product with another vector."""
        return float(jnp.dot(self.vec, other.vec))
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """Cross product with another vector."""
        result = jnp.cross(self.vec, other.vec)
        return Vector3D(*result)
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(*(self.vec + other.vec))
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(*(self.vec - other.vec))
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(*(self.vec * scalar))
    
    def __str__(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

@dataclass
class VectorImpulseProfile:
    """Configuration for vectorized impulse engine maneuver."""
    target_displacement: Vector3D = None  # Total displacement vector (meters)
    v_max: float = 1e-4                   # Maximum velocity magnitude (fraction of c)
    t_up: float = 10.0                    # Ramp-up time (seconds)
    t_hold: float = 20.0                  # Hold time at v_max (seconds)
    t_down: float = 10.0                  # Ramp-down time (seconds)
    n_steps: int = 500                    # Time discretization steps
    coordinate_frame: str = "cartesian"   # "cartesian", "cylindrical", "spherical"
    
    def __post_init__(self):
        if self.target_displacement is None:
            self.target_displacement = Vector3D(1000.0, 0.0, 0.0)  # 1 km in x-direction

@dataclass
class WarpBubbleVector:
    """Extended warp bubble configuration for vectorized operation."""
    R_max: float = 100.0                    # Maximum radius (meters)
    n_r: int = 1000                         # Radial grid points
    shape_params: jnp.ndarray = None        # Ansatz shape parameters
    thickness: float = 1.0                  # Bubble wall thickness
    orientation: Vector3D = None            # Bubble orientation vector
    asymmetry_factor: float = 1.0           # Directional asymmetry (1.0 = symmetric)
    
    def __post_init__(self):
        if self.shape_params is None:
            self.shape_params = jnp.array([1.0, 2.0, 0.5])
        if self.orientation is None:
            self.orientation = Vector3D(1.0, 0.0, 0.0)

def vector_velocity_profile(t: float, direction: Vector3D, v_max: float, 
                           t_up: float, t_hold: float, t_down: float) -> Vector3D:
    """
    Compute 3D velocity vector at time t for vectorized impulse profile.
    
    Args:
        t: Time coordinate
        direction: Unit direction vector
        v_max: Maximum velocity magnitude
        t_up: Ramp-up duration
        t_hold: Hold duration  
        t_down: Ramp-down duration
        
    Returns:
        3D velocity vector at time t
    """
    # Scalar velocity profile (same as before)
    if t < 0:
        v_scalar = 0.0
    elif t < t_up:
        v_scalar = v_max * (t / t_up)
    elif t < t_up + t_hold:
        v_scalar = v_max
    elif t < t_up + t_hold + t_down:
        v_scalar = v_max * (1.0 - (t - t_up - t_hold) / t_down)
    else:
        v_scalar = 0.0
    
    # Apply direction
    return direction * v_scalar

# Vectorize over time arrays
if JAX_AVAILABLE:
    def v_vector_batch(ts, direction, v_max, t_up, t_hold, t_down):
        def single_time(t):
            return vector_velocity_profile(t, direction, v_max, t_up, t_hold, t_down).vec
        return vmap(single_time)(ts)
else:
    def v_vector_batch(ts, direction, v_max, t_up, t_hold, t_down):
        return np.array([vector_velocity_profile(t, direction, v_max, t_up, t_hold, t_down).vec for t in ts])

@jit
def neg_energy_density_vector(r: jnp.ndarray, direction: jnp.ndarray, 
                             warp_params: WarpBubbleVector, velocity_mag: float) -> jnp.ndarray:
    """
    Compute negative energy density for vectorized warp bubble.
    
    Accounts for directional asymmetry and orientation effects.
    
    Args:
        r: Radial coordinates
        direction: Velocity direction vector (unit)
        warp_params: Bubble configuration
        velocity_mag: Velocity magnitude
        
    Returns:
        Energy density T00(r,θ,φ)
    """
    if velocity_mag == 0.0:
        return jnp.zeros_like(r)
    
    # Base energy density (v² scaling)
    base_density = -1e15 * velocity_mag**2
    
    # Spatial profile centered at bubble wall
    sigma = warp_params.thickness
    r_center = warp_params.R_max * 0.8
    spatial_profile = jnp.exp(-0.5 * ((r - r_center) / sigma)**2)
    
    # Directional asymmetry effect
    # Energy concentration depends on alignment with bubble orientation
    direction_dot = jnp.dot(direction, warp_params.orientation.vec)
    asymmetry_factor = 1.0 + warp_params.asymmetry_factor * direction_dot**2
    
    # Apply shape parameters
    if len(warp_params.shape_params) >= 2:
        amplitude_factor = warp_params.shape_params[0]
        width_factor = warp_params.shape_params[1]
        spatial_profile *= amplitude_factor * asymmetry_factor
        spatial_profile = jnp.exp(-0.5 * ((r - r_center) / (sigma * width_factor))**2)
    
    return base_density * spatial_profile

@jit
def compute_vector_energy_integral(velocity_vec: jnp.ndarray, warp_params: WarpBubbleVector) -> float:
    """
    Compute total negative energy for vectorized warp bubble.
    
    Args:
        velocity_vec: 3D velocity vector
        warp_params: Bubble configuration
        
    Returns:
        Total negative energy (J)
    """
    velocity_mag = jnp.linalg.norm(velocity_vec)
    if velocity_mag < 1e-12:
        return 0.0
    
    direction = velocity_vec / velocity_mag
    
    rs = jnp.linspace(0.0, warp_params.R_max, warp_params.n_r)
    densities = neg_energy_density_vector(rs, direction, warp_params, velocity_mag)
    
    # Integrate over spherical volume
    integrand = 4 * jnp.pi * rs**2 * jnp.abs(densities)
    return jnp.trapz(integrand, rs)

def simulate_vector_impulse_maneuver(profile: VectorImpulseProfile, 
                                   warp_params: WarpBubbleVector,
                                   enable_progress: bool = True) -> Dict[str, Any]:
    """
    Simulate complete vectorized impulse maneuver.
    
    Args:
        profile: Vector impulse profile configuration
        warp_params: Warp bubble parameters
        enable_progress: Enable progress tracking
        
    Returns:
        Simulation results with 3D trajectory data
    """
    # Initialize progress tracking
    progress = None
    if PROGRESS_AVAILABLE and enable_progress:
        try:
            progress = ProgressTracker(
                total_steps=6,
                description="Vector Impulse Simulation"
            )
            progress.set_stage("vector_trajectory_planning")
        except Exception as e:
            print(f"⚠️  ProgressTracker initialization failed: {e}")
    
    class DummyContext:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    with progress if progress else DummyContext():
        
        if progress: progress.update("Computing trajectory parameters", step_number=1)
        
        # Calculate trajectory parameters
        t_total = profile.t_up + profile.t_hold + profile.t_down
        displacement_mag = profile.target_displacement.magnitude
        direction = profile.target_displacement.unit
        
        # Time grid
        ts = jnp.linspace(0.0, t_total, profile.n_steps)
        
        if progress: 
            progress.update("Computing 3D velocity profile", step_number=2)
            progress.log_metric("displacement_magnitude", displacement_mag)
            progress.log_metric("direction_vector", str(direction))
        
        # Compute 3D velocity profile
        vs_3d = v_vector_batch(ts, direction, profile.v_max, profile.t_up, profile.t_hold, profile.t_down)
        
        if progress: progress.update("Integrating trajectory", step_number=3)
        
        # Integrate to get position trajectory
        dt = ts[1] - ts[0] if len(ts) > 1 else 1.0
        positions = jnp.cumsum(vs_3d * dt, axis=0)
        
        # Add starting position (origin)
        positions = jnp.concatenate([jnp.zeros((1, 3)), positions], axis=0)[:-1]
        
        if progress: progress.update("Computing vectorized energy densities", step_number=4)
        
        # Compute energy at each time step
        if JAX_AVAILABLE:
            energy_func = vmap(lambda v: compute_vector_energy_integral(v, warp_params))
            E_ts = energy_func(vs_3d)
        else:
            E_ts = np.array([compute_vector_energy_integral(v, warp_params) for v in vs_3d])
        
        if progress: progress.update("Computing trajectory metrics", step_number=5)
        
        # Trajectory analysis
        velocity_magnitudes = jnp.linalg.norm(vs_3d, axis=1)
        total_distance = jnp.sum(jnp.linalg.norm(jnp.diff(positions, axis=0), axis=1))
        final_position = positions[-1]
        trajectory_error = jnp.linalg.norm(final_position - profile.target_displacement.vec)
        
        # Energy analysis
        E_total = jnp.trapz(E_ts, ts)
        E_peak = jnp.max(E_ts)
        
        # Hold phase analysis
        hold_start_idx = jnp.argmin(jnp.abs(ts - profile.t_up))
        hold_end_idx = jnp.argmin(jnp.abs(ts - (profile.t_up + profile.t_hold)))
        E_hold_avg = jnp.mean(E_ts[hold_start_idx:hold_end_idx])
        
        if progress: 
            progress.update("Finalizing vector simulation", step_number=6)
            progress.log_metric("total_energy", float(E_total))
            progress.log_metric("trajectory_error", float(trajectory_error))
            progress.log_metric("total_distance", float(total_distance))
        
        results = {
            'time_grid': np.array(ts),
            'velocity_profile_3d': np.array(vs_3d),
            'velocity_magnitudes': np.array(velocity_magnitudes),
            'position_trajectory': np.array(positions),
            'energy_timeline': np.array(E_ts),
            'total_energy': float(E_total),
            'peak_energy': float(E_peak),
            'hold_avg_energy': float(E_hold_avg),
            'maneuver_duration': float(t_total),
            'total_distance': float(total_distance),
            'final_position': np.array(final_position),
            'target_position': profile.target_displacement.vec,
            'trajectory_error': float(trajectory_error),
            'trajectory_accuracy': float(1.0 - trajectory_error / displacement_mag),
            'direction_vector': direction.vec,
            'profile': profile,
            'warp_params': warp_params
        }
        
        if progress: 
            progress.complete({
                'total_energy_GJ': E_total/1e9,
                'trajectory_accuracy_percent': results['trajectory_accuracy']*100
            })
        
        return results

def coordinate_transform(position: jnp.ndarray, from_frame: str, to_frame: str) -> jnp.ndarray:
    """
    Transform coordinates between different frames.
    
    Args:
        position: Position vector [x, y, z]
        from_frame: Source coordinate system
        to_frame: Target coordinate system
        
    Returns:
        Transformed position vector
    """
    if from_frame == to_frame:
        return position
    
    x, y, z = position[0], position[1], position[2]
    
    if from_frame == "cartesian" and to_frame == "cylindrical":
        r = jnp.sqrt(x**2 + y**2)
        phi = jnp.arctan2(y, x)
        return jnp.array([r, phi, z])
    
    elif from_frame == "cartesian" and to_frame == "spherical":
        r = jnp.sqrt(x**2 + y**2 + z**2)
        theta = jnp.arccos(z / (r + 1e-12))
        phi = jnp.arctan2(y, x)
        return jnp.array([r, theta, phi])
    
    elif from_frame == "cylindrical" and to_frame == "cartesian":
        r, phi, z = position[0], position[1], position[2]
        x = r * jnp.cos(phi)
        y = r * jnp.sin(phi)
        return jnp.array([x, y, z])
    
    elif from_frame == "spherical" and to_frame == "cartesian":
        r, theta, phi = position[0], position[1], position[2]
        x = r * jnp.sin(theta) * jnp.cos(phi)
        y = r * jnp.sin(theta) * jnp.sin(phi)
        z = r * jnp.cos(theta)
        return jnp.array([x, y, z])
    
    else:
        raise ValueError(f"Coordinate transform {from_frame} → {to_frame} not implemented")

def plan_multi_segment_trajectory(waypoints: List[Vector3D], 
                                 segment_profiles: List[VectorImpulseProfile],
                                 warp_params: WarpBubbleVector) -> Dict[str, Any]:
    """
    Plan and simulate multi-segment vectorized trajectory.
    
    Args:
        waypoints: List of 3D waypoints to visit
        segment_profiles: Impulse profiles for each segment
        warp_params: Warp bubble configuration
        
    Returns:
        Combined trajectory simulation results
    """
    print(f"🛰️  Planning multi-segment trajectory: {len(waypoints)} waypoints")
    
    if len(waypoints) != len(segment_profiles) + 1:
        raise ValueError("Need N-1 profiles for N waypoints")
    
    # Initialize progress tracking
    progress = None
    if PROGRESS_AVAILABLE:
        try:
            progress = ProgressTracker(
                total_steps=len(segment_profiles),
                description="Multi-Segment Trajectory"
            )
            progress.set_stage("trajectory_planning")
        except:
            pass
    
    class DummyContext:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    with progress if progress else DummyContext():
        
        # Simulate each segment
        all_results = []
        cumulative_time = 0.0
        cumulative_energy = 0.0
        full_trajectory = []
        full_timeline = []
        
        for i, profile in enumerate(segment_profiles):
            if progress:
                progress.update(f"Simulating segment {i+1}/{len(segment_profiles)}", step_number=i+1)
            
            # Set displacement for this segment
            displacement = waypoints[i+1] - waypoints[i]
            profile.target_displacement = displacement
            
            # Simulate segment
            segment_results = simulate_vector_impulse_maneuver(profile, warp_params, enable_progress=False)
            
            # Adjust time coordinates
            adjusted_times = segment_results['time_grid'] + cumulative_time
            adjusted_positions = segment_results['position_trajectory'] + waypoints[i].vec
            
            # Store results
            segment_results['time_grid_adjusted'] = adjusted_times
            segment_results['position_trajectory_adjusted'] = adjusted_positions
            all_results.append(segment_results)
            
            # Accumulate for combined trajectory
            if i == 0:
                full_trajectory = adjusted_positions
                full_timeline = adjusted_times
            else:
                full_trajectory = np.concatenate([full_trajectory, adjusted_positions[1:]], axis=0)
                full_timeline = np.concatenate([full_timeline, adjusted_times[1:]], axis=0)
            
            cumulative_time += segment_results['maneuver_duration']
            cumulative_energy += segment_results['total_energy']
        
        # Combined results
        combined_results = {
            'segment_results': all_results,
            'waypoints': [wp.vec for wp in waypoints],
            'full_trajectory': full_trajectory,
            'full_timeline': full_timeline,
            'total_mission_time': cumulative_time,
            'total_mission_energy': cumulative_energy,
            'num_segments': len(segment_profiles),
            'trajectory_efficiency': sum(r['trajectory_accuracy'] for r in all_results) / len(all_results)
        }
        
        if progress:
            progress.complete({
                'mission_time_minutes': cumulative_time/60,
                'total_energy_GJ': cumulative_energy/1e9,
                'avg_accuracy_percent': combined_results['trajectory_efficiency']*100
            })
    
    print(f"✅ Multi-segment trajectory complete:")
    print(f"   Total time: {cumulative_time/60:.1f} minutes")
    print(f"   Total energy: {cumulative_energy/1e9:.2f} GJ") 
    print(f"   Average accuracy: {combined_results['trajectory_efficiency']*100:.1f}%")
    
    return combined_results

def visualize_vector_trajectory(results: Dict[str, Any], save_plots: bool = True):
    """
    Create comprehensive 3D visualization of vectorized trajectory.
    
    Args:
        results: Results from simulate_vector_impulse_maneuver()
        save_plots: Whether to save plots to files
    """
    print("📊 Generating 3D trajectory visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    positions = results['position_trajectory']
    
    # Color trajectory by velocity magnitude
    velocities = results['velocity_magnitudes']
    colors = plt.cm.viridis(velocities / np.max(velocities))
    
    for i in range(len(positions)-1):
        ax1.plot([positions[i,0], positions[i+1,0]], 
                [positions[i,1], positions[i+1,1]], 
                [positions[i,2], positions[i+1,2]], 
                color=colors[i], linewidth=2)
    
    # Mark start and end points
    ax1.scatter(*positions[0], color='green', s=100, label='Start')
    ax1.scatter(*positions[-1], color='red', s=100, label='End')
    ax1.scatter(*results['target_position'], color='orange', s=100, 
               marker='x', label='Target')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # Velocity components over time
    ax2 = fig.add_subplot(2, 3, 2)
    ts = results['time_grid']
    vs_3d = results['velocity_profile_3d']
    
    ax2.plot(ts, vs_3d[:, 0] * 299792458, label='Vx (m/s)', color='red')
    ax2.plot(ts, vs_3d[:, 1] * 299792458, label='Vy (m/s)', color='green') 
    ax2.plot(ts, vs_3d[:, 2] * 299792458, label='Vz (m/s)', color='blue')
    ax2.plot(ts, results['velocity_magnitudes'] * 299792458, 
             label='|V| (m/s)', color='black', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Energy timeline
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(ts, results['energy_timeline'] / 1e9, linewidth=2, color='red')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Energy Requirement (GJ)')
    ax3.set_title('Exotic Energy Timeline')
    ax3.grid(True, alpha=0.3)
    
    # Position components over time
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(ts, positions[:, 0], label='X (m)', color='red')
    ax4.plot(ts, positions[:, 1], label='Y (m)', color='green')
    ax4.plot(ts, positions[:, 2], label='Z (m)', color='blue')
    
    # Mark target position
    ax4.axhline(results['target_position'][0], color='red', linestyle=':', alpha=0.7)
    ax4.axhline(results['target_position'][1], color='green', linestyle=':', alpha=0.7)
    ax4.axhline(results['target_position'][2], color='blue', linestyle=':', alpha=0.7)
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('Position Components')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Trajectory error analysis
    ax5 = fig.add_subplot(2, 3, 5)
    target_positions = np.linspace(np.zeros(3), results['target_position'], len(positions))
    errors = np.linalg.norm(positions - target_positions, axis=1)
    
    ax5.plot(ts, errors, linewidth=2, color='purple')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Trajectory Error (m)')
    ax5.set_title('Trajectory Accuracy')
    ax5.grid(True, alpha=0.3)
    
    # Mission summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    profile = results['profile']
    summary_text = f"""
VECTORIZED IMPULSE MISSION SUMMARY

Target Displacement: {results['target_position']}
Final Position: {results['final_position']}
Trajectory Error: {results['trajectory_error']:.2f} m
Accuracy: {results['trajectory_accuracy']*100:.1f}%

Maximum Velocity: {profile.v_max:.2e} c
Direction: {results['direction_vector']}
Total Distance: {results['total_distance']:.1f} m
Mission Duration: {results['maneuver_duration']:.1f} s

Energy Requirements:
• Total Energy: {results['total_energy']/1e9:.2f} GJ
• Peak Energy: {results['peak_energy']/1e9:.2f} GJ
• Energy/km: {results['total_energy']/(results['total_distance']/1000)/1e6:.1f} MJ/km

Performance:
• Efficiency: {results['trajectory_accuracy']*100:.1f}%
• Avg Power: {results['total_energy']/(results['maneuver_duration']*1e6):.1f} MW
• Energy/Distance: {results['total_energy']/results['total_distance']/1e6:.1f} MJ/m
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plots:
        displacement_str = f"{np.linalg.norm(results['target_position']):.0f}m"
        filename = f"vector_impulse_trajectory_{displacement_str}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"📁 Plot saved: {filename}")
    
    plt.show()

# Example usage and testing
if __name__ == "__main__":
    print("🚀 VECTORIZED IMPULSE ENGINE SIMULATION")
    print("=" * 60)
    
    # Example 1: Simple vectorized maneuver
    print("\n1. 📊 Single Vector Impulse")
    target = Vector3D(1000.0, 500.0, -200.0)  # 1.1 km displacement
    
    profile = VectorImpulseProfile(
        target_displacement=target,
        v_max=5e-5,      # 15 km/s
        t_up=15.0,
        t_hold=30.0,
        t_down=15.0,
        n_steps=1000
    )
    
    warp_params = WarpBubbleVector(
        R_max=150.0,
        thickness=3.0,
        shape_params=jnp.array([1.2, 1.5, 0.9]),
        orientation=target.unit,
        asymmetry_factor=0.2
    )
    
    results = simulate_vector_impulse_maneuver(profile, warp_params)
    
    print(f"✅ Vector simulation complete!")
    print(f"   Target: {target}")
    print(f"   Final: {Vector3D(*results['final_position'])}")
    print(f"   Error: {results['trajectory_error']:.2f} m")
    print(f"   Accuracy: {results['trajectory_accuracy']*100:.1f}%")
    print(f"   Total energy: {results['total_energy']/1e9:.2f} GJ")
    
    # Example 2: Multi-segment trajectory
    print(f"\n2. 🛰️  Multi-Segment Mission")
    waypoints = [
        Vector3D(0.0, 0.0, 0.0),       # Start
        Vector3D(1000.0, 0.0, 0.0),    # Move east
        Vector3D(1000.0, 1000.0, 0.0), # Move north
        Vector3D(0.0, 1000.0, 500.0)   # Return west and up
    ]
    
    # Create profiles for each segment
    segment_profiles = []
    for i in range(len(waypoints)-1):
        segment_profiles.append(VectorImpulseProfile(
            v_max=3e-5,
            t_up=10.0,
            t_hold=15.0,
            t_down=10.0,
            n_steps=300
        ))
    
    multi_results = plan_multi_segment_trajectory(waypoints, segment_profiles, warp_params)
    
    # Example 3: Visualization
    print(f"\n3. 📊 Generating 3D Visualizations")
    visualize_vector_trajectory(results, save_plots=True)
    
    print(f"\n🎯 Vectorized impulse simulation complete!")
    print(f"💡 Features demonstrated:")
    print(f"   • 3D trajectory planning and control")
    print(f"   • Directional energy density calculation") 
    print(f"   • Multi-segment mission planning")
    print(f"   • Comprehensive 3D visualization")
    print(f"   • Trajectory accuracy analysis")
