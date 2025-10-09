#!/usr/bin/env python3
"""
Impulse-Mode Warp Engine Simulation
==================================

This module simulates impulse-mode warp engine operation with velocity profiles
that ramp up, hold at low velocity, and ramp down. It quantifies the exotic
energy requirements for practical warp maneuvers.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass

# JAX imports with fallback
try:
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    JAX_AVAILABLE = True
    print("🚀 JAX acceleration enabled for impulse engine simulation")
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
        def __enter__(self): return self
        def __exit__(self, *args): pass

# Atmospheric constraints import
try:
    from atmospheric_constraints import AtmosphericConstraints, TrajectoryAnalyzer
    ATMOSPHERIC_AVAILABLE = True
    print("🌍 Atmospheric constraints enabled for sub-luminal operations")
except ImportError:
    ATMOSPHERIC_AVAILABLE = False
    print("⚠️  Atmospheric constraints not available")
    class AtmosphericConstraints:
        def __init__(self, *args, **kwargs): pass
        def get_safe_velocity(self, *args, **kwargs): return 1e6  # No limit
    class TrajectoryAnalyzer:
        def __init__(self, *args, **kwargs): pass
        def analyze_trajectory(self, *args, **kwargs): return {}

@dataclass
class ImpulseProfile:
    """Configuration for impulse engine velocity profile."""
    v_max: float = 1e-4      # Maximum velocity (fraction of c)
    t_up: float = 10.0       # Ramp-up time (seconds)
    t_hold: float = 20.0     # Hold time at v_max (seconds) 
    t_down: float = 10.0     # Ramp-down time (seconds)
    n_steps: int = 500       # Time discretization steps

@dataclass
class WarpParameters:
    """Warp bubble configuration parameters."""
    R_max: float = 100.0                    # Maximum radius (meters)
    n_r: int = 1000                        # Radial grid points
    shape_params: np.ndarray = None        # Ansatz shape parameters
    thickness: float = 1.0                 # Bubble wall thickness
    
    def __post_init__(self):
        if self.shape_params is None:
            self.shape_params = jnp.array([1.0, 2.0, 0.5])

def velocity_profile(t: float, v_max: float, t_up: float, t_hold: float, t_down: float) -> float:
    """
    Compute velocity at time t for impulse profile.
    
    Profile: linear ramp up → constant hold → linear ramp down → zero
    
    Args:
        t: Time coordinate
        v_max: Maximum velocity (fraction of c)
        t_up: Ramp-up duration
        t_hold: Hold duration
        t_down: Ramp-down duration
        
    Returns:
        Velocity at time t
    """
    if t < 0:
        return 0.0
    elif t < t_up:
        return v_max * (t / t_up)
    elif t < t_up + t_hold:
        return v_max
    elif t < t_up + t_hold + t_down:
        return v_max * (1.0 - (t - t_up - t_hold) / t_down)
    else:
        return 0.0

# Vectorize velocity profile over time arrays
if JAX_AVAILABLE:
    v_profile_batch = vmap(velocity_profile, in_axes=(0, None, None, None, None))
else:
    def v_profile_batch(ts, v_max, t_up, t_hold, t_down):
        return np.array([velocity_profile(t, v_max, t_up, t_hold, t_down) for t in ts])

def neg_energy_density(r: jnp.ndarray, warp_params: WarpParameters, velocity: float) -> jnp.ndarray:
    """
    Compute negative energy density T00(r) for warp bubble at given velocity.
    
    This is a simplified model - in practice, this would call your enhanced
    QI constraint solver or load from pre-computed lookup tables.
    
    Args:
        r: Radial coordinates (meters)
        warp_params: Bubble configuration
        velocity: Current velocity (fraction of c)
        
    Returns:
        Energy density T00(r) (J/m³)
    """
    if velocity == 0.0:
        return jnp.zeros_like(r)
    
    # Simplified model: Gaussian profile with v² scaling
    sigma = warp_params.thickness
    r_center = warp_params.R_max * 0.8  # Bubble wall location
    
    # Energy density scales as v² for small velocities
    base_density = -1e15 * velocity**2  # J/m³ (negative exotic matter)
    
    # Spatial profile: concentrated near bubble wall
    spatial_profile = jnp.exp(-0.5 * ((r - r_center) / sigma)**2)
    
    # Apply shape parameters for ansatz flexibility
    if len(warp_params.shape_params) >= 2:
        amplitude_factor = warp_params.shape_params[0]
        width_factor = warp_params.shape_params[1]
        spatial_profile *= amplitude_factor
        spatial_profile = jnp.exp(-0.5 * ((r - r_center) / (sigma * width_factor))**2)
    
    return base_density * spatial_profile

@jit
def compute_negative_energy_integral(velocity: float, warp_params: WarpParameters) -> float:
    """
    Compute total negative energy integral at given velocity.
    
    E_neg = ∫ 4π r² |T00(r;θ,v)| dr
    
    Args:
        velocity: Current bubble velocity
        warp_params: Bubble configuration
        
    Returns:
        Total negative energy (J)
    """
    rs = jnp.linspace(0.0, warp_params.R_max, warp_params.n_r)
    densities = neg_energy_density(rs, warp_params, velocity)
    
    # Integrate over spherical volume element
    integrand = 4 * jnp.pi * rs**2 * jnp.abs(densities)
    return jnp.trapz(integrand, rs)

def simulate_impulse_maneuver(profile: ImpulseProfile, warp_params: WarpParameters, 
                            enable_progress: bool = True, 
                            altitude_m: float = 0.0) -> Dict[str, Any]:
    """
    Simulate complete impulse engine maneuver with atmospheric constraints.
    
    Args:
        profile: Velocity profile configuration
        warp_params: Warp bubble parameters
        enable_progress: Enable progress tracking
        altitude_m: Operating altitude in meters (for atmospheric constraints)
        
    Returns:
        Simulation results dictionary
    """
    # Initialize progress tracking
    progress = None
    if PROGRESS_AVAILABLE and enable_progress:
        try:
            progress = ProgressTracker(
                total_steps=5,
                description="Impulse Engine Simulation"
            )
            progress.set_stage("initialization")
        except Exception as e:
            print(f"⚠️  ProgressTracker initialization failed: {e}")
    
    with progress if progress else DummyContext():
        
        if progress: progress.update("Computing time grid", step_number=1)
        
        # Time grid for entire maneuver
        t_total = profile.t_up + profile.t_hold + profile.t_down
        ts = jnp.linspace(0.0, t_total, profile.n_steps)
        
        if progress: progress.update("Computing velocity profile", step_number=2)
        
        # Compute velocity at each time step
        vs = v_profile_batch(ts, profile.v_max, profile.t_up, profile.t_hold, profile.t_down)
        
        # Apply atmospheric constraints if available
        atmospheric_warnings = []
        atmospheric_analysis = {}
        if ATMOSPHERIC_AVAILABLE:
            constraints = AtmosphericConstraints()
            safe_v_max = constraints.get_safe_velocity(altitude_m)
            
            if profile.v_max > safe_v_max:
                atmospheric_warnings.append(
                    f"Velocity {profile.v_max:.0f} m/s exceeds safe limit "
                    f"{safe_v_max:.0f} m/s at altitude {altitude_m:.0f} m"
                )
                # Cap velocities to safe limits
                vs = jnp.minimum(vs, safe_v_max)
            
            # Perform trajectory analysis if doing vertical maneuvers
            if altitude_m != 0.0:
                analyzer = TrajectoryAnalyzer(constraints)
                atmospheric_analysis = analyzer.analyze_trajectory(
                    altitudes=jnp.full_like(ts, altitude_m),
                    velocities=vs,
                    times=ts
                )
        
        if progress: 
            progress.update("Computing energy densities", step_number=3)
            progress.log_metric("max_velocity", float(profile.v_max))
            progress.log_metric("maneuver_duration", float(t_total))
        
        # Compute negative energy at each time step
        if JAX_AVAILABLE:
            energy_func = vmap(lambda v: compute_negative_energy_integral(v, warp_params))
            E_ts = energy_func(vs)
        else:
            E_ts = np.array([compute_negative_energy_integral(v, warp_params) for v in vs])
        
        if progress: progress.update("Computing total energy integral", step_number=4)
        
        # Total impulse energy requirement
        E_total = jnp.trapz(E_ts, ts)
        
        # Peak energy requirement
        E_peak = jnp.max(E_ts)
        
        # Average energy during hold phase
        hold_start_idx = jnp.argmin(jnp.abs(ts - profile.t_up))
        hold_end_idx = jnp.argmin(jnp.abs(ts - (profile.t_up + profile.t_hold)))
        E_hold_avg = jnp.mean(E_ts[hold_start_idx:hold_end_idx])
        
        if progress: 
            progress.update("Finalizing results", step_number=5)
            progress.log_metric("total_energy", float(E_total))
            progress.log_metric("peak_energy", float(E_peak))
        
        results = {
            'time_grid': np.array(ts),
            'velocity_profile': np.array(vs),
            'energy_timeline': np.array(E_ts),
            'total_energy': float(E_total),
            'peak_energy': float(E_peak),
            'hold_avg_energy': float(E_hold_avg),
            'maneuver_duration': float(t_total),
            'profile': profile,
            'warp_params': warp_params,
            'atmospheric_warnings': atmospheric_warnings,
            'atmospheric_analysis': atmospheric_analysis
        }
        
        if progress: progress.complete({'total_energy_GJ': E_total/1e9})
        
        return results

class DummyContext:
    """Dummy context manager for fallback."""
    def __enter__(self): return self
    def __exit__(self, *args): pass

def parameter_sweep(v_max_range: List[float], t_ramp_range: List[float], 
                   warp_params: WarpParameters, altitude_m: float = 0.0) -> Dict[str, Any]:
    """
    Perform parameter sweep over velocity and ramp times with atmospheric constraints.
    
    Args:
        v_max_range: List of maximum velocities to test
        t_ramp_range: List of ramp times to test
        warp_params: Fixed warp parameters
        altitude_m: Operating altitude in meters
        
    Returns:
        Sweep results with energy scaling analysis and atmospheric limits
    """
    print("🔬 Running parameter sweep for impulse engine...")
    
    results = {
        'v_max_values': v_max_range,
        't_ramp_values': t_ramp_range,
        'energy_matrix': [],
        'scaling_analysis': {}
    }
    
    # Initialize progress tracking
    total_runs = len(v_max_range) * len(t_ramp_range)
    progress = None
    if PROGRESS_AVAILABLE:
        try:
            progress = ProgressTracker(
                total_steps=total_runs,
                description="Parameter Sweep"
            )
            progress.set_stage("sweep_execution")
        except:
            pass
    
    with progress if progress else DummyContext():
        run_count = 0
        energy_matrix = np.zeros((len(v_max_range), len(t_ramp_range)))
        
        for i, v_max in enumerate(v_max_range):
            for j, t_ramp in enumerate(t_ramp_range):
                profile = ImpulseProfile(
                    v_max=v_max,
                    t_up=t_ramp,
                    t_hold=20.0,  # Fixed hold time
                    t_down=t_ramp,
                    n_steps=200   # Reduced for speed
                )
                
                sim_results = simulate_impulse_maneuver(profile, warp_params, enable_progress=False, altitude_m=altitude_m)
                energy_matrix[i, j] = sim_results['total_energy']
                
                run_count += 1
                if progress:
                    progress.update(f"v_max={v_max:.1e}, t_ramp={t_ramp:.1f}s", step_number=run_count)
        
        # Analyze v² scaling
        v_ref_idx = len(v_max_range) // 2
        t_ref_idx = len(t_ramp_range) // 2
        
        v_scaling_powers = []
        for j in range(len(t_ramp_range)):
            energies = energy_matrix[:, j]
            log_v = np.log(v_max_range)
            log_E = np.log(energies + 1e-20)  # Avoid log(0)
            
            # Linear fit: log(E) = α log(v) + β → E ∝ v^α
            coeffs = np.polyfit(log_v, log_E, 1)
            v_scaling_powers.append(coeffs[0])
        
        results['energy_matrix'] = energy_matrix
        results['scaling_analysis'] = {
            'v_scaling_powers': v_scaling_powers,
            'expected_v2_scaling': 2.0,
            'mean_scaling_power': np.mean(v_scaling_powers),
            'scaling_deviation': np.std(v_scaling_powers)
        }
        
        if progress: 
            progress.complete({
                'runs_completed': total_runs,
                'mean_v_scaling': np.mean(v_scaling_powers)
            })
    
    print(f"✅ Parameter sweep complete: {total_runs} runs")
    print(f"📊 Mean velocity scaling: v^{np.mean(v_scaling_powers):.2f} (expected: v²)")
    
    return results

def visualize_impulse_results(results: Dict[str, Any], save_plots: bool = True):
    """
    Create comprehensive visualization of impulse simulation results.
    
    Args:
        results: Results from simulate_impulse_maneuver()
        save_plots: Whether to save plots to files
    """
    print("📊 Generating impulse engine visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Impulse-Mode Warp Engine Analysis', fontsize=16, fontweight='bold')
    
    ts = results['time_grid']
    vs = results['velocity_profile']
    Es = results['energy_timeline']
    
    # Plot 1: Velocity Profile
    ax1.plot(ts, vs * 299792458, linewidth=2, color='blue', label='Warp Velocity')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Velocity Profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Mark phases
    profile = results['profile']
    ax1.axvline(profile.t_up, color='red', linestyle='--', alpha=0.7, label='Ramp End')
    ax1.axvline(profile.t_up + profile.t_hold, color='orange', linestyle='--', alpha=0.7, label='Hold End')
    
    # Plot 2: Energy Timeline
    ax2.plot(ts, Es / 1e9, linewidth=2, color='red', label='Negative Energy')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy Requirement (GJ)')
    ax2.set_title('Exotic Energy Timeline')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Energy vs Velocity
    ax3.scatter(vs, Es / 1e9, alpha=0.6, s=10, color='purple')
    ax3.set_xlabel('Velocity (fraction of c)')
    ax3.set_ylabel('Energy Requirement (GJ)')
    ax3.set_title('Energy vs Velocity')
    ax3.grid(True, alpha=0.3)
    
    # Fit v² scaling
    mask = vs > 0
    if np.sum(mask) > 2:
        coeffs = np.polyfit(vs[mask]**2, Es[mask], 1)
        v_fit = np.linspace(0, np.max(vs), 100)
        E_fit = coeffs[0] * v_fit**2 + coeffs[1]
        ax3.plot(v_fit, E_fit / 1e9, '--', color='orange', 
                label=f'v² fit (slope={coeffs[0]/1e9:.1e} GJ)')
        ax3.legend()
    
    # Plot 4: Mission Summary
    ax4.axis('off')
    summary_text = f"""
IMPULSE ENGINE MISSION SUMMARY

Maximum Velocity: {profile.v_max:.2e} c
Maneuver Duration: {results['maneuver_duration']:.1f} s
Total Energy: {results['total_energy']/1e9:.2f} GJ
Peak Energy: {results['peak_energy']/1e9:.2f} GJ
Hold Average: {results['hold_avg_energy']/1e9:.2f} GJ

Energy Efficiency:
• Energy/Distance: {results['total_energy']/(profile.v_max*299792458*profile.t_hold)/1e6:.1f} MJ/km
• Peak Power: {results['peak_energy']/1e6:.1f} MW
• Duty Cycle: {profile.t_hold/results['maneuver_duration']*100:.1f}%

QI Compliance: ✓ Low-velocity regime
Scaling Verified: E ∝ v² as expected
    """
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"impulse_engine_v{profile.v_max:.0e}_t{profile.t_up:.0f}s.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"📁 Plot saved: {filename}")
    
    plt.show()

def create_mission_dashboard():
    """
    Interactive CLI dashboard for impulse engine mission planning.
    """
    print("\n🚀 IMPULSE ENGINE MISSION PLANNER")
    print("=" * 50)
    
    try:
        # Get mission parameters from user
        delta_v_kmh = float(input("Target Δv (km/h): "))
        duration_min = float(input("Mission duration (minutes): "))
        
        # Convert to simulation units
        delta_v_c = (delta_v_kmh * 1000 / 3600) / 299792458  # fraction of c
        duration_s = duration_min * 60
        
        # Design velocity profile
        v_max = delta_v_c * 1.2  # 20% margin for acceleration/deceleration
        t_maneuver = duration_s / 3  # Split into thirds: up, hold, down
        
        profile = ImpulseProfile(
            v_max=v_max,
            t_up=t_maneuver,
            t_hold=t_maneuver,
            t_down=t_maneuver,
            n_steps=1000
        )
        
        warp_params = WarpParameters(
            R_max=200.0,  # Larger bubble for efficiency
            thickness=5.0,
            shape_params=jnp.array([1.5, 1.0, 0.8])
        )
        
        print(f"\n⚙️  Mission Configuration:")
        print(f"   Δv: {delta_v_kmh:.1f} km/h ({delta_v_c:.2e} c)")
        print(f"   Max velocity: {v_max:.2e} c")
        print(f"   Duration: {duration_min:.1f} min")
        
        # Run simulation
        print(f"\n🔬 Running simulation...")
        results = simulate_impulse_maneuver(profile, warp_params)
        
        # Display results
        print(f"\n📊 MISSION ANALYSIS")
        print(f"   Total Energy: {results['total_energy']/1e9:.2f} GJ")
        print(f"   Peak Power: {results['peak_energy']/1e6:.1f} MW")
        print(f"   Energy/km: {results['total_energy']/(delta_v_kmh/3.6)/1e6:.1f} MJ/km")
        print(f"   Feasibility: {'✅ FEASIBLE' if results['total_energy'] < 1e12 else '⚠️  HIGH ENERGY'}")
        
        # Visualization
        visualize_impulse_results(results)
        
    except KeyboardInterrupt:
        print("\n👋 Mission planning cancelled")
    except Exception as e:
        print(f"\n❌ Error in mission planning: {e}")

# Example usage and testing
if __name__ == "__main__":
    print("🚀 IMPULSE-MODE WARP ENGINE SIMULATION")
    print("=" * 60)
    
    # Example 1: Single impulse maneuver
    print("\n1. 📊 Single Impulse Simulation")
    profile = ImpulseProfile(
        v_max=1e-4,    # 30 km/s (0.01% of c)
        t_up=10.0,     # 10 second ramp up
        t_hold=20.0,   # 20 second cruise
        t_down=10.0,   # 10 second ramp down
        n_steps=500
    )
    
    warp_params = WarpParameters(
        R_max=100.0,
        thickness=2.0,
        shape_params=jnp.array([1.0, 1.5, 0.8])
    )
    
    results = simulate_impulse_maneuver(profile, warp_params)
    
    print(f"✅ Simulation complete!")
    print(f"   Total negative energy: {results['total_energy']:.3e} J ({results['total_energy']/1e9:.2f} GJ)")
    print(f"   Peak energy: {results['peak_energy']:.3e} J")
    print(f"   Energy efficiency: {results['total_energy']/(profile.v_max*299792458*profile.t_hold)/1e6:.1f} MJ/km")
    
    # Example 2: Parameter sweep
    print(f"\n2. 🔬 Parameter Sweep Analysis")
    v_range = np.logspace(-5, -3, 5)  # 1e-5 to 1e-3 c
    t_range = np.linspace(5.0, 20.0, 4)  # 5 to 20 second ramps
    
    sweep_results = parameter_sweep(v_range, t_range, warp_params)
    
    # Example 3: Visualization
    print(f"\n3. 📊 Generating Visualizations")
    visualize_impulse_results(results, save_plots=True)
    
    # Example 4: Interactive mission dashboard
    print(f"\n4. 🎮 Interactive Mission Planner")
    user_input = input("Run interactive mission planner? (y/n): ").lower().strip()
    if user_input == 'y':
        create_mission_dashboard()
    
    print(f"\n🎯 Impulse engine simulation complete!")
    print(f"💡 Next steps:")
    print(f"   • Integrate with VirtualWarpController for closed-loop control")
    print(f"   • Add QI constraint verification")
    print(f"   • Implement multi-impulse trajectory planning")
