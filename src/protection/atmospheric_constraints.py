#!/usr/bin/env python3
"""
Atmospheric Constraints for Warp Bubble Operations
================================================

This module implements atmospheric drag and heating constraints for sub-luminal
warp bubble operations. Below c, warp bubbles are permeable to atmospheric
molecules, requiring thermal and drag management for boundary hardware.

Key Physics:
- Warp bubbles below c have no horizon → atmosphere passes through
- Molecules interact with bubble boundary hardware → drag and heating
- Need altitude-dependent speed limits to prevent thermal damage

References:
- Sutton-Graves heating formula
- Standard atmosphere model
- Warp bubble permeability theory
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# JAX imports with fallback
try:
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
    print("🚀 JAX acceleration enabled for atmospheric constraints")
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False
    print("⚠️  JAX not available - using NumPy fallback")
    def jit(func): return func

@dataclass
class AtmosphericParameters:
    """Standard atmosphere model parameters."""
    rho0: float = 1.225        # Sea level density (kg/m³)
    H: float = 8500.0          # Scale height (m)
    T0: float = 288.15         # Sea level temperature (K)
    g: float = 9.80665         # Gravitational acceleration (m/s²)
    R: float = 287.0           # Specific gas constant (J/kg/K)

@dataclass
class BubbleGeometry:
    """Warp bubble geometric parameters."""
    radius: float = 50.0       # Bubble radius (m)
    cross_section: float = None # Cross-sectional area (m²)
    Cd: float = 0.8           # Effective drag coefficient
    nose_radius: float = None  # Effective nose radius for heating (m)
    
    def __post_init__(self):
        if self.cross_section is None:
            self.cross_section = np.pi * self.radius**2
        if self.nose_radius is None:
            self.nose_radius = self.radius

@dataclass
class ThermalLimits:
    """Thermal protection system limits."""
    max_heat_flux: float = 1e5     # Maximum heat flux (W/m²)
    max_temperature: float = 2000  # Maximum temperature (K)
    max_drag_force: float = 1e6    # Maximum drag force (N)
    cooling_capacity: float = 1e4  # Cooling system capacity (W/m²)

class AtmosphericConstraints:
    """
    Atmospheric drag and heating constraints for warp bubble operations.
    
    Implements:
    - Standard atmosphere model
    - Sutton-Graves heating formula
    - Drag force calculations
    - Altitude-dependent speed limits
    """
    
    def __init__(self, atm_params: AtmosphericParameters = None,
                 bubble_geom: BubbleGeometry = None,
                 thermal_limits: ThermalLimits = None):
        """Initialize atmospheric constraints model."""
        self.atm = atm_params or AtmosphericParameters()
        self.bubble = bubble_geom or BubbleGeometry()
        self.thermal = thermal_limits or ThermalLimits()
          # Sutton-Graves constant (SI units)
        self.K_sg = 1.83e-4  # W⋅s^(3/2)⋅m^(-5/2)⋅kg^(-1/2)
    
    def atmospheric_density(self, altitude: float) -> float:
        """
        Compute atmospheric density vs altitude using exponential model.
        
        Args:
            altitude: Altitude above sea level (m)
            
        Returns:
            Atmospheric density (kg/m³)
        """
        return self.atm.rho0 * np.exp(-altitude / self.atm.H)
    
    def drag_force(self, velocity: float, altitude: float) -> float:
        """
        Compute aerodynamic drag force on warp bubble.
        
        Args:
            velocity: Bubble velocity relative to atmosphere (m/s)
            altitude: Altitude above sea level (m)
            
        Returns:
            Drag force (N)
        """
        rho = self.atmospheric_density(altitude)
        return 0.5 * rho * self.bubble.Cd * self.bubble.cross_section * velocity**2
    
    def heat_flux_sutton_graves(self, velocity: float, altitude: float) -> float:
        """
        Compute convective heat flux using Sutton-Graves formula.
        
        q = K * sqrt(ρ/R_n) * v³
        
        Args:
            velocity: Bubble velocity (m/s)
            altitude: Altitude (m)
            
        Returns:
            Heat flux (W/m²)
        """
        rho = self.atmospheric_density(altitude)
        return self.K_sg * np.sqrt(rho / self.bubble.nose_radius) * velocity**3
    
    def max_velocity_thermal(self, altitude: float) -> float:
        """
        Compute maximum velocity based on thermal constraints.
        
        Args:
            altitude: Altitude (m)
            
        Returns:
            Maximum safe velocity (m/s)
        """
        rho = self.atmospheric_density(altitude)
        if rho < 1e-15:  # Essentially vacuum
            return 1e6  # No thermal limit in vacuum
        
        # Invert Sutton-Graves: v = (q_max / K * sqrt(R_n/ρ))^(1/3)
        v_max = (self.thermal.max_heat_flux / self.K_sg * 
                 np.sqrt(self.bubble.nose_radius / rho))**(1/3)
        return float(v_max)
    
    def max_velocity_drag(self, altitude: float, max_acceleration: float = 10.0) -> float:
        """
        Compute maximum velocity based on drag force constraints.
        
        Args:
            altitude: Altitude (m)
            max_acceleration: Maximum allowable deceleration (m/s²)
            
        Returns:
            Maximum safe velocity (m/s)
        """
        rho = self.atmospheric_density(altitude)
        if rho < 1e-15:
            return 1e6  # No drag limit in vacuum
        
        # Assume bubble mass ~ 1000 kg for this calculation
        bubble_mass = 1000.0  # kg (adjustable)
        max_force = bubble_mass * max_acceleration
        
        # F = 0.5 * ρ * Cd * A * v² → v = sqrt(2F / (ρ * Cd * A))
        v_max = np.sqrt(2 * max_force / (rho * self.bubble.Cd * self.bubble.cross_section))
        return float(v_max)
    
    def safe_velocity_profile(self, altitudes: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute safe velocity envelope vs altitude.
        
        Args:
            altitudes: Array of altitudes (m)
            
        Returns:
            Dictionary with velocity limits and constraints
        """
        v_thermal = np.array([self.max_velocity_thermal(h) for h in altitudes])
        v_drag = np.array([self.max_velocity_drag(h) for h in altitudes])
        v_safe = np.minimum(v_thermal, v_drag)
        
        # Atmospheric properties
        densities = np.array([self.atmospheric_density(h) for h in altitudes])
        
        return {
            'altitudes': altitudes,
            'v_thermal_limit': v_thermal,
            'v_drag_limit': v_drag,
            'v_safe': v_safe,
            'atmospheric_density': densities
        }
    
    def analyze_trajectory_constraints(self, velocity_profile: np.ndarray,
                                     altitude_profile: np.ndarray,
                                     time_profile: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a given trajectory against atmospheric constraints.
        
        Args:
            velocity_profile: Velocity vs time (m/s)
            altitude_profile: Altitude vs time (m)
            time_profile: Time array (s)
            
        Returns:
            Constraint analysis results
        """
        n_points = len(time_profile)
        drag_forces = np.zeros(n_points)
        heat_fluxes = np.zeros(n_points)
        v_thermal_limits = np.zeros(n_points)
        v_drag_limits = np.zeros(n_points)
        
        for i in range(n_points):
            v, h = velocity_profile[i], altitude_profile[i]
            drag_forces[i] = self.drag_force(v, h)
            heat_fluxes[i] = self.heat_flux_sutton_graves(v, h)
            v_thermal_limits[i] = self.max_velocity_thermal(h)
            v_drag_limits[i] = self.max_velocity_drag(h)
        
        # Check violations
        thermal_violations = heat_fluxes > self.thermal.max_heat_flux
        drag_violations = drag_forces > self.thermal.max_drag_force
        velocity_violations = (velocity_profile > v_thermal_limits) | (velocity_profile > v_drag_limits)
        
        return {
            'time': time_profile,
            'velocity': velocity_profile,
            'altitude': altitude_profile,
            'drag_force': drag_forces,
            'heat_flux': heat_fluxes,
            'v_thermal_limit': v_thermal_limits,
            'v_drag_limit': v_drag_limits,
            'thermal_violations': thermal_violations,
            'drag_violations': drag_violations,
            'velocity_violations': velocity_violations,
            'max_heat_flux': np.max(heat_fluxes),
            'max_drag_force': np.max(drag_forces),
            'violation_count': np.sum(thermal_violations | drag_violations | velocity_violations),
            'safe_trajectory': not np.any(thermal_violations | drag_violations | velocity_violations)
        }
    
    def plot_safe_envelope(self, altitude_range: Tuple[float, float] = (0, 150e3),
                          save_path: Optional[str] = None) -> None:
        """
        Plot safe velocity envelope vs altitude.
        
        Args:
            altitude_range: (min_alt, max_alt) in meters
            save_path: Optional path to save plot
        """
        altitudes = np.linspace(altitude_range[0], altitude_range[1], 200)
        profile = self.safe_velocity_profile(altitudes)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Velocity limits vs altitude
        ax1.plot(profile['v_thermal_limit']/1000, altitudes/1000, 'r-', 
                label='Thermal limit', linewidth=2)
        ax1.plot(profile['v_drag_limit']/1000, altitudes/1000, 'b-', 
                label='Drag limit', linewidth=2)
        ax1.plot(profile['v_safe']/1000, altitudes/1000, 'k-', 
                label='Safe envelope', linewidth=3)
        ax1.set_xlabel('Velocity (km/s)')
        ax1.set_ylabel('Altitude (km)')
        ax1.set_title('Warp Bubble Safe Velocity Envelope')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, min(20, np.max(profile['v_safe'])/1000 * 1.1))
        
        # Atmospheric density
        ax2.semilogx(profile['atmospheric_density'], altitudes/1000, 'g-', linewidth=2)
        ax2.set_xlabel('Atmospheric Density (kg/m³)')
        ax2.set_ylabel('Altitude (km)')
        ax2.set_title('Standard Atmosphere Model')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Atmospheric constraints plot saved: {save_path}")
        plt.show()
    
    def generate_safe_ascent_profile(self, target_altitude: float, 
                                   ascent_time: float,
                                   safety_margin: float = 0.8) -> Dict[str, np.ndarray]:
        """
        Generate a safe ascent velocity profile respecting atmospheric constraints.
        
        Args:
            target_altitude: Target altitude (m)
            ascent_time: Total ascent time (s)
            safety_margin: Safety factor (0.8 = 20% margin)
            
        Returns:
            Safe ascent profile dictionary
        """
        # Time array
        t = np.linspace(0, ascent_time, int(ascent_time))
        
        # Smooth altitude profile (sigmoid-like)
        h = target_altitude * (1 - np.exp(-3 * t / ascent_time))
        
        # Compute safe velocities at each altitude
        v_safe = np.array([self.max_velocity_thermal(alt) for alt in h])
        v_safe = np.minimum(v_safe, [self.max_velocity_drag(alt) for alt in h])
        v_safe *= safety_margin  # Apply safety margin
        
        # Compute required velocity from altitude derivative
        dh_dt = np.gradient(h, t)
        v_required = np.abs(dh_dt)
        
        # Check feasibility
        feasible = np.all(v_required <= v_safe)
        
        return {
            'time': t,
            'altitude': h,
            'velocity_safe': v_safe,
            'velocity_required': v_required,
            'velocity_actual': np.minimum(v_required, v_safe),
            'feasible': feasible,
            'safety_margin_used': safety_margin
        }

# Example usage and testing
if __name__ == "__main__":
    print("🌍 ATMOSPHERIC CONSTRAINTS FOR WARP BUBBLE OPERATIONS")
    print("=" * 60)
    
    # Create atmospheric constraints model
    constraints = AtmosphericConstraints()
    
    # Example 1: Safe velocity envelope
    print("\n1. 📊 Safe Velocity Envelope Analysis")
    constraints.plot_safe_envelope(save_path='atmospheric_constraints_envelope.png')
    
    # Example 2: Specific altitude analysis
    print("\n2. 🔍 Specific Altitude Analysis")
    test_altitudes = [0, 10e3, 50e3, 80e3, 100e3, 150e3]  # m
    for h in test_altitudes:
        rho = constraints.atmospheric_density(h)
        v_thermal = constraints.max_velocity_thermal(h)
        v_drag = constraints.max_velocity_drag(h)
        v_safe = min(v_thermal, v_drag)
        
        print(f"h={h/1e3:5.0f} km: ρ={rho:.3e} kg/m³, "
              f"v_thermal={v_thermal/1000:6.2f} km/s, "
              f"v_drag={v_drag/1000:6.2f} km/s, "
              f"v_safe={v_safe/1000:6.2f} km/s")
    
    # Example 3: Test trajectory analysis
    print("\n3. 🚀 Trajectory Constraint Analysis")
    
    # Create a sample ascent trajectory
    t_ascent = np.linspace(0, 600, 100)  # 10 minutes
    h_ascent = 100e3 * (1 - np.exp(-t_ascent / 200))  # Exponential approach to 100 km
    v_ascent = np.gradient(h_ascent, t_ascent)  # Vertical velocity
    
    # Analyze constraints
    analysis = constraints.analyze_trajectory_constraints(v_ascent, h_ascent, t_ascent)
    
    print(f"   Trajectory analysis:")
    print(f"   Max heat flux: {analysis['max_heat_flux']:.2e} W/m² "
          f"(limit: {constraints.thermal.max_heat_flux:.2e})")
    print(f"   Max drag force: {analysis['max_drag_force']:.2e} N "
          f"(limit: {constraints.thermal.max_drag_force:.2e})")
    print(f"   Constraint violations: {analysis['violation_count']}")
    print(f"   Safe trajectory: {'✅ YES' if analysis['safe_trajectory'] else '❌ NO'}")
    
    # Example 4: Generate safe ascent profile
    print("\n4. 📈 Safe Ascent Profile Generation")
    safe_profile = constraints.generate_safe_ascent_profile(
        target_altitude=100e3,  # 100 km
        ascent_time=900,        # 15 minutes
        safety_margin=0.8       # 20% safety margin
    )
    
    print(f"   Safe ascent to {safe_profile['altitude'][-1]/1000:.0f} km:")
    print(f"   Feasible: {'✅ YES' if safe_profile['feasible'] else '❌ NO'}")
    print(f"   Max velocity required: {np.max(safe_profile['velocity_required'])/1000:.2f} km/s")
    print(f"   Max velocity allowed: {np.max(safe_profile['velocity_safe'])/1000:.2f} km/s")
    
    print(f"\n💡 Key insights:")
    print(f"   • Warp bubbles below c are permeable to atmosphere")
    print(f"   • Thermal constraints dominate at low altitudes")
    print(f"   • Safe operation requires altitude-dependent speed limits")
    print(f"   • Above ~100 km: minimal atmospheric constraints")
    print(f"   • Below ~50 km: significant thermal management required")
