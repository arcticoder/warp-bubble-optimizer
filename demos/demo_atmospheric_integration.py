#!/usr/bin/env python3
"""
Atmospheric Constraints Integration Demo
=======================================

Demonstrates integration of atmospheric constraints with the warp bubble
simulation suite for safe sub-luminal operations.

This script shows:
1. Safe velocity profiling for atmospheric ascent/descent
2. Integration with impulse engine control systems
3. Automatic trajectory optimization with atmospheric limits
4. Real-time constraint monitoring during operations
"""

import numpy as np
import matplotlib.pyplot as plt
from atmospheric_constraints import AtmosphericConstraints, AtmosphericParameters
from simulate_impulse_engine import ImpulseProfile, WarpParameters, simulate_impulse_maneuver
from progress_tracker import ProgressTracker
import time

def demo_atmospheric_integration():
    """Demonstrate atmospheric constraints integration with impulse engine."""
    
    print("🌍 ATMOSPHERIC CONSTRAINTS INTEGRATION DEMO")
    print("=" * 50)
      # Initialize atmospheric constraints
    constraints = AtmosphericConstraints()
      # Initialize warp parameters for impulse simulation
    warp_params = WarpParameters(
        R_max=100.0,          # Maximum radius (m)
        n_r=1000,             # Radial grid points
        thickness=1.0         # Bubble wall thickness
    )    
    with ProgressTracker(100, "Atmospheric Integration Demo") as tracker:
          # Demo 1: Safe ascent trajectory planning
        print("\n1. 🚀 Safe Ascent Trajectory Planning")
        tracker.update("Planning safe ascent trajectory", 10)
        
        target_altitude = 80e3  # 80 km target
        ascent_time = 900       # 15 minutes
        
        # Generate safe ascent profile
        safe_profile = constraints.generate_safe_ascent_profile(
            target_altitude=target_altitude,
            ascent_time=ascent_time,
            safety_margin=0.8
        )
        
        print(f"   Target altitude: {target_altitude/1000:.0f} km")
        print(f"   Ascent time: {ascent_time/60:.1f} minutes")
        print(f"   Feasible: {'✅ YES' if safe_profile['feasible'] else '❌ NO'}")
        print(f"   Max velocity required: {np.max(safe_profile['velocity_required'])/1000:.2f} km/s")
        print(f"   Max safe velocity: {np.max(safe_profile['velocity_safe'])/1000:.2f} km/s")        # Demo 2: Impulse sequence with atmospheric constraints
        print("\n2. ⚡ Atmospheric-Constrained Impulse Sequence")
        tracker.update("Executing atmospheric-constrained impulses", 30)
        
        # Plan impulse sequence for safe ascent
        altitudes = np.linspace(0, 50e3, 10)  # Up to 50 km
        impulse_sequence = []
        
        for i, h in enumerate(altitudes[1:], 1):
            # Get safe velocity at this altitude
            v_safe = constraints.max_velocity_thermal(h)
            v_safe = min(v_safe, constraints.max_velocity_drag(h))
            v_safe *= 0.8  # Safety margin
            
            # Plan vertical impulse
            delta_h = h - altitudes[i-1]
            v_required = np.sqrt(2 * 9.81 * delta_h)  # Simplified
            v_actual = min(v_required, v_safe)
              # Create impulse profile for this step
            v_max_fraction = v_actual / 299792458  # Convert to fraction of c
            impulse_profile = ImpulseProfile(
                v_max=v_max_fraction,
                t_up=0.5,        # Short ramp up
                t_hold=0.5,      # Brief hold
                t_down=0.5,      # Short ramp down
                n_steps=100      # Time steps
            )
            
            impulse_sequence.append({
                'altitude': h,
                'velocity_safe': v_safe,
                'velocity_required': v_required, 
                'velocity_actual': v_actual,
                'impulse_profile': impulse_profile
            })
            
            if i <= 3:  # Show first few impulses
                print(f"   Impulse {i}: h={h/1000:.0f}km, Δv={v_actual/1000:.2f}km/s "
                      f"(safe: {v_safe/1000:.2f}km/s)")
                
                # Simulate this impulse
                if i == 1:  # Simulate first impulse as example
                    result = simulate_impulse_maneuver(impulse_profile, warp_params)
                    print(f"      → Simulation: Δv achieved = {result['delta_v_achieved']:.3f} m/s")
          # Demo 3: Real-time constraint monitoring
        print("\n3. 📊 Real-Time Constraint Monitoring")
        tracker.update("Monitoring atmospheric constraints", 50)
        
        # Simulate a trajectory with constraint checking
        time_steps = np.linspace(0, 300, 100)  # 5 minutes
        altitudes_sim = 20e3 * (1 - np.exp(-time_steps / 100))  # Exponential ascent
        velocities_sim = np.gradient(altitudes_sim, time_steps)
        
        # Analyze constraints
        analysis = constraints.analyze_trajectory_constraints(
            velocities_sim, altitudes_sim, time_steps
        )
        
        print(f"   Trajectory points analyzed: {len(time_steps)}")
        print(f"   Max altitude reached: {np.max(altitudes_sim)/1000:.1f} km")
        print(f"   Max velocity: {np.max(velocities_sim)/1000:.3f} km/s")
        print(f"   Max heat flux: {analysis['max_heat_flux']:.2e} W/m²")
        print(f"   Max drag force: {analysis['max_drag_force']:.2e} N")
        print(f"   Constraint violations: {analysis['violation_count']}")
        print(f"   Safe trajectory: {'✅ YES' if analysis['safe_trajectory'] else '❌ NO'}")
          # Demo 4: Adaptive velocity control
        print("\n4. 🎯 Adaptive Velocity Control")
        tracker.update("Implementing adaptive velocity control", 70)
        
        # Demonstrate velocity adaptation based on altitude
        current_altitude = 0
        current_velocity = 0
        control_steps = []
        
        for step in range(10):
            # Get current safe velocity
            v_thermal_limit = constraints.max_velocity_thermal(current_altitude)
            v_drag_limit = constraints.max_velocity_drag(current_altitude)
            v_safe = min(v_thermal_limit, v_drag_limit) * 0.8
            
            # Adaptive control: increase velocity if safe, decrease if not
            if current_velocity < v_safe * 0.9:
                delta_v = min(100, v_safe * 0.1)  # Gentle acceleration
                current_velocity += delta_v
            elif current_velocity > v_safe:
                delta_v = -(current_velocity - v_safe)  # Emergency deceleration
                current_velocity = v_safe
            else:
                delta_v = 0  # Maintain velocity
            
            # Update position
            current_altitude += current_velocity * 10  # 10 second steps
            
            control_steps.append({
                'step': step,
                'altitude': current_altitude,
                'velocity': current_velocity,
                'v_safe': v_safe,
                'delta_v': delta_v
            })
            
            if step < 5:  # Show first few steps
                print(f"   Step {step}: h={current_altitude/1000:.1f}km, "
                      f"v={current_velocity/1000:.3f}km/s, "
                      f"Δv={delta_v/1000:.3f}km/s")
          # Demo 5: Create visualization
        print("\n5. 📈 Atmospheric Constraints Visualization")
        tracker.update("Generating visualizations", 90)
        
        create_atmospheric_integration_plot(constraints, analysis, safe_profile)
        
        tracker.update("Demo complete", 100)
    
    print(f"\n💡 Integration Summary:")
    print(f"   • Atmospheric constraints successfully integrated with impulse engine")
    print(f"   • Safe velocity profiles generated for ascent operations") 
    print(f"   • Real-time constraint monitoring implemented")
    print(f"   • Adaptive velocity control demonstrated")
    print(f"   • Below 50 km: significant atmospheric effects")
    print(f"   • Above 80 km: minimal atmospheric constraints")
    print(f"   • Thermal limits typically more restrictive than drag limits")

def create_atmospheric_integration_plot(constraints, analysis, safe_profile):
    """Create comprehensive atmospheric integration visualization."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Safe velocity envelope
    altitudes = np.linspace(0, 100e3, 200)
    profile = constraints.safe_velocity_profile(altitudes)
    
    ax1.plot(profile['v_thermal_limit']/1000, altitudes/1000, 'r-', 
             label='Thermal limit', linewidth=2)
    ax1.plot(profile['v_drag_limit']/1000, altitudes/1000, 'b-', 
             label='Drag limit', linewidth=2)
    ax1.plot(profile['v_safe']/1000, altitudes/1000, 'k-', 
             label='Safe envelope', linewidth=3)
    ax1.set_xlabel('Velocity (km/s)')
    ax1.set_ylabel('Altitude (km)')
    ax1.set_title('Safe Velocity Envelope')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)
    
    # Plot 2: Trajectory analysis
    ax2.plot(analysis['time']/60, analysis['altitude']/1000, 'g-', 
             label='Altitude', linewidth=2)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(analysis['time']/60, analysis['velocity']/1000, 'b-', 
                  label='Velocity', linewidth=2)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Altitude (km)', color='g')
    ax2_twin.set_ylabel('Velocity (km/s)', color='b')
    ax2.set_title('Example Trajectory')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Heat flux analysis
    ax3.semilogy(analysis['time']/60, analysis['heat_flux'], 'r-', 
                 label='Heat flux', linewidth=2)
    ax3.axhline(y=constraints.thermal.max_heat_flux, color='r', 
                linestyle='--', label='Thermal limit')
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('Heat Flux (W/m²)')
    ax3.set_title('Thermal Constraints')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Safe ascent profile
    ax4.plot(safe_profile['time']/60, safe_profile['altitude']/1000, 'k-', 
             label='Altitude profile', linewidth=2)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(safe_profile['time']/60, safe_profile['velocity_safe']/1000, 'g-', 
                  label='Safe velocity', linewidth=2)
    ax4_twin.plot(safe_profile['time']/60, safe_profile['velocity_required']/1000, 'b--', 
                  label='Required velocity', linewidth=2)
    ax4.set_xlabel('Time (min)')
    ax4.set_ylabel('Altitude (km)', color='k')
    ax4_twin.set_ylabel('Velocity (km/s)', color='g')
    ax4.set_title('Safe Ascent Profile')
    ax4_twin.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('atmospheric_integration_demo.png', dpi=300, bbox_inches='tight')
    print(f"   📊 Integration plot saved: atmospheric_integration_demo.png")
    plt.show()

if __name__ == "__main__":
    demo_atmospheric_integration()
