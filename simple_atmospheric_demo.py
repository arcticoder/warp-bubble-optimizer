#!/usr/bin/env python3
"""
Simple Atmospheric Constraints Demo
==================================

Demonstrates atmospheric constraints for warp bubble operations
without complex simulation dependencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from atmospheric_constraints import AtmosphericConstraints

def simple_atmospheric_demo():
    """Simple demonstration of atmospheric constraints."""
    
    print("🌍 SIMPLE ATMOSPHERIC CONSTRAINTS DEMO")
    print("=" * 45)
    
    # Initialize atmospheric constraints
    constraints = AtmosphericConstraints()
    
    print("\n1. 📊 Atmospheric Properties vs Altitude")
    print("-" * 40)
    
    # Test altitudes
    test_altitudes = [0, 5e3, 10e3, 20e3, 30e3, 50e3, 80e3, 100e3, 150e3]
    
    print(f"{'Altitude (km)':<12} {'Density (kg/m³)':<15} {'V_thermal (km/s)':<18} {'V_drag (km/s)':<15} {'V_safe (km/s)':<12}")
    print("-" * 85)
    
    for h in test_altitudes:
        rho = constraints.atmospheric_density(h)
        v_thermal = constraints.max_velocity_thermal(h)
        v_drag = constraints.max_velocity_drag(h)
        v_safe = min(v_thermal, v_drag)
        
        print(f"{h/1e3:<12.0f} {rho:<15.3e} {v_thermal/1000:<18.3f} {v_drag/1000:<15.3f} {v_safe/1000:<12.3f}")
    
    print("\n2. 🚀 Safe Ascent Analysis")
    print("-" * 30)
    
    # Test ascent scenarios
    scenarios = [
        {"target_alt": 50e3, "time": 300, "name": "Low Earth Orbit prep"},
        {"target_alt": 100e3, "time": 600, "name": "Karman line crossing"},
        {"target_alt": 150e3, "time": 900, "name": "Exosphere entry"}
    ]
    
    for scenario in scenarios:
        profile = constraints.generate_safe_ascent_profile(
            target_altitude=scenario["target_alt"],
            ascent_time=scenario["time"],
            safety_margin=0.8
        )
        
        print(f"\n   {scenario['name']}:")
        print(f"     Target: {scenario['target_alt']/1000:.0f} km in {scenario['time']/60:.1f} min")
        print(f"     Feasible: {'✅ YES' if profile['feasible'] else '❌ NO'}")
        print(f"     Max velocity required: {np.max(profile['velocity_required'])/1000:.3f} km/s")
        print(f"     Max safe velocity: {np.max(profile['velocity_safe'])/1000:.3f} km/s")
    
    print("\n3. 🔥 Thermal Analysis Example")
    print("-" * 32)
    
    # Example: High-speed atmospheric entry
    altitude = 80e3  # 80 km
    velocities = np.array([1000, 2000, 3000, 4000, 5000])  # m/s
    
    print(f"   Analysis at {altitude/1000:.0f} km altitude:")
    print(f"   {'Velocity (km/s)':<15} {'Heat Flux (W/m²)':<18} {'Drag Force (N)':<15} {'Safe?':<6}")
    print("   " + "-" * 60)
    
    for v in velocities:
        heat_flux = constraints.heat_flux_sutton_graves(v, altitude)
        drag_force = constraints.drag_force(v, altitude)
        v_thermal_limit = constraints.max_velocity_thermal(altitude)
        safe = v <= v_thermal_limit
        
        print(f"   {v/1000:<15.1f} {heat_flux:<18.2e} {drag_force:<15.2e} {'✅' if safe else '❌':<6}")
    
    print("\n4. 🎯 Mission Planning Insights")
    print("-" * 32)
    
    print("   💡 Key Findings:")
    print("     • Below 10 km: Very restrictive thermal limits (<2 km/s)")
    print("     • 10-50 km: Moderate constraints (2-4 km/s)")
    print("     • 50-100 km: Relaxed limits (4-10 km/s)")  
    print("     • Above 100 km: Minimal atmospheric effects (>10 km/s)")
    print()
    print("   📋 Operational Recommendations:")
    print("     • Use vertical ascent profiles to minimize atmospheric transit time")
    print("     • Implement thermal management below 50 km altitude")
    print("     • Plan longer ascent times for thermal safety")
    print("     • Consider atmospheric density variations with weather/season")
    print("     • Above 100 km: Standard warp operations feasible")
    
    # Create visualization
    print("\n5. 📈 Generating Visualization...")
    create_simple_plot(constraints)
    
    print("\n✅ Demo Complete!")
    print("   See atmospheric_constraints_simple.png for velocity envelope plot")

def create_simple_plot(constraints):
    """Create a simple atmospheric constraints plot."""
    
    # Altitude range
    altitudes = np.linspace(0, 150e3, 300)
    profile = constraints.safe_velocity_profile(altitudes)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Safe velocity envelope
    ax1.plot(profile['v_thermal_limit']/1000, altitudes/1000, 'r-', 
             label='Thermal limit', linewidth=2)
    ax1.plot(profile['v_drag_limit']/1000, altitudes/1000, 'b-', 
             label='Drag limit', linewidth=2)
    ax1.plot(profile['v_safe']/1000, altitudes/1000, 'k-', 
             label='Safe envelope', linewidth=3, alpha=0.8)
    
    # Add altitude markers
    key_altitudes = [10, 50, 80, 100]
    for alt in key_altitudes:
        ax1.axhline(y=alt, color='gray', linestyle='--', alpha=0.3)
        ax1.text(0.5, alt-2, f'{alt} km', fontsize=8, alpha=0.7)
    
    ax1.set_xlabel('Safe Velocity (km/s)')
    ax1.set_ylabel('Altitude (km)')
    ax1.set_title('Warp Bubble Safe Velocity Envelope')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 15)
    ax1.set_ylim(0, 150)
    
    # Atmospheric density
    ax2.semilogx(profile['atmospheric_density'], altitudes/1000, 'g-', linewidth=2)
    ax2.set_xlabel('Atmospheric Density (kg/m³)')
    ax2.set_ylabel('Altitude (km)')
    ax2.set_title('Standard Atmosphere Model')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 150)
    
    # Add key altitude markers
    for alt in key_altitudes:
        ax2.axhline(y=alt, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('atmospheric_constraints_simple.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    simple_atmospheric_demo()
