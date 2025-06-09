#!/usr/bin/env python3
"""
Test Digital Twin MVP Integration
================================

Simple validation script to test the complete digital twin MVP simulation
components and integration without running the full simulation.
"""

def test_mvp_components():
    """Test MVP digital twin component imports and basic functionality."""
    print("TESTING COMPLETE MVP DIGITAL TWIN INTEGRATION")
    print("=" * 50)
    
    # Test core systems
    try:
        from simulate_power_and_flight_computer import SimulatedPowerSystem, SimulatedFlightComputer
        from simulated_interfaces import create_simulated_sensor_suite
        print("✓ Core digital twin systems imported successfully")
    except ImportError as e:
        print(f"✗ Core systems import failed: {e}")
        return False
    
    # Test MVP simulation components
    try:
        from simulate_full_warp_MVP import (
            SimulatedNegativeEnergyGenerator, 
            SimulatedWarpFieldGenerator,
            SimulatedHullStructural
        )
        print("✓ MVP exotic physics digital twins imported successfully")
    except ImportError as e:
        print(f"✗ MVP components import failed: {e}")
        return False
    
    # Test component initialization
    try:
        power_sys = SimulatedPowerSystem()
        flight_cpu = SimulatedFlightComputer()
        neg_energy = SimulatedNegativeEnergyGenerator()
        warp_field = SimulatedWarpFieldGenerator()
        hull_struct = SimulatedHullStructural()
        sensors = create_simulated_sensor_suite()
        
        print("✓ All digital twin components initialized successfully")
    except Exception as e:
        print(f"✗ Component initialization failed: {e}")
        return False
    
    # Test basic functionality
    try:
        # Test power system
        power_result = power_sys.supply_power(100e3, 1.0)  # 100 kW for 1s
        print(f"✓ Power system test: {power_result['actual_power']/1e3:.1f} kW supplied")
        
        # Test negative energy generator
        neg_result = neg_energy.generate_exotic_pulse(-1e15, 1e-6)  # 1 PJ pulse
        print(f"✓ Negative energy test: {abs(neg_result['exotic_energy_generated'])/1e12:.1f} TJ generated")
        
        # Test warp field generator
        warp_field.set_warp_field({'R': 50.0, 'delta': 1.0}, [7500, 0, 0])
        field_result = warp_field.update_field(1.0)
        print(f"✓ Warp field test: {field_result['current_power']/1e6:.1f} MW field power")
        
        # Test hull structural
        struct_result = hull_struct.apply_warp_loads(1e6, [0, 0, -9.81e-6])
        print(f"✓ Structural test: {struct_result['total_stress']/1e6:.1f} MPa stress")
        
        print("\n🎯 MVP DIGITAL TWIN VALIDATION: COMPLETE SUCCESS")
        print("   All subsystems operational and ready for simulation")
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_mvp_components()
    if success:
        print("\n✅ Ready for full MVP simulation with simulate_full_warp_MVP.py")
    else:
        print("\n❌ MVP simulation requires debugging")
