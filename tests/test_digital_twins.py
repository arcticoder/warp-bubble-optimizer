#!/usr/bin/env python3
"""
Simple validation script for digital-twin integration
"""

def test_digital_twins():
    """Test basic digital-twin functionality."""
    print("Testing Digital-Twin Hardware Integration")
    print("=" * 45)
    
    try:
        # Test power system
        from simulate_power_and_flight_computer import SimulatedPowerSystem, SimulatedFlightComputer
        print("CHECK Power and flight computer imports successful")
        
        power_sys = SimulatedPowerSystem()
        flight_cpu = SimulatedFlightComputer()
        print("CHECK Digital twin objects created successfully")
        
        # Test basic power operation
        result = power_sys.supply_power(100e3, 1.0)  # 100 kW for 1 second
        print(f"CHECK Power system test: {result['actual_power']/1e3:.1f} kW supplied")
        
        # Test basic flight computer operation
        def simple_control(state):
            return state  # No-op control law
        
        test_state = {'time': 0.0, 'position': [0, 0, 0]}
        new_state = flight_cpu.execute_control_law(simple_control, test_state, 1.0)
        print(f"CHECK Flight computer test: control latency {new_state.get('execution_latency', 0)*1000:.2f} ms")
        
        print("\nDIGITAL-TWIN INTEGRATION: SUCCESSFUL")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    test_digital_twins()
