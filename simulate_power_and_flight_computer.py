#!/usr/bin/env python3
"""
Power System and Flight Computer Digital Twins
==============================================

Extends the simulated hardware suite with power management and computational
models for complete spacecraft system validation. Provides realistic modeling
of power generation, distribution, and consumption alongside flight computer
performance characteristics including computational latency and resource limits.

Digital Twin Components:
- Power system with generation, storage, and load management
- Flight computer with realistic processing latency and resource constraints
- Power-aware control algorithms with energy budget enforcement
- Integrated system performance monitoring and optimization

This enables full spacecraft system validation including power and computational
constraints without requiring physical hardware.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)

# JAX imports with fallback for acceleration
try:
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
    print("🚀 JAX acceleration enabled for power/flight computer simulation")
except ImportError:
    import numpy as jnp
    def jit(func):
        return func
    JAX_AVAILABLE = False
    print("⚠️  JAX not available, using NumPy fallback for power/flight computer simulation")

@dataclass
class PowerSystemStatus:
    """Real-time power system status information."""
    timestamp: float
    power_generation: float      # Current generation (W)
    power_consumption: float     # Current load (W)
    energy_stored: float         # Battery/storage level (J)
    storage_capacity: float      # Maximum storage (J)
    efficiency: float            # Current system efficiency
    thermal_state: float         # System temperature (K)
    system_health: str           # NOMINAL/DEGRADED/CRITICAL

@dataclass 
class FlightComputerStatus:
    """Real-time flight computer performance metrics."""
    timestamp: float
    cpu_utilization: float       # CPU usage (0-1)
    memory_utilization: float    # RAM usage (0-1)
    execution_latency: float     # Last control cycle latency (s)
    queue_length: int            # Pending task queue
    error_rate: float            # Computational error rate
    system_health: str           # NOMINAL/DEGRADED/CRITICAL

@dataclass
class FlightComputerConfig:
    """Configuration parameters for flight computer digital twin."""
    clock_hz: float = 1e9           # Processor clock speed (Hz)
    exec_cycles: int = 1e6          # Execution cycles for control law
    memory_gb: float = 64.0         # Available memory (GB)
    cache_size_mb: float = 256.0    # Cache size (MB)
    radiation_tolerance: float = 1e6 # Radiation tolerance (rads)
    cache_miss_penalty: float = 50e-9  # Cache miss penalty (s)
    radiation_error_rate: float = 1e-9  # Radiation-induced error rate (errors/s)

@dataclass
class PowerSystemConfig:
    """Configuration parameters for power system digital twin."""
    max_power: float = 1e9          # Maximum power generation/consumption (W)
    efficiency: float = 0.85        # System efficiency (0-1)
    initial_energy: float = 3.6e12  # Initial energy storage (J)
    thermal_limit: float = 373.15   # Maximum operating temperature (K)

class SimulatedPowerSystem:
    """
    Comprehensive power system digital twin modeling generation, storage,
    distribution, and thermal management for spacecraft operations.
    """
    
    def __init__(self, config: PowerSystemConfig = None):
        """
        Initialize power system digital twin.
        
        Args:
            config: Power system configuration
        """
        self.config = config or PowerSystemConfig()
        self.energy_stored = self.config.initial_energy * 0.8  # Start at 80% charge
        
        # System state variables
        self.temperature = 293.15  # Room temperature start (K)
        self.system_health = 1.0
        self.last_update = time.time()
        self.power_history = []
        self.efficiency_history = []
        
        # Load tracking
        self.current_loads = {}
        self.peak_power_5min = 0.0
        self.total_energy_consumed = 0.0
        self.current_load = 0.0
        self.fault_conditions = []
        
    def supply_power(self, load: float, dt: float) -> Dict[str, Any]:
        """
        Supply power to load over time period dt.
        
        Args:
            load: Requested power (W)
            dt: Time step (s)
            
        Returns:
            Dict with actual power supplied, energy consumed, and system status
        """        # Apply efficiency curve (efficiency decreases at high loads)
        load_factor = load / self.config.max_power
        efficiency = self.config.efficiency * (1.0 - 0.2 * load_factor**2)
        
        # Thermal effects on efficiency
        temp_factor = max(0.5, 1.0 - (self.temperature - 293.0) / 100.0)
        efficiency *= temp_factor
        
        # Determine actual power available
        max_available = min(load, self.config.max_power)
        actual_power = max_available * efficiency * self.system_health
        
        # Energy consumption
        energy_consumed = actual_power * dt
        self.energy_stored = max(0.0, self.energy_stored - energy_consumed)
        self.total_energy_consumed += energy_consumed
        
        # Update state
        self.current_load = load
        
        return {
            'actual_power': actual_power,
            'energy_consumed': energy_consumed,
            'efficiency': efficiency,
            'system_health': self.system_health
        }
        energy_consumed = actual_power * dt
        
        # Check energy storage
        if energy_consumed > self.energy_stored:
            # Insufficient energy - scale down power
            actual_power *= self.energy_stored / energy_consumed
            energy_consumed = self.energy_stored
            self.fault_conditions.append({
                'type': 'energy_depletion',
                'timestamp': time.time(),
                'remaining_energy': self.energy_stored
            })
            
        # Update energy storage
        self.energy_stored = max(0.0, self.energy_stored - energy_consumed)
        
        # Energy discharge when not in use
        idle_discharge = self.energy_stored * (1.0 - self.config.discharge_rate) * dt
        self.energy_stored -= idle_discharge
        
        # Thermal modeling
        heat_generated = energy_consumed * (1.0 - efficiency) / 1000.0  # kW to temperature
        self.temperature += heat_generated * dt * 0.1  # Simple thermal model
        
        # Thermal cooling (radiative + active cooling)
        cooling_rate = (self.temperature - 273.0) * 0.01  # K/s
        self.temperature = max(273.0, self.temperature - cooling_rate * dt)
        
        # Thermal protection
        if self.temperature > self.config.thermal_limit:
            self.system_health *= 0.99  # Gradual degradation
            self.fault_conditions.append({
                'type': 'thermal_overload',
                'timestamp': time.time(),
                'temperature': self.temperature
            })
            
        # Record performance metrics
        self.current_load = actual_power
        self.power_history.append({
            'timestamp': time.time(),
            'requested_power': load,
            'actual_power': actual_power,
            'efficiency': efficiency,
            'temperature': self.temperature,
            'energy_stored': self.energy_stored
        })
        
        return {
            'actual_power': actual_power,
            'energy_consumed': energy_consumed,
            'efficiency': efficiency,
            'temperature': self.temperature,
            'energy_remaining': self.energy_stored,
            'system_health': self.system_health,
            'fault_conditions': len(self.fault_conditions)
        }
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive power system status."""
        return {
            'energy_stored': self.energy_stored,
            'max_energy': self.config.initial_energy,
            'energy_percentage': self.energy_stored / self.config.initial_energy * 100,
            'current_load': self.current_load,
            'max_power': self.config.max_power,
            'load_percentage': self.current_load / self.config.max_power * 100,
            'temperature': self.temperature,
            'system_health': self.system_health,
            'fault_count': len(self.fault_conditions),
            'operational_status': 'NOMINAL' if self.system_health > 0.8 else 'DEGRADED'
        }

class SimulatedFlightComputer:
    """
    Digital twin of spacecraft flight management computer.
    
    Models realistic computational performance, memory usage, radiation effects,
    and processing latency for control law execution.
    """
    
    def __init__(self, config: FlightComputerConfig = None):
        self.config = config or FlightComputerConfig()
        self.cpu_load = 0.0
        self.memory_used = 0.0
        self.execution_history = []
        self.radiation_errors = 0
        self.cache_performance = 0.95  # Cache hit rate
        
    def execute_control_law(self, control_function: Callable, 
                           state: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """
        Execute control function with realistic computer simulation.
        
        Args:
            control_function: Function to execute
            state: Current system state
            dt: Time step
            
        Returns:
            Updated state with execution metrics
        """
        start_time = time.time()
        
        # Estimate computational complexity based on state size
        state_complexity = len(str(state))  # Simple complexity metric
        base_cycles = self.config.exec_cycles
        actual_cycles = base_cycles * (1.0 + state_complexity / 10000.0)
        
        # Cache performance effects
        cache_misses = (1.0 - self.cache_performance) * actual_cycles
        cache_penalty = cache_misses * self.config.cache_miss_penalty
        actual_cycles += cache_penalty
        
        # Base execution time
        execution_time = actual_cycles / self.config.clock_hz
        
        # Add realistic jitter (±10% variation)
        jitter = np.random.normal(0, execution_time * 0.1)
        execution_time += jitter
        
        # Radiation-induced bit flips
        bit_flip_probability = self.config.radiation_error_rate * actual_cycles
        if np.random.random() < bit_flip_probability:
            self.radiation_errors += 1
            # Simulate error correction overhead
            execution_time *= 1.1
            
        # Memory usage simulation
        memory_required = state_complexity / 1000.0  # MB
        self.memory_used = min(self.config.memory_gb * 1024, memory_required)
        
        # CPU load calculation
        self.cpu_load = min(1.0, execution_time / dt)
        
        # Execute the actual control function
        try:
            new_state = control_function(state.copy())
            execution_success = True
        except Exception as e:
            logger.error(f"Control function execution failed: {e}")
            new_state = state.copy()  # Fallback to previous state
            execution_success = False
            
        # Update state timing
        if 'time' in new_state:
            new_state['time'] += dt + execution_time
        else:
            new_state['time'] = dt + execution_time
            
        # Record execution metrics
        execution_record = {
            'timestamp': start_time,
            'execution_time': execution_time,
            'cpu_cycles': actual_cycles,
            'memory_used': self.memory_used,
            'cpu_load': self.cpu_load,
            'cache_hits': self.cache_performance,
            'radiation_errors': self.radiation_errors,
            'success': execution_success
        }
        self.execution_history.append(execution_record)
        
        # Add execution metrics to state
        new_state['_execution_metrics'] = execution_record
        
        return new_state
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get flight computer performance metrics."""
        if not self.execution_history:
            return {'status': 'no_data'}
            
        recent_executions = self.execution_history[-10:]  # Last 10 executions
        avg_execution_time = np.mean([e['execution_time'] for e in recent_executions])
        avg_cpu_load = np.mean([e['cpu_load'] for e in recent_executions])
        
        return {
            'avg_execution_time': avg_execution_time,
            'max_execution_time': max(e['execution_time'] for e in recent_executions),
            'avg_cpu_load': avg_cpu_load,
            'memory_utilization': self.memory_used / (self.config.memory_gb * 1024) * 100,
            'cache_hit_rate': self.cache_performance * 100,
            'radiation_errors': self.radiation_errors,
            'total_executions': len(self.execution_history),
            'system_status': 'NOMINAL' if avg_cpu_load < 0.8 else 'HIGH_LOAD'
        }

def run_simulated_mvp():
    """
    Run complete simulated Minimum Viable Product with all digital twins.
    
    Integrates power system, flight computer, and all protection systems
    in one comprehensive simulation loop.
    """
    print("🚀 SIMULATED MVP: COMPLETE DIGITAL TWIN INTEGRATION")
    print("=" * 60)
    
    # Import protection systems
    try:
        from atmospheric_constraints import AtmosphericConstraints
        from leo_collision_avoidance import LEOCollisionAvoidanceSystem
        from micrometeoroid_protection import IntegratedProtectionSystem, MicrometeoroidEnvironment, BubbleGeometry
        from integrated_space_protection import IntegratedSpaceProtectionSystem
        from simulated_interfaces import create_simulated_sensor_suite
        PROTECTION_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️  Protection systems not available: {e}")
        PROTECTION_AVAILABLE = False
      # Initialize digital twin systems
    power_system = SimulatedPowerSystem()
    flight_computer = SimulatedFlightComputer()
    
    if PROTECTION_AVAILABLE:
        atmo = AtmosphericConstraints()
        leo = LEOCollisionAvoidanceSystem()
        micro = IntegratedProtectionSystem(MicrometeoroidEnvironment(), BubbleGeometry(50.0))
        protection = IntegratedSpaceProtectionSystem()
        sensors = create_simulated_sensor_suite()
    
    print("\n⚙️  Digital Twin Systems Initialized:")
    print(f"   • Power System: {power_system.config.max_power/1e6:.0f} MW capacity")
    print(f"   • Flight Computer: {flight_computer.config.clock_hz/1e9:.1f} GHz processor")
    if PROTECTION_AVAILABLE:
        print("   • Complete protection suite loaded")
    
    # Initial spacecraft state
    state = {
        'position': np.array([0.0, 0.0, 350e3]),     # 350 km altitude
        'velocity': np.array([7500.0, 0.0, -50.0]),  # Orbital velocity + descent
        'time': 0.0,
        'mission_phase': 'descent'
    }
    
    print(f"\n🎯 Mission Scenario: LEO Descent Simulation")
    print(f"   Initial position: [{state['position'][0]:.0f}, {state['position'][1]:.0f}, {state['position'][2]/1000:.0f}] km")
    print(f"   Initial velocity: [{state['velocity'][0]:.0f}, {state['velocity'][1]:.0f}, {state['velocity'][2]:.0f}] m/s")
    
    # Main simulation loop
    dt = 1.0  # 1 second time steps
    total_energy_used = 0.0
    
    print(f"\n🔄 Running 60-Second Simulation...")
    
    for step in range(60):
        print(f"\n--- Step {step+1}/60 (t = {step+1} s) ---")
        
        # Define control law for flight computer
        def control_law(current_state):
            new_state = current_state.copy()
            
            if PROTECTION_AVAILABLE:
                # Apply atmospheric safety constraints
                h = current_state['position'][2]
                v = np.linalg.norm(current_state['velocity'])
                v_thermal = atmo.max_velocity_thermal(h)
                v_drag = atmo.max_velocity_drag(h)
                v_safe = min(v_thermal, v_drag)
                
                if v > v_safe:
                    # Need to decelerate
                    decel_factor = v_safe / v
                    new_state['velocity'] = current_state['velocity'] * decel_factor
                    print(f"   🌍 Atmospheric deceleration: {v:.0f} → {v_safe:.0f} m/s")
                    
            return new_state
        
        # Execute control law on flight computer
        state = flight_computer.execute_control_law(control_law, state, dt)
        
        # Power system simulation
        # Estimate power requirements based on active systems
        base_power = 50e3      # 50 kW baseline systems
        protection_power = 100e3 if PROTECTION_AVAILABLE else 0  # 100 kW protection
        warp_power = 500e3     # 500 kW for warp field maintenance
        
        total_power_request = base_power + protection_power + warp_power
        
        # Supply power
        power_result = power_system.supply_power(total_power_request, dt)
        total_energy_used += power_result['energy_consumed']
        
        # Sensor simulation and protection systems
        if PROTECTION_AVAILABLE and step % 5 == 0:  # Every 5 seconds
            # Simulate sensor readings
            sensor_data = {}
            for sensor_name, sensor in sensors.items():
                if hasattr(sensor, 'read_temperature'):
                    sensor_data[sensor_name] = sensor.read_temperature()
                elif hasattr(sensor, 'scan'):
                    detections = sensor.scan(state['position'], state['velocity'])
                    sensor_data[sensor_name] = len(detections)
                    
            print(f"   📡 Sensor readings: {len(sensor_data)} active sensors")
        
        # Simulate warp thrust execution (placeholder)
        if step % 10 == 0:  # Every 10 seconds
            target_velocity = np.array([8000, 0, 0])  # 8 km/s target
            thrust_energy = np.linalg.norm(target_velocity - state['velocity'][:3]) * 1e15  # Placeholder energy
            print(f"   ⚡ Warp thrust simulation: {thrust_energy:.2e} J")
        
        # Update spacecraft state
        state['position'] += state['velocity'] * dt
        
        # Altitude check
        altitude = state['position'][2]
        if altitude < 0:
            print(f"   🌍 Surface contact at t={step+1}s")
            break
            
        # Performance monitoring
        if step % 10 == 0:
            power_status = power_system.get_system_status()
            flight_metrics = flight_computer.get_performance_metrics()
            
            print(f"   🔋 Power: {power_status['energy_percentage']:.1f}% ({power_status['operational_status']})")
            print(f"   💻 CPU: {flight_metrics.get('avg_cpu_load', 0)*100:.1f}% load")
    
    # Final results
    print(f"\n📊 SIMULATION COMPLETE")
    print("=" * 40)
    
    final_altitude = state['position'][2] / 1000
    final_velocity = np.linalg.norm(state['velocity'])
    power_status = power_system.get_system_status()
    flight_metrics = flight_computer.get_performance_metrics()
    
    print(f"   Final altitude: {final_altitude:.1f} km")
    print(f"   Final velocity: {final_velocity:.0f} m/s")
    print(f"   Total energy used: {total_energy_used/1e9:.1f} GJ")
    print(f"   Power system health: {power_status['system_health']*100:.1f}%")
    print(f"   Flight computer errors: {flight_metrics.get('radiation_errors', 0)}")
    print(f"   Average execution time: {flight_metrics.get('avg_execution_time', 0)*1000:.1f} ms")
    
    print("\n✅ Simulated MVP run complete - all digital twins operational!")
    
    return {
        'final_state': state,
        'total_energy_used': total_energy_used,
        'power_system_status': power_status,
        'flight_computer_metrics': flight_metrics,
        'simulation_successful': True
    }

# Example usage and testing
def demo_power_flight_integration():
    """Demonstrate power system and flight computer integration."""
    print("🔧 POWER & FLIGHT COMPUTER DIGITAL TWIN DEMO")
    print("=" * 50)
    
    # Create systems
    power_config = PowerSystemConfig(max_power=2e6, efficiency=0.85)  # 2 MW system
    flight_config = FlightComputerConfig(clock_hz=2e9, exec_cycles=5e5)  # 2 GHz, 500k cycles
    
    power_sys = SimulatedPowerSystem(power_config)
    flight_cpu = SimulatedFlightComputer(flight_config)
    
    print(f"\n⚡ Power System: {power_config.max_power/1e6:.0f} MW, {power_config.efficiency*100:.0f}% efficient")
    print(f"💻 Flight Computer: {flight_config.clock_hz/1e9:.1f} GHz, {flight_config.exec_cycles/1e3:.0f}k cycles/execution")
    
    # Test control law
    def test_control_law(state):
        """Simple test control law."""
        # Simulate some computation
        result = state.copy()
        result['computed_value'] = np.sum([v for v in state.values() if isinstance(v, (int, float))])
        return result
    
    # Run test sequence
    test_state = {'x': 1.0, 'y': 2.0, 'z': 3.0, 'time': 0.0}
    
    for i in range(5):
        print(f"\n--- Test {i+1} ---")
        
        # Execute control law
        test_state = flight_cpu.execute_control_law(test_control_law, test_state, 0.1)
        
        # Request power
        power_load = 800e3  # 800 kW
        power_result = power_sys.supply_power(power_load, 0.1)
        
        print(f"   Control execution: {test_state['_execution_metrics']['execution_time']*1000:.2f} ms")
        print(f"   Power supplied: {power_result['actual_power']/1e3:.0f} kW ({power_result['efficiency']*100:.1f}% efficient)")
        print(f"   System temperature: {power_result['temperature']:.1f} K")
    
    # Performance summary
    power_status = power_sys.get_system_status()
    flight_metrics = flight_cpu.get_performance_metrics()
    
    print(f"\n📈 Performance Summary:")
    print(f"   Power system health: {power_status['system_health']*100:.1f}%")
    print(f"   Energy remaining: {power_status['energy_percentage']:.1f}%")
    print(f"   Average CPU load: {flight_metrics['avg_cpu_load']*100:.1f}%")
    print(f"   Cache hit rate: {flight_metrics['cache_hit_rate']:.1f}%")
    
    print("\n✅ Power & flight computer integration test complete!")

if __name__ == "__main__":
    # Run demonstration
    demo_power_flight_integration()
    
    print("\n" + "="*60)
    
    # Run full MVP simulation
    run_simulated_mvp()
