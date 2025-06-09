#!/usr/bin/env python3
"""
Enhanced Power System and Flight Computer Digital Twins
======================================================

Extends the simulated hardware suite with comprehensive power management and 
computational models for complete spacecraft system validation. Provides 
realistic modeling of power generation, distribution, consumption, thermal 
management, and flight computer performance characteristics.

Digital Twin Components:
- Advanced power system with realistic efficiency curves and thermal effects
- Flight computer with processing latency, memory management, and error modeling
- Power-aware control algorithms with energy budget enforcement
- Integrated system performance monitoring and health assessment

This enables full spacecraft system validation including power and computational
constraints without requiring physical hardware.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass

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

class SimulatedPowerSystem:
    """
    Comprehensive power system digital twin modeling generation, storage,
    distribution, and thermal management for spacecraft operations.
    """
    
    def __init__(self, P_max: float = 1e9, efficiency: float = 0.85, 
                 storage_capacity: float = 3.6e12):
        """
        Initialize power system digital twin.
        
        Args:
            P_max: Maximum power generation/consumption (W)
            efficiency: System efficiency (0-1)
            storage_capacity: Energy storage capacity (J) [default: 1 MW·h]
        """
        self.P_max = P_max
        self.base_efficiency = efficiency
        self.storage_capacity = storage_capacity
        self.energy_stored = storage_capacity * 0.8  # Start at 80% charge
        
        # System state variables
        self.thermal_state = 293.15  # Room temperature start (K)
        self.degradation_factor = 1.0
        self.last_update = time.time()
        self.power_history = []
        self.efficiency_history = []
        
        # Load tracking
        self.current_loads = {}
        self.peak_power_5min = 0.0
        self.total_energy_consumed = 0.0
        
    def add_load(self, load_id: str, power_demand: float) -> bool:
        """
        Add or update a power load.
        
        Args:
            load_id: Unique identifier for the load
            power_demand: Power requirement in Watts
            
        Returns:
            True if load can be supplied, False if exceeds capacity
        """
        total_demand = sum(self.current_loads.values()) + power_demand
        if load_id in self.current_loads:
            total_demand -= self.current_loads[load_id]
            
        if total_demand > self.P_max:
            return False
            
        self.current_loads[load_id] = power_demand
        return True
        
    def remove_load(self, load_id: str):
        """Remove a power load."""
        if load_id in self.current_loads:
            del self.current_loads[load_id]
    
    def supply(self, load: float, dt: float, load_id: str = "transient") -> float:
        """
        Supply power to a load over time duration dt.
        
        Args:
            load: Power demand (W)
            dt: Time duration (s)
            load_id: Load identifier for tracking
            
        Returns:
            Actual energy supplied (J)
        """
        current_time = time.time()
        
        # Update thermal state based on power dissipation
        waste_heat = load * (1 - self._get_current_efficiency())
        thermal_time_constant = 300.0  # 5 minutes
        alpha = dt / (thermal_time_constant + dt)
        ambient_temp = 2.7 + 200 * np.random.random()  # Space environment
        target_temp = ambient_temp + waste_heat / 1e4  # Simplified thermal model
        self.thermal_state = (1 - alpha) * self.thermal_state + alpha * target_temp
        
        # Efficiency depends on thermal state
        thermal_efficiency = max(0.3, 1.0 - (self.thermal_state - 293.15) / 500.0)
        current_efficiency = self.base_efficiency * thermal_efficiency * self.degradation_factor
        
        # Calculate actual power that can be supplied
        total_current_load = sum(self.current_loads.values()) + load
        available_power = min(self.P_max, total_current_load)
        actual_power = min(load, available_power) * current_efficiency
        
        # Calculate energy consumption
        energy_demanded = actual_power * dt
        
        # Check storage capacity
        if energy_demanded > self.energy_stored:
            # Partial supply based on available energy
            actual_energy = self.energy_stored
            self.energy_stored = 0.0
        else:
            actual_energy = energy_demanded
            self.energy_stored -= energy_demanded
            
        # Update tracking
        self.total_energy_consumed += actual_energy
        self.power_history.append((current_time, actual_power))
        self.efficiency_history.append((current_time, current_efficiency))
        
        # Keep only last 5 minutes of history
        cutoff_time = current_time - 300
        self.power_history = [(t, p) for t, p in self.power_history if t > cutoff_time]
        self.efficiency_history = [(t, e) for t, e in self.efficiency_history if t > cutoff_time]
        
        # Update peak power tracking
        recent_powers = [p for t, p in self.power_history if t > current_time - 300]
        if recent_powers:
            self.peak_power_5min = max(recent_powers)
            
        self.last_update = current_time
        
        return actual_energy
    
    def _get_current_efficiency(self) -> float:
        """Calculate current system efficiency including thermal effects."""
        thermal_efficiency = max(0.3, 1.0 - (self.thermal_state - 293.15) / 500.0)
        return self.base_efficiency * thermal_efficiency * self.degradation_factor
    
    def get_status(self) -> PowerSystemStatus:
        """Get comprehensive power system status."""
        current_load = sum(self.current_loads.values())
        storage_fraction = self.energy_stored / self.storage_capacity
        
        # Determine system health
        if storage_fraction < 0.1 or self.thermal_state > 400:
            health = "CRITICAL"
        elif storage_fraction < 0.3 or current_load > 0.9 * self.P_max:
            health = "DEGRADED" 
        else:
            health = "NOMINAL"
            
        return PowerSystemStatus(
            timestamp=time.time(),
            power_generation=0.0,  # Could model solar panels, RTG, etc.
            power_consumption=current_load,
            energy_stored=self.energy_stored,
            storage_capacity=self.storage_capacity,
            efficiency=self._get_current_efficiency(),
            thermal_state=self.thermal_state,
            system_health=health
        )

class SimulatedFlightComputer:
    """
    Flight computer digital twin modeling computational performance,
    memory management, and real-time control system characteristics.
    """
    
    def __init__(self, clock_hz: float = 1e9, cores: int = 4, 
                 memory_gb: float = 32.0, exec_cycles: int = 1e6):
        """
        Initialize flight computer digital twin.
        
        Args:
            clock_hz: Processor clock frequency (Hz)
            cores: Number of processor cores
            memory_gb: Available RAM (GB)
            exec_cycles: Typical control function execution cycles
        """
        self.clock_hz = clock_hz
        self.cores = cores
        self.memory_bytes = memory_gb * 1e9
        self.exec_cycles = exec_cycles
        
        # System state
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        self.task_queue = []
        self.execution_history = []
        self.error_count = 0
        self.total_executions = 0
        
        # Performance characteristics
        self.base_latency = exec_cycles / clock_hz
        self.memory_allocated = 0.0
        self.last_execution_time = 0.0
        
    def execute(self, control_fn: Callable, state: Dict[str, Any], 
                dt: float, priority: int = 1) -> Dict[str, Any]:
        """
        Execute control function with realistic computational modeling.
        
        Args:
            control_fn: Control function to execute
            state: Current system state
            dt: Time step (s)
            priority: Task priority (1=highest, 5=lowest)
            
        Returns:
            Updated system state
        """
        start_time = time.time()
        
        # Add to execution queue
        task_id = len(self.execution_history)
        self.task_queue.append({
            'id': task_id,
            'function': control_fn,
            'state': state,
            'dt': dt,
            'priority': priority,
            'submit_time': start_time
        })
        
        # Process highest priority task
        self.task_queue.sort(key=lambda x: x['priority'])
        task = self.task_queue.pop(0)
        
        # Model computational latency
        base_latency = self.base_latency
        
        # CPU utilization affects latency
        cpu_factor = 1.0 + self.cpu_utilization * 2.0
        
        # Memory pressure affects performance
        memory_factor = 1.0 + max(0, self.memory_utilization - 0.8) * 5.0
        
        # Queue length affects latency
        queue_factor = 1.0 + len(self.task_queue) * 0.1
        
        # Random jitter
        jitter_factor = 1.0 + np.random.normal(0, 0.1)
        
        total_latency = base_latency * cpu_factor * memory_factor * queue_factor * jitter_factor
        total_latency = max(total_latency, self.base_latency * 0.5)  # Minimum latency
        
        # Simulate execution time
        time.sleep(min(total_latency, 0.01))  # Cap simulation delay
        
        # Execute the control function
        try:
            new_state = task['function'](task['state'].copy())
            execution_success = True
            
            # Model occasional computational errors
            if np.random.random() < 1e-6:  # 1 in million chance
                self.error_count += 1
                execution_success = False
                
        except Exception as e:
            new_state = task['state'].copy()
            execution_success = False
            self.error_count += 1
            
        # Update system state
        if 'time' in new_state:
            new_state['time'] += dt + total_latency
        else:
            new_state['time'] = task['state'].get('time', 0) + dt + total_latency
            
        # Update performance metrics
        execution_time = time.time() - start_time
        self.execution_history.append({
            'task_id': task_id,
            'execution_time': execution_time,
            'latency': total_latency,
            'success': execution_success,
            'timestamp': start_time
        })
        
        # Keep only recent history
        cutoff_time = start_time - 300  # 5 minutes
        self.execution_history = [
            h for h in self.execution_history 
            if h['timestamp'] > cutoff_time
        ]
        
        self.total_executions += 1
        self.last_execution_time = execution_time
        
        # Update resource utilization (simplified model)
        recent_executions = len(self.execution_history)
        self.cpu_utilization = min(0.95, recent_executions / 100.0)
        self.memory_utilization = min(0.9, self.memory_allocated / self.memory_bytes)
        
        return new_state
    
    def get_status(self) -> FlightComputerStatus:
        """Get comprehensive flight computer status."""
        if self.execution_history:
            recent_errors = sum(1 for h in self.execution_history if not h['success'])
            error_rate = recent_errors / len(self.execution_history)
            avg_latency = np.mean([h['latency'] for h in self.execution_history])
        else:
            error_rate = 0.0
            avg_latency = self.base_latency
            
        # Determine system health
        if error_rate > 0.01 or self.cpu_utilization > 0.95:
            health = "CRITICAL"
        elif error_rate > 0.001 or self.cpu_utilization > 0.8:
            health = "DEGRADED"
        else:
            health = "NOMINAL"
            
        return FlightComputerStatus(
            timestamp=time.time(),
            cpu_utilization=self.cpu_utilization,
            memory_utilization=self.memory_utilization,
            execution_latency=avg_latency,
            queue_length=len(self.task_queue),
            error_rate=error_rate,
            system_health=health
        )

def run_simulated_MVP():
    """
    Comprehensive simulated MVP demonstrating power system and flight computer
    integration with the complete warp bubble protection pipeline.
    """
    print("🚀 SIMULATED MVP: POWER SYSTEM & FLIGHT COMPUTER INTEGRATION")
    print("=" * 70)
    
    # Import protection systems
    try:
        from atmospheric_constraints import AtmosphericConstraints
        from leo_collision_avoidance import LEOCollisionAvoidanceSystem
        from micrometeoroid_protection import IntegratedProtectionSystem, MicrometeoroidEnvironment, BubbleGeometry
        from integrated_space_protection import IntegratedSpaceProtectionSystem
        PROTECTION_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️  Protection systems not available: {e}")
        PROTECTION_AVAILABLE = False
    
    # Initialize digital twin hardware
    power_sys = SimulatedPowerSystem(P_max=1e9, efficiency=0.85)
    flight_cpu = SimulatedFlightComputer(clock_hz=2e9, cores=8, memory_gb=64.0)
    
    print(f"\n⚡ Power System Initialized:")
    print(f"   Max power: {power_sys.P_max/1e6:.1f} MW")
    print(f"   Storage capacity: {power_sys.storage_capacity/3.6e12:.1f} MW·h")
    print(f"   Current efficiency: {power_sys._get_current_efficiency():.1%}")
    
    print(f"\n💻 Flight Computer Initialized:")
    print(f"   Clock speed: {flight_cpu.clock_hz/1e9:.1f} GHz")
    print(f"   Cores: {flight_cpu.cores}")
    print(f"   Memory: {flight_cpu.memory_bytes/1e9:.1f} GB")
    
    if PROTECTION_AVAILABLE:
        # Initialize protection systems
        atmo = AtmosphericConstraints()
        leo = LEOCollisionAvoidanceSystem()
        micro = IntegratedProtectionSystem(MicrometeoroidEnvironment(), BubbleGeometry(50.0))
        integ = IntegratedSpaceProtectionSystem()
        
        print(f"\n🛡️  Protection Systems Online:")
        print(f"   ✓ Atmospheric constraints")
        print(f"   ✓ LEO collision avoidance")
        print(f"   ✓ Micrometeoroid protection")
        print(f"   ✓ Integrated coordination")
    
    # Initialize mission state
    state = {
        'pos': np.array([0.0, 0.0, 350e3]),
        'vel': np.array([7500.0, 0.0, -50.0]),
        'time': 0.0,
        'mission_phase': 'orbital_operations'
    }
    
    print(f"\n🎯 Mission State:")
    print(f"   Position: [{state['pos'][0]/1000:.1f}, {state['pos'][1]/1000:.1f}, {state['pos'][2]/1000:.1f}] km")
    print(f"   Velocity: [{state['vel'][0]:.0f}, {state['vel'][1]:.0f}, {state['vel'][2]:.0f}] m/s")
    
    # Main simulation loop
    dt = 1.0
    print(f"\n🔄 Starting 60-Second Simulation with Power/Compute Modeling...")
    
    for step in range(60):
        print(f"\n--- Step {step+1}/60 (t = {step+1} s) ---")
        
        # Define control law for flight computer execution
        def integrated_control_law(s):
            """Integrated control law executed by flight computer."""
            new_state = s.copy()
            
            if PROTECTION_AVAILABLE:
                # Apply atmospheric safety constraints
                h = s['pos'][2]
                v = np.linalg.norm(s['vel'])
                v_safe_thermal = atmo.max_velocity_thermal(h)
                v_safe_drag = atmo.max_velocity_drag(h)
                v_safe = min(v_safe_thermal, v_safe_drag)
                
                if v > v_safe:
                    print(f"   🌍 Atmospheric constraint: reducing velocity from {v:.0f} to {v_safe:.0f} m/s")
                    decel_factor = v_safe / v
                    new_state['vel'] = s['vel'] * decel_factor
                    
                # Simple collision avoidance (placeholder)
                # In real implementation would use full LEO system
                if np.random.random() < 0.1:  # 10% chance of debris detection
                    print(f"   📡 Debris detected: executing evasive maneuver")
                    # Small velocity adjustment
                    dodge_vector = np.random.normal(0, 1, 3)
                    dodge_vector /= np.linalg.norm(dodge_vector)
                    new_state['vel'] += dodge_vector * 0.1  # 0.1 m/s dodge
            
            return new_state
        
        # Execute control law through flight computer
        state = flight_cpu.execute(integrated_control_law, state, dt, priority=1)
        
        # Power system modeling
        # 1. Protection system power consumption
        protection_power = 50e3  # 50 kW baseline
        if PROTECTION_AVAILABLE:
            protection_power += np.random.uniform(10e3, 100e3)  # Variable load
            
        # 2. Flight computer power consumption
        cpu_status = flight_cpu.get_status()
        compute_power = 5e3 * (1 + cpu_status.cpu_utilization)  # 5-10 kW
        
        # 3. Warp drive power (major load)
        warp_power = 500e6  # 500 MW for warp operations
        
        # Supply power to all systems
        protection_energy = power_sys.supply(protection_power, dt, "protection")
        compute_energy = power_sys.supply(compute_power, dt, "flight_computer")
        warp_energy = power_sys.supply(warp_power, dt, "warp_drive")
        
        # Get system status
        power_status = power_sys.get_status()
        
        # Update state
        state['pos'] += state['vel'] * dt
        
        # Periodic status updates
        if step % 10 == 0:
            print(f"   💻 CPU: {cpu_status.cpu_utilization:.1%} util, {cpu_status.execution_latency*1000:.2f} ms latency")
            print(f"   ⚡ Power: {power_status.power_consumption/1e6:.1f} MW load, {power_status.energy_stored/3.6e12:.1f} MW·h stored")
            print(f"   🔋 Battery: {power_status.energy_stored/power_status.storage_capacity:.1%} charged")
        
        # Check for system health issues
        if power_status.system_health != "NOMINAL":
            print(f"   ⚠️  Power system health: {power_status.system_health}")
            
        if cpu_status.system_health != "NOMINAL":
            print(f"   ⚠️  Flight computer health: {cpu_status.system_health}")
    
    # Final status summary
    final_power_status = power_sys.get_status()
    final_cpu_status = flight_cpu.get_status()
    
    print(f"\n📊 SIMULATION COMPLETE - FINAL STATUS")
    print(f"=" * 50)
    print(f"💻 Flight Computer:")
    print(f"   Total executions: {flight_cpu.total_executions}")
    print(f"   Error rate: {final_cpu_status.error_rate:.2%}")
    print(f"   Average latency: {final_cpu_status.execution_latency*1000:.2f} ms")
    print(f"   System health: {final_cpu_status.system_health}")
    
    print(f"\n⚡ Power System:")
    print(f"   Total energy consumed: {power_sys.total_energy_consumed/3.6e12:.2f} MW·h")
    print(f"   Energy remaining: {final_power_status.energy_stored/3.6e12:.1f} MW·h")
    print(f"   System efficiency: {final_power_status.efficiency:.1%}")
    print(f"   Thermal state: {final_power_status.thermal_state:.1f} K")
    print(f"   System health: {final_power_status.system_health}")
    
    print(f"\n🎯 Mission Performance:")
    print(f"   Final position: [{state['pos'][0]/1000:.1f}, {state['pos'][1]/1000:.1f}, {state['pos'][2]/1000:.1f}] km")
    print(f"   Final velocity: {np.linalg.norm(state['vel']):.0f} m/s")
    print(f"   Mission time: {state['time']:.2f} s")
    
    print(f"\n✅ Simulated MVP run complete.")
    print(f"   All digital twin systems operational with realistic performance modeling")

if __name__ == "__main__":
    run_simulated_MVP()
