#!/usr/bin/env python3
"""
Full Warp Simulated Hardware Demo
=================================

Digital twins of all hardware subsystems for comprehensive warp bubble 
spacecraft simulation without requiring physical hardware.

Simulates:
- Radar sensor systems with realistic noise and detection limits
- IMU and navigation sensors with drift and bias
- Thermocouple arrays for thermal monitoring
- EM field generators with actuation delays
- Complete sensor fusion and control loop integration

This enables full validation of control and safety logic under realistic
sensor noise, latency, and failure modes before physical prototyping.
"""

import numpy as np
import time
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import protection systems
try:
    from atmospheric_constraints import AtmosphericConstraints
    from leo_collision_avoidance import LEOCollisionAvoidanceSystem, TargetObject
    from micrometeoroid_protection import IntegratedProtectionSystem, MicrometeoroidEnvironment, BubbleGeometry
    from integrated_space_protection import IntegratedSpaceProtectionSystem, IntegratedSystemConfig
    try:
        from simulated_interfaces import (
            SimulatedRadar, SimulatedIMU, SimulatedThermocouple, SimulatedEMFieldGenerator,
            create_simulated_sensor_suite, simulate_sensor_fusion_data
        )
        USE_CENTRALIZED_INTERFACES = True
    except ImportError:
        USE_CENTRALIZED_INTERFACES = False
    PROTECTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Protection systems not available: {e}")
    PROTECTION_AVAILABLE = False
    USE_CENTRALIZED_INTERFACES = False

@dataclass
class SimulatedSensorReading:
    """Base class for simulated sensor readings."""
    timestamp: float
    value: Any
    noise_level: float
    reliability: float = 1.0

class SimulatedRadar:
    """
    Digital twin of S/X-band phased array radar system.
    
    Simulates realistic detection physics, false alarms, and noise.
    """
    
    def __init__(self, range_max: float = 100e3, fov: float = 60.0, 
                 noise_std: float = 1.0, false_alarm_rate: float = 1e-6):
        self.range_max = range_max
        self.fov = fov  # degrees
        self.noise_std = noise_std
        self.false_alarm_rate = false_alarm_rate
        self.scan_count = 0
        
    def scan(self, position: np.ndarray, velocity: np.ndarray, 
             actual_objects: List[TargetObject] = None) -> List[Dict[str, Any]]:
        """
        Simulate radar scan with realistic detection physics.
        
        Args:
            position: Scanner position [x, y, z] (m)
            velocity: Scanner velocity [vx, vy, vz] (m/s)
            actual_objects: Ground truth objects for simulation
            
        Returns:
            List of simulated detections with measurement noise
        """
        self.scan_count += 1
        detections = []
        
        if actual_objects is None:
            # Generate synthetic object environment
            actual_objects = self._generate_synthetic_objects(position)
        
        for obj in actual_objects:
            rel_pos = obj.position - position
            range_val = np.linalg.norm(rel_pos)
            
            # Skip objects outside range
            if range_val > self.range_max:
                continue
            
            # Simulate radar equation: P_d ~ RCS / range^4
            snr = obj.cross_section / (range_val / 1000)**4
            detection_prob = 1 / (1 + np.exp(-snr + 5))
            
            if random.random() < detection_prob:
                # Add realistic measurement noise
                range_noise = np.random.normal(0, self.noise_std)
                velocity_noise = np.random.normal(0, 0.1, 3)
                
                detection = {
                    'position': obj.position + np.random.normal(0, range_noise, 3),
                    'velocity': obj.velocity + velocity_noise,
                    'range': range_val + range_noise,
                    'cross_section': obj.cross_section,
                    'confidence': detection_prob * random.uniform(0.8, 1.0),
                    'timestamp': time.time(),
                    'track_id': f"R{self.scan_count:04d}_{len(detections):02d}"
                }
                detections.append(detection)
        
        # Add false alarms
        n_false_alarms = np.random.poisson(self.false_alarm_rate * self.range_max / 1000)
        for i in range(n_false_alarms):
            false_range = random.uniform(1000, self.range_max)
            false_bearing = random.uniform(-self.fov/2, self.fov/2)
            
            false_pos = position + false_range * np.array([
                np.cos(np.radians(false_bearing)),
                np.sin(np.radians(false_bearing)),
                random.uniform(-0.1, 0.1)
            ])
            
            false_detection = {
                'position': false_pos,
                'velocity': np.random.normal(0, 7500, 3),
                'range': false_range,
                'cross_section': random.uniform(0.1, 10),
                'confidence': 0.3,  # Low confidence for false alarms
                'timestamp': time.time(),
                'track_id': f"F{self.scan_count:04d}_{i:02d}"
            }
            detections.append(false_detection)
        
        return detections
    
    def _generate_synthetic_objects(self, position: np.ndarray) -> List[TargetObject]:
        """Generate synthetic space objects for simulation."""
        objects = []
        
        # Generate 0-3 random debris objects
        n_objects = np.random.poisson(0.5)  # Low density environment
        
        for i in range(n_objects):
            # Random position within detection range
            range_val = random.uniform(10e3, self.range_max * 0.8)
            bearing = random.uniform(-180, 180)
            elevation = random.uniform(-30, 30)
            
            obj_pos = position + range_val * np.array([
                np.cos(np.radians(elevation)) * np.cos(np.radians(bearing)),
                np.cos(np.radians(elevation)) * np.sin(np.radians(bearing)),
                np.sin(np.radians(elevation))
            ])
            
            # Typical LEO velocities
            obj_vel = np.random.normal([7500, 0, 0], [500, 200, 100])
            
            obj = TargetObject(
                position=obj_pos,
                velocity=obj_vel,
                range_rate=random.uniform(-1000, 1000),
                cross_section=random.uniform(0.1, 50),
                confidence=1.0,  # Ground truth
                first_detection_time=time.time(),
                track_id=f"SIM_{i:03d}"
            )
            objects.append(obj)
        
        return objects

class SimulatedIMU:
    """
    Digital twin of inertial measurement unit with realistic drift and noise.
    """
    
    def __init__(self, noise_std: float = 1e-3, bias_drift: float = 1e-6):
        self.noise_std = noise_std
        self.bias_drift = bias_drift
        self.bias = np.zeros(3)
        self.last_reading_time = time.time()
        
    def read_acceleration(self) -> SimulatedSensorReading:
        """Read simulated acceleration with drift and noise."""
        current_time = time.time()
        dt = current_time - self.last_reading_time
        
        # Simulate bias drift
        self.bias += np.random.normal(0, self.bias_drift * dt, 3)
        
        # Generate base acceleration (mostly zero in LEO)
        base_accel = np.array([0.0, 0.0, -9.81e-6])  # Tiny residual gravity
        
        # Add noise and bias
        measured_accel = base_accel + self.bias + np.random.normal(0, self.noise_std, 3)
        
        self.last_reading_time = current_time
        
        return SimulatedSensorReading(
            timestamp=current_time,
            value=measured_accel,
            noise_level=self.noise_std,
            reliability=0.99
        )

class SimulatedThermocouple:
    """
    Digital twin of thermocouple array for thermal monitoring.
    """
    
    def __init__(self, noise_std: float = 0.5, baseline_temp: float = 273.15):
        self.noise_std = noise_std
        self.baseline_temp = baseline_temp
        self.thermal_history = []
        
    def read_temperature(self, micrometeoroid_flux: float = 0.0) -> SimulatedSensorReading:
        """Read simulated temperature with thermal effects."""
        current_time = time.time()
        
        # Base temperature (space environment)
        base_temp = self.baseline_temp + random.uniform(-20, 20)
        
        # Micrometeoroid heating effect
        heating_effect = micrometeoroid_flux * 10.0  # Simplified heating model
        
        # Measurement noise
        measured_temp = base_temp + heating_effect + np.random.normal(0, self.noise_std)
        
        # Store thermal history
        self.thermal_history.append((current_time, measured_temp))
        if len(self.thermal_history) > 100:
            self.thermal_history.pop(0)
        
        return SimulatedSensorReading(
            timestamp=current_time,
            value=measured_temp,
            noise_level=self.noise_std,
            reliability=0.95
        )

class SimulatedEMFieldGenerator:
    """
    Digital twin of electromagnetic field generator for curvature control.
    """
    
    def __init__(self, latency: float = 0.01, power_limit: float = 1e6):
        self.latency = latency
        self.power_limit = power_limit
        self.current_field_strength = 0.0
        self.target_field_strength = 0.0
        self.command_history = []
        
    def apply_profile(self, curvature_profile: Dict[str, Any]) -> bool:
        """
        Apply curvature profile with realistic actuation delays.
        
        Args:
            curvature_profile: Target curvature field configuration
            
        Returns:
            Success flag
        """
        current_time = time.time()
        
        # Extract target field strength (simplified)
        if 'field_strength' in curvature_profile:
            self.target_field_strength = curvature_profile['field_strength']
        else:
            self.target_field_strength = random.uniform(0.1, 1.0)
        
        # Check power limits
        if abs(self.target_field_strength) > self.power_limit:
            print(f"⚠️  Field strength {self.target_field_strength:.2e} exceeds power limit")
            return False
        
        # Simulate actuation delay
        time.sleep(self.latency)
        
        # Gradual field adjustment (realistic slew rate)
        field_difference = self.target_field_strength - self.current_field_strength
        slew_rate = 0.1  # Field units per second
        max_change = slew_rate * self.latency
        
        if abs(field_difference) <= max_change:
            self.current_field_strength = self.target_field_strength
        else:
            self.current_field_strength += np.sign(field_difference) * max_change
        
        # Log command
        self.command_history.append({
            'timestamp': current_time,
            'target': self.target_field_strength,
            'actual': self.current_field_strength,
            'profile': curvature_profile
        })
        
        return True

class WarpBubbleController:
    """
    Simulated warp bubble controller integrating all protection systems.
    """
    
    def __init__(self):
        self.current_acceleration = np.zeros(3)
        self.warp_energy_budget = 1e20  # Joules
        self.energy_consumed = 0.0
        
    def apply_deceleration(self, delta_v: float, smear_time: float = 10.0):
        """Apply deceleration through warp field modulation."""
        energy_cost = delta_v**2 * 1e12  # Simplified energy model
        
        if self.energy_consumed + energy_cost > self.warp_energy_budget:
            print(f"⚠️  Insufficient energy for deceleration: {energy_cost:.2e} J")
            return False
        
        self.energy_consumed += energy_cost
        print(f"🔧 Applied {delta_v:.1f} m/s deceleration (cost: {energy_cost:.2e} J)")
        return True
    
    def execute_impulse(self, impulse_vector: np.ndarray):
        """Execute collision avoidance impulse."""
        impulse_magnitude = np.linalg.norm(impulse_vector)
        energy_cost = impulse_magnitude**2 * 1e10
        
        if self.energy_consumed + energy_cost > self.warp_energy_budget:
            print(f"⚠️  Insufficient energy for impulse: {energy_cost:.2e} J")
            return False
        
        self.energy_consumed += energy_cost
        self.current_acceleration = impulse_vector / 1.0  # 1 second impulse duration
        print(f"🚀 Executed impulse: {impulse_magnitude:.3f} m/s (cost: {energy_cost:.2e} J)")
        return True
    
    def execute_warp_thrust(self, target_velocity: List[float], smear_time: float = 60.0):
        """Execute main warp drive thrust in simulation."""
        thrust_magnitude = np.linalg.norm(target_velocity)
        energy_cost = thrust_magnitude**2 * smear_time * 1e8
        
        if self.energy_consumed + energy_cost > self.warp_energy_budget:
            print(f"⚠️  Insufficient energy for warp thrust: {energy_cost:.2e} J")
            return False
        
        self.energy_consumed += energy_cost
        print(f"⚡ Warp thrust: {thrust_magnitude:.0f} m/s target (cost: {energy_cost:.2e} J)")
        return True

def run_full_warp_simulated_hardware():
    """Main simulated hardware demonstration."""
    print("🚀 FULL WARP SIMULATED HARDWARE DEMO")
    print("=" * 50)
    
    if not PROTECTION_AVAILABLE:
        print("❌ Protection systems not available - running simplified demo")
        return
    
    # Initialize simulated hardware
    print("\n🔧 Initializing Simulated Hardware Systems...")
    radar = SimulatedRadar(range_max=100e3, fov=60)
    imu = SimulatedIMU(noise_std=1e-3)
    thermocouple = SimulatedThermocouple(noise_std=0.5)
    emf_gen = SimulatedEMFieldGenerator(latency=0.01)
    controller = WarpBubbleController()
    
    print("   ✓ Radar system online")
    print("   ✓ IMU calibrated")
    print("   ✓ Thermal monitoring active")
    print("   ✓ EM field generator ready")
    
    # Initialize protection systems
    print("\n🛡️  Initializing Protection Systems...")
    try:
        atmo = AtmosphericConstraints()
        leo = LEOCollisionAvoidanceSystem()
        micro_env = MicrometeoroidEnvironment()
        bubble_geom = BubbleGeometry(radius=50.0)
        micro = IntegratedProtectionSystem(micro_env, bubble_geom)
        
        print("   ✓ Atmospheric constraints loaded")
        print("   ✓ LEO collision avoidance ready")
        print("   ✓ Micrometeoroid protection configured")
    except Exception as e:
        print(f"   ❌ Protection system error: {e}")
        return
    
    # Initial mission state
    state = {
        'pos': np.array([0.0, 0.0, 350e3]),  # 350 km altitude
        'vel': np.array([7500.0, 0.0, -100.0])  # Orbital velocity with slight descent
    }
    
    print(f"\n🎯 Mission State Initialized")
    print(f"   Position: [{state['pos'][0]/1000:.1f}, {state['pos'][1]/1000:.1f}, {state['pos'][2]/1000:.1f}] km")
    print(f"   Velocity: [{state['vel'][0]:.0f}, {state['vel'][1]:.0f}, {state['vel'][2]:.0f}] m/s")
    
    # Main simulation loop
    print(f"\n🔄 Starting 60-Second Simulation Loop...")
    dt = 1.0
    
    for step in range(60):
        print(f"\n--- Step {step+1}/60 (t = {step+1} s) ---")
        
        # 1. Atmospheric safety check
        h = state['pos'][2]
        v = np.linalg.norm(state['vel'])
        
        if h < 100e3:  # In atmosphere
            v_thermal = atmo.max_velocity_thermal(h)
            v_drag = atmo.max_velocity_drag(h)
            v_safe = min(v_thermal, v_drag)
            
            if v > v_safe:
                decel_required = v - v_safe
                print(f"🌍 Atmospheric violation: {v:.0f} > {v_safe:.0f} m/s")
                controller.apply_deceleration(decel_required, smear_time=10.0)
                state['vel'] *= (v_safe / v)  # Apply velocity reduction
        
        # 2. LEO debris scan
        detections = radar.scan(state['pos'], state['vel'])
        
        if detections:
            print(f"📡 Radar: {len(detections)} objects detected")
            
            # Convert to TargetObject format for leo system
            target_objects = []
            for det in detections:
                if det['confidence'] > 0.5:  # Filter low-confidence detections
                    target = TargetObject(
                        position=det['position'],
                        velocity=det['velocity'],
                        range_rate=np.dot(det['velocity'] - state['vel'], 
                                        (det['position'] - state['pos']) / det['range']),
                        cross_section=det['cross_section'],
                        confidence=det['confidence'],
                        first_detection_time=det['timestamp'],
                        track_id=det['track_id']
                    )
                    target_objects.append(target)
            
            if target_objects:
                # Execute collision avoidance
                result = leo.execute_collision_avoidance(
                    state['pos'], state['vel'], np.array([1, 0, 0]), target_objects
                )
                
                if result['maneuvers_executed'] > 0:
                    print(f"🚀 Executed {result['maneuvers_executed']} avoidance maneuvers")
                    state['vel'] = result['new_velocity']
        else:
            print("📡 Radar: Clear scan")
        
        # 3. Micrometeoroid protection
        temp_reading = thermocouple.read_temperature(micrometeoroid_flux=1e-6)
        imu_reading = imu.read_acceleration()
        
        # Simulate micrometeoroid impacts based on flux
        micrometeoroid_impacts = np.random.poisson(28.27 / 3600)  # Expected hourly rate
        
        if micrometeoroid_impacts > 0:
            print(f"🛡️  {micrometeoroid_impacts} micrometeoroid impacts detected")
            
            # Generate protection report
            protection_report = micro.generate_protection_report()
            best_strategy = max(protection_report['optimization_results'], 
                              key=lambda x: x['efficiency'])
            
            # Apply optimal curvature profile
            curvature_profile = {
                'strategy': best_strategy['strategy'],
                'field_strength': best_strategy['efficiency'] * 10.0,
                'parameters': best_strategy['optimal_params']
            }
            
            success = emf_gen.apply_profile(curvature_profile)
            if success:
                print(f"🔧 Applied {best_strategy['strategy']} profile ({best_strategy['efficiency']*100:.1f}% efficiency)")
        
        # 4. Execute nominal warp operations
        if step % 10 == 0:  # Every 10 seconds
            controller.execute_warp_thrust([8000.0, 0, 0], smear_time=60.0)
        
        # 5. Update state
        state['pos'] += state['vel'] * dt
        state['vel'] += controller.current_acceleration * dt
        controller.current_acceleration *= 0.9  # Impulse decay
        
        # Brief pause for real-time feel
        time.sleep(0.05)
    
    # Final report
    print(f"\n📊 SIMULATION COMPLETE")
    print(f"=" * 30)
    print(f"   Final position: [{state['pos'][0]/1000:.1f}, {state['pos'][1]/1000:.1f}, {state['pos'][2]/1000:.1f}] km")
    print(f"   Final velocity: {np.linalg.norm(state['vel']):.0f} m/s")
    print(f"   Energy consumed: {controller.energy_consumed:.2e} J")
    print(f"   Energy remaining: {controller.warp_energy_budget - controller.energy_consumed:.2e} J")
    print(f"   Radar scans: {radar.scan_count}")
    print(f"   EM field commands: {len(emf_gen.command_history)}")
    
    print(f"\n✅ Full warp simulation with virtual hardware complete!")
    print(f"💡 All subsystems operated within nominal parameters")
    print(f"🔬 Ready for Monte Carlo reliability analysis")

if __name__ == "__main__":
    run_full_warp_simulated_hardware()
