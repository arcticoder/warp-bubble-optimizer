#!/usr/bin/env python3
"""
Simulated Hardware Interfaces for Warp Bubble Digital Twin
=========================================================

Provides pure-Python "digital twin" implementations of key hardware interfaces
for warp bubble operations, including radar systems, IMUs, thermocouples, and
electromagnetic field generators. These simulate realistic sensor noise,
actuation delays, and hardware limitations without requiring physical hardware.

Used for validation of control algorithms and system integration testing
in the complete warp bubble protection and control system.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

@dataclass
class RadarDetection:
    """Simulated radar detection result."""
    range: float
    azimuth: float  
    elevation: float
    range_rate: float
    cross_section: float
    confidence: float
    timestamp: float

class SimulatedRadar:
    """
    Simulated S/X-band phased array radar system for space debris detection.
    Models realistic detection performance, noise characteristics, and limitations.
    """
    
    def __init__(self, range_max: float = 100e3, fov: float = 60.0, noise_std: float = 0.1):
        self.range_max = range_max
        self.fov = fov  # Field of view in degrees
        self.noise_std = noise_std
        self.last_scan_time = 0.0
        
    def scan(self, observer_pos: np.ndarray, observer_vel: np.ndarray, 
             scan_duration: float = 1.0) -> List[RadarDetection]:
        """
        Simulate a radar scan from the observer position.
        Generates synthetic debris detections based on realistic orbital parameters.
        """
        current_time = time.time()
        
        # Limit scan rate to realistic values
        if current_time - self.last_scan_time < 0.1:  # 10 Hz max
            return []
            
        self.last_scan_time = current_time
        
        # Generate 0-3 synthetic detections per scan
        n_detections = np.random.poisson(0.5)
        detections = []
        
        for _ in range(min(n_detections, 3)):
            # Generate realistic orbital debris parameters
            detection_range = np.random.uniform(10e3, self.range_max)
            azimuth = np.random.uniform(-self.fov/2, self.fov/2)
            elevation = np.random.uniform(-30, 30)
            
            # Simulate relative velocity (typical LEO: 7-15 km/s)
            range_rate = np.random.uniform(-500, -50)  # Approaching objects
            
            # Cross-section varies with object size
            cross_section = np.random.lognormal(mean=1.0, sigma=2.0)
            
            # Confidence decreases with range and small cross-sections
            base_confidence = 0.95
            range_factor = max(0.3, 1.0 - detection_range / self.range_max)
            size_factor = min(1.0, cross_section / 1.0)
            confidence = base_confidence * range_factor * size_factor
            
            # Add measurement noise
            range_noise = np.random.normal(0, self.noise_std * detection_range * 0.001)
            azimuth_noise = np.random.normal(0, self.noise_std * 0.5)
            elevation_noise = np.random.normal(0, self.noise_std * 0.5)
            range_rate_noise = np.random.normal(0, self.noise_std * 10)
            
            detection = RadarDetection(
                range=detection_range + range_noise,
                azimuth=azimuth + azimuth_noise,
                elevation=elevation + elevation_noise,
                range_rate=range_rate + range_rate_noise,
                cross_section=cross_section,
                confidence=max(0.1, confidence + np.random.normal(0, 0.05)),
                timestamp=current_time
            )
            detections.append(detection)
            
        return detections

class SimulatedIMU:
    """
    Simulated Inertial Measurement Unit for spacecraft attitude and acceleration.
    Models realistic noise characteristics and drift common in spacecraft IMUs.
    """
    
    def __init__(self, noise_std: float = 1e-3, drift_rate: float = 1e-6):
        self.noise_std = noise_std
        self.drift_rate = drift_rate
        self.bias = np.random.normal(0, drift_rate, 3)
        self.last_update = time.time()
        
    def read_acceleration(self) -> np.ndarray:
        """Read simulated acceleration vector in spacecraft body frame."""
        current_time = time.time()
        dt = current_time - self.last_update
        
        # Update bias due to drift
        self.bias += np.random.normal(0, self.drift_rate * dt, 3)
        self.last_update = current_time
        
        # Simulate spacecraft accelerations (mostly small perturbations)
        base_accel = np.random.normal(0, 1e-4, 3)  # Typical space environment
        noise = np.random.normal(0, self.noise_std, 3)
        
        return base_accel + self.bias + noise
        
    def read_angular_velocity(self) -> np.ndarray:
        """Read simulated angular velocity vector."""
        # Typical spacecraft angular rates are very small
        base_rate = np.random.normal(0, 1e-5, 3)
        noise = np.random.normal(0, self.noise_std * 0.1, 3)
        
        return base_rate + noise

class SimulatedThermocouple:
    """
    Simulated thermocouple for monitoring bubble boundary temperature.
    Models realistic thermal response and noise characteristics.
    """
    
    def __init__(self, noise_std: float = 0.5, response_time: float = 2.0):
        self.noise_std = noise_std
        self.response_time = response_time
        self.temperature = 293.15  # Room temperature start
        self.target_temp = 293.15
        self.last_update = time.time()
        
    def read_temperature(self) -> float:
        """Read current temperature with realistic thermal response."""
        current_time = time.time()
        dt = current_time - self.last_update
        
        # Update target temperature based on simulated environment
        # In space: radiative cooling vs solar heating vs internal heating
        ambient_temp = 2.7 + 200 * np.random.random()  # Space environment variation
        self.target_temp = ambient_temp + np.random.normal(0, 10)
        
        # First-order thermal response
        tau = self.response_time
        alpha = dt / (tau + dt)
        self.temperature = (1 - alpha) * self.temperature + alpha * self.target_temp
        
        # Add measurement noise
        noise = np.random.normal(0, self.noise_std)
        measured_temp = self.temperature + noise
        
        self.last_update = current_time
        return measured_temp

class SimulatedEMFieldGenerator:
    """
    Simulated electromagnetic field generator for curvature deflection.
    Models realistic actuation delays and power limitations.
    """
    
    def __init__(self, latency: float = 0.01, max_power: float = 1e6):
        self.latency = latency
        self.max_power = max_power
        self.current_profile = np.zeros(10)  # 10-element field profile
        self.command_queue = []
        self.last_command_time = 0.0
        
    def apply_profile(self, new_profile: np.ndarray) -> bool:
        """
        Apply a new electromagnetic field profile with realistic delays.
        Returns True if command accepted, False if rejected.
        """
        current_time = time.time()
        
        # Check rate limits (typical actuator response ~100 Hz)
        if current_time - self.last_command_time < 0.01:
            return False
            
        # Check power constraints
        profile_power = np.sum(new_profile**2)
        if profile_power > self.max_power:
            # Scale down to maximum available power
            new_profile = new_profile * np.sqrt(self.max_power / profile_power)
            
        # Add command to queue with latency
        command_time = current_time + self.latency + np.random.normal(0, 0.002)
        self.command_queue.append((command_time, new_profile.copy()))
        self.last_command_time = current_time
        
        # Process any ready commands
        self._process_command_queue(current_time)
        
        return True
        
    def _process_command_queue(self, current_time: float):
        """Process any commands that are ready to execute."""
        ready_commands = [cmd for cmd in self.command_queue if cmd[0] <= current_time]
        
        if ready_commands:
            # Take the most recent ready command
            latest_command = max(ready_commands, key=lambda x: x[0])
            self.current_profile = latest_command[1]
            
            # Remove processed commands
            self.command_queue = [cmd for cmd in self.command_queue if cmd[0] > current_time]
            
    def get_current_profile(self) -> np.ndarray:
        """Get the currently active field profile."""
        self._process_command_queue(time.time())
        return self.current_profile.copy()
        
    def get_status(self) -> Dict[str, Any]:
        """Get system status information."""
        return {
            'current_power': np.sum(self.current_profile**2),
            'max_power': self.max_power,
            'queue_length': len(self.command_queue),
            'latency': self.latency,
            'last_command_time': self.last_command_time
        }

# Utility functions for integration testing

def create_simulated_sensor_suite() -> Dict[str, Any]:
    """Create a complete simulated sensor suite for testing."""
    return {
        'radar': SimulatedRadar(range_max=100e3, fov=60.0),
        'imu': SimulatedIMU(noise_std=1e-3),
        'thermocouple': SimulatedThermocouple(noise_std=0.5),
        'em_generator': SimulatedEMFieldGenerator(latency=0.01)
    }

def simulate_sensor_fusion_data(sensor_suite: Dict[str, Any], 
                               spacecraft_state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate fused sensor data for spacecraft state estimation."""
    
    # Get individual sensor readings
    radar_data = sensor_suite['radar'].scan(
        spacecraft_state['position'], 
        spacecraft_state['velocity']
    )
    
    imu_accel = sensor_suite['imu'].read_acceleration()
    imu_gyro = sensor_suite['imu'].read_angular_velocity()
    temperature = sensor_suite['thermocouple'].read_temperature()
    em_status = sensor_suite['em_generator'].get_status()
    
    return {
        'radar_detections': radar_data,
        'acceleration': imu_accel,
        'angular_velocity': imu_gyro,
        'temperature': temperature,
        'em_field_status': em_status,
        'timestamp': time.time(),
        'sensor_health': {
            'radar': 'OPERATIONAL',
            'imu': 'OPERATIONAL', 
            'thermocouple': 'OPERATIONAL',
            'em_generator': 'OPERATIONAL'
        }
    }

if __name__ == "__main__":
    print("🔧 SIMULATED HARDWARE INTERFACE DEMO")
    print("=" * 50)
    
    # Create sensor suite
    sensors = create_simulated_sensor_suite()
    
    # Simulate spacecraft state
    spacecraft_state = {
        'position': np.array([0, 0, 350e3]),  # 350 km altitude
        'velocity': np.array([7500, 0, 0])    # Orbital velocity
    }
    
    print("\n📡 Testing Sensor Suite...")
    
    # Run sensor fusion for 10 seconds
    for i in range(10):
        fused_data = simulate_sensor_fusion_data(sensors, spacecraft_state)
        
        print(f"\nTime step {i+1}:")
        print(f"  Radar detections: {len(fused_data['radar_detections'])}")
        print(f"  Acceleration: [{fused_data['acceleration'][0]:.2e}, "
              f"{fused_data['acceleration'][1]:.2e}, {fused_data['acceleration'][2]:.2e}] m/s²")
        print(f"  Temperature: {fused_data['temperature']:.1f} K")
        print(f"  EM field power: {fused_data['em_field_status']['current_power']:.2e} W")
        
        # Test EM field generator with random profile
        test_profile = np.random.normal(0, 100, 10)
        success = sensors['em_generator'].apply_profile(test_profile)
        print(f"  EM command success: {success}")
        
        time.sleep(0.5)
    
    print("\n✅ Simulated hardware interface testing complete.")
    print("   All digital twin components operational.")
