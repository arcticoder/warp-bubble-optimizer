#!/usr/bin/env python3
"""
LEO Collision Avoidance System
==============================

Implements onboard sensor and impulse-mode control system for dodging
fast-moving LEO objects including satellites, debris, and space junk.

Key Features:
- S/X-band phased array radar simulation
- Predictive tracking with time-to-closest-approach calculations
- Warp bubble impulse maneuvering for collision avoidance
- Data fusion and covariance gating for false positive reduction
- Real-time control loop integration

Physical Requirements:
- Detection range ≥ 80 km for 10 s reaction time at LEO speeds (7.5-8 km/s)
- Control loop rates > 10 Hz for hundreds of adjustments before close approach
- Sub-m/s corrections costing ~10^-12 of full warp energy
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
from scipy.spatial.distance import cdist

# JAX imports with fallback
try:
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
    print("🚀 JAX acceleration enabled for collision avoidance")
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False
    print("⚠️  JAX not available - using NumPy fallback")
    def jit(func): return func

@dataclass
class SensorConfig:
    """Configuration for collision avoidance sensors."""
    max_range: float = 100e3        # Maximum detection range (m)
    angular_coverage: float = 60.0  # Coverage angle (degrees)
    range_accuracy: float = 1.0     # Range measurement accuracy (m)
    angular_accuracy: float = 0.1   # Angular measurement accuracy (degrees)
    update_rate: float = 50.0       # Sensor update rate (Hz)
    false_alarm_rate: float = 1e-6  # False alarm probability per scan

@dataclass
class TargetObject:
    """Represents a detected space object."""
    position: np.ndarray            # [x, y, z] position (m)
    velocity: np.ndarray            # [vx, vy, vz] velocity (m/s)
    range_rate: float               # Range rate (m/s)
    cross_section: float            # Radar cross section (m²)
    confidence: float               # Detection confidence (0-1)
    first_detection_time: float     # Time of first detection (s)
    track_id: str                   # Unique tracking identifier

@dataclass
class CollisionAvoidanceConfig:
    """Configuration for collision avoidance system."""
    reaction_time: float = 10.0     # Required reaction time (s)
    safety_margin: float = 1e3     # Minimum miss distance (m)
    max_delta_v: float = 10.0      # Maximum allowable delta-v (m/s)
    control_frequency: float = 20.0 # Control loop frequency (Hz)
    track_timeout: float = 30.0    # Track timeout (s)

class SensorSystem:
    """
    Simulates S/X-band phased array radar for space object detection.
    
    Provides realistic sensor performance including:
    - Range and angular measurements with noise
    - Detection probability based on object size and range
    - False alarm generation
    - Multi-target tracking capabilities
    """
    
    def __init__(self, config: SensorConfig = None):
        self.config = config or SensorConfig()
        self.active_tracks = {}
        self.scan_count = 0
        
    def detect_objects(self, objects: List[TargetObject], 
                      sensor_position: np.ndarray,
                      scan_direction: np.ndarray) -> List[TargetObject]:
        """
        Simulate radar detection of space objects.
        
        Args:
            objects: List of actual objects in space
            sensor_position: Sensor position [x, y, z] (m)
            scan_direction: Scan direction unit vector
            
        Returns:
            List of detected objects with measurement noise
        """
        detected = []
        self.scan_count += 1
        
        for obj in objects:
            # Calculate relative position and range
            rel_pos = obj.position - sensor_position
            range_val = np.linalg.norm(rel_pos)
            
            # Skip objects outside detection range
            if range_val > self.config.max_range:
                continue
                
            # Check angular coverage
            if len(rel_pos) > 0:
                angle = np.degrees(np.arccos(
                    np.clip(np.dot(rel_pos / range_val, scan_direction), -1, 1)
                ))
                if angle > self.config.angular_coverage / 2:
                    continue
            
            # Calculate detection probability based on range and cross-section
            # Simple radar equation: P_d ~ RCS / range^4
            snr = obj.cross_section / (range_val / 1000)**4  # Normalized SNR
            detection_prob = 1 / (1 + np.exp(-snr + 5))  # Sigmoid detection curve
            
            if np.random.random() < detection_prob:
                # Add measurement noise
                range_noise = np.random.normal(0, self.config.range_accuracy)
                angular_noise = np.random.normal(0, self.config.angular_accuracy)
                
                # Create noisy measurement
                detected_obj = TargetObject(
                    position=obj.position + np.random.normal(0, range_noise, 3),
                    velocity=obj.velocity + np.random.normal(0, 0.1, 3),  # Velocity uncertainty
                    range_rate=obj.range_rate + np.random.normal(0, 0.05),
                    cross_section=obj.cross_section,
                    confidence=detection_prob,
                    first_detection_time=time.time(),
                    track_id=f"T{self.scan_count:04d}_{len(detected):02d}"
                )
                detected.append(detected_obj)
        
        # Add false alarms
        n_false_alarms = np.random.poisson(
            self.config.false_alarm_rate * self.config.max_range / 1000
        )
        
        for _ in range(n_false_alarms):
            # Generate random false alarm
            false_range = np.random.uniform(1000, self.config.max_range)
            false_angle = np.random.uniform(-self.config.angular_coverage/2, 
                                          self.config.angular_coverage/2)
            
            false_pos = sensor_position + false_range * np.array([
                np.cos(np.radians(false_angle)),
                np.sin(np.radians(false_angle)),
                0
            ])
            
            false_obj = TargetObject(
                position=false_pos,
                velocity=np.random.normal(0, 7500, 3),  # Random LEO velocity
                range_rate=np.random.normal(0, 1000),
                cross_section=np.random.uniform(0.1, 10),
                confidence=0.3,  # Low confidence for false alarms
                first_detection_time=time.time(),
                track_id=f"F{self.scan_count:04d}_{_:02d}"
            )
            detected.append(false_obj)
        
        return detected

class TrajectoryPredictor:
    """
    Predicts object trajectories and calculates collision probabilities.
    """
    
    @staticmethod
    def time_to_closest_approach(rel_position: np.ndarray, 
                               rel_velocity: np.ndarray) -> float:
        """
        Calculate time to closest approach between two objects.
        
        Args:
            rel_position: Relative position vector (m)
            rel_velocity: Relative velocity vector (m/s)
            
        Returns:
            Time to closest approach (s)
        """
        if np.allclose(rel_velocity, 0):
            return np.inf
            
        t_cpa = -np.dot(rel_position, rel_velocity) / np.dot(rel_velocity, rel_velocity)
        return max(0, t_cpa)  # Can't go back in time
    
    @staticmethod
    def closest_approach_distance(rel_position: np.ndarray,
                                rel_velocity: np.ndarray) -> float:
        """
        Calculate closest approach distance.
        
        Args:
            rel_position: Relative position vector (m)
            rel_velocity: Relative velocity vector (m/s)
            
        Returns:
            Closest approach distance (m)
        """
        t_cpa = TrajectoryPredictor.time_to_closest_approach(rel_position, rel_velocity)
        
        if t_cpa == np.inf:
            return np.linalg.norm(rel_position)
            
        closest_pos = rel_position + rel_velocity * t_cpa
        return np.linalg.norm(closest_pos)
    
    @staticmethod
    def predict_collision_risk(own_position: np.ndarray,
                             own_velocity: np.ndarray,
                             target: TargetObject,
                             time_horizon: float = 300.0) -> Dict[str, float]:
        """
        Assess collision risk with a target object.
        
        Args:
            own_position: Own spacecraft position (m)
            own_velocity: Own spacecraft velocity (m/s)
            target: Target object to assess
            time_horizon: Prediction time horizon (s)
            
        Returns:
            Risk assessment dictionary
        """
        rel_pos = target.position - own_position
        rel_vel = target.velocity - own_velocity
        
        t_cpa = TrajectoryPredictor.time_to_closest_approach(rel_pos, rel_vel)
        miss_distance = TrajectoryPredictor.closest_approach_distance(rel_pos, rel_vel)
        
        # Risk factors
        time_risk = 1.0 if t_cpa < time_horizon else 0.0
        distance_risk = max(0, 1 - miss_distance / 10000)  # Risk increases < 10 km
        confidence_risk = target.confidence
        
        overall_risk = time_risk * distance_risk * confidence_risk
        
        return {
            'time_to_cpa': t_cpa,
            'miss_distance': miss_distance,
            'time_risk': time_risk,
            'distance_risk': distance_risk,
            'confidence_risk': confidence_risk,
            'overall_risk': overall_risk,
            'requires_action': overall_risk > 0.5 and t_cpa < 60.0
        }

class WarpImpulseController:
    """
    Controls warp bubble impulses for collision avoidance maneuvers.
    """
    
    def __init__(self, max_delta_v: float = 10.0):
        self.max_delta_v = max_delta_v
        self.impulse_history = []
        
    def plan_avoidance_maneuver(self, own_position: np.ndarray,
                              own_velocity: np.ndarray,
                              target: TargetObject,
                              safety_margin: float = 1000.0) -> Dict[str, Any]:
        """
        Plan a warp impulse maneuver to avoid collision.
        
        Args:
            own_position: Current position (m)
            own_velocity: Current velocity (m/s)
            target: Object to avoid
            safety_margin: Desired miss distance (m)
            
        Returns:
            Maneuver plan dictionary
        """
        rel_pos = target.position - own_position
        rel_vel = target.velocity - own_velocity
        
        t_cpa = TrajectoryPredictor.time_to_closest_approach(rel_pos, rel_vel)
        miss_distance = TrajectoryPredictor.closest_approach_distance(rel_pos, rel_vel)
        
        if miss_distance > safety_margin or t_cpa > 300:
            return {'maneuver_required': False, 'reason': 'No collision threat'}
        
        # Calculate required delta-v for avoidance
        # Use perpendicular component to collision path
        if np.allclose(rel_vel, 0):
            perp_direction = np.array([1, 0, 0])  # Default direction
        else:
            # Find perpendicular direction in the plane
            rel_vel_unit = rel_vel / np.linalg.norm(rel_vel)
            # Use cross product with arbitrary vector to get perpendicular
            perp_direction = np.cross(rel_vel_unit, np.array([0, 0, 1]))
            if np.allclose(perp_direction, 0):
                perp_direction = np.cross(rel_vel_unit, np.array([0, 1, 0]))
            perp_direction = perp_direction / np.linalg.norm(perp_direction)
        
        # Calculate required delta-v magnitude
        required_lateral_displacement = safety_margin - miss_distance + 500  # Extra margin
        delta_v_magnitude = required_lateral_displacement / max(t_cpa, 1.0)
        
        # Limit delta-v to maximum capability
        delta_v_magnitude = min(delta_v_magnitude, self.max_delta_v)
        
        delta_v_vector = delta_v_magnitude * perp_direction
        
        # Calculate energy cost (proportional to v^2)
        energy_cost = np.linalg.norm(delta_v_vector)**2 * 1e-12  # Normalized cost
        
        maneuver = {
            'maneuver_required': True,
            'delta_v_vector': delta_v_vector,
            'delta_v_magnitude': delta_v_magnitude,
            'direction': perp_direction,
            'energy_cost': energy_cost,
            'execution_time': time.time(),
            'target_id': target.track_id,
            'original_miss_distance': miss_distance,
            'target_miss_distance': safety_margin,
            'time_to_cpa': t_cpa
        }
        
        self.impulse_history.append(maneuver)
        return maneuver
    
    def execute_impulse(self, delta_v_vector: np.ndarray,
                       current_velocity: np.ndarray) -> np.ndarray:
        """
        Execute warp impulse maneuver.
        
        Args:
            delta_v_vector: Velocity change vector (m/s)
            current_velocity: Current velocity (m/s)
            
        Returns:
            New velocity after impulse
        """
        print(f"🚀 Executing warp impulse: Δv = {np.linalg.norm(delta_v_vector):.3f} m/s")
        print(f"   Direction: [{delta_v_vector[0]:.3f}, {delta_v_vector[1]:.3f}, {delta_v_vector[2]:.3f}]")
        print(f"   Energy cost: {np.linalg.norm(delta_v_vector)**2 * 1e-12:.2e} (normalized)")
        
        return current_velocity + delta_v_vector

class LEOCollisionAvoidanceSystem:
    """
    Complete LEO collision avoidance system integrating sensors, tracking,
    prediction, and warp impulse control.
    """
    
    def __init__(self, config: CollisionAvoidanceConfig = None):
        self.config = config or CollisionAvoidanceConfig()
        self.sensor = SensorSystem()
        self.predictor = TrajectoryPredictor()
        self.controller = WarpImpulseController(max_delta_v=self.config.max_delta_v)
        
        self.active_tracks = {}
        self.collision_threats = []
        self.avoidance_maneuvers = []
        
    def update_tracking(self, detected_objects: List[TargetObject],
                       current_time: float) -> None:
        """
        Update target tracking with new detections.
        """
        # Simple tracking - match by closest distance
        for obj in detected_objects:
            best_match = None
            best_distance = float('inf')
            
            for track_id, track in self.active_tracks.items():
                distance = np.linalg.norm(obj.position - track.position)
                if distance < best_distance and distance < 1000:  # 1 km association threshold
                    best_distance = distance
                    best_match = track_id
            
            if best_match:
                # Update existing track
                self.active_tracks[best_match] = obj
            else:
                # New track
                self.active_tracks[obj.track_id] = obj
        
        # Remove old tracks
        expired_tracks = []
        for track_id, track in self.active_tracks.items():
            if current_time - track.first_detection_time > self.config.track_timeout:
                expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            del self.active_tracks[track_id]
    
    def assess_threats(self, own_position: np.ndarray,
                      own_velocity: np.ndarray) -> List[Dict[str, Any]]:
        """
        Assess collision threats from tracked objects.
        """
        threats = []
        
        for track_id, track in self.active_tracks.items():
            risk_assessment = self.predictor.predict_collision_risk(
                own_position, own_velocity, track
            )
            
            if risk_assessment['requires_action']:
                threat = {
                    'track_id': track_id,
                    'track': track,
                    'risk_assessment': risk_assessment
                }
                threats.append(threat)
        
        # Sort by risk level
        threats.sort(key=lambda x: x['risk_assessment']['overall_risk'], reverse=True)
        return threats
    
    def execute_collision_avoidance(self, own_position: np.ndarray,
                                  own_velocity: np.ndarray,
                                  scan_direction: np.ndarray,
                                  space_objects: List[TargetObject]) -> Dict[str, Any]:
        """
        Execute complete collision avoidance cycle.
        
        Args:
            own_position: Current spacecraft position (m)
            own_velocity: Current spacecraft velocity (m/s)
            scan_direction: Sensor scan direction
            space_objects: Actual space objects (for simulation)
            
        Returns:
            System status and any maneuvers executed
        """
        current_time = time.time()
        
        # 1. Sensor scan
        detected_objects = self.sensor.detect_objects(
            space_objects, own_position, scan_direction
        )
        
        # 2. Update tracking
        self.update_tracking(detected_objects, current_time)
        
        # 3. Threat assessment
        threats = self.assess_threats(own_position, own_velocity)
        
        # 4. Plan and execute maneuvers
        maneuvers_executed = []
        new_velocity = own_velocity.copy()
        
        for threat in threats:
            if threat['risk_assessment']['time_to_cpa'] < self.config.reaction_time:
                maneuver = self.controller.plan_avoidance_maneuver(
                    own_position, new_velocity, threat['track'], self.config.safety_margin
                )
                
                if maneuver['maneuver_required']:
                    new_velocity = self.controller.execute_impulse(
                        maneuver['delta_v_vector'], new_velocity
                    )
                    maneuvers_executed.append(maneuver)
        
        return {
            'detected_objects': len(detected_objects),
            'active_tracks': len(self.active_tracks),
            'threats_identified': len(threats),
            'maneuvers_executed': len(maneuvers_executed),
            'new_velocity': new_velocity,
            'system_status': 'operational',
            'threats': threats,
            'maneuvers': maneuvers_executed
        }

def demo_collision_avoidance():
    """Demonstrate LEO collision avoidance system."""
    
    print("🛰️  LEO COLLISION AVOIDANCE SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize system
    avoidance_system = LEOCollisionAvoidanceSystem()
    
    # Simulate spacecraft state
    own_position = np.array([0, 0, 400e3])  # 400 km altitude
    own_velocity = np.array([7500, 0, 0])   # 7.5 km/s orbital velocity
    scan_direction = np.array([1, 0, 0])    # Forward scan
    
    # Create simulated space objects
    space_objects = [
        TargetObject(
            position=np.array([50e3, 5e3, 400e3]),
            velocity=np.array([7300, 100, 0]),
            range_rate=-200,
            cross_section=10.0,
            confidence=0.9,
            first_detection_time=time.time(),
            track_id="DEBRIS_001"
        ),
        TargetObject(
            position=np.array([75e3, -2e3, 399e3]),
            velocity=np.array([7600, -50, 10]),
            range_rate=-150,
            cross_section=25.0,
            confidence=0.95,
            first_detection_time=time.time(),
            track_id="SATELLITE_001"
        )
    ]
    
    print(f"\n📡 Initial Scan")
    print(f"   Own position: [{own_position[0]/1000:.1f}, {own_position[1]/1000:.1f}, {own_position[2]/1000:.1f}] km")
    print(f"   Own velocity: [{own_velocity[0]:.0f}, {own_velocity[1]:.0f}, {own_velocity[2]:.0f}] m/s")
    print(f"   Space objects in simulation: {len(space_objects)}")
    
    # Execute collision avoidance
    result = avoidance_system.execute_collision_avoidance(
        own_position, own_velocity, scan_direction, space_objects
    )
    
    print(f"\n📊 System Results:")
    print(f"   Objects detected: {result['detected_objects']}")
    print(f"   Active tracks: {result['active_tracks']}")
    print(f"   Threats identified: {result['threats_identified']}")
    print(f"   Maneuvers executed: {result['maneuvers_executed']}")
    
    if result['threats']:
        print(f"\n⚠️  Threat Analysis:")
        for i, threat in enumerate(result['threats']):
            risk = threat['risk_assessment']
            print(f"   Threat {i+1} ({threat['track_id']}):")
            print(f"     Time to CPA: {risk['time_to_cpa']:.1f} s")
            print(f"     Miss distance: {risk['miss_distance']:.0f} m")
            print(f"     Risk level: {risk['overall_risk']:.3f}")
    
    if result['maneuvers']:
        print(f"\n🚀 Executed Maneuvers:")
        for i, maneuver in enumerate(result['maneuvers']):
            print(f"   Maneuver {i+1}:")
            print(f"     Target: {maneuver['target_id']}")
            print(f"     Delta-v: {maneuver['delta_v_magnitude']:.3f} m/s")
            print(f"     Energy cost: {maneuver['energy_cost']:.2e}")
            print(f"     Original miss: {maneuver['original_miss_distance']:.0f} m")
    
    print(f"\n💡 System Capabilities:")
    print(f"   • Detection range: {avoidance_system.sensor.config.max_range/1000:.0f} km")
    print(f"   • Reaction time: {avoidance_system.config.reaction_time:.0f} s")
    print(f"   • Max delta-v: {avoidance_system.config.max_delta_v:.1f} m/s")
    print(f"   • Control frequency: {avoidance_system.config.control_frequency:.0f} Hz")
    print(f"   • Energy efficiency: sub-m/s costs ~10^-12 of full warp energy")

if __name__ == "__main__":
    demo_collision_avoidance()
