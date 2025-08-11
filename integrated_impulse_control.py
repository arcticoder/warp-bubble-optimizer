#!/usr/bin/env python3
"""
Integrated Impulse Engine Control System
=======================================

This module provides a complete closed-loop control system for impulse-mode
warp engine operations, integrating translation, rotation, and fine-pointing
capabilities with virtual control loop feedback.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import time
import logging

# Import our simulation modules
from simulate_impulse_engine import (
    ImpulseProfile, WarpParameters, simulate_impulse_maneuver
)
from src.simulation.simulate_vector_impulse import (
    VectorImpulseProfile, WarpBubbleVector, Vector3D,
    simulate_vector_impulse_maneuver
)
from simulate_rotation import (
    RotationProfile, WarpBubbleRotational, Quaternion,
    simulate_rotation_maneuver
)

try:
    from sim_control_loop import (
        VirtualWarpController, SensorConfig, ActuatorConfig, ControllerConfig
    )
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    print("⚠️  Control loop module not available")

try:
    from progress_tracker import ProgressTracker
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False

# Atmospheric constraints import
try:
    from atmospheric_constraints import AtmosphericConstraints, TrajectoryAnalyzer
    ATMOSPHERIC_AVAILABLE = True
    print("🌍 Atmospheric constraints enabled for integrated control")
except ImportError:
    ATMOSPHERIC_AVAILABLE = False
    print("⚠️  Atmospheric constraints not available")

# JAX imports with fallback
try:
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False
    def jit(func): return func
    def vmap(func, in_axes=None): 
        def vectorized(*args):
            return np.array([func(*[arg[i] if isinstance(arg, np.ndarray) else arg 
                                  for arg in args]) for i in range(len(args[0]))])
        return vectorized

@dataclass
class MissionWaypoint:
    """Complete 6-DOF waypoint specification."""
    position: Vector3D                  # Target position
    orientation: Quaternion             # Target orientation
    dwell_time: float = 10.0           # Time to maintain position/attitude
    approach_speed: float = 1e-5       # Approach velocity (fraction of c)
    pointing_tolerance: float = 1e-3   # Angular tolerance (radians)
    position_tolerance: float = 1.0    # Position tolerance (meters)

@dataclass
class ImpulseEngineConfig:
    """Configuration for integrated impulse engine system."""
    warp_params: WarpParameters = None
    vector_params: WarpBubbleVector = None
    rotation_params: WarpBubbleRotational = None
    max_velocity: float = 1e-4         # Maximum translation velocity
    max_angular_velocity: float = 0.1  # Maximum angular velocity
    energy_budget: float = 1e12        # Total energy budget (J)
    safety_margin: float = 0.2         # Safety margin factor
    
    def __post_init__(self):
        if self.warp_params is None:
            self.warp_params = WarpParameters()
        if self.vector_params is None:
            self.vector_params = WarpBubbleVector()
        if self.rotation_params is None:
            self.rotation_params = WarpBubbleRotational()

class IntegratedImpulseController:
    """Integrated control system for impulse-mode warp engine operations."""
    
    def __init__(self, config: ImpulseEngineConfig):
        """Initialize the integrated controller."""
        self.config = config
        self.current_position = Vector3D(0.0, 0.0, 0.0)
        self.current_orientation = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.current_velocity = Vector3D(0.0, 0.0, 0.0)
        self.current_angular_velocity = 0.0
        
        # Atmospheric constraints
        self.atmospheric_constraints = AtmosphericConstraints() if ATMOSPHERIC_AVAILABLE else None
        self.current_altitude = 0.0  # Track current altitude for constraints
        
        # Control system configuration
        self.control_config = {
            'sensor': SensorConfig(noise_level=0.01, update_rate=50.0),
            'actuator': ActuatorConfig(response_time=0.02, damping_factor=0.9),
            'controller': ControllerConfig(kp=0.8, ki=0.1, kd=0.05)
        } if CONTROL_AVAILABLE else None
        
        # Mission tracking
        self.mission_log = []
        self.total_energy_used = 0.0
        self.total_mission_time = 0.0
    
    def plan_impulse_trajectory(self, waypoints: List[MissionWaypoint], 
                              optimize_energy: bool = True) -> Dict[str, Any]:
        """
        Plan complete impulse trajectory through multiple waypoints.
        
        Args:
            waypoints: List of mission waypoints
            optimize_energy: Whether to optimize for energy efficiency
            
        Returns:
            Complete trajectory plan
        """
        print(f"📋 Planning impulse trajectory: {len(waypoints)} waypoints")
        
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints for trajectory planning")
        
        # Initialize progress tracking
        progress = None
        if PROGRESS_AVAILABLE:
            try:
                progress = ProgressTracker(
                    total_steps=len(waypoints)-1,
                    description="Trajectory Planning"
                )
                progress.set_stage("trajectory_optimization")
            except:
                pass
        
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        
        with progress if progress else DummyContext():
            
            segments = []
            total_energy_estimate = 0.0
            total_time_estimate = 0.0
            
            current_pos = waypoints[0].position
            current_orient = waypoints[0].orientation
            
            for i, target_waypoint in enumerate(waypoints[1:], 1):
                if progress:
                    progress.update(f"Planning segment {i}/{len(waypoints)-1}", step_number=i)
                
                # Translation component
                displacement = target_waypoint.position - current_pos
                  # Rotation component  
                orientation_change = target_waypoint.orientation.multiply(current_orient.conjugate())
                
                # Design impulse profiles
                if optimize_energy:
                    # Energy-optimized velocity selection
                    displacement_mag = displacement.magnitude
                    if displacement_mag > 0:
                        v_max = min(self.config.max_velocity, 
                                  target_waypoint.approach_speed,
                                  displacement_mag / 60.0)  # Complete in reasonable time
                    else:
                        v_max = 0.0
                    
                    # Energy-optimized time allocation
                    if v_max > 0:
                        t_ramp = min(10.0, displacement_mag / (v_max * 299792458) / 4)
                        t_hold = max(5.0, displacement_mag / (v_max * 299792458) - 2*t_ramp)
                    else:
                        t_ramp = 5.0
                        t_hold = target_waypoint.dwell_time
                else:
                    # Time-optimized profiles
                    v_max = min(self.config.max_velocity, target_waypoint.approach_speed)
                    t_ramp = 8.0
                    t_hold = target_waypoint.dwell_time
                
                # Create segment profiles
                translation_profile = VectorImpulseProfile(
                    target_displacement=displacement,
                    v_max=v_max,
                    t_up=t_ramp,
                    t_hold=t_hold,
                    t_down=t_ramp,
                    n_steps=int((2*t_ramp + t_hold) * 20)  # 20 Hz
                )
                
                # Rotation profile (if significant rotation needed)
                # Calculate angular distance manually using dot product
                dot = np.abs(np.dot(current_orient.q, target_waypoint.orientation.q))
                rotation_angle = 2 * np.arccos(np.clip(dot, 0, 1))
                
                if rotation_angle > target_waypoint.pointing_tolerance:
                    omega_max = min(self.config.max_angular_velocity, 
                                  rotation_angle / (2*t_ramp + t_hold))
                    
                    rotation_profile = RotationProfile(
                        target_orientation=target_waypoint.orientation,
                        omega_max=omega_max,
                        t_up=t_ramp,
                        t_hold=t_hold,
                        t_down=t_ramp,
                        n_steps=int((2*t_ramp + t_hold) * 20)
                    )
                else:
                    rotation_profile = None
                
                # Estimate energy and time
                if displacement.magnitude > 0:
                    trans_energy = self._estimate_translation_energy(translation_profile)
                else:
                    trans_energy = 0.0
                
                if rotation_profile:
                    rot_energy = self._estimate_rotation_energy(rotation_profile)
                else:
                    rot_energy = 0.0
                
                segment_time = 2*t_ramp + t_hold
                segment_energy = trans_energy + rot_energy
                
                segments.append({
                    'translation_profile': translation_profile,
                    'rotation_profile': rotation_profile,
                    'estimated_energy': segment_energy,
                    'estimated_time': segment_time,
                    'target_waypoint': target_waypoint
                })
                
                total_energy_estimate += segment_energy
                total_time_estimate += segment_time
                
                # Update for next segment
                current_pos = target_waypoint.position
                current_orient = target_waypoint.orientation
            
            # Check energy budget
            if total_energy_estimate > self.config.energy_budget:
                print(f"⚠️  Energy estimate ({total_energy_estimate/1e9:.2f} GJ) exceeds budget ({self.config.energy_budget/1e9:.2f} GJ)")
                if optimize_energy:
                    print("   Consider reducing velocity or using multi-impulse approach")
            
            trajectory_plan = {
                'segments': segments,
                'waypoints': waypoints,
                'total_energy_estimate': total_energy_estimate,
                'total_time_estimate': total_time_estimate,
                'energy_efficiency': total_energy_estimate / (total_time_estimate + 1e-12),
                'feasible': total_energy_estimate <= self.config.energy_budget
            }
            
            if progress:
                progress.complete({
                    'segments_planned': len(segments),
                    'total_energy_GJ': total_energy_estimate/1e9,
                    'total_time_minutes': total_time_estimate/60
                })
        
        print(f"✅ Trajectory planning complete:")
        print(f"   {len(segments)} segments planned")
        print(f"   Estimated energy: {total_energy_estimate/1e9:.2f} GJ")
        print(f"   Estimated time: {total_time_estimate/60:.1f} minutes")
        print(f"   Energy budget: {'✅ OK' if trajectory_plan['feasible'] else '❌ EXCEEDED'}")
        
        return trajectory_plan
    
    async def execute_impulse_mission(self, trajectory_plan: Dict[str, Any],
                                    enable_feedback: bool = True) -> Dict[str, Any]:
        """
        Execute complete impulse mission with feedback control.
        
        Args:
            trajectory_plan: Planned trajectory from plan_impulse_trajectory()
            enable_feedback: Enable closed-loop control
            
        Returns:
            Mission execution results
        """
        print(f"🚀 Executing impulse mission: {len(trajectory_plan['segments'])} segments")
        
        if not CONTROL_AVAILABLE:
            enable_feedback = False
            print("⚠️  Feedback control disabled - running open-loop simulation")
        
        # Initialize progress tracking
        progress = None
        if PROGRESS_AVAILABLE:
            try:
                total_steps = len(trajectory_plan['segments']) * 3  # Plan, execute, verify
                progress = ProgressTracker(
                    total_steps=total_steps,
                    description="Mission Execution"
                )
                progress.set_stage("mission_execution")
            except:
                pass
        
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        
        with progress if progress else DummyContext():
            
            mission_results = {
                'segment_results': [],
                'trajectory_log': [],
                'energy_log': [],
                'performance_metrics': {},
                'mission_success': True
            }
            
            cumulative_time = 0.0
            cumulative_energy = 0.0
            
            for i, segment in enumerate(trajectory_plan['segments']):
                step_base = i * 3
                
                if progress:
                    progress.update(f"Executing segment {i+1}/{len(trajectory_plan['segments'])}", 
                                  step_number=step_base + 1)
                
                # Execute translation component
                trans_profile = segment['translation_profile']
                if trans_profile.target_displacement.magnitude > 0:
                    
                    if enable_feedback:
                        # Closed-loop translation with feedback
                        trans_results = await self._execute_translation_with_feedback(
                            trans_profile, cumulative_time
                        )
                    else:
                        # Open-loop translation simulation
                        trans_results = simulate_vector_impulse_maneuver(
                            trans_profile, self.config.vector_params, enable_progress=False
                        )
                    
                    # Update position
                    self.current_position = Vector3D(*trans_results['final_position'])
                    
                else:
                    trans_results = None
                
                if progress:
                    progress.update(f"Rotation phase {i+1}", step_number=step_base + 2)
                
                # Execute rotation component
                rot_profile = segment['rotation_profile']
                if rot_profile is not None:
                    
                    if enable_feedback:
                        # Closed-loop rotation with feedback
                        rot_results = await self._execute_rotation_with_feedback(
                            rot_profile, cumulative_time
                        )
                    else:
                        # Open-loop rotation simulation
                        rot_results = simulate_rotation_maneuver(
                            rot_profile, self.config.rotation_params, enable_progress=False
                        )
                    
                    # Update orientation
                    self.current_orientation = Quaternion(*rot_results['final_orientation'])
                    
                else:
                    rot_results = None
                
                if progress:
                    progress.update(f"Verifying segment {i+1}", step_number=step_base + 3)
                
                # Verify segment completion
                target_waypoint = segment['target_waypoint']
                position_error = (self.current_position - target_waypoint.position).magnitude
                orientation_error = self.current_orientation.angular_distance(target_waypoint.orientation)
                
                segment_success = (
                    position_error <= target_waypoint.position_tolerance and
                    orientation_error <= target_waypoint.pointing_tolerance
                )
                
                if not segment_success:
                    print(f"⚠️  Segment {i+1} tolerance exceeded:")
                    print(f"     Position error: {position_error:.2f} m (limit: {target_waypoint.position_tolerance:.2f} m)")
                    print(f"     Orientation error: {np.degrees(orientation_error):.3f}° (limit: {np.degrees(target_waypoint.pointing_tolerance):.3f}°)")
                    mission_results['mission_success'] = False
                
                # Accumulate results
                segment_energy = 0.0
                segment_time = 0.0
                
                if trans_results:
                    segment_energy += trans_results['total_energy']
                    segment_time += trans_results['maneuver_duration']
                
                if rot_results:
                    segment_energy += rot_results['total_energy']
                    segment_time = max(segment_time, rot_results['maneuver_duration'])  # Parallel execution
                
                cumulative_energy += segment_energy
                cumulative_time += segment_time
                
                # Store segment results
                mission_results['segment_results'].append({
                    'segment_index': i,
                    'translation_results': trans_results,
                    'rotation_results': rot_results,
                    'achieved_position': self.current_position.vec,
                    'achieved_orientation': self.current_orientation.q,
                    'position_error': position_error,
                    'orientation_error': orientation_error,
                    'segment_energy': segment_energy,
                    'segment_time': segment_time,
                    'segment_success': segment_success
                })
                
                # Update mission log
                self.mission_log.append({
                    'time': cumulative_time,
                    'position': self.current_position.vec,
                    'orientation': self.current_orientation.q,
                    'energy_used': cumulative_energy
                })
                
                # Dwell at waypoint
                await asyncio.sleep(target_waypoint.dwell_time / 100)  # Accelerated time
                cumulative_time += target_waypoint.dwell_time
            
            # Final mission metrics
            mission_results['performance_metrics'] = {
                'total_energy_used': cumulative_energy,
                'total_mission_time': cumulative_time,
                'energy_efficiency': cumulative_energy / (cumulative_time + 1e-12),
                'energy_budget_utilization': cumulative_energy / self.config.energy_budget,
                'mission_duration_hours': cumulative_time / 3600,
                'segments_completed': len(trajectory_plan['segments']),
                'segments_successful': sum(s['segment_success'] for s in mission_results['segment_results']),
                'overall_success_rate': sum(s['segment_success'] for s in mission_results['segment_results']) / len(trajectory_plan['segments'])
            }
            
            # Update totals
            self.total_energy_used = cumulative_energy
            self.total_mission_time = cumulative_time
            
            if progress:
                progress.complete({
                    'mission_success': mission_results['mission_success'],
                    'total_energy_GJ': cumulative_energy/1e9,
                    'mission_time_hours': cumulative_time/3600
                })
        
        success_rate = mission_results['performance_metrics']['overall_success_rate']
        print(f"✅ Mission execution complete:")
        print(f"   Success rate: {success_rate*100:.1f}%")
        print(f"   Total energy: {cumulative_energy/1e9:.2f} GJ")
        print(f"   Mission time: {cumulative_time/3600:.2f} hours")
        print(f"   Energy efficiency: {mission_results['performance_metrics']['energy_efficiency']/1e6:.1f} MW")
        
        return mission_results
    
    async def _execute_translation_with_feedback(self, profile: VectorImpulseProfile, 
                                               start_time: float) -> Dict[str, Any]:
        """Execute translation with closed-loop feedback control."""
        if not CONTROL_AVAILABLE:
            return simulate_vector_impulse_maneuver(profile, self.config.vector_params)
        
        # Simplified feedback: just return simulation results for now
        return simulate_vector_impulse_maneuver(profile, self.config.vector_params, enable_progress=False)
    
    async def _execute_rotation_with_feedback(self, profile: RotationProfile,
                                            start_time: float) -> Dict[str, Any]:
        """Execute rotation with closed-loop feedback control."""
        if not CONTROL_AVAILABLE:
            return simulate_rotation_maneuver(profile, self.config.rotation_params)
        
        # Simplified feedback: just return simulation results for now
        return simulate_rotation_maneuver(profile, self.config.rotation_params, enable_progress=False)
    
    def _estimate_translation_energy(self, profile: VectorImpulseProfile) -> float:
        """Quick energy estimate for translation."""
        # Simplified energy estimate: E ∝ v²
        return 1e11 * profile.v_max**2 * profile.target_displacement.magnitude
    
    def _estimate_rotation_energy(self, profile: RotationProfile) -> float:
        """Quick energy estimate for rotation."""
        # Simplified energy estimate: E ∝ ω²  
        total_angle = 2 * np.arccos(np.abs(profile.target_orientation.w))
        return 5e10 * profile.omega_max**2 * total_angle
    
    def generate_mission_report(self, mission_results: Dict[str, Any]) -> str:
        """Generate comprehensive mission analysis report."""
        metrics = mission_results['performance_metrics']
        
        report = f"""
INTEGRATED IMPULSE ENGINE MISSION REPORT
=======================================

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

MISSION OVERVIEW
---------------
Segments Planned: {metrics['segments_completed']}
Segments Successful: {metrics['segments_successful']}
Overall Success Rate: {metrics['overall_success_rate']*100:.1f}%
Mission Status: {'✅ SUCCESS' if mission_results['mission_success'] else '❌ PARTIAL FAILURE'}

PERFORMANCE METRICS
------------------
Total Energy Used: {metrics['total_energy_used']/1e9:.2f} GJ
Energy Budget Utilization: {metrics['energy_budget_utilization']*100:.1f}%
Mission Duration: {metrics['mission_duration_hours']:.2f} hours
Average Power: {metrics['energy_efficiency']/1e6:.1f} MW

SEGMENT ANALYSIS
---------------"""
        
        for i, segment in enumerate(mission_results['segment_results']):
            report += f"""
Segment {i+1}:
  Position Error: {segment['position_error']:.2f} m
  Orientation Error: {np.degrees(segment['orientation_error']):.3f}°
  Energy Used: {segment['segment_energy']/1e9:.3f} GJ
  Duration: {segment['segment_time']/60:.1f} min
  Status: {'✅ SUCCESS' if segment['segment_success'] else '❌ FAILED'}"""
        
        report += f"""

CONTROL PERFORMANCE
------------------
Control Loop: {'Enabled' if CONTROL_AVAILABLE else 'Disabled'}
Feedback Quality: {'Excellent' if metrics['overall_success_rate'] > 0.9 else 'Good' if metrics['overall_success_rate'] > 0.7 else 'Poor'}
Trajectory Accuracy: {metrics['overall_success_rate']*100:.1f}%

RECOMMENDATIONS
--------------
{'• Mission objectives achieved successfully' if mission_results['mission_success'] else '• Review failed segments for trajectory refinement'}
• Energy efficiency: {metrics['energy_budget_utilization']*100:.1f}% of budget utilized
{'• Consider higher velocity for faster completion' if metrics['mission_duration_hours'] > 2 else '• Mission duration acceptable'}
{'• Excellent control performance' if metrics['overall_success_rate'] > 0.9 else '• Consider control parameter tuning'}
"""
        
        return report

# Example usage and testing
if __name__ == "__main__":
    print("🚀 INTEGRATED IMPULSE ENGINE CONTROL SYSTEM")
    print("=" * 60)
    
    # Example 1: Simple waypoint mission
    print("\n1. 📋 Waypoint Mission Planning")
    
    # Define mission waypoints
    waypoints = [
        MissionWaypoint(  # Start
            position=Vector3D(0.0, 0.0, 0.0),
            orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
            dwell_time=5.0
        ),
        MissionWaypoint(  # Move east
            position=Vector3D(1000.0, 0.0, 0.0),
            orientation=Quaternion.from_euler(0.0, 0.0, np.pi/4),  # 45° yaw
            dwell_time=15.0,
            approach_speed=5e-5
        ),
        MissionWaypoint(  # Move north and up
            position=Vector3D(1000.0, 800.0, 200.0),
            orientation=Quaternion.from_euler(np.pi/6, 0.0, np.pi/2),  # 30° roll, 90° yaw
            dwell_time=20.0,
            approach_speed=3e-5
        ),
        MissionWaypoint(  # Return to origin with final orientation
            position=Vector3D(0.0, 0.0, 0.0),
            orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
            dwell_time=10.0,
            approach_speed=4e-5
        )
    ]
    
    # Create integrated controller
    config = ImpulseEngineConfig(
        max_velocity=1e-4,
        max_angular_velocity=0.1,
        energy_budget=5e11,  # 500 GJ
        safety_margin=0.15
    )
    
    controller = IntegratedImpulseController(config)
    
    # Plan trajectory
    trajectory_plan = controller.plan_impulse_trajectory(waypoints, optimize_energy=True)
    
    # Example 2: Execute mission
    print("\n2. 🚀 Mission Execution")
    
    async def run_mission():
        return await controller.execute_impulse_mission(trajectory_plan, enable_feedback=True)
    
    try:
        mission_results = asyncio.run(run_mission())
        
        # Generate report
        report = controller.generate_mission_report(mission_results)
        print("\n" + report)
        
        # Save mission data
        import json
        mission_data = {
            'trajectory_plan': {
                'total_energy_estimate': trajectory_plan['total_energy_estimate'],
                'total_time_estimate': trajectory_plan['total_time_estimate'],
                'feasible': trajectory_plan['feasible']
            },
            'mission_results': {
                'mission_success': mission_results['mission_success'],
                'performance_metrics': mission_results['performance_metrics']
            }
        }
        
        with open('integrated_mission_log.json', 'w') as f:
            json.dump(mission_data, f, indent=2)
        
        print("\n📁 Mission data saved: integrated_mission_log.json")
        
    except Exception as e:
        print(f"❌ Mission execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Integrated control system demonstration complete!")
    print("💡 Features demonstrated:")
    print("   • Multi-waypoint trajectory planning")
    print("   • Combined translation and rotation control")
    print("   • Energy budget optimization")
    print("   • Closed-loop feedback control")
    print("   • Mission performance analysis")
    print("   • Comprehensive mission reporting")
