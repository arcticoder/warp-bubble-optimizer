#!/usr/bin/env python3
"""
Integrated Warp Engine Control System
====================================

This module integrates impulse engine simulation, vectorized translation,
rotation control, and the VirtualWarpController into a unified control
system for complex 6-DOF warp bubble maneuvers.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
import json

# JAX imports with fallback
try:
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    JAX_AVAILABLE = True
    print("🚀 JAX acceleration enabled for integrated control system")
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False
    print("⚠️  JAX not available - using NumPy fallback")
    # Fallback decorators
    def jit(func): return func
    def vmap(func, in_axes=None): 
        def vectorized(*args):
            return np.array([func(*[arg[i] if isinstance(arg, np.ndarray) else arg 
                                  for arg in args]) for i in range(len(args[0]))])
        return vectorized
    def grad(func): 
        def grad_func(x, *args):
            h = 1e-8
            return (func(x + h, *args) - func(x - h, *args)) / (2 * h)
        return grad_func

# Import simulation modules with fallback
try:
    from simulate_impulse_engine import simulate_impulse_maneuver, ImpulseProfile, WarpParameters
    from simulate_vector_impulse import simulate_vector_impulse_maneuver, VectorImpulseProfile, WarpBubbleVector, Vector3D
    from simulate_rotation import simulate_rotation_maneuver, RotationProfile, WarpBubbleRotational, Quaternion
    SIMULATION_MODULES_AVAILABLE = True
except ImportError:
    SIMULATION_MODULES_AVAILABLE = False
    print("⚠️  Simulation modules not available - using mock implementations")
    
    # Mock classes for fallback
    class Vector3D:
        def __init__(self, x=0, y=0, z=0): 
            self.x, self.y, self.z = x, y, z
            self.vec = np.array([x, y, z])
        def magnitude(self): return np.linalg.norm(self.vec)
        def __str__(self): return f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    class Quaternion:
        def __init__(self, w=1, x=0, y=0, z=0): 
            self.w, self.x, self.y, self.z = w, x, y, z
        def __str__(self): return f"Q({self.w:.3f}, {self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

# ProgressTracker import with fallback
try:
    from progress_tracker import ProgressTracker
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False
    class ProgressTracker:
        def __init__(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def set_stage(self, *args, **kwargs): pass
        def complete(self, *args, **kwargs): pass
        def log_metric(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

# VirtualWarpController import with fallback
try:
    from enhanced_virtual_control_loop import VirtualWarpController
    CONTROL_AVAILABLE = True
except ImportError:
    try:
        from sim_control_loop import VirtualWarpController
        CONTROL_AVAILABLE = True
    except ImportError:
        CONTROL_AVAILABLE = False
        print("⚠️  VirtualWarpController not available - using mock implementation")
        
        class VirtualWarpController:
            def __init__(self, *args, **kwargs): 
                self.trajectory_log = []
            def execute_maneuver(self, *args, **kwargs): 
                return {'status': 'mock', 'energy': 1e9}

@dataclass
class MissionObjective:
    """High-level mission objective specification."""
    objective_type: str = "translation"        # "translation", "rotation", "combined", "waypoint"
    target_position: Vector3D = None          # Target position (m)
    target_orientation: Quaternion = None     # Target orientation
    waypoints: List[Vector3D] = None          # For multi-waypoint missions
    time_constraint: float = 120.0            # Maximum mission time (s)
    energy_constraint: float = 1e12          # Maximum energy budget (J)
    accuracy_requirement: float = 0.95       # Minimum accuracy (0-1)
    priority: str = "balanced"                # "speed", "energy", "accuracy", "balanced"
    
    def __post_init__(self):
        if self.target_position is None:
            self.target_position = Vector3D(1000.0, 0.0, 0.0)
        if self.target_orientation is None:
            self.target_orientation = Quaternion(1.0, 0.0, 0.0, 0.0)
        if self.waypoints is None:
            self.waypoints = []

@dataclass
class SystemConfiguration:
    """Integrated system configuration."""
    bubble_radius: float = 100.0              # Warp bubble radius (m)
    bubble_thickness: float = 2.0             # Bubble wall thickness (m)
    moment_of_inertia: float = 1e6           # System moment of inertia (kg⋅m²)
    max_velocity: float = 1e-4               # Maximum velocity (fraction of c)
    max_angular_velocity: float = 0.1        # Maximum angular velocity (rad/s)
    energy_efficiency: float = 0.8           # System efficiency factor
    control_precision: float = 0.01          # Control system precision
    simulation_steps: int = 1000             # Time discretization steps

class IntegratedWarpController:
    """
    Integrated warp engine control system combining all simulation capabilities.
    """
    
    def __init__(self, config: SystemConfiguration):
        """
        Initialize integrated control system.
        
        Args:
            config: System configuration parameters
        """
        self.config = config
        self.mission_log = []
        self.performance_metrics = {}
        
        # Initialize subsystem controllers
        if CONTROL_AVAILABLE:
            self.virtual_controller = VirtualWarpController()
        else:
            self.virtual_controller = None
        
        # Performance tracking
        self.total_energy_used = 0.0
        self.total_mission_time = 0.0
        self.successful_missions = 0
        self.failed_missions = 0
        
        print(f"🎮 Integrated Warp Controller initialized")
        print(f"   Bubble radius: {config.bubble_radius} m")
        print(f"   Max velocity: {config.max_velocity:.2e} c")
        print(f"   Max angular velocity: {config.max_angular_velocity:.3f} rad/s")
    
    def analyze_mission_feasibility(self, objective: MissionObjective) -> Dict[str, Any]:
        """
        Analyze mission feasibility before execution.
        
        Args:
            objective: Mission objective specification
            
        Returns:
            Feasibility analysis results
        """
        print(f"🔍 Analyzing mission feasibility...")
        
        analysis = {
            'feasible': True,
            'estimated_time': 0.0,
            'estimated_energy': 0.0,
            'risk_factors': [],
            'recommendations': [],
            'confidence': 1.0
        }
        
        # Analyze translation requirements
        if objective.objective_type in ["translation", "combined", "waypoint"]:
            if objective.waypoints:
                total_distance = sum(
                    np.linalg.norm((wp2 - wp1).vec if hasattr(wp2, 'vec') else np.array([wp2.x, wp2.y, wp2.z]) - np.array([wp1.x, wp1.y, wp1.z]))
                    for wp1, wp2 in zip(objective.waypoints[:-1], objective.waypoints[1:])
                )
            else:
                total_distance = objective.target_position.magnitude() if hasattr(objective.target_position, 'magnitude') else np.linalg.norm([objective.target_position.x, objective.target_position.y, objective.target_position.z])
            
            # Estimate translation time and energy
            avg_velocity = self.config.max_velocity * 0.6  # 60% duty cycle
            translation_time = total_distance / (avg_velocity * 299792458)
            translation_energy = 1e15 * avg_velocity**2 * translation_time
            
            analysis['estimated_time'] += translation_time
            analysis['estimated_energy'] += translation_energy
            
            if translation_time > objective.time_constraint:
                analysis['feasible'] = False
                analysis['risk_factors'].append(f"Translation time {translation_time:.1f}s exceeds constraint {objective.time_constraint:.1f}s")
        
        # Analyze rotation requirements
        if objective.objective_type in ["rotation", "combined"]:
            # Estimate rotation angle (simplified)
            rotation_angle = np.pi / 4  # Default estimate
            rotation_time = rotation_angle / (self.config.max_angular_velocity * 0.6)
            rotation_energy = 1e12 * self.config.max_angular_velocity**2 * rotation_time
            
            analysis['estimated_time'] += rotation_time
            analysis['estimated_energy'] += rotation_energy
            
            if rotation_time > objective.time_constraint:
                analysis['feasible'] = False
                analysis['risk_factors'].append(f"Rotation time {rotation_time:.1f}s exceeds constraint {objective.time_constraint:.1f}s")
        
        # Energy constraint check
        if analysis['estimated_energy'] > objective.energy_constraint:
            analysis['feasible'] = False
            analysis['risk_factors'].append(f"Energy requirement {analysis['estimated_energy']/1e9:.2f}GJ exceeds budget {objective.energy_constraint/1e9:.2f}GJ")
        
        # Generate recommendations
        if not analysis['feasible']:
            analysis['recommendations'].append("Consider reducing mission scope or extending constraints")
            analysis['confidence'] = 0.3
        elif analysis['estimated_energy'] > objective.energy_constraint * 0.8:
            analysis['recommendations'].append("Mission near energy limit - consider efficiency optimizations")
            analysis['confidence'] = 0.7
        else:
            analysis['recommendations'].append("Mission appears feasible with current parameters")
            analysis['confidence'] = 0.9
        
        print(f"📊 Feasibility Analysis:")
        print(f"   Feasible: {'✅' if analysis['feasible'] else '❌'}")
        print(f"   Estimated time: {analysis['estimated_time']:.1f} s")
        print(f"   Estimated energy: {analysis['estimated_energy']/1e9:.2f} GJ")
        print(f"   Confidence: {analysis['confidence']*100:.0f}%")
        
        return analysis
    
    def plan_mission_profile(self, objective: MissionObjective) -> Dict[str, Any]:
        """
        Generate optimal mission profile for given objective.
        
        Args:
            objective: Mission objective specification
            
        Returns:
            Mission profile with optimized parameters
        """
        print(f"📋 Planning mission profile for {objective.objective_type} objective...")
        
        profile = {
            'mission_type': objective.objective_type,
            'phases': [],
            'total_duration': 0.0,
            'optimization_strategy': objective.priority
        }
        
        # Phase planning based on objective type
        if objective.objective_type == "translation":
            # Single translation maneuver
            distance = objective.target_position.magnitude() if hasattr(objective.target_position, 'magnitude') else np.linalg.norm([objective.target_position.x, objective.target_position.y, objective.target_position.z])
            
            # Optimize velocity profile based on priority
            if objective.priority == "speed":
                v_max = self.config.max_velocity
                t_ramp = 5.0
            elif objective.priority == "energy":
                v_max = self.config.max_velocity * 0.5
                t_ramp = 15.0
            else:  # balanced
                v_max = self.config.max_velocity * 0.8
                t_ramp = 10.0
            
            # Calculate hold time for desired distance
            cruise_speed = v_max * 299792458  # m/s
            t_hold = max(5.0, distance / cruise_speed - t_ramp)
            
            profile['phases'].append({
                'type': 'translation',
                'target': objective.target_position,
                'v_max': v_max,
                't_up': t_ramp,
                't_hold': t_hold,
                't_down': t_ramp
            })
            profile['total_duration'] = 2 * t_ramp + t_hold
            
        elif objective.objective_type == "rotation":
            # Single rotation maneuver
            if objective.priority == "speed":
                omega_max = self.config.max_angular_velocity
                t_ramp = 3.0
            elif objective.priority == "energy":
                omega_max = self.config.max_angular_velocity * 0.4
                t_ramp = 10.0
            else:  # balanced
                omega_max = self.config.max_angular_velocity * 0.7
                t_ramp = 6.0
            
            t_hold = 8.0  # Fine pointing time
            
            profile['phases'].append({
                'type': 'rotation',
                'target': objective.target_orientation,
                'omega_max': omega_max,
                't_up': t_ramp,
                't_hold': t_hold,
                't_down': t_ramp
            })
            profile['total_duration'] = 2 * t_ramp + t_hold
            
        elif objective.objective_type == "combined":
            # Simultaneous translation and rotation
            # Use conservative parameters for stability
            distance = objective.target_position.magnitude() if hasattr(objective.target_position, 'magnitude') else np.linalg.norm([objective.target_position.x, objective.target_position.y, objective.target_position.z])
            
            v_max = self.config.max_velocity * 0.6
            omega_max = self.config.max_angular_velocity * 0.5
            t_ramp = 12.0
            cruise_speed = v_max * 299792458
            t_hold = max(10.0, distance / cruise_speed - t_ramp)
            
            profile['phases'].append({
                'type': 'combined',
                'translation_target': objective.target_position,
                'rotation_target': objective.target_orientation,
                'v_max': v_max,
                'omega_max': omega_max,
                't_up': t_ramp,
                't_hold': t_hold,
                't_down': t_ramp
            })
            profile['total_duration'] = 2 * t_ramp + t_hold
            
        elif objective.objective_type == "waypoint":
            # Multi-waypoint mission
            for i in range(len(objective.waypoints) - 1):
                segment_distance = np.linalg.norm((objective.waypoints[i+1] - objective.waypoints[i]).vec if hasattr(objective.waypoints[i+1], 'vec') else np.array([objective.waypoints[i+1].x, objective.waypoints[i+1].y, objective.waypoints[i+1].z]) - np.array([objective.waypoints[i].x, objective.waypoints[i].y, objective.waypoints[i].z]))
                
                v_max = self.config.max_velocity * 0.7
                t_ramp = 8.0
                cruise_speed = v_max * 299792458
                t_hold = max(5.0, segment_distance / cruise_speed - t_ramp)
                
                profile['phases'].append({
                    'type': 'translation',
                    'target': objective.waypoints[i+1],
                    'v_max': v_max,
                    't_up': t_ramp,
                    't_hold': t_hold,
                    't_down': t_ramp
                })
                profile['total_duration'] += 2 * t_ramp + t_hold
        
        print(f"✅ Mission profile planned:")
        print(f"   Phases: {len(profile['phases'])}")
        print(f"   Total duration: {profile['total_duration']:.1f} s")
        print(f"   Strategy: {profile['optimization_strategy']}")
        
        return profile
    
    def execute_mission(self, objective: MissionObjective, 
                       enable_simulation: bool = True,
                       enable_visualization: bool = False) -> Dict[str, Any]:
        """
        Execute complete mission with integrated control.
        
        Args:
            objective: Mission objective specification
            enable_simulation: Run detailed simulation
            enable_visualization: Generate plots
            
        Returns:
            Mission execution results
        """
        print(f"\n🚀 EXECUTING MISSION: {objective.objective_type.upper()}")
        print("=" * 60)
        
        # Initialize progress tracking
        progress = None
        if PROGRESS_AVAILABLE:
            try:
                progress = ProgressTracker(
                    total_steps=6,
                    description=f"Mission Execution: {objective.objective_type}"
                )
                progress.set_stage("mission_planning")
            except:
                pass
        
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        
        with progress if progress else DummyContext():
            
            # Step 1: Feasibility analysis
            if progress: progress.update("Analyzing mission feasibility", step_number=1)
            feasibility = self.analyze_mission_feasibility(objective)
            
            if not feasibility['feasible']:
                result = {
                    'success': False,
                    'error': 'Mission not feasible',
                    'feasibility': feasibility,
                    'energy_used': 0.0,
                    'mission_time': 0.0
                }
                self.failed_missions += 1
                return result
            
            # Step 2: Mission planning
            if progress: progress.update("Planning mission profile", step_number=2)
            mission_profile = self.plan_mission_profile(objective)
            
            # Step 3: Simulation execution
            if progress: progress.update("Executing simulation", step_number=3)
            simulation_results = []
            total_energy = 0.0
            total_time = 0.0
            
            if enable_simulation and SIMULATION_MODULES_AVAILABLE:
                for i, phase in enumerate(mission_profile['phases']):
                    if phase['type'] == 'translation':
                        # Vector impulse simulation
                        vector_profile = VectorImpulseProfile(
                            target_displacement=phase['target'],
                            v_max=phase['v_max'],
                            t_up=phase['t_up'],
                            t_hold=phase['t_hold'],
                            t_down=phase['t_down'],
                            n_steps=self.config.simulation_steps // len(mission_profile['phases'])
                        )
                        
                        warp_params = WarpBubbleVector(
                            R_max=self.config.bubble_radius,
                            thickness=self.config.bubble_thickness
                        )
                        
                        phase_result = simulate_vector_impulse_maneuver(
                            vector_profile, warp_params, enable_progress=False
                        )
                        
                    elif phase['type'] == 'rotation':
                        # Rotation simulation
                        rotation_profile = RotationProfile(
                            target_orientation=phase['target'],
                            omega_max=phase['omega_max'],
                            t_up=phase['t_up'],
                            t_hold=phase['t_hold'],
                            t_down=phase['t_down'],
                            n_steps=self.config.simulation_steps // len(mission_profile['phases'])
                        )
                        
                        warp_params = WarpBubbleRotational(
                            R_max=self.config.bubble_radius,
                            thickness=self.config.bubble_thickness,
                            moment_of_inertia=self.config.moment_of_inertia
                        )
                        
                        phase_result = simulate_rotation_maneuver(
                            rotation_profile, warp_params, enable_progress=False
                        )
                        
                    else:  # combined or other
                        # Simplified combined simulation
                        phase_result = {
                            'total_energy': feasibility['estimated_energy'] / len(mission_profile['phases']),
                            'maneuver_duration': phase['t_up'] + phase['t_hold'] + phase['t_down'],
                            'trajectory_accuracy': 0.95,
                            'rotation_accuracy': 0.93
                        }
                    
                    simulation_results.append(phase_result)
                    total_energy += phase_result['total_energy']
                    total_time += phase_result['maneuver_duration']
            else:
                # Use feasibility estimates
                total_energy = feasibility['estimated_energy']
                total_time = feasibility['estimated_time']
            
            # Step 4: Control system integration
            if progress: progress.update("Integrating control system", step_number=4)
            
            if self.virtual_controller:
                control_result = self.virtual_controller.execute_maneuver(
                    objective, mission_profile
                )
            else:
                control_result = {'status': 'mock_success', 'control_energy': total_energy * 0.1}
            
            # Step 5: Performance analysis
            if progress: progress.update("Analyzing performance", step_number=5)
            
            # Calculate overall mission success metrics
            energy_efficiency = min(1.0, objective.energy_constraint / total_energy)
            time_efficiency = min(1.0, objective.time_constraint / total_time)
            
            if simulation_results:
                avg_accuracy = np.mean([
                    r.get('trajectory_accuracy', r.get('rotation_accuracy', 0.9)) 
                    for r in simulation_results
                ])
            else:
                avg_accuracy = 0.9  # Default estimate
            
            success = (avg_accuracy >= objective.accuracy_requirement and 
                      total_energy <= objective.energy_constraint and
                      total_time <= objective.time_constraint)
            
            # Step 6: Results compilation
            if progress: 
                progress.update("Compiling results", step_number=6)
                progress.log_metric("total_energy_GJ", total_energy/1e9)
                progress.log_metric("mission_time_min", total_time/60)
                progress.log_metric("accuracy_percent", avg_accuracy*100)
            
            # Update system statistics
            if success:
                self.successful_missions += 1
            else:
                self.failed_missions += 1
            
            self.total_energy_used += total_energy
            self.total_mission_time += total_time
            
            # Compile final results
            results = {
                'success': success,
                'objective': objective,
                'mission_profile': mission_profile,
                'feasibility': feasibility,
                'simulation_results': simulation_results,
                'control_result': control_result,
                'performance': {
                    'total_energy': total_energy,
                    'total_time': total_time,
                    'average_accuracy': avg_accuracy,
                    'energy_efficiency': energy_efficiency,
                    'time_efficiency': time_efficiency
                },
                'system_status': {
                    'successful_missions': self.successful_missions,
                    'failed_missions': self.failed_missions,
                    'total_energy_used': self.total_energy_used,
                    'total_mission_time': self.total_mission_time
                }
            }
            
            # Log mission
            self.mission_log.append({
                'timestamp': time.time(),
                'objective_type': objective.objective_type,
                'success': success,
                'energy': total_energy,
                'time': total_time,
                'accuracy': avg_accuracy
            })
            
            if progress:
                progress.complete({
                    'success': success,
                    'energy_GJ': total_energy/1e9,
                    'time_min': total_time/60
                })
            
            # Display results
            print(f"\n📊 MISSION RESULTS:")
            print(f"   Success: {'✅' if success else '❌'}")
            print(f"   Total Energy: {total_energy/1e9:.2f} GJ")
            print(f"   Mission Time: {total_time/60:.1f} minutes")
            print(f"   Average Accuracy: {avg_accuracy*100:.1f}%")
            print(f"   Energy Efficiency: {energy_efficiency*100:.1f}%")
            print(f"   Time Efficiency: {time_efficiency*100:.1f}%")
            
            return results
    
    def generate_mission_report(self, results: Dict[str, Any], save_report: bool = True) -> str:
        """
        Generate comprehensive mission report.
        
        Args:
            results: Mission execution results
            save_report: Whether to save report to file
            
        Returns:
            Report content as string
        """
        print("📄 Generating mission report...")
        
        report = f"""
INTEGRATED WARP ENGINE MISSION REPORT
=====================================

Mission Overview:
• Objective Type: {results['objective'].objective_type.upper()}
• Mission Status: {'SUCCESS' if results['success'] else 'FAILED'}
• Execution Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Mission Parameters:
• Target Position: {results['objective'].target_position}
• Target Orientation: {results['objective'].target_orientation}
• Time Constraint: {results['objective'].time_constraint:.1f} s
• Energy Budget: {results['objective'].energy_constraint/1e9:.2f} GJ
• Accuracy Requirement: {results['objective'].accuracy_requirement*100:.1f}%
• Priority: {results['objective'].priority}

Feasibility Analysis:
• Feasible: {'Yes' if results['feasibility']['feasible'] else 'No'}
• Estimated Time: {results['feasibility']['estimated_time']:.1f} s
• Estimated Energy: {results['feasibility']['estimated_energy']/1e9:.2f} GJ
• Confidence: {results['feasibility']['confidence']*100:.0f}%

Mission Profile:
• Number of Phases: {len(results['mission_profile']['phases'])}
• Total Duration: {results['mission_profile']['total_duration']:.1f} s
• Optimization Strategy: {results['mission_profile']['optimization_strategy']}

Performance Results:
• Actual Energy Used: {results['performance']['total_energy']/1e9:.2f} GJ
• Actual Mission Time: {results['performance']['total_time']/60:.1f} minutes
• Average Accuracy: {results['performance']['average_accuracy']*100:.1f}%
• Energy Efficiency: {results['performance']['energy_efficiency']*100:.1f}%
• Time Efficiency: {results['performance']['time_efficiency']*100:.1f}%

System Statistics:
• Successful Missions: {results['system_status']['successful_missions']}
• Failed Missions: {results['system_status']['failed_missions']}
• Total Energy Used: {results['system_status']['total_energy_used']/1e9:.1f} GJ
• Total Mission Time: {results['system_status']['total_mission_time']/3600:.1f} hours

Recommendations:
"""
        
        # Add recommendations based on performance
        if results['success']:
            if results['performance']['energy_efficiency'] > 0.9:
                report += "• Excellent energy efficiency - consider more aggressive mission parameters\n"
            if results['performance']['time_efficiency'] > 0.9:
                report += "• Mission completed well within time constraints\n"
        else:
            if results['performance']['total_energy'] > results['objective'].energy_constraint:
                report += "• Energy budget exceeded - consider mission scope reduction\n"
            if results['performance']['total_time'] > results['objective'].time_constraint:
                report += "• Time constraint exceeded - optimize velocity profiles\n"
            if results['performance']['average_accuracy'] < results['objective'].accuracy_requirement:
                report += "• Accuracy requirement not met - improve control algorithms\n"
        
        report += f"\nReport generated by Integrated Warp Engine Control System v1.0\n"
        
        if save_report:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"mission_report_{results['objective'].objective_type}_{timestamp}.txt"
            with open(filename, 'w') as f:
                f.write(report)
            print(f"📁 Report saved: {filename}")
        
        return report
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics."""
        total_missions = self.successful_missions + self.failed_missions
        success_rate = self.successful_missions / total_missions if total_missions > 0 else 0.0
        
        return {
            'total_missions': total_missions,
            'successful_missions': self.successful_missions,
            'failed_missions': self.failed_missions,
            'success_rate': success_rate,
            'total_energy_used': self.total_energy_used,
            'total_mission_time': self.total_mission_time,
            'average_energy_per_mission': self.total_energy_used / total_missions if total_missions > 0 else 0.0,
            'average_time_per_mission': self.total_mission_time / total_missions if total_missions > 0 else 0.0,
            'mission_log_length': len(self.mission_log)
        }

# Example usage and testing
if __name__ == "__main__":
    print("🎮 INTEGRATED WARP ENGINE CONTROL SYSTEM")
    print("=" * 60)
    
    # Initialize system
    config = SystemConfiguration(
        bubble_radius=150.0,
        bubble_thickness=3.0,
        max_velocity=8e-5,
        max_angular_velocity=0.08,
        simulation_steps=1500
    )
    
    controller = IntegratedWarpController(config)
    
    # Example 1: Translation mission
    print("\n1. 🚀 Translation Mission")
    translation_objective = MissionObjective(
        objective_type="translation",
        target_position=Vector3D(2000.0, 1000.0, -500.0),
        time_constraint=180.0,
        energy_constraint=5e11,
        accuracy_requirement=0.95,
        priority="balanced"
    )
    
    translation_results = controller.execute_mission(translation_objective, enable_simulation=True)
    translation_report = controller.generate_mission_report(translation_results)
    
    # Example 2: Rotation mission
    print("\n2. 🔄 Rotation Mission")
    rotation_objective = MissionObjective(
        objective_type="rotation",
        target_orientation=Quaternion(0.9, 0.1, 0.2, 0.3),  # Arbitrary rotation
        time_constraint=120.0,
        energy_constraint=1e11,
        accuracy_requirement=0.92,
        priority="accuracy"
    )
    
    rotation_results = controller.execute_mission(rotation_objective, enable_simulation=True)
    
    # Example 3: Combined 6-DOF mission
    print("\n3. 🛰️  Combined 6-DOF Mission")
    combined_objective = MissionObjective(
        objective_type="combined",
        target_position=Vector3D(1500.0, -800.0, 300.0),
        target_orientation=Quaternion(0.8, 0.3, -0.2, 0.4),
        time_constraint=240.0,
        energy_constraint=8e11,
        accuracy_requirement=0.90,
        priority="balanced"
    )
    
    combined_results = controller.execute_mission(combined_objective, enable_simulation=True)
    
    # Example 4: Multi-waypoint mission
    print("\n4. 🗺️  Multi-Waypoint Mission")
    waypoints = [
        Vector3D(0.0, 0.0, 0.0),
        Vector3D(1000.0, 0.0, 0.0),
        Vector3D(1000.0, 1000.0, 0.0),
        Vector3D(0.0, 1000.0, 500.0),
        Vector3D(0.0, 0.0, 500.0)
    ]
    
    waypoint_objective = MissionObjective(
        objective_type="waypoint",
        waypoints=waypoints,
        time_constraint=600.0,
        energy_constraint=2e12,
        accuracy_requirement=0.88,
        priority="energy"
    )
    
    waypoint_results = controller.execute_mission(waypoint_objective, enable_simulation=True)
    
    # System status summary
    print("\n📊 SYSTEM STATUS SUMMARY")
    status = controller.get_system_status()
    print(f"   Total Missions: {status['total_missions']}")
    print(f"   Success Rate: {status['success_rate']*100:.1f}%")
    print(f"   Total Energy Used: {status['total_energy_used']/1e9:.1f} GJ")
    print(f"   Total Mission Time: {status['total_mission_time']/3600:.1f} hours")
    print(f"   Average Energy/Mission: {status['average_energy_per_mission']/1e9:.2f} GJ")
    print(f"   Average Time/Mission: {status['average_time_per_mission']/60:.1f} minutes")
    
    print(f"\n🎯 Integrated control system demonstration complete!")
    print(f"💡 Features demonstrated:")
    print(f"   • Mission feasibility analysis")
    print(f"   • Automated mission profile planning")
    print(f"   • Integrated simulation execution")
    print(f"   • Performance monitoring and optimization")
    print(f"   • Comprehensive mission reporting")
    print(f"   • Multi-mission system statistics")
