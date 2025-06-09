# src/warp_engine/mission_planner.py
"""
Mission-Profile & Energy Budgeting Module
=========================================

This module implements comprehensive mission planning and energy management
for warp drive operations. Includes:

1. Mission trajectory optimization and planning
2. Energy budget calculation and resource allocation
3. Multi-phase mission profiles (launch, cruise, arrival)
4. Propellant and exotic matter consumption tracking
5. Emergency contingency planning
6. Performance optimization across mission phases

Key Features:
- Delta-v calculations for warp trajectories
- Energy efficiency optimization
- Mission risk assessment
- Automated resource scheduling
- Real-time mission adaptation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import json

logger = logging.getLogger(__name__)

class MissionPhase(Enum):
    """Mission phases for warp drive operations."""
    PREPARATION = "preparation"
    LAUNCH = "launch"
    ACCELERATION = "acceleration"
    CRUISE = "cruise"
    DECELERATION = "deceleration"
    ARRIVAL = "arrival"
    EMERGENCY = "emergency"

class ResourceType(Enum):
    """Types of resources required for mission."""
    EXOTIC_MATTER = "exotic_matter"
    ENERGY = "energy"
    PROPELLANT = "propellant"
    COOLANT = "coolant"
    STRUCTURAL_INTEGRITY = "structural_integrity"

@dataclass
class Waypoint:
    """Single waypoint in mission trajectory."""
    position: np.ndarray  # 3D position [x, y, z] in appropriate units
    time: float          # Time to reach this waypoint
    velocity: float      # Required velocity at waypoint
    warp_factor: float   # Warp factor for this segment
    phase: MissionPhase  # Mission phase at this waypoint

@dataclass
class ResourceConsumption:
    """Resource consumption rate for different mission phases."""
    resource_type: ResourceType
    base_rate: float              # Base consumption rate
    warp_factor_scaling: float    # How consumption scales with warp factor
    phase_modifiers: Dict[MissionPhase, float] = field(default_factory=dict)

@dataclass
class MissionConstraints:
    """Constraints for mission planning."""
    max_warp_factor: float = 10.0
    max_acceleration: float = 100.0  # m/s²
    max_mission_duration: float = 365.0 * 24 * 3600  # 1 year in seconds
    crew_g_tolerance: float = 10.0   # Maximum acceleration in g's
    safety_margin: float = 0.2       # 20% safety margin on resources
    
    # Resource limits
    total_exotic_matter: float = 1000.0  # kg
    total_energy: float = 1e15          # Joules
    structural_stress_limit: float = 1e9 # Pascal

@dataclass
class MissionParameters:
    """Parameters defining a mission."""
    origin: np.ndarray
    destination: np.ndarray
    departure_time: float
    arrival_deadline: Optional[float] = None
    crew_size: int = 1
    cargo_mass: float = 0.0
    priority: str = "balanced"  # speed, efficiency, safety

class TrajectoryPoint(NamedTuple):
    """Point along mission trajectory."""
    time: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    warp_factor: float
    phase: MissionPhase

@dataclass
class MissionResults:
    """Results from mission planning optimization."""
    trajectory: List[TrajectoryPoint]
    total_duration: float
    total_energy: float
    total_exotic_matter: float
    resource_schedule: Dict[ResourceType, List[Tuple[float, float]]]  # (time, consumption_rate)
    risk_assessment: Dict[str, float]
    success_probability: float
    waypoints: List[Waypoint]

class WarpTrajectoryOptimizer:
    """
    Optimizer for warp drive trajectories.
    
    Computes optimal paths considering:
    - Space-time curvature effects
    - Energy consumption
    - Exotic matter requirements
    - Safety constraints
    """
    
    def __init__(self, constraints: MissionConstraints):
        self.constraints = constraints
        
    def compute_warp_metrics(self, warp_factor: float, distance: float, 
                           mass: float) -> Dict[str, float]:
        """Compute key metrics for warp travel segment."""
        # Alcubierre warp drive energy scaling (simplified)
        # E ∝ (warp_factor)^4 for field generation
        # Plus exotic matter requirements ∝ (warp_factor)^3
        
        # Energy calculations
        base_energy_density = 1e14  # J/m³ (rough estimate)
        volume_factor = (warp_factor / 10.0) ** 3  # Bubble volume scaling
        energy_density = base_energy_density * (warp_factor / 10.0) ** 4
        
        # Estimate bubble size based on ship mass
        ship_radius = (3 * mass / (4 * np.pi * 2700)) ** (1/3)  # Assume aluminum density
        bubble_volume = (4/3) * np.pi * (ship_radius * warp_factor) ** 3
        
        total_energy = energy_density * bubble_volume
        
        # Exotic matter requirements
        exotic_matter_density = 1e-6 * warp_factor ** 3  # kg/m³ 
        exotic_matter_total = exotic_matter_density * bubble_volume
        
        # Effective velocity (faster than light for warp > 1)
        c = 3e8  # Speed of light
        if warp_factor <= 1:
            effective_velocity = warp_factor * c
        else:
            # Alcubierre metric gives v_eff = c * warp_factor³
            effective_velocity = c * warp_factor ** 3
            
        travel_time = distance / effective_velocity
        
        return {
            "energy": total_energy,
            "exotic_matter": exotic_matter_total,
            "travel_time": travel_time,
            "effective_velocity": effective_velocity,
            "bubble_volume": bubble_volume
        }
        
    def optimize_single_segment(self, start_pos: np.ndarray, end_pos: np.ndarray,
                              ship_mass: float) -> Tuple[float, Dict]:
        """Optimize single trajectory segment."""
        distance = np.linalg.norm(end_pos - start_pos)
        
        def objective(warp_factor_array):
            wf = warp_factor_array[0]
            if wf <= 0 or wf > self.constraints.max_warp_factor:
                return 1e10  # Invalid warp factor
                
            metrics = self.compute_warp_metrics(wf, distance, ship_mass)
            
            # Multi-objective: minimize time and energy consumption
            time_weight = 1.0
            energy_weight = 1e-14  # Scale energy to comparable magnitude
            exotic_weight = 1e3    # Penalize exotic matter heavily
            
            cost = (time_weight * metrics["travel_time"] + 
                   energy_weight * metrics["energy"] +
                   exotic_weight * metrics["exotic_matter"])
                   
            # Add penalty for exceeding resource limits
            if metrics["energy"] > self.constraints.total_energy:
                cost += 1e6
            if metrics["exotic_matter"] > self.constraints.total_exotic_matter:
                cost += 1e6
                
            return cost
            
        # Optimize warp factor
        result = minimize(objective, [5.0], bounds=[(0.1, self.constraints.max_warp_factor)], 
                         method='L-BFGS-B')
        
        optimal_wf = result.x[0]
        optimal_metrics = self.compute_warp_metrics(optimal_wf, distance, ship_mass)
        
        return optimal_wf, optimal_metrics
        
    def plan_multi_segment_trajectory(self, waypoints: List[np.ndarray], 
                                    ship_mass: float) -> List[Tuple[float, Dict]]:
        """Plan trajectory through multiple waypoints."""
        segments = []
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            warp_factor, metrics = self.optimize_single_segment(start, end, ship_mass)
            segments.append((warp_factor, metrics))
            
        return segments

class ResourceManager:
    """
    Manages resource consumption and allocation throughout mission.
    
    Tracks consumption of exotic matter, energy, propellant, and
    other critical resources across all mission phases.
    """
    
    def __init__(self, constraints: MissionConstraints):
        self.constraints = constraints
        self.consumption_models = self._create_default_consumption_models()
        
    def _create_default_consumption_models(self) -> Dict[ResourceType, ResourceConsumption]:
        """Create default resource consumption models."""
        return {
            ResourceType.EXOTIC_MATTER: ResourceConsumption(
                resource_type=ResourceType.EXOTIC_MATTER,
                base_rate=0.1,  # kg/s base consumption
                warp_factor_scaling=3.0,  # Cubic scaling with warp factor
                phase_modifiers={
                    MissionPhase.ACCELERATION: 2.0,
                    MissionPhase.CRUISE: 1.0,
                    MissionPhase.DECELERATION: 2.0,
                    MissionPhase.EMERGENCY: 5.0
                }
            ),
            ResourceType.ENERGY: ResourceConsumption(
                resource_type=ResourceType.ENERGY,
                base_rate=1e9,  # J/s (1 GW)
                warp_factor_scaling=4.0,  # Quartic scaling
                phase_modifiers={
                    MissionPhase.ACCELERATION: 3.0,
                    MissionPhase.CRUISE: 1.0,
                    MissionPhase.DECELERATION: 3.0,
                    MissionPhase.LAUNCH: 10.0
                }
            ),
            ResourceType.PROPELLANT: ResourceConsumption(
                resource_type=ResourceType.PROPELLANT,
                base_rate=0.01,  # kg/s
                warp_factor_scaling=1.0,  # Linear scaling
                phase_modifiers={
                    MissionPhase.LAUNCH: 100.0,
                    MissionPhase.ACCELERATION: 10.0,
                    MissionPhase.DECELERATION: 10.0
                }
            ),
            ResourceType.COOLANT: ResourceConsumption(
                resource_type=ResourceType.COOLANT,
                base_rate=0.005,  # kg/s
                warp_factor_scaling=2.0,  # Quadratic scaling
                phase_modifiers={
                    MissionPhase.ACCELERATION: 1.5,
                    MissionPhase.CRUISE: 1.0,
                    MissionPhase.DECELERATION: 1.5,
                    MissionPhase.EMERGENCY: 3.0
                }
            ),
            ResourceType.STRUCTURAL_INTEGRITY: ResourceConsumption(
                resource_type=ResourceType.STRUCTURAL_INTEGRITY,
                base_rate=0.001,  # units/s (degradation rate)
                warp_factor_scaling=2.5,  # High warp factors stress structure
                phase_modifiers={
                    MissionPhase.ACCELERATION: 2.0,
                    MissionPhase.CRUISE: 0.5,
                    MissionPhase.DECELERATION: 2.0,
                    MissionPhase.EMERGENCY: 10.0
                }
            )
        }
        
    def compute_consumption_rate(self, resource_type: ResourceType, 
                               warp_factor: float, phase: MissionPhase,
                               ship_mass: float = 1000.0) -> float:
        """Compute instantaneous resource consumption rate."""
        model = self.consumption_models[resource_type]
        
        # Base consumption scaled by warp factor
        base_consumption = model.base_rate * (warp_factor ** model.warp_factor_scaling)
        
        # Apply phase modifier
        phase_modifier = model.phase_modifiers.get(phase, 1.0)
        
        # Scale by ship mass (larger ships consume more)
        mass_scaling = (ship_mass / 1000.0) ** 0.5
        
        return base_consumption * phase_modifier * mass_scaling
        
    def integrate_consumption(self, trajectory: List[TrajectoryPoint],
                            ship_mass: float) -> Dict[ResourceType, float]:
        """Integrate total resource consumption over trajectory."""
        total_consumption = {rt: 0.0 for rt in ResourceType}
        
        for i in range(len(trajectory) - 1):
            current = trajectory[i]
            next_point = trajectory[i + 1]
            dt = next_point.time - current.time
            
            for resource_type in ResourceType:
                rate = self.compute_consumption_rate(
                    resource_type, current.warp_factor, current.phase, ship_mass
                )
                total_consumption[resource_type] += rate * dt
                
        return total_consumption
        
    def check_resource_feasibility(self, trajectory: List[TrajectoryPoint],
                                 ship_mass: float) -> Dict[str, bool]:
        """Check if trajectory is feasible given resource constraints."""
        total_consumption = self.integrate_consumption(trajectory, ship_mass)
        
        feasibility = {}
        feasibility["exotic_matter"] = total_consumption[ResourceType.EXOTIC_MATTER] <= self.constraints.total_exotic_matter
        feasibility["energy"] = total_consumption[ResourceType.ENERGY] <= self.constraints.total_energy
        
        # Add safety margins
        safety_factor = 1.0 + self.constraints.safety_margin
        feasibility["exotic_matter_with_margin"] = total_consumption[ResourceType.EXOTIC_MATTER] <= self.constraints.total_exotic_matter / safety_factor
        feasibility["energy_with_margin"] = total_consumption[ResourceType.ENERGY] <= self.constraints.total_energy / safety_factor
        
        return feasibility

class RiskAssessment:
    """
    Comprehensive risk assessment for warp drive missions.
    
    Evaluates technical, operational, and safety risks
    throughout the mission profile.
    """
    
    def __init__(self):
        self.risk_factors = {
            "warp_field_instability": 0.0,
            "exotic_matter_depletion": 0.0,
            "structural_failure": 0.0,
            "navigation_error": 0.0,
            "crew_safety": 0.0,
            "emergency_scenarios": 0.0
        }
        
    def assess_warp_field_risks(self, max_warp_factor: float, 
                              mission_duration: float) -> float:
        """Assess risks related to warp field stability."""
        # Higher warp factors and longer durations increase instability risk
        warp_risk = min(0.9, (max_warp_factor / 10.0) ** 2)
        duration_risk = min(0.3, mission_duration / (365 * 24 * 3600))  # 1 year reference
        
        return min(0.95, warp_risk + duration_risk * 0.1)
        
    def assess_resource_risks(self, resource_utilization: Dict[ResourceType, float]) -> float:
        """Assess risks from resource consumption patterns."""
        # Risk increases as we approach resource limits
        risk = 0.0
        
        for resource_type, utilization in resource_utilization.items():
            if utilization > 0.9:  # Using >90% of available resource
                risk += 0.3
            elif utilization > 0.8:
                risk += 0.1
            elif utilization > 0.7:
                risk += 0.05
                
        return min(0.8, risk)
        
    def assess_structural_risks(self, max_acceleration: float, 
                              max_warp_factor: float) -> float:
        """Assess structural integrity risks."""
        accel_risk = min(0.4, max_acceleration / 100.0)  # 100 m/s² reference
        field_stress_risk = min(0.3, (max_warp_factor / 10.0) ** 1.5)
        
        return min(0.6, accel_risk + field_stress_risk)
        
    def assess_crew_safety(self, max_acceleration: float, 
                         mission_duration: float) -> float:
        """Assess crew safety risks."""
        # G-force exposure
        g_force = max_acceleration / 9.81
        g_risk = min(0.5, g_force / 10.0)  # 10g reference
        
        # Radiation exposure during extended warp
        radiation_risk = min(0.2, mission_duration / (30 * 24 * 3600))  # 30 days reference
        
        return min(0.6, g_risk + radiation_risk)
        
    def compute_overall_risk(self, trajectory: List[TrajectoryPoint],
                           resource_consumption: Dict[ResourceType, float],
                           constraints: MissionConstraints) -> Dict[str, float]:
        """Compute comprehensive risk assessment."""
        # Extract mission parameters
        max_warp = max(point.warp_factor for point in trajectory)
        max_accel = max(np.linalg.norm(point.acceleration) for point in trajectory)
        duration = trajectory[-1].time - trajectory[0].time
        
        # Compute resource utilization ratios
        resource_utilization = {
            ResourceType.EXOTIC_MATTER: resource_consumption.get(ResourceType.EXOTIC_MATTER, 0) / constraints.total_exotic_matter,
            ResourceType.ENERGY: resource_consumption.get(ResourceType.ENERGY, 0) / constraints.total_energy
        }
        
        # Individual risk assessments
        risks = {
            "warp_field_instability": self.assess_warp_field_risks(max_warp, duration),
            "exotic_matter_depletion": self.assess_resource_risks(resource_utilization),
            "structural_failure": self.assess_structural_risks(max_accel, max_warp),
            "crew_safety": self.assess_crew_safety(max_accel, duration),
            "navigation_error": min(0.1, max_warp / 20.0),  # Simple model
        }
        
        # Overall mission risk (not simple sum due to correlations)
        weights = {
            "warp_field_instability": 0.3,
            "exotic_matter_depletion": 0.25,
            "structural_failure": 0.2,
            "crew_safety": 0.15,
            "navigation_error": 0.1
        }
        
        overall_risk = sum(risk * weights[factor] for factor, risk in risks.items())
        risks["overall"] = min(0.95, overall_risk)
        
        return risks

class MissionPlanningManager:
    """
    Main mission planning and optimization system.
    
    Integrates trajectory optimization, resource management,
    and risk assessment for comprehensive mission planning.
    """
    
    def __init__(self, constraints: MissionConstraints = None):
        self.constraints = constraints or MissionConstraints()
        self.trajectory_optimizer = WarpTrajectoryOptimizer(self.constraints)
        self.resource_manager = ResourceManager(self.constraints)
        self.risk_assessor = RiskAssessment()
        
    def plan_mission(self, mission_params: MissionParameters) -> MissionResults:
        """Plan complete mission from origin to destination."""
        logger.info(f"Planning mission from {mission_params.origin} to {mission_params.destination}")
        
        # Basic trajectory waypoints
        waypoints = [mission_params.origin, mission_params.destination]
        
        # Estimate ship mass
        base_mass = 1000.0  # kg base ship mass
        crew_mass = mission_params.crew_size * 80.0  # 80 kg per person
        total_mass = base_mass + crew_mass + mission_params.cargo_mass
        
        # Optimize trajectory segments
        segments = self.trajectory_optimizer.plan_multi_segment_trajectory(waypoints, total_mass)
        
        # Build detailed trajectory
        trajectory = self._build_detailed_trajectory(waypoints, segments, mission_params)
        
        # Compute resource requirements
        resource_consumption = self.resource_manager.integrate_consumption(trajectory, total_mass)
        
        # Assess risks
        risk_assessment = self.risk_assessor.compute_overall_risk(
            trajectory, resource_consumption, self.constraints
        )
        
        # Compute success probability
        success_probability = 1.0 - risk_assessment["overall"]
        
        # Create resource schedule
        resource_schedule = self._create_resource_schedule(trajectory, total_mass)
        
        # Create waypoint list
        mission_waypoints = []
        for i, point in enumerate(trajectory):
            if i % (len(trajectory) // 10) == 0:  # 10 major waypoints
                mission_waypoints.append(Waypoint(
                    position=point.position,
                    time=point.time,
                    velocity=np.linalg.norm(point.velocity),
                    warp_factor=point.warp_factor,
                    phase=point.phase
                ))
        
        return MissionResults(
            trajectory=trajectory,
            total_duration=trajectory[-1].time - trajectory[0].time,
            total_energy=resource_consumption.get(ResourceType.ENERGY, 0),
            total_exotic_matter=resource_consumption.get(ResourceType.EXOTIC_MATTER, 0),
            resource_schedule=resource_schedule,
            risk_assessment=risk_assessment,
            success_probability=success_probability,
            waypoints=mission_waypoints
        )
        
    def _build_detailed_trajectory(self, waypoints: List[np.ndarray], 
                                 segments: List[Tuple[float, Dict]],
                                 mission_params: MissionParameters) -> List[TrajectoryPoint]:
        """Build detailed trajectory from optimized segments."""
        trajectory = []
        current_time = mission_params.departure_time
        
        for i, (warp_factor, metrics) in enumerate(segments):
            start_pos = waypoints[i]
            end_pos = waypoints[i + 1]
            
            # Travel time for this segment
            travel_time = metrics["travel_time"]
            
            # Create trajectory points for this segment
            n_points = max(10, int(travel_time / 3600))  # At least 10 points, or 1 per hour
            times = np.linspace(current_time, current_time + travel_time, n_points)
            
            for j, t in enumerate(times):
                # Linear interpolation for position (simplified)
                alpha = j / (len(times) - 1)
                position = start_pos + alpha * (end_pos - start_pos)
                
                # Velocity and acceleration (simplified)
                velocity_magnitude = metrics["effective_velocity"]
                direction = (end_pos - start_pos) / np.linalg.norm(end_pos - start_pos)
                velocity = velocity_magnitude * direction
                
                # Determine mission phase
                if alpha < 0.1:
                    phase = MissionPhase.ACCELERATION
                elif alpha > 0.9:
                    phase = MissionPhase.DECELERATION
                else:
                    phase = MissionPhase.CRUISE
                    
                # Simplified acceleration
                if phase == MissionPhase.ACCELERATION:
                    acceleration = velocity / (travel_time * 0.1)  # Accelerate over 10% of journey
                elif phase == MissionPhase.DECELERATION:
                    acceleration = -velocity / (travel_time * 0.1)  # Decelerate over 10% of journey
                else:
                    acceleration = np.zeros(3)
                    
                trajectory.append(TrajectoryPoint(
                    time=t,
                    position=position,
                    velocity=velocity,
                    acceleration=acceleration,
                    warp_factor=warp_factor,
                    phase=phase
                ))
                
            current_time += travel_time
            
        return trajectory
        
    def _create_resource_schedule(self, trajectory: List[TrajectoryPoint],
                                ship_mass: float) -> Dict[ResourceType, List[Tuple[float, float]]]:
        """Create detailed resource consumption schedule."""
        schedule = {rt: [] for rt in ResourceType}
        
        for point in trajectory:
            for resource_type in ResourceType:
                rate = self.resource_manager.compute_consumption_rate(
                    resource_type, point.warp_factor, point.phase, ship_mass
                )
                schedule[resource_type].append((point.time, rate))
                
        return schedule
        
    def optimize_mission_profile(self, mission_params: MissionParameters,
                               optimization_target: str = "balanced") -> MissionResults:
        """Optimize mission profile for specific objectives."""
        
        def objective_function(params):
            # params = [warp_factor_1, warp_factor_2, ...]
            # For simplicity, assume single segment for now
            modified_params = mission_params
            
            # Plan mission with these parameters
            result = self.plan_mission(modified_params)
            
            if optimization_target == "speed":
                return result.total_duration
            elif optimization_target == "efficiency":
                return result.total_energy
            elif optimization_target == "safety":
                return result.risk_assessment["overall"]
            elif optimization_target == "balanced":
                # Multi-objective: normalize and combine
                duration_norm = result.total_duration / (365 * 24 * 3600)  # Normalize to 1 year
                energy_norm = result.total_energy / 1e15  # Normalize to reference energy
                risk_norm = result.risk_assessment["overall"]
                
                return duration_norm + energy_norm + 2 * risk_norm  # Weight risk heavily
            else:
                return result.total_duration  # Default to speed
                
        # For now, return basic plan (optimization would require more complex setup)
        return self.plan_mission(mission_params)
        
    def generate_contingency_plans(self, primary_mission: MissionResults,
                                 mission_params: MissionParameters) -> Dict[str, MissionResults]:
        """Generate contingency plans for various failure scenarios."""
        contingencies = {}
        
        # Emergency return plan
        emergency_params = MissionParameters(
            origin=mission_params.destination,  # Return from destination
            destination=mission_params.origin,
            departure_time=primary_mission.trajectory[-1].time,
            crew_size=mission_params.crew_size,
            cargo_mass=0.0,  # Dump cargo in emergency
            priority="safety"
        )
        contingencies["emergency_return"] = self.plan_mission(emergency_params)
        
        # Reduced capacity plan (if exotic matter runs low)
        reduced_constraints = MissionConstraints(
            max_warp_factor=self.constraints.max_warp_factor * 0.5,
            total_exotic_matter=self.constraints.total_exotic_matter * 0.3,
            total_energy=self.constraints.total_energy * 0.5
        )
        
        reduced_planner = MissionPlanningManager(reduced_constraints)
        contingencies["reduced_capacity"] = reduced_planner.plan_mission(mission_params)
        
        return contingencies
        
    def visualize_mission_plan(self, mission_result: MissionResults, save_path: str = None):
        """Create comprehensive visualization of mission plan."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Trajectory in 3D (projected to 2D)
        positions = np.array([point.position for point in mission_result.trajectory])
        axes[0, 0].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
        axes[0, 0].scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start')
        axes[0, 0].scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End')
        axes[0, 0].set_xlabel('X Position')
        axes[0, 0].set_ylabel('Y Position')
        axes[0, 0].set_title('Mission Trajectory')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Warp factor vs time
        times = [point.time for point in mission_result.trajectory]
        warp_factors = [point.warp_factor for point in mission_result.trajectory]
        axes[0, 1].plot(np.array(times) / 3600, warp_factors, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time (hours)')
        axes[0, 1].set_ylabel('Warp Factor')
        axes[0, 1].set_title('Warp Factor Profile')
        axes[0, 1].grid(True)
        
        # 3. Resource consumption over time
        for resource_type, schedule in mission_result.resource_schedule.items():
            if len(schedule) > 0:
                times_r = [item[0] for item in schedule]
                rates = [item[1] for item in schedule]
                axes[0, 2].plot(np.array(times_r) / 3600, rates, 
                              label=resource_type.value, linewidth=2)
        axes[0, 2].set_xlabel('Time (hours)')
        axes[0, 2].set_ylabel('Consumption Rate')
        axes[0, 2].set_title('Resource Consumption')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Risk assessment
        risk_factors = list(mission_result.risk_assessment.keys())
        risk_values = list(mission_result.risk_assessment.values())
        axes[1, 0].bar(range(len(risk_factors)), risk_values, alpha=0.7)
        axes[1, 0].set_xticks(range(len(risk_factors)))
        axes[1, 0].set_xticklabels([factor.replace('_', '\n') for factor in risk_factors], 
                                  rotation=45, ha='right')
        axes[1, 0].set_ylabel('Risk Level')
        axes[1, 0].set_title('Risk Assessment')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Mission phases
        phase_colors = {
            MissionPhase.ACCELERATION: 'red',
            MissionPhase.CRUISE: 'blue',
            MissionPhase.DECELERATION: 'orange'
        }
        
        for phase in MissionPhase:
            phase_times = [point.time for point in mission_result.trajectory if point.phase == phase]
            phase_velocities = [np.linalg.norm(point.velocity) for point in mission_result.trajectory if point.phase == phase]
            
            if phase_times:
                axes[1, 1].scatter(np.array(phase_times) / 3600, phase_velocities,
                                 c=phase_colors.get(phase, 'gray'), label=phase.value, alpha=0.7)
                                 
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Velocity (m/s)')
        axes[1, 1].set_title('Velocity by Mission Phase')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 6. Energy budget
        energy_data = {
            'Total Available': self.constraints.total_energy,
            'Mission Requirement': mission_result.total_energy,
            'Safety Margin': self.constraints.total_energy * self.constraints.safety_margin
        }
        
        bars = axes[1, 2].bar(energy_data.keys(), energy_data.values(), 
                             color=['green', 'blue', 'orange'], alpha=0.7)
        axes[1, 2].set_ylabel('Energy (J)')
        axes[1, 2].set_title('Energy Budget')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, energy_data.values()):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1e}', ha='center', va='bottom')
                           
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Mission plan visualization saved to {save_path}")
            
        plt.show()

# Example usage and testing
# Utility functions for backwards compatibility
def compute_fuel_budget(sweep_csv: str, R_target: float, v_target: float) -> Tuple[float, float]:
    """
    Utility function to compute fuel budget from parameter sweep.
    
    Args:
        sweep_csv: Path to CSV file with parameter sweep results
        R_target: Target bubble radius in meters
        v_target: Target velocity in units of c
        
    Returns:
        Tuple of (fuel_energy_J, power_W)
    """
    import pandas as pd
    import os
    
    if not os.path.exists(sweep_csv):
        # If CSV doesn't exist, use analytical estimate
        logger.warning(f"Sweep CSV {sweep_csv} not found, using analytical estimate")
        
        # Rough estimate based on Alcubierre drive energy requirements
        volume = (4/3) * np.pi * R_target**3
        lorentz_factor = 1 / np.sqrt(1 - v_target**2)
        
        # Base energy density estimate (very rough)
        energy_density = 1e15  # J/m³ (order of magnitude)
        fuel_energy = volume * energy_density * lorentz_factor
        
        # Power estimate assuming 2-hour operation
        power = fuel_energy / (2 * 3600)
        
        return fuel_energy, power
        
    try:
        df = pd.read_csv(sweep_csv)
        
        # Find closest match to target parameters
        df['distance'] = np.sqrt((df['R_m'] - R_target)**2 + (df['v_c'] - v_target)**2)
        closest_row = df.loc[df['distance'].idxmin()]
        
        # Extract energy with 10% safety margin
        fuel_energy = abs(closest_row['energy_J']) * 1.1
        
        # Estimate power assuming 2-hour flight
        power = fuel_energy / (3600 * 2)
        
        return fuel_energy, power
        
    except Exception as e:
        logger.error(f"Error reading sweep CSV: {e}")
        # Fallback to analytical estimate
        volume = (4/3) * np.pi * R_target**3
        fuel_energy = volume * 1e15  # J/m³
        power = fuel_energy / (2 * 3600)
        return fuel_energy, power


if __name__ == "__main__":
    # Define mission parameters
    mission_params = MissionParameters(
        origin=np.array([0.0, 0.0, 0.0]),  # Earth
        destination=np.array([4.37, 0.0, 0.0]),  # Proxima Centauri (4.37 light-years)
        departure_time=0.0,
        crew_size=4,
        cargo_mass=500.0,  # kg
        priority="balanced"
    )
    
    # Create mission planner
    constraints = MissionConstraints(
        max_warp_factor=9.9,
        total_exotic_matter=2000.0,  # kg
        total_energy=1e16,  # Joules
        safety_margin=0.3
    )
    
    planner = MissionPlanningManager(constraints)
    
    # Plan the mission
    print("Planning interstellar mission to Proxima Centauri...")
    mission_result = planner.plan_mission(mission_params)
    
    # Display results
    print("\nMission Planning Results:")
    print("=" * 50)
    print(f"Total Duration: {mission_result.total_duration / (24*3600):.1f} days")
    print(f"Total Energy: {mission_result.total_energy:.2e} J")
    print(f"Total Exotic Matter: {mission_result.total_exotic_matter:.1f} kg")
    print(f"Success Probability: {mission_result.success_probability:.1%}")
    
    print(f"\nRisk Assessment:")
    for factor, risk in mission_result.risk_assessment.items():
        print(f"  {factor}: {risk:.1%}")
        
    print(f"\nMajor Waypoints:")
    for i, wp in enumerate(mission_result.waypoints):
        print(f"  {i+1}. Time: {wp.time/3600:.1f}h, Warp: {wp.warp_factor:.1f}, Phase: {wp.phase.value}")
        
    # Generate contingency plans
    print("\nGenerating contingency plans...")
    contingencies = planner.generate_contingency_plans(mission_result, mission_params)
    
    for scenario, plan in contingencies.items():
        print(f"{scenario}: Duration {plan.total_duration/(24*3600):.1f} days, "
              f"Success Rate {plan.success_probability:.1%}")
    
    # Create visualization
    planner.visualize_mission_plan(mission_result, "mission_plan_visualization.png")
    
    print("\nMission planning complete!")
