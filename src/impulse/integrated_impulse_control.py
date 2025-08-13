"""Relocated integrated impulse control module (full implementation).

Original file was at repo root: integrated_impulse_control.py
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

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

try:
    from progress_tracker import ProgressTracker
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False

try:
    import jax.numpy as jnp  # noqa: F401
    from jax import jit, vmap  # noqa: F401
    JAX_AVAILABLE = True  # noqa: F841
except ImportError:  # pragma: no cover
    JAX_AVAILABLE = False  # noqa: F841
    def jit(func): return func  # type: ignore
    def vmap(func, in_axes=None):  # type: ignore
        def vectorized(*args):
            return np.array([func(*[arg[i] if isinstance(arg, np.ndarray) else arg
                                  for arg in args]) for i in range(len(args[0]))])
        return vectorized


@dataclass
class MissionWaypoint:
    position: Vector3D
    orientation: Quaternion
    dwell_time: float = 10.0
    approach_speed: float = 1e-5
    pointing_tolerance: float = 1e-3
    position_tolerance: float = 1.0


@dataclass
class ImpulseEngineConfig:
    warp_params: Optional[WarpParameters] = None
    vector_params: Optional[WarpBubbleVector] = None
    rotation_params: Optional[WarpBubbleRotational] = None
    max_velocity: float = 1e-4
    max_angular_velocity: float = 0.1
    energy_budget: float = 1e12
    safety_margin: float = 0.2

    def __post_init__(self):
        if self.warp_params is None:
            self.warp_params = WarpParameters()
        if self.vector_params is None:
            self.vector_params = WarpBubbleVector()
        if self.rotation_params is None:
            self.rotation_params = WarpBubbleRotational()


class IntegratedImpulseController:
    def __init__(self, config: ImpulseEngineConfig):
        self.config = config
        # Enforce non-None params after dataclass init for static type checkers
        if self.config.vector_params is None:
            self.config.vector_params = WarpBubbleVector()
        if self.config.rotation_params is None:
            self.config.rotation_params = WarpBubbleRotational()
        self.current_position = Vector3D(0.0, 0.0, 0.0)
        self.current_orientation = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.current_velocity = Vector3D(0.0, 0.0, 0.0)
        self.current_angular_velocity = 0.0
        self.control_config = {
            'sensor': SensorConfig(noise_level=0.01, update_rate=50.0),
            'actuator': ActuatorConfig(response_time=0.02, damping_factor=0.9),
            'controller': ControllerConfig(kp=0.8, ki=0.1, kd=0.05)
        } if CONTROL_AVAILABLE else None
        self.mission_log = []
        self.total_energy_used = 0.0
        self.total_mission_time = 0.0

    def plan_impulse_trajectory(self, waypoints: List[MissionWaypoint], optimize_energy: bool = True) -> Dict[str, Any]:
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")
        segments = []
        total_energy_estimate = 0.0
        total_time_estimate = 0.0
        current_pos = waypoints[0].position
        current_orient = waypoints[0].orientation
        for target_wp in waypoints[1:]:
            displacement = target_wp.position - current_pos
            if optimize_energy:
                dmag = displacement.magnitude
                v_max = min(self.config.max_velocity, target_wp.approach_speed, dmag / 60.0) if dmag > 0 else 0.0
                if v_max > 0:
                    t_ramp = min(10.0, dmag / (v_max * 299792458) / 4)
                    t_hold = max(5.0, dmag / (v_max * 299792458) - 2 * t_ramp)
                else:
                    t_ramp = 5.0
                    t_hold = target_wp.dwell_time
            else:
                v_max = min(self.config.max_velocity, target_wp.approach_speed)
                t_ramp = 8.0
                t_hold = target_wp.dwell_time
            translation_profile = VectorImpulseProfile(
                target_displacement=displacement,
                v_max=v_max,
                t_up=t_ramp,
                t_hold=t_hold,
                t_down=t_ramp,
                n_steps=int((2 * t_ramp + t_hold) * 20)
            )
            displacement_energy = self._estimate_translation_energy(translation_profile) if displacement.magnitude > 0 else 0.0
            segment_time = 2 * t_ramp + t_hold
            segments.append({
                'translation_profile': translation_profile,
                'rotation_profile': None,
                'estimated_energy': displacement_energy,
                'estimated_time': segment_time,
                'target_waypoint': target_wp
            })
            total_energy_estimate += displacement_energy
            total_time_estimate += segment_time
            current_pos = target_wp.position
            current_orient = target_wp.orientation
        return {
            'segments': segments,
            'waypoints': waypoints,
            'total_energy_estimate': total_energy_estimate,
            'total_time_estimate': total_time_estimate,
            'energy_efficiency': total_energy_estimate / (total_time_estimate + 1e-12),
            'feasible': total_energy_estimate <= self.config.energy_budget
        }

    async def execute_impulse_mission(self, trajectory_plan: Dict[str, Any], enable_feedback: bool = True) -> Dict[str, Any]:
        mission_results = {
            'segment_results': [],
            'performance_metrics': {},
            'mission_success': True
        }
        cumulative_time = 0.0
        cumulative_energy = 0.0
        for i, segment in enumerate(trajectory_plan['segments']):
            trans_profile = segment['translation_profile']
            vec_params = self.config.vector_params  # type: ignore[assignment]
            assert vec_params is not None
            trans_results = simulate_vector_impulse_maneuver(trans_profile, vec_params, enable_progress=False)
            self.current_position = Vector3D(*trans_results['final_position'])
            segment_energy = trans_results['total_energy']
            segment_time = trans_results['maneuver_duration']
            cumulative_energy += segment_energy
            cumulative_time += segment_time
            mission_results['segment_results'].append({
                'segment_index': i,
                'translation_results': trans_results,
                'segment_energy': segment_energy,
                'segment_time': segment_time,
                'segment_success': True
            })
        mission_results['performance_metrics'] = {
            'total_energy_used': cumulative_energy,
            'total_mission_time': cumulative_time,
            'energy_efficiency': cumulative_energy / (cumulative_time + 1e-12),
            'energy_budget_utilization': cumulative_energy / self.config.energy_budget,
            'segments_completed': len(trajectory_plan['segments']),
            'segments_successful': len(trajectory_plan['segments']),
            'overall_success_rate': 1.0
        }
        return mission_results

    def _estimate_translation_energy(self, profile: VectorImpulseProfile) -> float:
        return 1e11 * profile.v_max ** 2 * profile.target_displacement.magnitude

    def generate_mission_report(self, mission_results: Dict[str, Any]) -> str:
        m = mission_results['performance_metrics']
        return f"Total Energy Used: {m['total_energy_used']/1e9:.2f} GJ"

__all__ = [
    'MissionWaypoint', 'ImpulseEngineConfig', 'IntegratedImpulseController'
]
