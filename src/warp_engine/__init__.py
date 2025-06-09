"""
Warp Engine: Complete Implementation Framework
===========================================

This package implements the complete warp engine concept evolution from 
validated negative-energy sources to a full operational warp drive system.

Modules:
--------
1. backreaction: Einstein field equation solver with full back-reaction
2. dynamic_sim: Time-dependent dynamic bubble simulation  
3. tidal_analysis: Crew safety and geodesic deviation analysis
4. control_loop: Real-time feedback control architecture
5. analog_prototype: Laboratory analog and table-top prototyping
6. hardware_loop: Hardware-in-the-loop integration
7. mission_planner: Mission profiling and energy budgeting
8. failure_modes: Safety analysis and failure recovery

Usage:
------
from warp_engine import WarpEngineOrchestrator
engine = WarpEngineOrchestrator()
result = engine.run_full_simulation()
"""

from .backreaction import EinsteinSolver, BackreactionAnalyzer
from .dynamic_sim import DynamicBubbleSimulator, TrajectoryProfile
from .tidal_analysis import TidalForceAnalyzer, CrewSafetyAssessment
from .control_loop import WarpControlLoop, StabilityController, SensorInterface, ActuatorInterface
from .analog_prototype import AnalogPrototypeManager, WaterTankAnalogue
from .hardware_loop import HardwareInTheLoopManager, SensorInterface, ActuatorInterface
from .mission_planner import MissionPlanningManager, WarpTrajectoryOptimizer
from .failure_modes import FailureModeManager, RecoveryManager
from .orchestrator import WarpEngineOrchestrator

__version__ = "1.0.0"
__author__ = "Warp Engine Development Team"

__all__ = [
    'EinsteinSolver', 'BackreactionAnalyzer',
    'DynamicBubbleSimulator', 'TrajectoryProfile', 
    'TidalForceAnalyzer', 'CrewSafetyAssessment',
    'WarpControlLoop', 'StabilityController', 'MockSensor', 'MockActuator',
    'AnalogPrototypeManager', 'WaterTankAnalogue',
    'HardwareInTheLoopManager', 'SensorInterface', 'ActuatorInterface',
    'MissionPlanningManager', 'WarpTrajectoryOptimizer', 
    'FailureModeManager', 'RecoveryManager',
    'WarpEngineOrchestrator'
]
