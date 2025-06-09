# src/warp_engine/failure_modes.py
"""
Failure Modes & Recovery Module
==============================

This module implements comprehensive failure mode analysis and recovery
systems for warp drive operations. Includes:

1. Fault tree analysis and failure mode identification
2. Real-time fault detection and diagnosis
3. Automated recovery procedures and contingency protocols
4. Emergency shutdown and safe-mode operations
5. System health monitoring and predictive maintenance
6. Post-failure analysis and system restoration

Key Features:
- Hierarchical fault detection
- Automated recovery sequences
- Emergency protocols
- System diagnostics and health monitoring
- Failure prevention through predictive analysis
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
from collections import deque
import threading

logger = logging.getLogger(__name__)

class FailureSeverity(Enum):
    """Severity levels for system failures."""
    MINOR = "minor"          # Non-critical, continue operation
    MODERATE = "moderate"    # Degraded performance, monitor closely
    MAJOR = "major"         # Significant impact, initiate recovery
    CRITICAL = "critical"   # Mission threatening, emergency protocols
    CATASTROPHIC = "catastrophic"  # Immediate danger, emergency shutdown

class FailureType(Enum):
    """Types of failures in warp drive systems."""
    WARP_FIELD_COLLAPSE = "warp_field_collapse"
    EXOTIC_MATTER_LEAK = "exotic_matter_leak"
    ENERGY_SYSTEM_FAILURE = "energy_system_failure"
    STRUCTURAL_INTEGRITY = "structural_integrity"
    NAVIGATION_ERROR = "navigation_error"
    CONTROL_SYSTEM_FAULT = "control_system_fault"
    SENSOR_MALFUNCTION = "sensor_malfunction"
    COOLING_SYSTEM_FAILURE = "cooling_system_failure"
    COMMUNICATION_LOSS = "communication_loss"
    CREW_MEDICAL_EMERGENCY = "crew_medical_emergency"

class SystemState(Enum):
    """Overall system operational states."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    SAFE_MODE = "safe_mode"
    SHUTDOWN = "shutdown"
    RECOVERY = "recovery"

@dataclass
class FailureEvent:
    """Individual failure event record."""
    failure_id: str
    failure_type: FailureType
    severity: FailureSeverity
    timestamp: float
    description: str
    affected_systems: List[str]
    sensor_readings: Dict[str, float] = field(default_factory=dict)
    recovery_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[float] = None

@dataclass
class RecoveryAction:
    """Automated recovery action."""
    action_id: str
    action_type: str
    priority: int  # Lower numbers = higher priority
    target_system: str
    parameters: Dict = field(default_factory=dict)
    timeout: float = 30.0  # Maximum time to wait for action completion
    prerequisite_actions: List[str] = field(default_factory=list)

@dataclass
class SystemHealth:
    """System health metrics."""
    system_name: str
    health_score: float  # 0.0 (failed) to 1.0 (perfect)
    status: str
    last_maintenance: float
    operating_hours: float
    failure_count: int
    predicted_failure_time: Optional[float] = None

class FailureDetector(ABC):
    """Abstract base class for failure detection algorithms."""
    
    @abstractmethod
    def detect_failures(self, sensor_data: Dict[str, float], 
                       system_state: Dict) -> List[FailureEvent]:
        """Detect potential failures from sensor data."""
        pass
    
    @abstractmethod
    def get_health_score(self, sensor_data: Dict[str, float]) -> float:
        """Compute health score for monitored system."""
        pass

class WarpFieldFailureDetector(FailureDetector):
    """Failure detector for warp field systems."""
    
    def __init__(self, field_stability_threshold: float = 0.95,
                 collapse_warning_threshold: float = 0.8):
        self.field_stability_threshold = field_stability_threshold
        self.collapse_warning_threshold = collapse_warning_threshold
        self.field_history = deque(maxlen=100)
        
    def detect_failures(self, sensor_data: Dict[str, float], 
                       system_state: Dict) -> List[FailureEvent]:
        """Detect warp field related failures."""
        failures = []
        current_time = time.time()
        
        # Field strength monitoring
        field_strength = sensor_data.get("warp_field_strength", 0.0)
        field_stability = sensor_data.get("warp_field_stability", 1.0)
        
        self.field_history.append(field_stability)
        
        # Detect field collapse
        if field_stability < self.collapse_warning_threshold:
            severity = FailureSeverity.CRITICAL if field_stability < 0.5 else FailureSeverity.MAJOR
            
            failures.append(FailureEvent(
                failure_id=f"warp_field_instability_{current_time}",
                failure_type=FailureType.WARP_FIELD_COLLAPSE,
                severity=severity,
                timestamp=current_time,
                description=f"Warp field stability dropped to {field_stability:.2f}",
                affected_systems=["warp_core", "field_generators", "exotic_matter_containment"],
                sensor_readings={"field_stability": field_stability, "field_strength": field_strength}
            ))
            
        # Detect field oscillations
        if len(self.field_history) >= 10:
            field_variance = np.var(list(self.field_history)[-10:])
            if field_variance > 0.01:  # High variance indicates oscillations
                failures.append(FailureEvent(
                    failure_id=f"warp_field_oscillation_{current_time}",
                    failure_type=FailureType.WARP_FIELD_COLLAPSE,
                    severity=FailureSeverity.MODERATE,
                    timestamp=current_time,
                    description=f"Warp field oscillations detected (variance: {field_variance:.4f})",
                    affected_systems=["field_generators"],
                    sensor_readings={"field_variance": field_variance}
                ))
                
        return failures
        
    def get_health_score(self, sensor_data: Dict[str, float]) -> float:
        """Compute warp field system health score."""
        field_stability = sensor_data.get("warp_field_stability", 0.0)
        field_strength = sensor_data.get("warp_field_strength", 0.0)
        
        # Combine stability and strength metrics
        stability_score = min(1.0, field_stability / self.field_stability_threshold)
        strength_score = min(1.0, field_strength / 10.0)  # Assume max field strength of 10
        
        return (stability_score + strength_score) / 2.0

class ExoticMatterDetector(FailureDetector):
    """Failure detector for exotic matter containment systems."""
    
    def __init__(self, containment_pressure_limit: float = 1e6,
                 leak_detection_threshold: float = 0.01):
        self.containment_pressure_limit = containment_pressure_limit
        self.leak_detection_threshold = leak_detection_threshold
        self.pressure_history = deque(maxlen=50)
        
    def detect_failures(self, sensor_data: Dict[str, float], 
                       system_state: Dict) -> List[FailureEvent]:
        """Detect exotic matter containment failures."""
        failures = []
        current_time = time.time()
        
        # Containment pressure monitoring
        containment_pressure = sensor_data.get("exotic_matter_pressure", 0.0)
        matter_density = sensor_data.get("exotic_matter_density", 0.0)
        
        self.pressure_history.append(containment_pressure)
        
        # Detect pressure loss (potential leak)
        if len(self.pressure_history) >= 5:
            pressure_trend = np.polyfit(range(5), list(self.pressure_history)[-5:], 1)[0]
            if pressure_trend < -self.leak_detection_threshold:
                failures.append(FailureEvent(
                    failure_id=f"exotic_matter_leak_{current_time}",
                    failure_type=FailureType.EXOTIC_MATTER_LEAK,
                    severity=FailureSeverity.CRITICAL,
                    timestamp=current_time,
                    description=f"Exotic matter leak detected (pressure drop rate: {pressure_trend:.4f})",
                    affected_systems=["containment_field", "exotic_matter_storage"],
                    sensor_readings={"pressure_trend": pressure_trend, "current_pressure": containment_pressure}
                ))
                
        # Detect over-pressurization
        if containment_pressure > self.containment_pressure_limit:
            failures.append(FailureEvent(
                failure_id=f"exotic_matter_overpressure_{current_time}",
                failure_type=FailureType.EXOTIC_MATTER_LEAK,
                severity=FailureSeverity.MAJOR,
                timestamp=current_time,
                description=f"Exotic matter containment over-pressurized: {containment_pressure:.0f} Pa",
                affected_systems=["containment_field"],
                sensor_readings={"containment_pressure": containment_pressure}
            ))
            
        return failures
        
    def get_health_score(self, sensor_data: Dict[str, float]) -> float:
        """Compute exotic matter system health score."""
        pressure = sensor_data.get("exotic_matter_pressure", 0.0)
        density = sensor_data.get("exotic_matter_density", 0.0)
        
        # Pressure should be within safe operating range
        pressure_score = 1.0 - abs(pressure - self.containment_pressure_limit * 0.7) / (self.containment_pressure_limit * 0.3)
        pressure_score = max(0.0, min(1.0, pressure_score))
        
        # Density should be stable
        density_score = min(1.0, density / 1000.0) if density > 0 else 0.0
        
        return (pressure_score + density_score) / 2.0

class StructuralIntegrityDetector(FailureDetector):
    """Failure detector for structural integrity monitoring."""
    
    def __init__(self, stress_limit: float = 1e8, vibration_threshold: float = 10.0):
        self.stress_limit = stress_limit
        self.vibration_threshold = vibration_threshold
        self.vibration_history = deque(maxlen=20)
        
    def detect_failures(self, sensor_data: Dict[str, float], 
                       system_state: Dict) -> List[FailureEvent]:
        """Detect structural integrity issues."""
        failures = []
        current_time = time.time()
        
        # Structural stress monitoring
        hull_stress = sensor_data.get("hull_stress", 0.0)
        vibration_level = sensor_data.get("vibration_amplitude", 0.0)
        
        self.vibration_history.append(vibration_level)
        
        # Detect excessive stress
        if hull_stress > self.stress_limit:
            severity = FailureSeverity.CRITICAL if hull_stress > self.stress_limit * 1.2 else FailureSeverity.MAJOR
            
            failures.append(FailureEvent(
                failure_id=f"structural_stress_{current_time}",
                failure_type=FailureType.STRUCTURAL_INTEGRITY,
                severity=severity,
                timestamp=current_time,
                description=f"Hull stress exceeded safe limits: {hull_stress:.0f} Pa",
                affected_systems=["hull", "structural_framework"],
                sensor_readings={"hull_stress": hull_stress}
            ))
            
        # Detect excessive vibration
        if vibration_level > self.vibration_threshold:
            failures.append(FailureEvent(
                failure_id=f"excessive_vibration_{current_time}",
                failure_type=FailureType.STRUCTURAL_INTEGRITY,
                severity=FailureSeverity.MODERATE,
                timestamp=current_time,
                description=f"Excessive structural vibration: {vibration_level:.2f} m/s²",
                affected_systems=["structural_framework", "equipment_mounts"],
                sensor_readings={"vibration_amplitude": vibration_level}
            ))
            
        return failures
        
    def get_health_score(self, sensor_data: Dict[str, float]) -> float:
        """Compute structural system health score."""
        stress = sensor_data.get("hull_stress", 0.0)
        vibration = sensor_data.get("vibration_amplitude", 0.0)
        
        stress_score = max(0.0, 1.0 - stress / self.stress_limit)
        vibration_score = max(0.0, 1.0 - vibration / self.vibration_threshold)
        
        return (stress_score + vibration_score) / 2.0

class RecoveryManager:
    """
    Manages automated recovery procedures for various failure modes.
    
    Implements hierarchical recovery strategies from simple parameter
    adjustments to emergency shutdown procedures.
    """
    
    def __init__(self):
        self.recovery_procedures: Dict[FailureType, List[RecoveryAction]] = {}
        self.active_recoveries: Dict[str, RecoveryAction] = {}
        self.recovery_history: List[Tuple[str, bool, float]] = []
        
        self._initialize_recovery_procedures()
        
    def _initialize_recovery_procedures(self):
        """Initialize default recovery procedures for common failure modes."""
        
        # Warp field collapse recovery
        self.recovery_procedures[FailureType.WARP_FIELD_COLLAPSE] = [
            RecoveryAction(
                action_id="reduce_warp_factor",
                action_type="parameter_adjustment",
                priority=1,
                target_system="warp_core",
                parameters={"warp_factor_reduction": 0.5, "ramp_rate": 0.1},
                timeout=10.0
            ),
            RecoveryAction(
                action_id="stabilize_field_generators",
                action_type="system_recalibration",
                priority=2,
                target_system="field_generators",
                parameters={"calibration_mode": "emergency", "power_level": 0.8},
                timeout=30.0
            ),
            RecoveryAction(
                action_id="emergency_field_shutdown",
                action_type="emergency_shutdown",
                priority=3,
                target_system="warp_core",
                parameters={"shutdown_type": "controlled"},
                timeout=5.0,
                prerequisite_actions=["reduce_warp_factor"]
            )
        ]
        
        # Exotic matter leak recovery
        self.recovery_procedures[FailureType.EXOTIC_MATTER_LEAK] = [
            RecoveryAction(
                action_id="seal_containment_breach",
                action_type="containment_repair",
                priority=1,
                target_system="containment_field",
                parameters={"field_strength_increase": 1.5, "backup_systems": True},
                timeout=15.0
            ),
            RecoveryAction(
                action_id="isolate_damaged_section",
                action_type="system_isolation",
                priority=2,
                target_system="exotic_matter_storage",
                parameters={"isolation_valves": ["primary", "secondary"]},
                timeout=10.0
            ),
            RecoveryAction(
                action_id="emergency_matter_jettison",
                action_type="emergency_jettison",
                priority=3,
                target_system="exotic_matter_storage",
                parameters={"jettison_amount": 0.8, "safety_distance": 1000.0},
                timeout=20.0
            )
        ]
        
        # Energy system failure recovery
        self.recovery_procedures[FailureType.ENERGY_SYSTEM_FAILURE] = [
            RecoveryAction(
                action_id="switch_to_backup_power",
                action_type="power_source_switch",
                priority=1,
                target_system="power_management",
                parameters={"backup_systems": ["fusion_reactor_2", "battery_bank"]},
                timeout=5.0
            ),
            RecoveryAction(
                action_id="reduce_power_consumption",
                action_type="load_shedding",
                priority=2,
                target_system="power_management",
                parameters={"non_essential_systems": ["life_support_secondary", "communications"]},
                timeout=2.0
            )
        ]
        
        # Structural integrity recovery
        self.recovery_procedures[FailureType.STRUCTURAL_INTEGRITY] = [
            RecoveryAction(
                action_id="reduce_acceleration",
                action_type="parameter_adjustment",
                priority=1,
                target_system="propulsion",
                parameters={"max_acceleration": 5.0},  # Reduce to 5 m/s²
                timeout=5.0
            ),
            RecoveryAction(
                action_id="activate_structural_reinforcement",
                action_type="system_activation",
                priority=2,
                target_system="structural_framework",
                parameters={"reinforcement_level": "maximum"},
                timeout=10.0
            )
        ]
        
    def get_recovery_actions(self, failure: FailureEvent) -> List[RecoveryAction]:
        """Get appropriate recovery actions for a specific failure."""
        base_actions = self.recovery_procedures.get(failure.failure_type, [])
        
        # Filter actions based on failure severity
        if failure.severity == FailureSeverity.MINOR:
            # Only minor adjustments for minor failures
            return [action for action in base_actions if action.priority <= 1]
        elif failure.severity == FailureSeverity.MODERATE:
            return [action for action in base_actions if action.priority <= 2]
        else:
            # All actions available for major/critical failures
            return base_actions
            
    def execute_recovery_action(self, action: RecoveryAction, 
                              system_interface: Dict[str, Callable]) -> bool:
        """Execute a recovery action using available system interfaces."""
        try:
            logger.info(f"Executing recovery action: {action.action_id}")
            
            # Check prerequisites
            for prereq in action.prerequisite_actions:
                if prereq not in [a.action_id for a in self.active_recoveries.values()]:
                    logger.warning(f"Prerequisite {prereq} not met for {action.action_id}")
                    return False
                    
            # Get system interface
            system_func = system_interface.get(action.target_system)
            if not system_func:
                logger.error(f"No interface available for system: {action.target_system}")
                return False
                
            # Execute action with timeout
            start_time = time.time()
            self.active_recoveries[action.action_id] = action
            
            # Call system interface
            success = system_func(action.action_type, action.parameters)
            
            execution_time = time.time() - start_time
            
            if success:
                logger.info(f"Recovery action {action.action_id} completed successfully in {execution_time:.2f}s")
                self.recovery_history.append((action.action_id, True, execution_time))
            else:
                logger.error(f"Recovery action {action.action_id} failed")
                self.recovery_history.append((action.action_id, False, execution_time))
                
            # Remove from active recoveries
            self.active_recoveries.pop(action.action_id, None)
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing recovery action {action.action_id}: {e}")
            self.active_recoveries.pop(action.action_id, None)
            return False
            
    def add_custom_recovery(self, failure_type: FailureType, action: RecoveryAction):
        """Add custom recovery action for specific failure type."""
        if failure_type not in self.recovery_procedures:
            self.recovery_procedures[failure_type] = []
        self.recovery_procedures[failure_type].append(action)
        
        # Sort by priority
        self.recovery_procedures[failure_type].sort(key=lambda x: x.priority)


class HealthMonitor:
    """
    System health monitoring and predictive maintenance component.
    
    Tracks system health metrics, predicts failure times, and 
    recommends maintenance schedules.
    """
    
    def __init__(self):
        self.system_health: Dict[str, SystemHealth] = {}
        
    def update_system_health(self, system_name: str, health_score: float,
                           status: str, operating_hours: float):
        """Update health information for a system."""
        if system_name not in self.system_health:
            self.system_health[system_name] = SystemHealth(
                health_score=health_score,
                status=status,
                last_maintenance=operating_hours,
                operating_hours=operating_hours,
                failure_count=0,
                predicted_failure_time=None
            )
        else:
            health = self.system_health[system_name]
            health.health_score = health_score
            health.status = status
            health.operating_hours = operating_hours
            
        # Update failure prediction
        self._update_failure_prediction(system_name)
        
    def _update_failure_prediction(self, system_name: str):
        """Update failure prediction for a system based on health trends."""
        health = self.system_health[system_name]
        
        # Simple linear degradation model
        # In practice, this would use more sophisticated ML models
        degradation_rate = (1.0 - health.health_score) / max(1.0, health.operating_hours)
        
        if degradation_rate > 0:
            # Time until health score reaches critical threshold (0.3)
            remaining_health = health.health_score - 0.3
            if remaining_health > 0:
                predicted_time = remaining_health / degradation_rate
                health.predicted_failure_time = time.time() + predicted_time * 3600  # Convert to absolute time
            else:
                health.predicted_failure_time = time.time()  # Already critical
        else:
            health.predicted_failure_time = None  # System improving
            
    def get_systems_needing_maintenance(self, threshold: float = 0.7) -> List[str]:
        """Get list of systems with health below threshold."""
        return [name for name, health in self.system_health.items() 
                if health.health_score < threshold]
        
    def get_systems_at_risk(self, time_horizon: float = 7*24*3600) -> List[str]:
        """Get systems predicted to fail within time horizon."""
        current_time = time.time()
        at_risk = []
        
        for name, health in self.system_health.items():
            if (health.predicted_failure_time and 
                health.predicted_failure_time - current_time < time_horizon):
                at_risk.append(name)
                
        return at_risk


class FailureModeManager:
    """
    Main failure mode analysis and recovery management system.
    
    Integrates failure detection, recovery management, and health
    monitoring for comprehensive system reliability.
    """
    
    def __init__(self):
        self.detectors: Dict[str, FailureDetector] = {}
        self.recovery_manager = RecoveryManager()
        self.health_monitor = HealthMonitor()
        
        self.active_failures: List[FailureEvent] = []
        self.failure_history: List[FailureEvent] = []
        self.system_state = SystemState.NORMAL
        
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        self._initialize_detectors()
        
    def _initialize_detectors(self):
        """Initialize failure detectors for various subsystems."""
        self.detectors["warp_field"] = WarpFieldFailureDetector()
        self.detectors["exotic_matter"] = ExoticMatterDetector()
        self.detectors["structural"] = StructuralIntegrityDetector()
        
    def add_detector(self, name: str, detector: FailureDetector):
        """Add custom failure detector."""
        self.detectors[name] = detector
        
    def start_monitoring(self, sensor_interface: Callable[[], Dict[str, float]],
                        system_interface: Dict[str, Callable],
                        monitoring_interval: float = 1.0):
        """Start continuous failure monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.sensor_interface = sensor_interface
        self.system_interface = system_interface
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Get current sensor data
                    sensor_data = self.sensor_interface()
                    
                    # Run failure detection
                    self._run_failure_detection(sensor_data)
                    
                    # Update system health
                    self._update_system_health(sensor_data)
                    
                    # Process active failures
                    self._process_active_failures()
                    
                    time.sleep(monitoring_interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    
        self.monitoring_thread = threading.Thread(target=monitoring_loop)
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop the failure monitoring system."""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join()
            logger.info("Failure monitoring stopped")
        else:
            logger.info("Monitoring was not active")
            
    def _run_failure_detection(self, sensor_data: Dict[str, float]):
        """Run failure detection on all registered detectors."""
        for detector_name, detector in self.detectors.items():
            try:
                failures = detector.detect_failures(sensor_data, {})
                
                for failure in failures:
                    # Check if this is a duplicate failure
                    if not self._is_duplicate_failure(failure):
                        self.active_failures.append(failure)
                        self.failure_history.append(failure)
                        
                        logger.warning(f"Failure detected: {failure.description}")
                        
                        # Trigger recovery if needed
                        if failure.severity in [FailureSeverity.MAJOR, FailureSeverity.CRITICAL, FailureSeverity.CATASTROPHIC]:
                            self._initiate_recovery(failure)
                            
            except Exception as e:
                logger.error(f"Error in detector {detector_name}: {e}")
                
    def _update_system_health(self, sensor_data: Dict[str, float]):
        """Update health scores for all monitored systems."""
        for detector_name, detector in self.detectors.items():
            try:
                health_score = detector.get_health_score(sensor_data)
                operating_hours = time.time() / 3600.0  # Simple approximation
                
                self.health_monitor.update_system_health(
                    detector_name, health_score, "operational", operating_hours
                )
                
            except Exception as e:
                logger.error(f"Error updating health for {detector_name}: {e}")
                
    def _is_duplicate_failure(self, new_failure: FailureEvent) -> bool:
        """Check if failure is duplicate of existing active failure."""
        for active_failure in self.active_failures:
            if (active_failure.failure_type == new_failure.failure_type and
                not active_failure.resolved and
                new_failure.timestamp - active_failure.timestamp < 60.0):  # Within 1 minute
                return True
        return False
        
    def _initiate_recovery(self, failure: FailureEvent):
        """Initiate recovery procedures for a failure."""
        logger.info(f"Initiating recovery for failure: {failure.failure_id}")
        
        recovery_actions = self.recovery_manager.get_recovery_actions(failure)
        
        for action in recovery_actions:
            success = self.recovery_manager.execute_recovery_action(
                action, self.system_interface
            )
            
            if success:
                failure.recovery_actions.append(action.action_id)
                logger.info(f"Recovery action {action.action_id} successful")
                
                # Check if failure is resolved
                if self._check_failure_resolution(failure):
                    failure.resolved = True
                    failure.resolution_time = time.time()
                    logger.info(f"Failure {failure.failure_id} resolved")
                    break
            else:
                logger.error(f"Recovery action {action.action_id} failed")
                
        # Update system state based on failure severity
        if failure.severity == FailureSeverity.CATASTROPHIC:
            self.system_state = SystemState.EMERGENCY
        elif failure.severity == FailureSeverity.CRITICAL:
            self.system_state = SystemState.DEGRADED
            
    def _check_failure_resolution(self, failure: FailureEvent) -> bool:
        """Check if a failure has been resolved."""
        # Simple check based on recovery actions taken
        # In practice, this would re-evaluate sensor data
        return len(failure.recovery_actions) > 0
        
    def _process_active_failures(self):
        """Process and clean up resolved failures."""
        # Remove resolved failures from active list
        self.active_failures = [f for f in self.active_failures if not f.resolved]
        
        # Update system state based on active failures
        if not self.active_failures:
            self.system_state = SystemState.NORMAL
        elif any(f.severity == FailureSeverity.CATASTROPHIC for f in self.active_failures):
            self.system_state = SystemState.EMERGENCY
        elif any(f.severity in [FailureSeverity.CRITICAL, FailureSeverity.MAJOR] for f in self.active_failures):
            self.system_state = SystemState.DEGRADED
            
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        return {
            "system_state": self.system_state.value,
            "active_failures": len(self.active_failures),
            "total_failures": len(self.failure_history),
            "systems_needing_maintenance": self.health_monitor.get_systems_needing_maintenance(),
            "systems_at_risk": self.health_monitor.get_systems_at_risk(),
            "monitoring_active": self.monitoring_active
        }
        
    def generate_failure_report(self) -> Dict:
        """Generate comprehensive failure analysis report."""
        # Failure statistics
        failure_counts = {}
        for failure in self.failure_history:
            failure_type = failure.failure_type.value
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
            
        # Resolution statistics
        resolved_failures = [f for f in self.failure_history if f.resolved]
        resolution_times = [f.resolution_time - f.timestamp for f in resolved_failures if f.resolution_time]
        
        avg_resolution_time = np.mean(resolution_times) if resolution_times else 0.0
        
        # System health summary
        health_summary = {}
        for system_name, health in self.health_monitor.system_health.items():
            health_summary[system_name] = {
                "health_score": health.health_score,
                "status": health.status,
                "failure_count": health.failure_count,
                "predicted_failure_time": health.predicted_failure_time
            }
            
        return {
            "total_failures": len(self.failure_history),
            "failure_counts_by_type": failure_counts,
            "resolution_rate": len(resolved_failures) / len(self.failure_history) if self.failure_history else 1.0,
            "average_resolution_time": avg_resolution_time,
            "system_health": health_summary,
            "current_state": self.system_state.value,
            "active_failures": len(self.active_failures)
        }

    def design_recovery_procedure(self, scenario: str, mission_params: Dict) -> Dict:
        """
        Design recovery procedures for specific failure scenarios.
        
        Args:
            scenario: Failure scenario identifier
            mission_params: Mission parameters that may affect recovery
            
        Returns:
            Dict containing recovery procedure details
        """
        logger.info(f"Designing recovery procedure for scenario: {scenario}")
        
        # Common recovery steps based on scenario type
        base_procedures = {
            "warp_field_collapse": [
                "emergency_shutdown_warp_core",
                "stabilize_exotic_matter_containment", 
                "assess_structural_integrity",
                "restart_warp_core_safe_mode",
                "gradual_field_restoration"
            ],
            "exotic_matter_leak": [
                "emergency_containment_protocol",
                "evacuate_affected_areas",
                "activate_backup_containment",
                "assess_leak_severity",
                "repair_containment_systems"
            ],
            "power_system_failure": [
                "switch_to_backup_power",
                "isolate_failed_components",
                "assess_power_requirements",
                "redistribute_power_loads",
                "restore_primary_power"
            ],
            "navigation_error": [
                "emergency_stop_warp_drive",
                "recalibrate_navigation_sensors",
                "verify_position_fix",
                "plot_corrective_course",
                "resume_normal_operations"
            ]
        }
        
        # Get base procedure or create generic one
        procedure_steps = base_procedures.get(scenario, [
            "assess_situation",
            "implement_emergency_protocols", 
            "isolate_failed_systems",
            "execute_recovery_actions",
            "verify_system_restoration"
        ])
        
        # Calculate estimated recovery time based on mission parameters
        base_recovery_time = 300.0  # 5 minutes base
        complexity_factor = mission_params.get("mission_complexity", 1.0)
        estimated_time = base_recovery_time * complexity_factor
        
        return {
            "scenario": scenario,
            "procedure_steps": procedure_steps,
            "estimated_recovery_time": estimated_time,
            "required_personnel": 2,
            "critical_systems": ["warp_core", "life_support", "navigation"],
            "success_probability": 0.85,
            "backup_procedures": ["manual_override", "emergency_evacuation"],
            "resource_requirements": {
                "power": mission_params.get("power_requirement", 1000.0) * 0.1,
                "time": estimated_time,
                "crew": 2
            }
        }
    
    def generate_emergency_protocols(self, threat_level: str, available_resources: Dict) -> Dict:
        """
        Generate emergency protocols based on threat level and available resources.
        
        Args:
            threat_level: Level of emergency threat (low, medium, high, critical)
            available_resources: Dict of currently available resources
            
        Returns:
            Dict containing emergency protocol details
        """
        logger.info(f"Generating emergency protocols for threat level: {threat_level}")
        
        # Protocol templates based on threat level
        protocol_templates = {
            "low": {
                "response_time": 60.0,  # 1 minute
                "evacuation_required": False,
                "automatic_shutdown": False,
                "crew_alert_level": "yellow",
                "priority_systems": ["life_support", "navigation"]
            },
            "medium": {
                "response_time": 30.0,  # 30 seconds
                "evacuation_required": False,
                "automatic_shutdown": False,
                "crew_alert_level": "orange", 
                "priority_systems": ["life_support", "warp_core", "navigation"]
            },
            "high": {
                "response_time": 15.0,  # 15 seconds
                "evacuation_required": True,
                "automatic_shutdown": True,
                "crew_alert_level": "red",
                "priority_systems": ["life_support", "emergency_systems"]
            },
            "critical": {
                "response_time": 5.0,   # 5 seconds
                "evacuation_required": True,
                "automatic_shutdown": True,
                "crew_alert_level": "critical",
                "priority_systems": ["life_support", "emergency_evacuation"]
            }
        }
        
        # Get base protocol
        base_protocol = protocol_templates.get(threat_level, protocol_templates["medium"])
        
        # Adapt protocol based on available resources
        power_available = available_resources.get("power", 0.0)
        crew_available = available_resources.get("crew", 0)
        
        # Adjust protocol based on resource constraints
        if power_available < 1000.0:  # Low power
            base_protocol["priority_systems"] = ["life_support"]
            base_protocol["automatic_shutdown"] = True
            
        if crew_available < 2:  # Minimal crew
            base_protocol["evacuation_required"] = False  # Can't evacuate with no crew
            
        # Generate action sequence
        action_sequence = [
            "sound_general_alarm",
            "notify_all_crew",
            f"implement_{threat_level}_threat_protocols",
            "secure_critical_systems",
            "prepare_emergency_systems"
        ]
        
        if base_protocol["evacuation_required"]:
            action_sequence.extend([
                "initiate_evacuation_procedures",
                "secure_escape_pods",
                "transmit_distress_signal"
            ])
            
        if base_protocol["automatic_shutdown"]:
            action_sequence.extend([
                "shutdown_warp_drive",
                "secure_exotic_matter_containment",
                "activate_emergency_power"
            ])
            
        return {
            "threat_level": threat_level,
            "response_time": base_protocol["response_time"],
            "evacuation_required": base_protocol["evacuation_required"],
            "automatic_shutdown": base_protocol["automatic_shutdown"],
            "crew_alert_level": base_protocol["crew_alert_level"],
            "priority_systems": base_protocol["priority_systems"],
            "action_sequence": action_sequence,
            "resource_allocation": {
                "emergency_power": min(power_available, 5000.0),
                "crew_assignments": min(crew_available, 4),
                "backup_systems": ["emergency_life_support", "distress_beacon"]
            },
            "communication_protocols": [
                "internal_ship_announcement",
                "external_distress_call",
                "mission_control_notification"
            ],
            "success_criteria": [
                "all_crew_accounted_for",
                "critical_systems_stable", 
                "emergency_systems_operational"
            ]
        }
        
# Mock system interfaces for testing
def mock_sensor_interface() -> Dict[str, float]:
    """Mock sensor interface returning simulated sensor data."""
    return {
        "warp_field_strength": 8.5 + 0.5 * np.random.randn(),
        "warp_field_stability": 0.95 + 0.05 * np.random.randn(),
        "exotic_matter_pressure": 5e5 + 1e4 * np.random.randn(),
        "exotic_matter_density": 100.0 + 5.0 * np.random.randn(),
        "hull_stress": 5e7 + 1e6 * np.random.randn(),
        "vibration_amplitude": 2.0 + 0.5 * np.random.randn(),
    }

def mock_system_interface(system: str, action: str, parameters: Dict) -> bool:
    """Mock system interface for testing recovery actions."""
    logger.info(f"Mock system interface: {system} -> {action} with params {parameters}")
    # Simulate action execution time
    time.sleep(0.1)
    # Simulate 90% success rate
    return np.random.random() > 0.1

# Example usage
# Utility functions for backwards compatibility
def simulate_collapse(bubble_radius: float, warp_velocity: float, pump_off_time: float) -> Dict:
    """
    Utility function to simulate bubble collapse for backwards compatibility.
    
    Args:
        bubble_radius: Initial bubble radius in meters
        warp_velocity: Warp velocity in units of c
        pump_off_time: Time when pump turns off in seconds
        
    Returns:
        Dict with collapse_time and horizon_formation
    """
    # Simple collapse simulation based on energy decay
    # Collapse time scales with bubble size and available energy
    energy_density = 1e30 * warp_velocity**2 / bubble_radius**3  # Rough estimate
    
    # Time for energy to dissipate
    collapse_time = pump_off_time + bubble_radius / (0.1 * 3e8)  # Sound speed propagation
    
    # Horizon formation depends on collapse speed vs bubble size
    horizon_formation = collapse_time < bubble_radius / 3e8  # Light crossing time
    
    return {
        'collapse_time': collapse_time,
        'horizon_formation': horizon_formation,
        'energy_density': energy_density
    }


if __name__ == "__main__":
    # Create failure mode manager
    manager = FailureModeManager()
    
    # Create mock system interface
    system_interfaces = {
        "warp_core": lambda action, params: mock_system_interface("warp_core", action, params),
        "field_generators": lambda action, params: mock_system_interface("field_generators", action, params),
        "containment_field": lambda action, params: mock_system_interface("containment_field", action, params),
        "exotic_matter_storage": lambda action, params: mock_system_interface("exotic_matter_storage", action, params),
        "power_management": lambda action, params: mock_system_interface("power_management", action, params),
        "propulsion": lambda action, params: mock_system_interface("propulsion", action, params),
        "structural_framework": lambda action, params: mock_system_interface("structural_framework", action, params),
    }
    
    # Start monitoring
    print("Starting failure monitoring system...")
    manager.start_monitoring(mock_sensor_interface, system_interfaces, monitoring_interval=0.5)
    
    try:
        # Run for a short time
        print("Monitoring system for 10 seconds...")
        time.sleep(10)
        
        # Check status
        status = manager.get_system_status()
        print(f"\nSystem Status:")
        print(f"State: {status['system_state']}")
        print(f"Active Failures: {status['active_failures']}")
        print(f"Total Failures: {status['total_failures']}")
        print(f"Systems Needing Maintenance: {status['systems_needing_maintenance']}")
        
        # Generate failure report
        report = manager.generate_failure_report()
        print(f"\nFailure Report:")
        print(f"Total Failures: {report['total_failures']}")
        print(f"Resolution Rate: {report['resolution_rate']:.1%}")
        print(f"Average Resolution Time: {report['average_resolution_time']:.2f}s")
        
    finally:
        # Stop monitoring
        print("\nStopping failure monitoring...")
        manager.stop_monitoring()
        print("Failure monitoring stopped.")
