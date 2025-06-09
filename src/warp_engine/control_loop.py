"""
Control-Loop & Feedback Architecture
===================================

This module implements real-time control systems for warp bubble stability:
- PID controllers for bubble parameters
- Stability monitoring and feedback
- Metric strain gauge simulation
- Real-time parameter adjustment
- Control bandwidth and latency analysis

Features:
- Multi-input multi-output (MIMO) control system
- Adaptive control algorithms
- Sensor fusion for stability estimation
- Actuator control for Ghost EFT pumps
- Safety interlocks and emergency shutdown
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import logging
from collections import deque
from scipy import signal
from scipy.optimize import minimize
import threading
import queue

logger = logging.getLogger(__name__)

@dataclass
class ControlState:
    """Current state of warp bubble control system."""
    timestamp: float
    bubble_radius: float  # m
    bubble_speed: float   # m/s  
    stability_metric: float  # [0,1]
    energy_density: float    # J/m³
    tidal_forces: float      # m/s²
    pump_power: float        # W
    pump_phase: float        # rad

@dataclass
class ControlCommand:
    """Control commands for actuators."""
    pump_power_delta: float  # W
    pump_phase_delta: float  # rad
    bubble_radius_target: float  # m
    emergency_shutdown: bool

@dataclass
class ControllerParams:
    """PID controller parameters."""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain
    setpoint: float  # Target value
    output_limits: Tuple[float, float]  # (min, max) output

@dataclass
class ControlConfig:
    """Configuration for control system."""
    target_stability: float = 0.95
    target_radius: float = 10.0
    control_frequency: float = 10.0  # Hz
    max_pump_power: float = 5000.0   # W
    max_phase_adjustment: float = 0.1  # rad
    safety_margins: Dict[str, float] = None
    
    def __post_init__(self):
        if self.safety_margins is None:
            self.safety_margins = {
                "min_stability": 0.3,
                "max_tidal_forces": 50.0,
                "max_pump_power": 5000.0,
                "min_bubble_radius": 2.0
            }

class SensorInterface:
    """
    Simulates sensor readings for warp bubble monitoring.
    """
    
    def __init__(self, noise_level: float = 0.01):
        self.noise_level = noise_level
        self.last_reading_time = time.time()
        
    def read_stability_metric(self) -> float:
        """Read bubble stability metric from sensors."""
        # Simulate noisy sensor reading
        base_stability = 0.85 + 0.1 * np.sin(time.time() * 0.1)  # Slow drift
        noise = self.noise_level * np.random.randn()
        return max(0.0, min(1.0, base_stability + noise))
    
    def read_bubble_radius(self) -> float:
        """Read bubble radius from interferometric measurements."""
        base_radius = 10.0 + 2.0 * np.sin(time.time() * 0.05)  # Slow oscillation
        noise = self.noise_level * np.random.randn()
        return max(1.0, base_radius + noise)
    
    def read_energy_density(self) -> float:
        """Read negative energy density."""
        base_density = -1e20 * (1 + 0.1 * np.random.randn())
        return base_density
    
    def read_tidal_forces(self) -> float:
        """Read tidal force measurements."""
        base_tidal = 5.0 + 2.0 * np.random.randn()  # m/s²
        return max(0.0, base_tidal)
    
    def read_pump_status(self) -> Tuple[float, float]:
        """Read pump power and phase."""
        power = 1000.0 + 100.0 * np.random.randn()  # W
        phase = 0.1 + 0.05 * np.random.randn()      # rad
        return max(0.0, power), phase

class ActuatorInterface:
    """
    Interface for controlling warp bubble actuators.
    """
    
    def __init__(self, response_time: float = 0.1):
        self.response_time = response_time  # seconds
        self.current_pump_power = 1000.0   # W
        self.current_pump_phase = 0.0      # rad
        self.emergency_stop = False
        
    def adjust_pump_power(self, delta_power: float):
        """Adjust pump power by delta amount."""
        if not self.emergency_stop:
            self.current_pump_power = max(0.0, self.current_pump_power + delta_power)
            time.sleep(self.response_time * 0.1)  # Simulate response delay
    
    def adjust_pump_phase(self, delta_phase: float):
        """Adjust pump phase by delta amount."""
        if not self.emergency_stop:
            self.current_pump_phase += delta_phase
            # Keep phase in [-π, π] range
            self.current_pump_phase = np.mod(self.current_pump_phase + np.pi, 2*np.pi) - np.pi
            time.sleep(self.response_time * 0.1)
    
    def emergency_shutdown(self):
        """Emergency shutdown of all systems."""
        self.emergency_stop = True
        self.current_pump_power = 0.0
        logger.critical("EMERGENCY SHUTDOWN ACTIVATED")
    
    def reset_emergency(self):
        """Reset emergency stop."""
        self.emergency_stop = False
        logger.info("Emergency stop reset")

class PIDController:
    """
    PID controller implementation.
    """
    
    def __init__(self, params: ControllerParams, dt: float = 0.1):
        self.params = params
        self.dt = dt
        
        # Controller state
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        
    def update(self, measured_value: float) -> float:
        """Update controller and return control output."""
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Calculate error
        error = self.params.setpoint - measured_value
        
        # Proportional term
        proportional = self.params.kp * error
        
        # Integral term
        self.integral += error * dt
        integral_term = self.params.ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative = (error - self.last_error) / dt
        else:
            derivative = 0.0
        derivative_term = self.params.kd * derivative
        
        # Combine terms
        output = proportional + integral_term + derivative_term
        
        # Apply output limits
        output = max(self.params.output_limits[0], 
                    min(self.params.output_limits[1], output))
        
        # Update state
        self.last_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()

class StabilityController:
    """
    Multi-variable controller for warp bubble stability.
    """
    
    def __init__(self, 
                 target_stability: float = 0.95,
                 target_radius: float = 10.0,
                 control_frequency: float = 10.0):  # Hz
        
        self.target_stability = target_stability
        self.target_radius = target_radius
        self.control_period = 1.0 / control_frequency
        
        # Initialize PID controllers
        self.stability_controller = PIDController(
            ControllerParams(kp=1000.0, ki=100.0, kd=50.0, 
                           setpoint=target_stability,
                           output_limits=(-500.0, 500.0))  # Power adjustment limits
        )
        
        self.radius_controller = PIDController(
            ControllerParams(kp=50.0, ki=10.0, kd=5.0,
                           setpoint=target_radius, 
                           output_limits=(-0.1, 0.1))  # Phase adjustment limits
        )
        
        # State history for analysis
        self.state_history = deque(maxlen=1000)  # Keep last 1000 states
        
        # Safety limits
        self.safety_limits = {
            "min_stability": 0.3,
            "max_tidal_forces": 50.0,  # m/s²
            "max_pump_power": 5000.0,  # W
            "min_bubble_radius": 2.0   # m
        }
        
    def compute_control_action(self, current_state: ControlState) -> ControlCommand:
        """Compute control action based on current state."""
        
        # Safety checks
        if self._safety_check(current_state):
            return ControlCommand(
                pump_power_delta=0.0,
                pump_phase_delta=0.0,
                bubble_radius_target=current_state.bubble_radius,
                emergency_shutdown=True
            )
        
        # Stability control (adjust pump power)
        stability_output = self.stability_controller.update(current_state.stability_metric)
        
        # Radius control (adjust pump phase)
        radius_output = self.radius_controller.update(current_state.bubble_radius)
        
        # Store state for analysis
        self.state_history.append(current_state)
        
        return ControlCommand(
            pump_power_delta=stability_output,
            pump_phase_delta=radius_output,
            bubble_radius_target=self.target_radius,
            emergency_shutdown=False
        )
    
    def _safety_check(self, state: ControlState) -> bool:
        """Check if state violates safety limits."""
        if state.stability_metric < self.safety_limits["min_stability"]:
            logger.warning(f"Stability too low: {state.stability_metric:.3f}")
            return True
        
        if state.tidal_forces > self.safety_limits["max_tidal_forces"]:
            logger.warning(f"Tidal forces too high: {state.tidal_forces:.1f} m/s²")
            return True
        
        if state.pump_power > self.safety_limits["max_pump_power"]:
            logger.warning(f"Pump power too high: {state.pump_power:.0f} W")
            return True
        
        if state.bubble_radius < self.safety_limits["min_bubble_radius"]:
            logger.warning(f"Bubble radius too small: {state.bubble_radius:.1f} m")
            return True
        
        return False
    
    def analyze_performance(self) -> Dict[str, float]:
        """Analyze controller performance from state history."""
        if len(self.state_history) < 10:
            return {"error": "Insufficient data"}
        
        # Extract metrics
        stabilities = [s.stability_metric for s in self.state_history]
        radii = [s.bubble_radius for s in self.state_history]
        
        # Calculate performance metrics
        stability_error = np.mean([(self.target_stability - s)**2 for s in stabilities])
        radius_error = np.mean([(self.target_radius - r)**2 for r in radii])
        
        stability_std = np.std(stabilities)
        radius_std = np.std(radii)
        
        return {
            "stability_rmse": np.sqrt(stability_error),
            "radius_rmse": np.sqrt(radius_error),
            "stability_std": stability_std,
            "radius_std": radius_std,
            "mean_stability": np.mean(stabilities),
            "mean_radius": np.mean(radii)
        }

class WarpControlLoop:
    """
    Main control loop for warp bubble system.
    """
    
    def __init__(self,
                 sensor: SensorInterface,
                 actuator: ActuatorInterface,
                 controller: StabilityController,
                 control_frequency: float = 10.0):
        
        self.sensor = sensor
        self.actuator = actuator
        self.controller = controller
        self.control_period = 1.0 / control_frequency
        
        # Control loop state
        self.running = False
        self.loop_thread = None
        self.command_queue = queue.Queue()
        
        # Performance monitoring
        self.loop_times = deque(maxlen=100)
        self.missed_deadlines = 0
        
    def start(self):
        """Start control loop in separate thread."""
        self.running = True
        self.loop_thread = threading.Thread(target=self._control_loop)
        self.loop_thread.start()
        logger.info("Control loop started")
    
    def stop(self):
        """Stop control loop."""
        self.running = False
        if self.loop_thread:
            self.loop_thread.join()
        logger.info("Control loop stopped")
    
    def _control_loop(self):
        """Main control loop implementation."""
        next_loop_time = time.time()
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Read sensors
                current_state = self._read_sensors()
                
                # Compute control action
                command = self.controller.compute_control_action(current_state)
                
                # Execute control command
                self._execute_command(command)
                
                # Check for emergency shutdown
                if command.emergency_shutdown:
                    self.actuator.emergency_shutdown()
                    logger.critical("Emergency shutdown triggered by controller")
                    break
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                continue
            
            # Timing analysis
            loop_end = time.time()
            loop_duration = loop_end - loop_start
            self.loop_times.append(loop_duration)
            
            # Sleep until next control period
            next_loop_time += self.control_period
            sleep_time = next_loop_time - time.time()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self.missed_deadlines += 1
                next_loop_time = time.time()  # Reset timing
    
    def _read_sensors(self) -> ControlState:
        """Read all sensors and create state object."""
        stability = self.sensor.read_stability_metric()
        radius = self.sensor.read_bubble_radius()
        energy_density = self.sensor.read_energy_density()
        tidal_forces = self.sensor.read_tidal_forces()
        pump_power, pump_phase = self.sensor.read_pump_status()
        
        return ControlState(
            timestamp=time.time(),
            bubble_radius=radius,
            bubble_speed=1000.0,  # Assume constant for now
            stability_metric=stability,
            energy_density=energy_density,
            tidal_forces=tidal_forces,
            pump_power=pump_power,
            pump_phase=pump_phase
        )
    
    def _execute_command(self, command: ControlCommand):
        """Execute control command via actuators."""
        if abs(command.pump_power_delta) > 1.0:  # W threshold
            self.actuator.adjust_pump_power(command.pump_power_delta)
        
        if abs(command.pump_phase_delta) > 0.01:  # rad threshold
            self.actuator.adjust_pump_phase(command.pump_phase_delta)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get control loop performance metrics."""
        if not self.loop_times:
            return {}
        
        avg_loop_time = np.mean(self.loop_times) * 1000  # ms
        max_loop_time = np.max(self.loop_times) * 1000   # ms
        loop_jitter = np.std(self.loop_times) * 1000     # ms
        
        controller_metrics = self.controller.analyze_performance()
        
        return {
            "avg_loop_time_ms": avg_loop_time,
            "max_loop_time_ms": max_loop_time,
            "loop_jitter_ms": loop_jitter,
            "missed_deadlines": self.missed_deadlines,
            "control_period_ms": self.control_period * 1000,
            **controller_metrics
        }
    
    def step(self, dt: float) -> Tuple[float, float]:
        """
        Execute one control step manually.
        
        Args:
            dt: Time step for this control iteration
            
        Returns:
            Tuple of (stability_metric, pump_power_adjustment)
        """
        try:
            # Read current sensor state
            current_state = self._read_sensors()
            
            # Compute control action
            command = self.controller.compute_control_action(current_state)
            
            # Execute control command
            self._execute_command(command)
            
            # Handle emergency conditions
            if command.emergency_shutdown:
                logger.warning("Emergency shutdown triggered in manual step")
                self.actuator.emergency_shutdown()
                return 0.0, 0.0  # Return safe values
            
            # Return stability and pump adjustment
            stability = current_state.stability_metric
            pump_adjustment = command.pump_power_delta
            
            return stability, pump_adjustment
            
        except Exception as e:
            logger.warning(f"Control step failed: {e}")
            # Return conservative values
            return 0.5, 0.0

class AdaptiveController:
    """
    Adaptive control system that learns optimal parameters.
    """
    
    def __init__(self, base_controller: StabilityController):
        self.base_controller = base_controller
        self.learning_rate = 0.01
        self.adaptation_history = []
        
    def adapt_parameters(self, performance_metrics: Dict[str, float]):
        """Adapt controller parameters based on performance."""
        # Simple adaptive rule: adjust gains based on error
        stability_error = performance_metrics.get("stability_rmse", 0.0)
        
        if stability_error > 0.05:  # High error - increase gains
            self.base_controller.stability_controller.params.kp *= (1 + self.learning_rate)
            self.base_controller.stability_controller.params.ki *= (1 + self.learning_rate * 0.5)
        elif stability_error < 0.01:  # Low error - decrease gains to reduce oscillation
            self.base_controller.stability_controller.params.kp *= (1 - self.learning_rate * 0.5)
            self.base_controller.stability_controller.params.ki *= (1 - self.learning_rate * 0.25)
        
        # Store adaptation history
        self.adaptation_history.append({
            "timestamp": time.time(),
            "stability_error": stability_error,
            "kp": self.base_controller.stability_controller.params.kp,
            "ki": self.base_controller.stability_controller.params.ki
        })
        
        # Limit history size
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-500:]

# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Control Loop System ===")
    
    # Initialize components
    sensor = SensorInterface(noise_level=0.02)
    actuator = ActuatorInterface(response_time=0.05)
    controller = StabilityController(
        target_stability=0.90,
        target_radius=12.0,
        control_frequency=20.0  # 20 Hz control
    )
    
    # Create control loop
    control_loop = WarpControlLoop(sensor, actuator, controller, control_frequency=20.0)
    
    print("Starting control loop for 10 seconds...")
    control_loop.start()
    
    # Let it run for 10 seconds
    time.sleep(10.0)
    
    # Stop control loop
    control_loop.stop()
    
    # Analyze performance
    performance = control_loop.get_performance_metrics()
    
    print("\n=== Control Loop Performance ===")
    print(f"Average loop time: {performance.get('avg_loop_time_ms', 0):.2f} ms")
    print(f"Maximum loop time: {performance.get('max_loop_time_ms', 0):.2f} ms")
    print(f"Loop jitter: {performance.get('loop_jitter_ms', 0):.2f} ms")
    print(f"Missed deadlines: {performance.get('missed_deadlines', 0)}")
    print(f"Control period: {performance.get('control_period_ms', 0):.1f} ms")
    
    print("\n=== Controller Performance ===")
    print(f"Stability RMSE: {performance.get('stability_rmse', 0):.4f}")
    print(f"Radius RMSE: {performance.get('radius_rmse', 0):.2f} m")
    print(f"Mean stability: {performance.get('mean_stability', 0):.3f}")
    print(f"Mean radius: {performance.get('mean_radius', 0):.1f} m")
    
    # Test adaptive controller
    print("\n=== Testing Adaptive Controller ===")
    adaptive = AdaptiveController(controller)
    
    # Simulate adaptation
    for i in range(5):
        # Get current performance
        current_performance = control_loop.get_performance_metrics()
        
        # Adapt parameters
        adaptive.adapt_parameters(current_performance)
        
        print(f"Adaptation {i+1}:")
        print(f"  Kp: {controller.stability_controller.params.kp:.1f}")
        print(f"  Ki: {controller.stability_controller.params.ki:.1f}")
    
    print("\nControl system testing complete!")
    
    # Test emergency scenarios
    print("\n=== Testing Emergency Scenarios ===")
    
    # Simulate low stability
    test_state = ControlState(
        timestamp=time.time(),
        bubble_radius=5.0,
        bubble_speed=1000.0,
        stability_metric=0.2,  # Low stability
        energy_density=-1e20,
        tidal_forces=60.0,     # High tidal forces
        pump_power=2000.0,
        pump_phase=0.1
    )
    
    emergency_command = controller.compute_control_action(test_state)
    print(f"Emergency shutdown triggered: {emergency_command.emergency_shutdown}")
    
    if emergency_command.emergency_shutdown:
        print("Emergency shutdown would be activated!")
        print("Reason: Safety limits exceeded")
