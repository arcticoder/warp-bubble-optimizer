#!/usr/bin/env python3
"""
Simulated Real-Time Control Loop for Warp Engine
==============================================

This module simulates a real-time control system for warp bubble management,
including sensor noise, actuator delays, and feedback control algorithms.
This replaces actual hardware interfaces with realistic simulation models.
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
import logging
from dataclasses import dataclass

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
        def log_metric(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

class DummyContext:
    """Dummy context manager for fallback."""
    def __enter__(self): return self
    def __exit__(self, *args): pass

# JAX imports with fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad
    JAX_AVAILABLE = True
except ImportError:
    jnp = np
    JAX_AVAILABLE = False
    
    def jit(func):
        return func
    def grad(func):
        def grad_func(x, *args, **kwargs):
            eps = 1e-8
            x = np.asarray(x)
            grads = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                grads[i] = (func(x_plus, *args, **kwargs) - func(x_minus, *args, **kwargs)) / (2 * eps)
            return grads
        return grad_func

logger = logging.getLogger(__name__)

@dataclass
class SensorConfig:
    """Configuration for simulated sensors."""
    noise_level: float = 1e-3
    update_rate: float = 100.0  # Hz
    latency: float = 0.001  # seconds
    drift_rate: float = 1e-6  # parameter drift per second

@dataclass 
class ActuatorConfig:
    """Configuration for simulated actuators."""
    response_time: float = 0.05  # seconds
    damping_factor: float = 0.9
    saturation_limit: float = 10.0
    bandwidth: float = 50.0  # Hz

@dataclass
class ControllerConfig:
    """Configuration for control algorithm."""
    kp: float = 0.1  # Proportional gain
    ki: float = 0.01  # Integral gain
    kd: float = 0.001  # Derivative gain
    learning_rate: float = 1e-2
    stability_threshold: float = 1e-6

class VirtualSensorInterface:
    """Simulates sensor interface with realistic noise and latency."""
    
    def __init__(self, config: SensorConfig):
        self.config = config
        self.last_reading = None
        self.drift_accumulator = 0.0
        self.reading_history = []
        
    def sense(self, true_state: np.ndarray) -> np.ndarray:
        """Simulate sensor reading with noise and drift.
        
        Args:
            true_state: Actual system state
            
        Returns:
            Noisy sensor measurement
        """
        # Add sensor noise
        noise = np.random.normal(0, self.config.noise_level, size=true_state.shape)
        
        # Add drift
        self.drift_accumulator += self.config.drift_rate * (1.0 / self.config.update_rate)
        drift = self.drift_accumulator * np.ones_like(true_state)
        
        # Simulate reading
        measurement = true_state + noise + drift
        
        # Store history
        self.reading_history.append({
            'timestamp': time.time(),
            'measurement': measurement.copy(),
            'noise': noise,
            'drift': drift
        })
        
        # Keep only recent history
        if len(self.reading_history) > 1000:
            self.reading_history = self.reading_history[-1000:]
        
        self.last_reading = measurement
        return measurement
    
    def get_sensor_diagnostics(self) -> Dict:
        """Get sensor diagnostic information."""
        if not self.reading_history:
            return {}
        
        recent_readings = [h['measurement'] for h in self.reading_history[-100:]]
        recent_noise = [np.linalg.norm(h['noise']) for h in self.reading_history[-100:]]
        
        return {
            'drift_accumulator': self.drift_accumulator,
            'noise_rms': np.std(recent_noise),
            'reading_mean': np.mean(recent_readings, axis=0),
            'reading_std': np.std(recent_readings, axis=0),
            'update_rate': self.config.update_rate
        }

class VirtualActuatorInterface:
    """Simulates actuator interface with realistic dynamics."""
    
    def __init__(self, config: ActuatorConfig, n_params: int):
        self.config = config
        self.current_state = np.zeros(n_params)
        self.target_state = np.zeros(n_params)
        self.velocity = np.zeros(n_params)
        self.command_history = []
        
    def actuate(self, command: np.ndarray) -> np.ndarray:
        """Simulate actuator response to command.
        
        Args:
            command: Desired actuator state
            
        Returns:
            Actual actuator state after dynamics
        """
        # Apply saturation limits
        command_limited = np.clip(command, -self.config.saturation_limit, 
                                 self.config.saturation_limit)
        
        # Store command
        self.command_history.append({
            'timestamp': time.time(),
            'command': command.copy(),
            'command_limited': command_limited.copy()
        })
        
        # Keep only recent history
        if len(self.command_history) > 1000:
            self.command_history = self.command_history[-1000:]
        
        # Update target
        self.target_state = command_limited
        
        # Simulate first-order actuator dynamics
        dt = 1.0 / self.config.bandwidth
        tau = self.config.response_time
        
        # First-order response: dx/dt = (target - current) / tau
        error = self.target_state - self.current_state
        self.velocity = error / tau
        self.current_state += self.velocity * dt
        
        # Apply damping
        self.current_state *= self.config.damping_factor
        
        return self.current_state.copy()
    
    def get_actuator_diagnostics(self) -> Dict:
        """Get actuator diagnostic information."""
        if not self.command_history:
            return {}
        
        recent_commands = [h['command'] for h in self.command_history[-100:]]
        
        return {
            'current_state': self.current_state.copy(),
            'target_state': self.target_state.copy(),
            'velocity': self.velocity.copy(),
            'command_mean': np.mean(recent_commands, axis=0),
            'command_std': np.std(recent_commands, axis=0),
            'response_time': self.config.response_time
        }

class VirtualWarpController:
    """Virtual warp bubble controller with realistic control algorithms."""
    
    def __init__(self, 
                 objective_func: Callable,
                 initial_params: np.ndarray,
                 sensor_config: Optional[SensorConfig] = None,
                 actuator_config: Optional[ActuatorConfig] = None,
                 controller_config: Optional[ControllerConfig] = None):
        """Initialize virtual controller.
        
        Args:
            objective_func: Objective function to optimize
            initial_params: Initial parameter values
            sensor_config: Sensor configuration
            actuator_config: Actuator configuration  
            controller_config: Controller configuration
        """
        self.objective_func = objective_func
        self.true_params = np.array(initial_params)
        
        # Use defaults if not provided
        self.sensor_config = sensor_config or SensorConfig()
        self.actuator_config = actuator_config or ActuatorConfig()
        self.controller_config = controller_config or ControllerConfig()
        
        # Initialize interfaces
        self.sensor = VirtualSensorInterface(self.sensor_config)
        self.actuator = VirtualActuatorInterface(self.actuator_config, len(initial_params))
        
        # Control algorithm state
        self.error_integral = np.zeros_like(initial_params)
        self.previous_error = np.zeros_like(initial_params)
        self.control_history = []
        
        # Optimization state
        if JAX_AVAILABLE:
            self.grad_objective = jit(grad(objective_func))
        else:
            self.grad_objective = grad(objective_func)
    
    async def control_step(self) -> Dict:
        """Execute one control step.
        
        Returns:
            Dictionary with step information
        """
        step_start = time.time()
        
        # Sensor reading with latency
        await asyncio.sleep(self.sensor_config.latency)
        measured_params = self.sensor.sense(self.true_params)
        
        # Compute objective and gradient
        current_objective = self.objective_func(measured_params)
        gradient = self.grad_objective(measured_params)
        
        # PID control algorithm
        error = -gradient  # Want to move opposite to gradient
        
        # Proportional term
        proportional = self.controller_config.kp * error
        
        # Integral term
        self.error_integral += error * (1.0 / self.sensor_config.update_rate)
        integral = self.controller_config.ki * self.error_integral
        
        # Derivative term
        derivative = self.controller_config.kd * (error - self.previous_error) * self.sensor_config.update_rate
        self.previous_error = error
        
        # Combined control signal
        control_signal = proportional + integral + derivative
        
        # Apply learning rate
        control_command = measured_params + self.controller_config.learning_rate * control_signal
        
        # Actuate with realistic dynamics
        actual_params = self.actuator.actuate(control_command)
        
        # Update true system state (simulate perfect actuation to true state)
        # In reality, this would be the physical response
        self.true_params = actual_params
        
        # Record step information
        step_info = {
            'timestamp': step_start,
            'measured_params': measured_params.copy(),
            'true_params': self.true_params.copy(),
            'objective_value': float(current_objective),
            'gradient': gradient.copy(),
            'control_signal': control_signal.copy(),
            'control_command': control_command.copy(),
            'step_duration': time.time() - step_start        }
        
        self.control_history.append(step_info)
        
        # Keep only recent history
        if len(self.control_history) > 10000:
            self.control_history = self.control_history[-10000:]
        
        return step_info
    
    async def run_control_loop(self, 
                              duration: float = 10.0,
                              target_rate: float = None) -> Dict:
        """Run the control loop for specified duration.
        
        Args:
            duration: Control loop duration in seconds
            target_rate: Target control rate (Hz)
            
        Returns:
            Complete control session results
        """
        if target_rate is None:
            target_rate = self.sensor_config.update_rate
        
        dt = 1.0 / target_rate
        start_time = time.time()
        step_count = 0
        
        logger.info(f"Starting control loop: {duration}s at {target_rate}Hz")
        
        # Initialize progress tracking
        total_steps = int(duration * target_rate)
        progress = None
        if PROGRESS_AVAILABLE:
            try:
                progress = ProgressTracker(
                    total_iterations=total_steps,
                    description="Virtual Control Loop",
                    log_level=logging.INFO
                )
                progress.set_stage("control_execution")
            except Exception as e:
                logger.warning(f"Failed to initialize ProgressTracker: {e}")
                progress = None
        
        with progress if progress else DummyContext():
            while (time.time() - start_time) < duration:
                step_start = time.time()
                
                # Execute control step
                step_info = await self.control_step()
                step_count += 1
                
                # Update progress
                if progress:
                    try:
                        progress.update(1)
                        if step_count % int(target_rate) == 0:  # Every second
                            progress.log_metric("objective_value", step_info['objective_value'])
                            progress.log_metric("control_rate_hz", step_count / (time.time() - start_time))
                    except Exception as e:
                        logger.warning(f"Progress update failed: {e}")
                
                # Progress reporting
                if step_count % int(target_rate) == 0:  # Every second
                    elapsed = time.time() - start_time
                    obj_val = step_info['objective_value']
                    logger.info(f"t={elapsed:.1f}s, step={step_count}, obj={obj_val:.6e}")
                
                # Maintain target rate
                step_duration = time.time() - step_start
                sleep_time = max(0, dt - step_duration)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        
        # Compile results
        elapsed_time = time.time() - start_time
        actual_rate = step_count / elapsed_time
        
        results = {
            'duration': elapsed_time,
            'steps': step_count,
            'actual_rate': actual_rate,
            'target_rate': target_rate,
            'final_params': self.true_params.copy(),
            'final_objective': float(self.objective_func(self.true_params)),
            'control_history': self.control_history.copy(),
            'sensor_diagnostics': self.sensor.get_sensor_diagnostics(),
            'actuator_diagnostics': self.actuator.get_actuator_diagnostics()
        }
        
        logger.info(f"Control loop complete: {step_count} steps in {elapsed_time:.2f}s")
        logger.info(f"Rate: {actual_rate:.1f}Hz (target: {target_rate:.1f}Hz)")
        logger.info(f"Final objective: {results['final_objective']:.6e}")
        
        return results

def demo_virtual_control():
    """Demonstrate virtual control loop simulation."""
    print("=" * 60)
    print("VIRTUAL WARP CONTROL LOOP DEMO")
    print("=" * 60)
    
    if not JAX_AVAILABLE:
        print("⚠️  JAX not available - using NumPy fallback")
    
    # Define test objective (quadratic with minimum at [1, 2])
    def test_objective(params):
        x, y = params
        return (x - 1)**2 + (y - 2)**2 + 0.1 * np.sin(10*x) * np.sin(10*y)
    
    # Controller configuration
    sensor_config = SensorConfig(
        noise_level=0.01,
        update_rate=50.0,
        latency=0.002,
        drift_rate=1e-5
    )
    
    actuator_config = ActuatorConfig(
        response_time=0.1,
        damping_factor=0.95,
        saturation_limit=5.0,
        bandwidth=25.0
    )
    
    controller_config = ControllerConfig(
        kp=0.5,
        ki=0.1,
        kd=0.01,
        learning_rate=0.1
    )
    
    # Create controller
    initial_params = np.array([0.0, 0.0])
    controller = VirtualWarpController(
        objective_func=test_objective,
        initial_params=initial_params,
        sensor_config=sensor_config,
        actuator_config=actuator_config,
        controller_config=controller_config
    )
    
    # Run control loop
    async def run_demo():
        return await controller.run_control_loop(duration=5.0, target_rate=50.0)
    
    # Execute demo
    results = asyncio.run(run_demo())
    
    # Report results
    print(f"\n📊 CONTROL LOOP RESULTS:")
    print(f"   Duration: {results['duration']:.2f}s")
    print(f"   Steps: {results['steps']}")
    print(f"   Actual rate: {results['actual_rate']:.1f}Hz")
    print(f"   Initial objective: {test_objective(initial_params):.6f}")
    print(f"   Final objective: {results['final_objective']:.6f}")
    print(f"   Improvement: {test_objective(initial_params) - results['final_objective']:.6f}")
    
    # Convergence analysis
    objectives = [step['objective_value'] for step in results['control_history']]
    final_10_percent = objectives[int(0.9*len(objectives)):]
    convergence_stability = np.std(final_10_percent)
    
    print(f"\n🎯 CONVERGENCE ANALYSIS:")
    print(f"   Final 10% std: {convergence_stability:.6f}")
    print(f"   Converged: {'✅' if convergence_stability < 0.01 else '❌'}")
    
    return results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run demonstration
    demo_virtual_control()
