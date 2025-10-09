#!/usr/bin/env python3
"""
Enhanced Virtual Control Loop Simulation
======================================

Advanced warp field control simulation with comprehensive progress tracking,
realistic sensor/actuator modeling, and hardware-in-the-loop simulation.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import asyncio
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Callable, List
import logging
from pathlib import Path
import time
from dataclasses import dataclass

# Import ProgressTracker for enhanced progress monitoring
try:
    from progress_tracker import ProgressTracker, MultiProcessProgressTracker
    PROGRESS_TRACKER_AVAILABLE = True
except ImportError:
    PROGRESS_TRACKER_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ControlSystemConfig:
    """Configuration for virtual control system."""
    # PID Controller Parameters
    kp: float = 1.0          # Proportional gain
    ki: float = 0.1          # Integral gain
    kd: float = 0.01         # Derivative gain
    
    # System Parameters
    sampling_rate: float = 1000.0  # Hz
    control_bandwidth: float = 100.0  # Hz
    actuator_bandwidth: float = 50.0   # Hz
    sensor_noise_level: float = 0.01   # RMS noise level
    actuator_delay: float = 0.001      # seconds
    
    # Simulation Parameters
    simulation_duration: float = 10.0  # seconds
    enable_disturbances: bool = True
    disturbance_amplitude: float = 0.1
    adaptive_gains: bool = False
    
    # Warp Field Parameters
    field_energy_scale: float = 1e12  # Joules
    field_spatial_scale: float = 100.0  # meters
    field_time_constant: float = 0.1   # seconds

class VirtualSensorArray:
    """Simulate realistic sensor measurements with noise and bandwidth limits."""
    
    def __init__(self, config: ControlSystemConfig):
        self.config = config
        self.noise_level = config.sensor_noise_level
        self.bandwidth = config.control_bandwidth
        
        # Sensor transfer function coefficients (1st order low-pass)
        self.alpha = 2 * np.pi * self.bandwidth * (1.0 / config.sampling_rate)
        self.filter_state = 0.0
        
        # Measurement history for derivative computation
        self.measurement_history = []
        
    def measure(self, true_value: float, timestamp: float) -> Dict:
        """Simulate sensor measurement with realistic characteristics."""
        
        # Add measurement noise
        noise = np.random.normal(0, self.noise_level)
        noisy_measurement = true_value + noise
        
        # Apply bandwidth filtering (1st order low-pass)
        self.filter_state += self.alpha * (noisy_measurement - self.filter_state)
        filtered_measurement = self.filter_state
        
        # Store measurement history
        self.measurement_history.append({
            'timestamp': timestamp,
            'raw_measurement': noisy_measurement,
            'filtered_measurement': filtered_measurement,
            'true_value': true_value
        })
        
        # Limit history size
        if len(self.measurement_history) > 1000:
            self.measurement_history.pop(0)
        
        return {
            'value': filtered_measurement,
            'raw_value': noisy_measurement,
            'noise_estimate': noise,
            'filter_state': self.filter_state,
            'measurement_quality': 1.0 - abs(noise) / (abs(true_value) + 1e-10)
        }

class VirtualActuatorArray:
    """Simulate realistic actuator response with bandwidth limits and saturation."""
    
    def __init__(self, config: ControlSystemConfig):
        self.config = config
        self.bandwidth = config.actuator_bandwidth
        self.delay = config.actuator_delay
        
        # Actuator transfer function (2nd order with damping)
        self.omega_n = 2 * np.pi * self.bandwidth
        self.zeta = 0.7  # Damping ratio
        
        # State variables for 2nd order system
        self.position = 0.0
        self.velocity = 0.0
        
        # Command history for delay simulation
        self.command_history = []
        
        # Actuator limits
        self.max_output = 10.0
        self.max_rate = 100.0
        
    def apply(self, command: float, timestamp: float) -> Dict:
        """Apply control command through realistic actuator dynamics."""
        
        # Store command with timestamp for delay
        self.command_history.append({
            'timestamp': timestamp,
            'command': command
        })
        
        # Apply delay by retrieving delayed command
        delayed_command = self._get_delayed_command(timestamp)
        
        # Apply saturation limits
        saturated_command = np.clip(delayed_command, -self.max_output, self.max_output)
        
        # Apply rate limits
        dt = 1.0 / self.config.sampling_rate
        max_change = self.max_rate * dt
        command_change = saturated_command - self.position
        limited_change = np.clip(command_change, -max_change, max_change)
        limited_command = self.position + limited_change
        
        # 2nd order actuator dynamics
        acceleration = self.omega_n**2 * (limited_command - self.position) - \
                      2 * self.zeta * self.omega_n * self.velocity
        
        # Update state
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        return {
            'output': self.position,
            'command': command,
            'delayed_command': delayed_command,
            'saturated_command': saturated_command,
            'saturation_active': abs(command) > self.max_output,
            'rate_limit_active': abs(command_change) > max_change,
            'velocity': self.velocity,
            'acceleration': acceleration
        }
    
    def _get_delayed_command(self, current_time: float) -> float:
        """Retrieve command with appropriate delay."""
        target_time = current_time - self.delay
        
        # Find closest command in history
        for i, cmd_data in enumerate(reversed(self.command_history)):
            if cmd_data['timestamp'] <= target_time:
                return cmd_data['command']
        
        # Return zero if no suitable command found
        return 0.0

class WarpFieldPlantModel:
    """Physical model of warp field dynamics for control simulation."""
    
    def __init__(self, config: ControlSystemConfig):
        self.config = config
        self.time_constant = config.field_time_constant
        self.energy_scale = config.field_energy_scale
        self.spatial_scale = config.field_spatial_scale
        
        # Field state variables
        self.field_amplitude = 0.0
        self.field_energy = 0.0
        self.field_gradient = 0.0
        
        # Physical constants
        self.c = 2.998e8  # Speed of light
        
        # Disturbance model
        self.disturbance_state = 0.0
        
        # Target setpoint
        self.setpoint = 1.0
        
    def evolve(self, control_input: float, dt: float) -> Dict:
        """Evolve warp field dynamics with control input."""
        
        # Field evolution equation (simplified)
        # dΦ/dt = -Φ/τ + α*u + disturbance
        alpha = 1.0 / self.energy_scale  # Control effectiveness
        
        # Add disturbances if enabled
        disturbance = 0.0
        if self.config.enable_disturbances:
            # Generate correlated disturbance
            disturbance_bandwidth = 10.0  # Hz
            noise_strength = self.config.disturbance_amplitude
            
            # 1st order colored noise
            tau_dist = 1.0 / (2 * np.pi * disturbance_bandwidth)
            self.disturbance_state += (-self.disturbance_state / tau_dist + 
                                     np.random.normal(0, noise_strength)) * dt
            disturbance = self.disturbance_state
        
        # Field evolution
        field_derivative = (-self.field_amplitude / self.time_constant + 
                          alpha * control_input + disturbance)
        
        self.field_amplitude += field_derivative * dt
        
        # Compute derived quantities
        self.field_energy = 0.5 * self.field_amplitude**2 * self.energy_scale
        self.field_gradient = self.field_amplitude / self.spatial_scale
        
        return {
            'field_amplitude': self.field_amplitude,
            'field_energy': self.field_energy,
            'field_gradient': self.field_gradient,
            'field_derivative': field_derivative,
            'disturbance': disturbance,
            'control_effectiveness': alpha * control_input
        }
    
    def get_measurement_output(self) -> float:
        """Get the measurement output (what sensors would see)."""
        return self.field_amplitude
    
    def get_state_vector(self) -> Dict:
        """Get complete state vector for logging."""
        return {
            'field_amplitude': self.field_amplitude,
            'field_energy': self.field_energy,
            'field_gradient': self.field_gradient,
            'disturbance_state': self.disturbance_state,
            'setpoint': self.setpoint
        }

class PIDController:
    """Advanced PID controller with anti-windup and adaptive gains."""
    
    def __init__(self, config: ControlSystemConfig):
        self.config = config
        self.kp = config.kp
        self.ki = config.ki
        self.kd = config.kd
        
        # PID state variables
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = 0.0
        
        # Anti-windup parameters
        self.integral_limit = 10.0
        self.output_limit = 10.0
        
        # Adaptive gain parameters
        self.adaptive_gains = config.adaptive_gains
        self.gain_adaptation_rate = 0.01
        
        # Performance tracking
        self.error_history = []
        
    def compute(self, measurement: float, setpoint: float, timestamp: float) -> Dict:
        """Compute PID control output."""
        
        # Compute error
        error = setpoint - measurement
        
        # Time delta
        dt = timestamp - self.last_time if self.last_time > 0 else 1.0 / self.config.sampling_rate
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = 0.0
        if dt > 0:
            derivative = self.kd * (error - self.last_error) / dt
        
        # Total output
        output = proportional + integral + derivative
        
        # Apply output saturation
        saturated_output = np.clip(output, -self.output_limit, self.output_limit)
        
        # Anti-windup: back-calculate integral if saturated
        if abs(output) > self.output_limit:
            excess = output - saturated_output
            self.integral -= excess / (self.ki + 1e-10)
        
        # Adaptive gain adjustment
        if self.adaptive_gains:
            self._adapt_gains(error, dt)
        
        # Store error history
        self.error_history.append({
            'timestamp': timestamp,
            'error': error,
            'proportional': proportional,
            'integral': integral,
            'derivative': derivative,
            'output': output,
            'saturated_output': saturated_output
        })
        
        # Limit history size
        if len(self.error_history) > 1000:
            self.error_history.pop(0)
        
        # Update state
        self.last_error = error
        self.last_time = timestamp
        
        return {
            'output': saturated_output,
            'error': error,
            'proportional': proportional,
            'integral': integral,
            'derivative': derivative,
            'gains': {'kp': self.kp, 'ki': self.ki, 'kd': self.kd},
            'saturation_active': abs(output) > self.output_limit
        }
    
    def _adapt_gains(self, error: float, dt: float):
        """Adaptive gain adjustment based on performance."""
        
        # Simple adaptive rule: adjust gains based on error magnitude
        error_magnitude = abs(error)
        
        if error_magnitude > 1.0:
            # Large error: increase proportional gain
            self.kp += self.gain_adaptation_rate * dt
        elif error_magnitude < 0.1:
            # Small error: increase derivative gain for stability
            self.kd += self.gain_adaptation_rate * dt * 0.1
        
        # Prevent gains from becoming negative or too large
        self.kp = np.clip(self.kp, 0.1, 10.0)
        self.ki = np.clip(self.ki, 0.01, 1.0)
        self.kd = np.clip(self.kd, 0.001, 0.1)

class EnhancedVirtualControlLoop:
    """Enhanced virtual control loop with comprehensive simulation and tracking."""
    
    def __init__(self, config: Optional[ControlSystemConfig] = None):
        self.config = config or ControlSystemConfig()
        
        # Initialize subsystems
        self.plant = WarpFieldPlantModel(self.config)
        self.controller = PIDController(self.config)
        self.sensors = VirtualSensorArray(self.config)
        self.actuators = VirtualActuatorArray(self.config)
        
        # Simulation state
        self.simulation_time = 0.0
        self.simulation_data = []
        
        logger.info("Enhanced virtual control loop initialized")
    
    async def run_simulation(self, duration: Optional[float] = None, 
                           progress_callback: Optional[Callable] = None) -> Dict:
        """Run complete control loop simulation with progress tracking."""
        
        duration = duration or self.config.simulation_duration
        dt = 1.0 / self.config.sampling_rate
        n_steps = int(duration / dt)
        
        # Initialize progress tracking
        if PROGRESS_TRACKER_AVAILABLE:
            progress = ProgressTracker(total_steps=n_steps, description="Virtual Control Loop Simulation")
            progress.start({
                'duration': f"{duration:.2f} seconds",
                'sampling_rate': f"{self.config.sampling_rate} Hz",
                'control_bandwidth': f"{self.config.control_bandwidth} Hz",
                'controller_type': 'PID',
                'plant_model': 'Warp Field Dynamics',
                'disturbances_enabled': self.config.enable_disturbances
            })
        
        # Simulation loop
        for step in range(n_steps):
            current_time = step * dt
            
            # Get plant measurement
            true_output = self.plant.get_measurement_output()
            
            # Sensor measurement with noise and filtering
            sensor_data = self.sensors.measure(true_output, current_time)
            measurement = sensor_data['value']
            
            # Controller computation
            control_data = self.controller.compute(
                measurement, self.plant.setpoint, current_time
            )
            control_signal = control_data['output']
            
            # Actuator response
            actuator_data = self.actuators.apply(control_signal, current_time)
            actuator_output = actuator_data['output']
            
            # Plant evolution
            plant_data = self.plant.evolve(actuator_output, dt)
            
            # Log simulation data
            step_data = {
                'time': current_time,
                'setpoint': self.plant.setpoint,
                'measurement': measurement,
                'control_signal': control_signal,
                'actuator_output': actuator_output,
                'plant_output': true_output,
                'error': control_data['error'],
                'field_energy': plant_data['field_energy'],
                'disturbance': plant_data['disturbance'],
                'sensor_data': sensor_data,
                'control_data': control_data,
                'actuator_data': actuator_data,
                'plant_data': plant_data
            }
            self.simulation_data.append(step_data)
            
            # Progress reporting
            if PROGRESS_TRACKER_AVAILABLE and step % (n_steps // 20) == 0:
                step_info = {
                    'time': f"{current_time:.3f}s",
                    'error': f"{control_data['error']:.4f}",
                    'control_output': f"{control_signal:.4f}",
                    'field_energy': f"{plant_data['field_energy']:.2e}",
                    'system_stable': abs(control_data['error']) < 0.1
                }
                progress.update(f"Simulation step {step}", step, step_info)
            
            # Small delay to simulate real-time operation
            await asyncio.sleep(0.001)
        
        # Complete simulation
        if PROGRESS_TRACKER_AVAILABLE:
            final_error = self.simulation_data[-1]['error']
            settling_time = self._calculate_settling_time()
            overshoot = self._calculate_overshoot()
            
            progress.complete({
                'final_error': f"{final_error:.6f}",
                'settling_time': f"{settling_time:.3f}s",
                'overshoot': f"{overshoot:.1%}",
                'simulation_success': abs(final_error) < 0.05,
                'total_steps': n_steps
            })
        
        # Analyze results
        results = self._analyze_simulation_results()
        
        return results
    
    def _calculate_settling_time(self, tolerance: float = 0.05) -> float:
        """Calculate settling time (time to reach within tolerance of setpoint)."""
        setpoint = self.plant.setpoint
        
        for i in reversed(range(len(self.simulation_data))):
            if abs(self.simulation_data[i]['error']) > tolerance:
                return self.simulation_data[min(i + 1, len(self.simulation_data) - 1)]['time']
        
        return 0.0
    
    def _calculate_overshoot(self) -> float:
        """Calculate maximum overshoot as percentage of setpoint."""
        if not self.simulation_data:
            return 0.0
        
        setpoint = self.plant.setpoint
        measurements = [data['measurement'] for data in self.simulation_data]
        max_measurement = max(measurements)
        
        if setpoint > 0:
            overshoot = (max_measurement - setpoint) / setpoint
        else:
            overshoot = 0.0
        
        return max(0.0, overshoot)
    
    def _analyze_simulation_results(self) -> Dict:
        """Comprehensive analysis of simulation results."""
        
        if not self.simulation_data:
            return {}
        
        # Extract time series data
        times = np.array([d['time'] for d in self.simulation_data])
        errors = np.array([d['error'] for d in self.simulation_data])
        measurements = np.array([d['measurement'] for d in self.simulation_data])
        control_signals = np.array([d['control_signal'] for d in self.simulation_data])
        field_energies = np.array([d['field_energy'] for d in self.simulation_data])
        
        # Performance metrics
        settling_time = self._calculate_settling_time()
        overshoot = self._calculate_overshoot()
        steady_state_error = np.mean(np.abs(errors[-100:]))  # Last 100 samples
        
        # Control effort metrics
        total_control_effort = np.sum(np.abs(control_signals)) * (times[1] - times[0])
        max_control_signal = np.max(np.abs(control_signals))
        
        # System identification metrics
        rise_time = self._calculate_rise_time(measurements)
        peak_time = times[np.argmax(measurements)]
        
        results = {
            'performance_metrics': {
                'settling_time': settling_time,
                'overshoot': overshoot,
                'rise_time': rise_time,
                'peak_time': peak_time,
                'steady_state_error': steady_state_error
            },
            'control_metrics': {
                'total_control_effort': total_control_effort,
                'max_control_signal': max_control_signal,
                'control_signal_std': np.std(control_signals)
            },
            'energy_metrics': {
                'final_field_energy': field_energies[-1],
                'max_field_energy': np.max(field_energies),
                'energy_efficiency': field_energies[-1] / (total_control_effort + 1e-10)
            },
            'simulation_data': self.simulation_data,
            'time_series': {
                'times': times,
                'errors': errors,
                'measurements': measurements,
                'control_signals': control_signals,
                'field_energies': field_energies
            }
        }
        
        return results
    
    def _calculate_rise_time(self, measurements: np.ndarray, 
                           start_percent: float = 0.1, 
                           end_percent: float = 0.9) -> float:
        """Calculate rise time (time to go from 10% to 90% of final value)."""
        
        if len(measurements) < 2:
            return 0.0
        
        final_value = measurements[-1]
        start_value = measurements[0]
        
        start_threshold = start_value + start_percent * (final_value - start_value)
        end_threshold = start_value + end_percent * (final_value - start_value)
        
        start_idx = np.argmax(measurements >= start_threshold)
        end_idx = np.argmax(measurements >= end_threshold)
        
        if end_idx <= start_idx:
            return 0.0
        
        times = np.linspace(0, len(measurements) / self.config.sampling_rate, len(measurements))
        return times[end_idx] - times[start_idx]
    
    def visualize_results(self, results: Dict, save_path: Optional[str] = None):
        """Create comprehensive visualization of control loop results."""
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        times = results['time_series']['times']
        measurements = results['time_series']['measurements']
        errors = results['time_series']['errors']
        control_signals = results['time_series']['control_signals']
        field_energies = results['time_series']['field_energies']
        
        # Response plot
        ax = axes[0, 0]
        ax.plot(times, measurements, label='Measurement', linewidth=2)
        ax.axhline(y=self.plant.setpoint, color='r', linestyle='--', label='Setpoint')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Field Amplitude')
        ax.set_title('System Response')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error plot
        ax = axes[0, 1]
        ax.plot(times, errors, label='Error', color='red', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error')
        ax.set_title('Tracking Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Control signal
        ax = axes[1, 0]
        ax.plot(times, control_signals, label='Control Signal', color='green', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Control Output')
        ax.set_title('Control Signal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Field energy
        ax = axes[1, 1]
        ax.plot(times, field_energies, label='Field Energy', color='purple', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy (J)')
        ax.set_title('Warp Field Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Phase portrait (error vs error rate)
        ax = axes[2, 0]
        error_rate = np.gradient(errors, times)
        ax.plot(errors, error_rate, alpha=0.7)
        ax.set_xlabel('Error')
        ax.set_ylabel('Error Rate')
        ax.set_title('Phase Portrait')
        ax.grid(True, alpha=0.3)
        
        # Performance summary
        ax = axes[2, 1]
        metrics = results['performance_metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Create text summary
        summary_text = []
        for name, value in metrics.items():
            if isinstance(value, float):
                if abs(value) < 0.001:
                    summary_text.append(f"{name}: {value:.2e}")
                else:
                    summary_text.append(f"{name}: {value:.4f}")
            else:
                summary_text.append(f"{name}: {value}")
        
        ax.text(0.1, 0.9, "Performance Metrics:", fontsize=12, fontweight='bold', 
                transform=ax.transAxes)
        
        for i, text in enumerate(summary_text):
            ax.text(0.1, 0.8 - i*0.1, text, fontsize=10, transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Control loop visualization saved to {save_path}")
        
        plt.show()
        
        return fig

# Demonstration and testing
async def demonstrate_enhanced_control_loop():
    """Demonstrate enhanced virtual control loop simulation."""
    
    print("\n🚀 Enhanced Virtual Control Loop Demonstration")
    print("="*60)
    
    # Setup configuration
    config = ControlSystemConfig(
        kp=2.0,
        ki=0.5,
        kd=0.1,
        sampling_rate=1000.0,
        simulation_duration=5.0,
        enable_disturbances=True,
        disturbance_amplitude=0.1,
        adaptive_gains=True
    )
    
    # Initialize control loop
    control_loop = EnhancedVirtualControlLoop(config)
    
    # Run simulation
    results = await control_loop.run_simulation()
    
    # Display results
    metrics = results['performance_metrics']
    print(f"\n📊 Simulation Results:")
    print(f"   • Settling Time: {metrics['settling_time']:.3f} seconds")
    print(f"   • Overshoot: {metrics['overshoot']:.1%}")
    print(f"   • Steady-State Error: {metrics['steady_state_error']:.6f}")
    print(f"   • Rise Time: {metrics['rise_time']:.3f} seconds")
    
    control_metrics = results['control_metrics']
    print(f"\n🎮 Control Performance:")
    print(f"   • Total Control Effort: {control_metrics['total_control_effort']:.3f}")
    print(f"   • Max Control Signal: {control_metrics['max_control_signal']:.3f}")
    
    energy_metrics = results['energy_metrics']
    print(f"\n⚡ Energy Metrics:")
    print(f"   • Final Field Energy: {energy_metrics['final_field_energy']:.2e} J")
    print(f"   • Energy Efficiency: {energy_metrics['energy_efficiency']:.2e}")
    
    # Create visualization
    try:
        control_loop.visualize_results(results, "control_loop_results.png")
    except Exception as e:
        print(f"⚠️  Visualization skipped: {e}")
    
    return results

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstration
    asyncio.run(demonstrate_enhanced_control_loop())
