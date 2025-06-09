# src/warp_engine/hardware_loop.py
"""
Hardware-in-the-Loop Planning Module
===================================

This module implements hardware-in-the-loop (HIL) planning and integration
for warp drive systems. Includes:

1. Real-time sensor data acquisition and processing
2. Hardware control interfaces for experimental setups
3. Safety interlocks and monitoring systems
4. Digital twin synchronization with physical experiments
5. Automated calibration and validation protocols

Key Features:
- Multi-sensor data fusion
- Real-time control loop integration
- Safety monitoring and emergency shutdown
- Automated experimental parameter optimization
- Hardware/software co-simulation
"""

import numpy as np
import time
import threading
import queue
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import json
from enum import Enum

logger = logging.getLogger(__name__)

class HardwareState(Enum):
    """Hardware system states."""
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"

class SensorType(Enum):
    """Types of sensors in the HIL system."""
    ELECTROMAGNETIC = "electromagnetic"
    GRAVITATIONAL = "gravitational"
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    OPTICAL = "optical"
    POWER = "power"

@dataclass
class SensorReading:
    """Individual sensor reading with metadata."""
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    value: float
    unit: str
    quality: float = 1.0  # 0-1 quality score
    metadata: Dict = field(default_factory=dict)

@dataclass
class HardwareConfig:
    """Configuration for hardware-in-the-loop system."""
    sampling_frequency: float = 1000.0  # Hz
    control_frequency: float = 100.0    # Hz
    safety_check_frequency: float = 10.0  # Hz
    max_buffer_size: int = 10000
    emergency_thresholds: Dict[str, float] = field(default_factory=dict)
    calibration_schedule: Dict[str, int] = field(default_factory=dict)  # sensor_id: interval_seconds

@dataclass
class ControlCommand:
    """Command sent to hardware actuators."""
    actuator_id: str
    command_type: str
    value: float
    timestamp: float
    priority: int = 0  # Higher priority commands executed first

class SensorInterface(ABC):
    """Abstract base class for sensor interfaces."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the sensor. Returns True if successful."""
        pass
    
    @abstractmethod
    def read(self) -> SensorReading:
        """Read current sensor value."""
        pass
    
    @abstractmethod
    def calibrate(self) -> bool:
        """Calibrate the sensor. Returns True if successful."""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Safely shutdown the sensor."""
        pass

class ActuatorInterface(ABC):
    """Abstract base class for actuator interfaces."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the actuator. Returns True if successful."""
        pass
    
    @abstractmethod
    def execute_command(self, command: ControlCommand) -> bool:
        """Execute a control command. Returns True if successful."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict:
        """Get current actuator status."""
        pass
    
    @abstractmethod
    def emergency_stop(self):
        """Immediately stop actuator in emergency."""
        pass

class MockSensor(SensorInterface):
    """Mock sensor for testing and simulation."""
    
    def __init__(self, sensor_id: str, sensor_type: SensorType, 
                 noise_level: float = 0.01):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.noise_level = noise_level
        self.baseline_value = 0.0
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize mock sensor."""
        self.is_initialized = True
        logger.info(f"Mock sensor {self.sensor_id} initialized")
        return True
        
    def read(self) -> SensorReading:
        """Generate mock sensor reading."""
        if not self.is_initialized:
            raise RuntimeError(f"Sensor {self.sensor_id} not initialized")
            
        # Generate realistic-looking data with some dynamics
        t = time.time()
        base_signal = self.baseline_value + 0.1 * np.sin(2 * np.pi * 0.1 * t)
        noise = np.random.normal(0, self.noise_level)
        value = base_signal + noise
        
        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=t,
            value=value,
            unit=self._get_unit(),
            quality=0.95 + 0.05 * np.random.random()
        )
        
    def _get_unit(self) -> str:
        """Get appropriate unit for sensor type."""
        unit_map = {
            SensorType.ELECTROMAGNETIC: "T",    # Tesla
            SensorType.GRAVITATIONAL: "m/s²",  # Acceleration
            SensorType.PRESSURE: "Pa",         # Pascal
            SensorType.TEMPERATURE: "K",       # Kelvin
            SensorType.VIBRATION: "m/s²",      # Acceleration
            SensorType.OPTICAL: "W/m²",        # Power density
            SensorType.POWER: "W",             # Watts
        }
        return unit_map.get(self.sensor_type, "units")
        
    def calibrate(self) -> bool:
        """Mock calibration."""
        logger.info(f"Calibrating mock sensor {self.sensor_id}")
        time.sleep(0.1)  # Simulate calibration time
        return True
        
    def shutdown(self):
        """Shutdown mock sensor."""
        self.is_initialized = False
        logger.info(f"Mock sensor {self.sensor_id} shutdown")

class MockActuator(ActuatorInterface):
    """Mock actuator for testing and simulation."""
    
    def __init__(self, actuator_id: str, max_value: float = 100.0):
        self.actuator_id = actuator_id
        self.max_value = max_value
        self.current_value = 0.0
        self.is_initialized = False
        self.emergency_stopped = False
        
    def initialize(self) -> bool:
        """Initialize mock actuator."""
        self.is_initialized = True
        self.emergency_stopped = False
        logger.info(f"Mock actuator {self.actuator_id} initialized")
        return True
        
    def execute_command(self, command: ControlCommand) -> bool:
        """Execute mock command."""
        if not self.is_initialized or self.emergency_stopped:
            return False
            
        # Clamp to safe range
        new_value = np.clip(command.value, -self.max_value, self.max_value)
        self.current_value = new_value
        
        logger.debug(f"Actuator {self.actuator_id} set to {new_value}")
        return True
        
    def get_status(self) -> Dict:
        """Get actuator status."""
        return {
            "actuator_id": self.actuator_id,
            "current_value": self.current_value,
            "max_value": self.max_value,
            "initialized": self.is_initialized,
            "emergency_stopped": self.emergency_stopped
        }
        
    def emergency_stop(self):
        """Emergency stop actuator."""
        self.current_value = 0.0
        self.emergency_stopped = True
        logger.warning(f"EMERGENCY STOP: Actuator {self.actuator_id}")

class DataLogger:
    """High-performance data logging system."""
    
    def __init__(self, log_file: str, max_buffer_size: int = 10000):
        self.log_file = log_file
        self.max_buffer_size = max_buffer_size
        self.data_buffer = queue.Queue(maxsize=max_buffer_size)
        self.logging_thread = None
        self.stop_logging = threading.Event()
        
    def start_logging(self):
        """Start background logging thread."""
        self.stop_logging.clear()
        self.logging_thread = threading.Thread(target=self._log_worker)
        self.logging_thread.daemon = True
        self.logging_thread.start()
        logger.info(f"Data logging started to {self.log_file}")
        
    def log_data(self, data: Dict):
        """Add data to logging queue."""
        try:
            self.data_buffer.put_nowait({
                "timestamp": time.time(),
                "data": data
            })
        except queue.Full:
            logger.warning("Data buffer full, dropping oldest data")
            try:
                self.data_buffer.get_nowait()  # Remove oldest
                self.data_buffer.put_nowait({
                    "timestamp": time.time(),
                    "data": data
                })
            except queue.Empty:
                pass
                
    def stop_logging(self):
        """Stop logging and flush buffer."""
        self.stop_logging.set()
        if self.logging_thread:
            self.logging_thread.join(timeout=5.0)
        self._flush_buffer()
        logger.info("Data logging stopped")
        
    def _log_worker(self):
        """Background worker for writing log data."""
        with open(self.log_file, 'w') as f:
            while not self.stop_logging.is_set():
                try:
                    log_entry = self.data_buffer.get(timeout=1.0)
                    json.dump(log_entry, f)
                    f.write('\n')
                    f.flush()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Logging error: {e}")
                    
    def _flush_buffer(self):
        """Flush remaining buffer data to file."""
        with open(self.log_file, 'a') as f:
            while not self.data_buffer.empty():
                try:
                    log_entry = self.data_buffer.get_nowait()
                    json.dump(log_entry, f)
                    f.write('\n')
                except queue.Empty:
                    break

class SafetyMonitor:
    """Safety monitoring and emergency response system."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.emergency_callbacks: List[Callable] = []
        self.safety_violations: List[Dict] = []
        
    def add_emergency_callback(self, callback: Callable):
        """Add callback function for emergency situations."""
        self.emergency_callbacks.append(callback)
        
    def check_safety(self, sensor_readings: Dict[str, SensorReading]) -> bool:
        """
        Check all safety conditions.
        Returns True if safe, False if emergency stop needed.
        """
        for sensor_id, reading in sensor_readings.items():
            threshold = self.config.emergency_thresholds.get(sensor_id)
            if threshold is not None and abs(reading.value) > threshold:
                violation = {
                    "sensor_id": sensor_id,
                    "value": reading.value,
                    "threshold": threshold,
                    "timestamp": reading.timestamp
                }
                self.safety_violations.append(violation)
                
                logger.critical(f"SAFETY VIOLATION: {sensor_id} = {reading.value} > {threshold}")
                
                # Trigger emergency callbacks
                for callback in self.emergency_callbacks:
                    try:
                        callback(violation)
                    except Exception as e:
                        logger.error(f"Emergency callback error: {e}")
                        
                return False
                
        return True

class HardwareInTheLoopManager:
    """
    Main Hardware-in-the-Loop management system.
    
    Coordinates sensors, actuators, safety systems, and control loops
    for real-time warp drive experimentation and testing.
    """
    
    def __init__(self, config: HardwareConfig = None):
        self.config = config or HardwareConfig()
        self.state = HardwareState.OFFLINE
        
        # Hardware components
        self.sensors: Dict[str, SensorInterface] = {}
        self.actuators: Dict[str, ActuatorInterface] = {}
        
        # Data management
        self.sensor_readings: Dict[str, SensorReading] = {}
        self.data_logger = DataLogger("hil_data.jsonl", self.config.max_buffer_size)
        
        # Safety and monitoring
        self.safety_monitor = SafetyMonitor(self.config)
        self.safety_monitor.add_emergency_callback(self._emergency_shutdown)
        
        # Threading
        self.sensor_thread = None
        self.control_thread = None
        self.safety_thread = None
        self.stop_threads = threading.Event()
        
        # Control system
        self.control_commands = queue.PriorityQueue()
        self.external_controller: Optional[Callable] = None
        
    def add_sensor(self, sensor: SensorInterface) -> bool:
        """Add a sensor to the HIL system."""
        if sensor.initialize():
            self.sensors[sensor.sensor_id] = sensor
            logger.info(f"Added sensor: {sensor.sensor_id}")
            return True
        else:
            logger.error(f"Failed to initialize sensor: {sensor.sensor_id}")
            return False
            
    def add_actuator(self, actuator: ActuatorInterface) -> bool:
        """Add an actuator to the HIL system."""
        if actuator.initialize():
            self.actuators[actuator.actuator_id] = actuator
            logger.info(f"Added actuator: {actuator.actuator_id}")
            return True
        else:
            logger.error(f"Failed to initialize actuator: {actuator.actuator_id}")
            return False
            
    def set_external_controller(self, controller: Callable[[Dict[str, SensorReading]], List[ControlCommand]]):
        """Set external control function."""
        self.external_controller = controller
        
    def start_system(self) -> bool:
        """Start the complete HIL system."""
        if self.state != HardwareState.OFFLINE:
            logger.warning("System already running")
            return False
            
        self.state = HardwareState.INITIALIZING
        
        try:
            # Start data logging
            self.data_logger.start_logging()
            
            # Start background threads
            self.stop_threads.clear()
            
            self.sensor_thread = threading.Thread(target=self._sensor_loop)
            self.sensor_thread.daemon = True
            self.sensor_thread.start()
            
            self.control_thread = threading.Thread(target=self._control_loop)
            self.control_thread.daemon = True
            self.control_thread.start()
            
            self.safety_thread = threading.Thread(target=self._safety_loop)
            self.safety_thread.daemon = True
            self.safety_thread.start()
            
            self.state = HardwareState.READY
            logger.info("HIL system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start HIL system: {e}")
            self.state = HardwareState.ERROR
            return False
            
    def stop_system(self):
        """Stop the HIL system safely."""
        logger.info("Stopping HIL system...")
        
        self.stop_threads.set()
        
        # Wait for threads to finish
        for thread in [self.sensor_thread, self.control_thread, self.safety_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
                
        # Stop data logging
        self.data_logger.stop_logging()
        
        # Shutdown hardware
        for sensor in self.sensors.values():
            sensor.shutdown()
        for actuator in self.actuators.values():
            actuator.emergency_stop()
            
        self.state = HardwareState.OFFLINE
        logger.info("HIL system stopped")
        
    def send_command(self, command: ControlCommand):
        """Send command to actuator."""
        self.control_commands.put((command.priority, command))
        
    def get_latest_readings(self) -> Dict[str, SensorReading]:
        """Get latest sensor readings."""
        return self.sensor_readings.copy()
        
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        return {
            "state": self.state.value,
            "num_sensors": len(self.sensors),
            "num_actuators": len(self.actuators),
            "latest_readings": {k: v.value for k, v in self.sensor_readings.items()},
            "actuator_status": {k: v.get_status() for k, v in self.actuators.items()},
            "safety_violations": len(self.safety_monitor.safety_violations),
            "control_queue_size": self.control_commands.qsize()
        }
        
    def _sensor_loop(self):
        """Background thread for sensor data acquisition."""
        dt = 1.0 / self.config.sampling_frequency
        
        while not self.stop_threads.is_set():
            start_time = time.time()
            
            # Read all sensors
            for sensor_id, sensor in self.sensors.items():
                try:
                    reading = sensor.read()
                    self.sensor_readings[sensor_id] = reading
                    
                    # Log sensor data
                    self.data_logger.log_data({
                        "type": "sensor_reading",
                        "sensor_id": sensor_id,
                        "value": reading.value,
                        "unit": reading.unit,
                        "quality": reading.quality
                    })
                    
                except Exception as e:
                    logger.error(f"Error reading sensor {sensor_id}: {e}")
                    
            # Maintain sampling frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def _control_loop(self):
        """Background thread for control command execution."""
        dt = 1.0 / self.config.control_frequency
        
        while not self.stop_threads.is_set():
            start_time = time.time()
            
            # Execute queued commands
            commands_executed = 0
            while not self.control_commands.empty() and commands_executed < 10:
                try:
                    priority, command = self.control_commands.get_nowait()
                    
                    actuator = self.actuators.get(command.actuator_id)
                    if actuator:
                        success = actuator.execute_command(command)
                        
                        # Log command execution
                        self.data_logger.log_data({
                            "type": "command_executed",
                            "actuator_id": command.actuator_id,
                            "command_type": command.command_type,
                            "value": command.value,
                            "success": success
                        })
                        
                    commands_executed += 1
                    
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error executing command: {e}")
                    
            # Call external controller if available
            if self.external_controller and self.sensor_readings:
                try:
                    commands = self.external_controller(self.sensor_readings)
                    for cmd in commands:
                        self.send_command(cmd)
                except Exception as e:
                    logger.error(f"External controller error: {e}")
                    
            # Maintain control frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def _safety_loop(self):
        """Background thread for safety monitoring."""
        dt = 1.0 / self.config.safety_check_frequency
        
        while not self.stop_threads.is_set():
            start_time = time.time()
            
            # Check safety conditions
            if self.sensor_readings:
                is_safe = self.safety_monitor.check_safety(self.sensor_readings)
                
                if not is_safe and self.state != HardwareState.EMERGENCY_STOP:
                    self.state = HardwareState.EMERGENCY_STOP
                    logger.critical("EMERGENCY STOP ACTIVATED")
                    
            # Maintain safety check frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def _emergency_shutdown(self, violation: Dict):
        """Emergency shutdown procedure."""
        logger.critical(f"Emergency shutdown triggered by: {violation}")
        
        # Stop all actuators immediately
        for actuator in self.actuators.values():
            actuator.emergency_stop()
            
        # Log emergency event
        self.data_logger.log_data({
            "type": "emergency_shutdown",
            "violation": violation,
            "timestamp": time.time()
        })

# Example usage and integration with warp drive systems
def create_warp_drive_hil_system() -> HardwareInTheLoopManager:
    """Create a complete HIL system for warp drive testing."""
    
    # Configure system
    config = HardwareConfig(
        sampling_frequency=1000.0,
        control_frequency=100.0,
        emergency_thresholds={
            "field_strength": 10.0,      # Tesla
            "power_consumption": 1000.0,  # Watts
            "temperature": 373.0,         # Kelvin (100°C)
            "pressure": 200000.0          # Pascal (2 atm)
        }
    )
    
    # Create HIL manager
    hil_manager = HardwareInTheLoopManager(config)
    
    # Add sensors
    sensors = [
        MockSensor("field_strength", SensorType.ELECTROMAGNETIC, 0.01),
        MockSensor("power_consumption", SensorType.POWER, 0.1),
        MockSensor("temperature", SensorType.TEMPERATURE, 0.5),
        MockSensor("pressure", SensorType.PRESSURE, 10.0),
        MockSensor("vibration", SensorType.VIBRATION, 0.001),
    ]
    
    for sensor in sensors:
        hil_manager.add_sensor(sensor)
        
    # Add actuators
    actuators = [
        MockActuator("field_coil_1", 100.0),
        MockActuator("field_coil_2", 100.0),
        MockActuator("cooling_pump", 50.0),
        MockActuator("power_regulator", 1000.0),
    ]
    
    for actuator in actuators:
        hil_manager.add_actuator(actuator)
        
    return hil_manager

def warp_drive_controller(readings: Dict[str, SensorReading]) -> List[ControlCommand]:
    """Example warp drive control function."""
    commands = []
    current_time = time.time()
    
    # Get current readings
    field_strength = readings.get("field_strength")
    temperature = readings.get("temperature")
    power = readings.get("power_consumption")
    
    # Simple control logic
    if field_strength and field_strength.value < 5.0:
        # Increase field strength
        commands.append(ControlCommand(
            actuator_id="field_coil_1",
            command_type="set_current",
            value=min(field_strength.value * 1.1, 90.0),
            timestamp=current_time,
            priority=1
        ))
        
    if temperature and temperature.value > 350.0:
        # Increase cooling
        commands.append(ControlCommand(
            actuator_id="cooling_pump",
            command_type="set_flow_rate",
            value=45.0,
            timestamp=current_time,
            priority=2  # Higher priority for safety
        ))
        
    if power and power.value > 800.0:
        # Reduce power consumption
        commands.append(ControlCommand(
            actuator_id="power_regulator",
            command_type="reduce_power",
            value=power.value * 0.9,
            timestamp=current_time,
            priority=2
        ))
        
    return commands

# Example usage
if __name__ == "__main__":
    # Create warp drive HIL system
    print("Creating warp drive HIL system...")
    hil_system = create_warp_drive_hil_system()
    
    # Set control function
    hil_system.set_external_controller(warp_drive_controller)
    
    # Start system
    print("Starting HIL system...")
    if hil_system.start_system():
        print("HIL system started successfully!")
        
        try:
            # Run for a short time
            print("Running system for 10 seconds...")
            time.sleep(10)
            
            # Check status
            status = hil_system.get_system_status()
            print(f"\nSystem Status:")
            print(f"State: {status['state']}")
            print(f"Sensors: {status['num_sensors']}")
            print(f"Actuators: {status['num_actuators']}")
            print(f"Latest readings: {status['latest_readings']}")
            print(f"Safety violations: {status['safety_violations']}")
            
        finally:
            # Stop system
            print("\nStopping HIL system...")
            hil_system.stop_system()
            print("HIL system stopped.")
            
    else:
        print("Failed to start HIL system!")
