#!/usr/bin/env python3
"""
Progress Tracker Utility for Warp Engine Simulation
==================================================

A comprehensive progress tracking system that provides real-time feedback
for long-running simulations and optimizations across all warp engine subsystems.
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class ProgressStep:
    """Individual progress step tracking."""
    name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
    
    @property
    def duration(self) -> Optional[float]:
        """Get step duration if completed."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return None


class ProgressTracker:
    """
    Enhanced progress tracker with detailed logging and performance metrics.
    
    Features:
    - Real-time progress updates with ETA calculation
    - Performance metrics and timing analysis
    - Step-by-step breakdown with status tracking
    - Error handling and recovery suggestions
    - Integration across all warp engine subsystems
    """
    
    def __init__(self, total_steps: int, description: str = "Processing", 
                 enable_detailed_logging: bool = True):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.enable_detailed_logging = enable_detailed_logging
        
        # Enhanced tracking
        self.steps: List[ProgressStep] = []
        self.performance_data: Dict[str, Any] = {
            'total_start_time': time.time(),
            'step_durations': [],
            'memory_usage': [],
            'cpu_usage': [],
            'errors': [],
            'warnings': []
        }
    
    def start(self, additional_info: Optional[Dict] = None):
        """Initialize progress tracking with optional metadata."""
        print(f"\n🚀 {self.description}")
        if additional_info:
            for key, value in additional_info.items():
                print(f"   {key}: {value}")
        print(f"📊 Progress: 0% (0/{self.total_steps} steps)")
        print("="*80)
        
        # Record start metadata
        self.performance_data['start_metadata'] = additional_info or {}
        self.performance_data['total_start_time'] = time.time()
    
    def update(self, step_name: str, step_number: int = None, 
               step_data: Optional[Dict] = None, status: str = "running"):
        """Update progress with enhanced tracking."""
        # Complete previous step if exists
        if self.steps and self.steps[-1].status == "running":
            self.steps[-1].end_time = time.time()
            self.steps[-1].status = "completed"
        
        # Update step tracking
        if step_number is not None:
            self.current_step = step_number
        else:
            self.current_step += 1
        
        # Create new step
        new_step = ProgressStep(
            name=step_name,
            start_time=time.time(),
            status=status,
            data=step_data or {}
        )
        self.steps.append(new_step)
        
        # Calculate progress metrics
        percentage = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        step_time = time.time() - self.step_start_time
        
        # Display progress
        if self.current_step > 1:
            print(f"   ✅ Previous step completed in {step_time:.1f}s")
            self.performance_data['step_durations'].append(step_time)
        
        print(f"\n📊 Progress: {percentage:.1f}% ({self.current_step}/{self.total_steps} steps)")
        print(f"🔄 Current: {step_name}")
        print(f"⏱️  Elapsed: {elapsed:.1f}s")
        
        # Calculate and display ETA
        if self.current_step < self.total_steps and self.current_step > 0:
            avg_step_time = elapsed / self.current_step
            eta = avg_step_time * (self.total_steps - self.current_step)
            print(f"🕒 ETA: {eta:.1f}s remaining")
        
        # Display step-specific data if provided
        if step_data and self.enable_detailed_logging:
            print(f"📈 Step Data:")
            for key, value in step_data.items():
                print(f"   • {key}: {value}")
        
        print("-" * 80)
        self.step_start_time = time.time()
    
    def add_warning(self, message: str):
        """Add a warning message."""
        warning = {
            'timestamp': time.time(),
            'step': self.current_step,
            'message': message
        }
        self.performance_data['warnings'].append(warning)
        print(f"⚠️  Warning: {message}")
    
    def add_error(self, message: str, error_data: Optional[Dict] = None):
        """Add an error message."""
        error = {
            'timestamp': time.time(),
            'step': self.current_step,
            'message': message,
            'data': error_data or {}
        }
        self.performance_data['errors'].append(error)
        print(f"❌ Error: {message}")
    
    def mark_step_failed(self, error_message: str):
        """Mark current step as failed."""
        if self.steps and self.steps[-1].status == "running":
            self.steps[-1].status = "failed"
            self.steps[-1].end_time = time.time()
            self.steps[-1].data['error'] = error_message
        self.add_error(error_message)
    
    def complete(self, summary_data: Optional[Dict] = None):
        """Mark progress as complete with performance summary."""
        # Complete final step
        if self.steps and self.steps[-1].status == "running":
            self.steps[-1].end_time = time.time()
            self.steps[-1].status = "completed"
        
        total_time = time.time() - self.start_time
        step_time = time.time() - self.step_start_time
        
        print(f"   ✅ Final step completed in {step_time:.1f}s")
        print(f"\n🎉 {self.description} COMPLETE!")
        print(f"📊 Progress: 100% ({self.total_steps}/{self.total_steps} steps)")
        print(f"⏱️  Total time: {total_time:.1f}s")
        
        # Performance summary
        if self.performance_data['step_durations']:
            avg_step = sum(self.performance_data['step_durations']) / len(self.performance_data['step_durations'])
            print(f"📈 Average step time: {avg_step:.1f}s")
        
        if self.performance_data['warnings']:
            print(f"⚠️  Warnings: {len(self.performance_data['warnings'])}")
        
        if self.performance_data['errors']:
            print(f"❌ Errors: {len(self.performance_data['errors'])}")
        
        # Display summary data if provided
        if summary_data:
            print(f"📋 Summary:")
            for key, value in summary_data.items():
                print(f"   • {key}: {value}")
        
        print("="*80)
        
        # Store completion metadata
        self.performance_data['total_duration'] = total_time
        self.performance_data['completion_summary'] = summary_data or {}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate detailed performance report."""
        report = {
            'description': self.description,
            'total_steps': self.total_steps,
            'completed_steps': len([s for s in self.steps if s.status == "completed"]),
            'failed_steps': len([s for s in self.steps if s.status == "failed"]),
            'total_duration': time.time() - self.performance_data['total_start_time'],
            'steps': [
                {
                    'name': step.name,
                    'duration': step.duration,
                    'status': step.status,
                    'data': step.data
                }
                for step in self.steps
            ],
            'performance_data': self.performance_data
        }
        return report
    
    def save_report(self, filename: str):
        """Save performance report to file."""
        import json
        report = self.get_performance_report()
        
        # Convert any non-serializable data
        def make_serializable(obj):
            if hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            elif hasattr(obj, '__dict__'):  # custom objects
                return str(obj)
            return obj
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=make_serializable)
            print(f"📁 Performance report saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save report: {e}")


class MultiProcessProgressTracker:
    """
    Advanced progress tracker for multi-process simulations.
    
    Handles parallel operations across multiple warp engine subsystems
    with centralized progress coordination.
    """
    
    def __init__(self, process_configs: List[Dict[str, Any]]):
        self.process_configs = process_configs
        self.trackers: Dict[str, ProgressTracker] = {}
        self.start_time = time.time()
        
        for config in process_configs:
            process_id = config['id']
            self.trackers[process_id] = ProgressTracker(
                total_steps=config['steps'],
                description=config['description'],
                enable_detailed_logging=config.get('detailed_logging', True)
            )
    
    def start_all(self):
        """Start all process trackers."""
        print(f"\n🚀 MULTI-PROCESS WARP ENGINE SIMULATION")
        print(f"📊 Processes: {len(self.trackers)}")
        print("="*80)
        
        for process_id, tracker in self.trackers.items():
            print(f"🔄 Starting process: {process_id}")
            tracker.start()
    
    def update_process(self, process_id: str, step_name: str, 
                      step_data: Optional[Dict] = None):
        """Update specific process progress."""
        if process_id in self.trackers:
            self.trackers[process_id].update(step_name, step_data=step_data)
    
    def complete_all(self):
        """Complete all process trackers."""
        total_time = time.time() - self.start_time
        
        print(f"\n🎉 ALL PROCESSES COMPLETE!")
        print(f"⏱️  Total multi-process time: {total_time:.1f}s")
        
        # Summary statistics
        total_steps = sum(t.total_steps for t in self.trackers.values())
        total_warnings = sum(len(t.performance_data['warnings']) for t in self.trackers.values())
        total_errors = sum(len(t.performance_data['errors']) for t in self.trackers.values())
        
        print(f"📊 Total steps completed: {total_steps}")
        print(f"⚠️  Total warnings: {total_warnings}")
        print(f"❌ Total errors: {total_errors}")
        print("="*80)
    
    def get_combined_report(self) -> Dict[str, Any]:
        """Generate combined report from all processes."""
        return {
            'multi_process_summary': {
                'total_processes': len(self.trackers),
                'total_duration': time.time() - self.start_time,
                'start_time': self.start_time
            },
            'process_reports': {
                process_id: tracker.get_performance_report()
                for process_id, tracker in self.trackers.items()
            }
        }


# GPU/CPU acceleration progress tracking
class AccelerationProgressTracker(ProgressTracker):
    """
    Specialized progress tracker for JAX/GPU acceleration monitoring.
    
    Tracks GPU utilization, memory usage, and acceleration performance.
    """
    
    def __init__(self, total_steps: int, description: str = "GPU Acceleration"):
        super().__init__(total_steps, description)
        self.gpu_available = False
        self.device_info = {}
        
        # Check for JAX and GPU availability
        try:
            import jax
            self.gpu_available = 'gpu' in str(jax.devices()).lower()
            self.device_info = {
                'devices': str(jax.devices()),
                'default_backend': jax.default_backend(),
                'gpu_available': self.gpu_available
            }
        except ImportError:
            self.device_info = {'jax_available': False}
    
    def start(self, additional_info: Optional[Dict] = None):
        """Start with GPU/acceleration information."""
        gpu_info = {
            'GPU Available': "✅ Yes" if self.gpu_available else "❌ No (CPU only)",
            'JAX Available': "✅ Yes" if 'jax_available' not in self.device_info else "❌ No",
            'Acceleration Mode': "GPU" if self.gpu_available else "CPU with JIT"
        }
        
        if additional_info:
            gpu_info.update(additional_info)
            
        super().start(gpu_info)
    
    def update_with_performance(self, step_name: str, performance_metrics: Dict[str, Any]):
        """Update with GPU performance metrics."""
        step_data = {
            'GPU Memory': f"{performance_metrics.get('gpu_memory_mb', 0):.1f} MB",
            'Computation Time': f"{performance_metrics.get('compute_time_ms', 0):.1f} ms",
            'Speedup': f"{performance_metrics.get('speedup_factor', 1.0):.1f}x",
            'Device': performance_metrics.get('device', 'Unknown')
        }
        
        self.update(step_name, step_data=step_data)


if __name__ == "__main__":
    # Demo of enhanced progress tracking
    print("🧪 PROGRESS TRACKER DEMO")
    print("="*50)
    
    # Simple progress tracking demo
    tracker = ProgressTracker(5, "Warp Engine Initialization")
    tracker.start({'Bubble Radius': '10.0 m', 'Warp Velocity': '5000c'})
    
    for i in range(5):
        step_data = {
            'Energy': f"{-1e30 * (i+1):.2e} J",
            'Stability': f"{0.95 - i*0.01:.3f}",
            'Temperature': f"{300 + i*50} K"
        }
        tracker.update(f"Subsystem {i+1} Online", step_data=step_data)
        time.sleep(0.5)  # Simulate work
    
    summary = {
        'Total Energy': "-5.0e30 J",
        'Final Stability': "0.91",
        'System Status': "READY"
    }
    tracker.complete(summary)
    
    # Save performance report
    tracker.save_report("demo_progress_report.json")
