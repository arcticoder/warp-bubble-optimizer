#!/usr/bin/env python3
"""
Adaptive Fidelity Runner for Warp Bubble MVP Simulation
======================================================

This runner provides progressive fidelity enhancement for the complete digital-twin
simulation suite, starting from coarse resolution and stepping up to high-fidelity
as all hardware models are validated.

Fidelity Parameters:
- Spatial resolution: Grid size for field calculations and spatial sampling
- Temporal resolution: Time step size for integration and control loops
- Sensor noise levels: Realistic noise modeling for digital-twin validation
- Monte Carlo sampling: Statistical analysis of system reliability

Performance Monitoring:
- Control loop frequencies and latency measurements
- Energy overhead tracking across fidelity levels
- Structural health monitoring under varying simulation precision
- Memory and computational resource utilization
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FidelityConfig:
    """Configuration for simulation fidelity levels."""
    spatial_resolution: int = 100        # Grid points for spatial calculations
    temporal_dt: float = 1.0              # Time step in seconds
    sensor_noise_level: float = 0.01     # Sensor noise standard deviation
    monte_carlo_samples: int = 1         # Number of MC samples (1 = deterministic)
    enable_jax_acceleration: bool = True  # Use JAX for performance
    enable_detailed_logging: bool = False # Detailed performance logging

class AdaptiveFidelityRunner:
    """
    Adaptive fidelity runner for progressive simulation enhancement.
    
    Starts with coarse resolution for rapid prototyping and validation,
    then progressively increases fidelity for detailed analysis and
    Monte Carlo reliability assessment.
    """
    
    def __init__(self):
        self.performance_history = []
        self.fidelity_levels = [
            ("Coarse", FidelityConfig(100, 1.0, 0.05, 1)),
            ("Medium", FidelityConfig(500, 0.5, 0.02, 1)), 
            ("Fine", FidelityConfig(1000, 0.1, 0.01, 1)),
            ("Ultra-Fine", FidelityConfig(2000, 0.05, 0.005, 1)),
            ("Monte Carlo", FidelityConfig(1000, 0.1, 0.01, 100))
        ]
    
    def setup_environment(self, config: FidelityConfig):
        """Configure environment variables for simulation fidelity."""
        os.environ["SIM_GRID_RESOLUTION"] = str(config.spatial_resolution)
        os.environ["SIM_TIME_STEP"] = str(config.temporal_dt)
        os.environ["SIM_SENSOR_NOISE"] = str(config.sensor_noise_level)
        os.environ["SIM_MONTE_CARLO_SAMPLES"] = str(config.monte_carlo_samples)
        os.environ["SIM_ENABLE_JAX"] = str(config.enable_jax_acceleration)
        os.environ["SIM_DETAILED_LOGGING"] = str(config.enable_detailed_logging)
    
    def run_with_fidelity(self, level_name: str, config: FidelityConfig) -> Dict:
        """
        Run simulation at specified fidelity level.
        
        Args:
            level_name: Descriptive name for fidelity level
            config: Fidelity configuration parameters
            
        Returns:
            Performance metrics and results
        """
        print(f"\n{'='*60}")
        print(f"🔬 RUNNING ADAPTIVE FIDELITY SIMULATION: {level_name.upper()}")
        print(f"{'='*60}")
        print(f"   Spatial Resolution: {config.spatial_resolution} grid points")
        print(f"   Temporal Step: {config.temporal_dt} seconds")
        print(f"   Sensor Noise: {config.sensor_noise_level*100:.1f}%")
        print(f"   Monte Carlo Samples: {config.monte_carlo_samples}")
        print(f"   JAX Acceleration: {config.enable_jax_acceleration}")
        
        # Setup environment
        self.setup_environment(config)        # Import and run simulation
        try:
            from simulate_full_warp_MVP import run_full_simulation_with_config, SimulationConfig
            
            start_time = time.time()
            memory_before = self.get_memory_usage()
            
            # Convert FidelityConfig to SimulationConfig
            sim_config = SimulationConfig(
                spatial_resolution=config.spatial_resolution,
                temporal_dt=config.temporal_dt,
                sensor_noise_level=config.sensor_noise_level,
                monte_carlo_samples=config.monte_carlo_samples,
                enable_jax_acceleration=config.enable_jax_acceleration,
                detailed_logging=config.enable_detailed_logging
            )
            
            # Run simulation with converted configuration  
            results = run_full_simulation_with_config(sim_config)
            
            end_time = time.time()
            memory_after = self.get_memory_usage()
            
            # Calculate performance metrics
            simulation_time = end_time - start_time
            memory_usage = memory_after - memory_before
            
            metrics = {
                'level_name': level_name,
                'config': config,
                'simulation_time': simulation_time,
                'memory_usage_mb': memory_usage,
                'results': results
            }
            
            print(f"\n📊 FIDELITY LEVEL PERFORMANCE METRICS:")
            print(f"   Simulation Time: {simulation_time:.2f} seconds")
            print(f"   Memory Usage: {memory_usage:.1f} MB")
            if results:
                print(f"   Control Frequency: {results.get('control_frequency', 0):.1f} Hz")
                print(f"   Energy Overhead: {results.get('energy_overhead', 0)*100:.2f}%")
                print(f"   Structural Health: {results.get('final_structural_health', 1.0):.3f}")
            
            self.performance_history.append(metrics)
            return metrics
            
        except ImportError:
            print("⚠️  MVP simulation not available, using fallback")
            return self.run_fallback_simulation(level_name, config)
        except Exception as e:
            print(f"❌ Simulation failed at {level_name} fidelity: {e}")
            return {'level_name': level_name, 'error': str(e)}
    
    def run_fallback_simulation(self, level_name: str, config: FidelityConfig) -> Dict:
        """Fallback simulation when MVP is not available."""
        from simulate_full_warp_MVP import run_full_simulation
        
        start_time = time.time()
        
        print(f"   Running fallback simulation...")
        run_full_simulation()
        
        end_time = time.time()
        simulation_time = end_time - start_time
        
        # Estimate performance based on fidelity
        estimated_frequency = 1000.0 / config.spatial_resolution
        estimated_overhead = config.spatial_resolution / 10000.0
        
        return {
            'level_name': level_name,
            'simulation_time': simulation_time,
            'control_frequency': estimated_frequency,
            'energy_overhead': estimated_overhead,
            'final_structural_health': 1.0
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0  # Fallback if psutil not available
    
    def run_progressive_fidelity_sweep(self, max_level: Optional[int] = None):
        """
        Run complete progressive fidelity sweep from coarse to fine.
        
        Args:
            max_level: Maximum fidelity level to run (None = all levels)
        """
        print("🚀 ADAPTIVE FIDELITY PROGRESSIVE SWEEP")
        print("=" * 50)
        
        levels_to_run = self.fidelity_levels
        if max_level is not None:
            levels_to_run = self.fidelity_levels[:max_level+1]
        
        for i, (level_name, config) in enumerate(levels_to_run):
            print(f"\n🎯 FIDELITY LEVEL {i+1}/{len(levels_to_run)}: {level_name}")
            
            metrics = self.run_with_fidelity(level_name, config)
            
            # Check for failures
            if 'error' in metrics:
                print(f"❌ Stopping sweep due to failure at {level_name}")
                break
            
            # Performance validation
            if 'results' in metrics and metrics['results']:
                results = metrics['results']
                frequency = results.get('control_frequency', 0)
                health = results.get('final_structural_health', 1.0)
                
                if frequency < 1.0:
                    print(f"⚠️  Warning: Low control frequency ({frequency:.1f} Hz)")
                if health < 0.8:
                    print(f"⚠️  Warning: Structural health degraded ({health:.3f})")
        
        self.generate_fidelity_analysis_report()
    
    def run_monte_carlo_analysis(self, base_config: FidelityConfig, num_samples: int = 100):
        """
        Run Monte Carlo analysis for reliability assessment.
        
        Args:
            base_config: Base configuration for MC analysis
            num_samples: Number of Monte Carlo samples
        """
        print(f"\n🎲 MONTE CARLO RELIABILITY ANALYSIS")
        print(f"   Samples: {num_samples}")
        print(f"   Base Configuration: {base_config.spatial_resolution}x grid, dt={base_config.temporal_dt}s")
        
        mc_config = FidelityConfig(
            spatial_resolution=base_config.spatial_resolution,
            temporal_dt=base_config.temporal_dt,
            sensor_noise_level=base_config.sensor_noise_level,
            monte_carlo_samples=num_samples,
            enable_jax_acceleration=True,
            enable_detailed_logging=False
        )
        
        metrics = self.run_with_fidelity("Monte Carlo", mc_config)
        
        if 'results' in metrics and metrics['results']:
            self.analyze_monte_carlo_results(metrics['results'])
    
    def analyze_monte_carlo_results(self, results: Dict):
        """Analyze Monte Carlo simulation results."""
        print(f"\n📈 MONTE CARLO ANALYSIS RESULTS:")
        
        # Extract reliability metrics
        success_rate = results.get('mission_success_rate', 0.0)
        mean_health = results.get('mean_structural_health', 1.0)
        std_health = results.get('std_structural_health', 0.0)
        
        print(f"   Mission Success Rate: {success_rate*100:.1f}%")
        print(f"   Mean Structural Health: {mean_health:.3f} ± {std_health:.3f}")
        
        # Failure mode analysis
        failure_modes = results.get('failure_modes', {})
        if failure_modes:
            print(f"   Primary Failure Modes:")
            for mode, frequency in failure_modes.items():
                print(f"     {mode}: {frequency*100:.1f}%")
    
    def generate_fidelity_analysis_report(self):
        """Generate comprehensive fidelity analysis report."""
        print(f"\n📋 FIDELITY ANALYSIS SUMMARY REPORT")
        print(f"=" * 50)
        
        if not self.performance_history:
            print("   No performance data available")
            return
        
        print(f"   Levels Completed: {len(self.performance_history)}")
        
        # Performance scaling analysis
        for i, metrics in enumerate(self.performance_history):
            level_name = metrics['level_name']
            sim_time = metrics['simulation_time']
            memory = metrics.get('memory_usage_mb', 0)
            
            print(f"\n   {i+1}. {level_name}:")
            print(f"      Simulation Time: {sim_time:.2f}s")
            print(f"      Memory Usage: {memory:.1f} MB")
            
            if 'results' in metrics and metrics['results']:
                results = metrics['results']
                freq = results.get('control_frequency', 0)
                overhead = results.get('energy_overhead', 0)
                health = results.get('final_structural_health', 1.0)
                
                print(f"      Control Frequency: {freq:.1f} Hz")
                print(f"      Energy Overhead: {overhead*100:.2f}%")
                print(f"      Final Health: {health:.3f}")
        
        # Scaling recommendations
        print(f"\n🎯 FIDELITY SCALING RECOMMENDATIONS:")
        if len(self.performance_history) >= 2:
            coarse = self.performance_history[0]
            fine = self.performance_history[-1]
            time_ratio = fine['simulation_time'] / coarse['simulation_time']
            print(f"   Time Scaling Factor: {time_ratio:.1f}x")
            
            if time_ratio < 10:
                print("   ✅ Good scaling - suitable for routine analysis")
            elif time_ratio < 100:
                print("   ⚠️  Moderate scaling - use fine fidelity selectively")
            else:
                print("   ❌ Poor scaling - optimize before production use")

def main():
    """Main execution function."""
    runner = AdaptiveFidelityRunner()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            # Quick test - only coarse and medium
            runner.run_progressive_fidelity_sweep(max_level=1)
        elif sys.argv[1] == "monte-carlo":
            # Monte Carlo analysis
            base_config = FidelityConfig(1000, 0.1, 0.01, 1)
            runner.run_monte_carlo_analysis(base_config, 50)
        else:
            print("Usage: python fidelity_runner.py [quick|monte-carlo]")
    else:
        # Full progressive sweep
        runner.run_progressive_fidelity_sweep()

if __name__ == "__main__":
    main()
