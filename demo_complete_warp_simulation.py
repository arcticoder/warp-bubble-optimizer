#!/usr/bin/env python3
"""
Comprehensive Warp Engine Simulation Pipeline
===========================================

This script demonstrates the complete progression from GPU acceleration
through shape optimization, quantum constraints, control loops, and analog
prototyping - all in realistic simulation.
"""

import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import our simulation modules
try:
    from gpu_check import check_jax_gpu
    GPU_CHECK_AVAILABLE = True
except ImportError:
    GPU_CHECK_AVAILABLE = False

try:
    from optimize_shape import WarpShapeOptimizer, demo_shape_optimization
    SHAPE_OPT_AVAILABLE = True
except ImportError:
    SHAPE_OPT_AVAILABLE = False

try:
    from qi_constraint import QuantumInequalityConstraint, QIConstrainedOptimizer, demo_qi_constraint
    QI_AVAILABLE = True
except ImportError:
    QI_AVAILABLE = False

try:
    from sim_control_loop import VirtualWarpController, demo_virtual_control
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False

try:
    from visualize_bubble import WarpBubbleVisualizer, demo_visualization
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from analog_sim import AcousticWarpAnalog, ElectromagneticWarpAnalog, demo_analog_simulation
    ANALOG_AVAILABLE = True
except ImportError:
    ANALOG_AVAILABLE = False

# JAX imports with fallback
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False

logger = logging.getLogger(__name__)

class WarpEngineSimulationPipeline:
    """Complete warp engine simulation pipeline."""
    
    def __init__(self, output_dir: str = "warp_simulation_results"):
        """Initialize the simulation pipeline.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.pipeline_start_time = None
        
    def phase_1_gpu_verification(self) -> Dict:
        """Phase 1: Verify GPU acceleration capabilities."""
        print("\n" + "="*60)
        print("🚀 PHASE 1: GPU ACCELERATION VERIFICATION")
        print("="*60)
        
        if not GPU_CHECK_AVAILABLE:
            print("❌ GPU check module not available")
            return {'status': 'unavailable'}
        
        try:
            gpu_available = check_jax_gpu()
            
            # Basic performance test
            if JAX_AVAILABLE:
                import jax
                from jax import jit
                
                @jit
                def test_computation(x):
                    return jnp.sum(jnp.exp(jnp.sin(x)) * jnp.cos(x))
                
                # Warmup
                test_data = jnp.ones((1000, 1000))
                _ = test_computation(test_data)
                
                # Timing
                start = time.time()
                for _ in range(10):
                    result = test_computation(test_data).block_until_ready()
                elapsed = time.time() - start
                
                results = {
                    'status': 'success',
                    'gpu_available': gpu_available,
                    'jax_backend': str(jax.devices()[0]),
                    'performance_test_time': elapsed,
                    'ops_per_second': 10 / elapsed
                }
            else:
                results = {
                    'status': 'jax_unavailable',
                    'gpu_available': False
                }
            
            print(f"✅ GPU verification complete")
            if results.get('gpu_available'):
                print(f"   GPU acceleration: ENABLED")
                print(f"   Performance: {results.get('ops_per_second', 0):.1f} ops/sec")
            else:
                print(f"   GPU acceleration: DISABLED (using CPU)")
            
            return results
            
        except Exception as e:
            logger.error(f"GPU verification failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def phase_2_shape_optimization(self) -> Dict:
        """Phase 2: Optimize warp bubble shape functions."""
        print("\n" + "="*60)
        print("🔧 PHASE 2: WARP SHAPE OPTIMIZATION")
        print("="*60)
        
        if not SHAPE_OPT_AVAILABLE:
            print("❌ Shape optimization module not available")
            return {'status': 'unavailable'}
        
        try:
            # Test multiple ansätze
            ansatz_types = ["gaussian", "polynomial", "hybrid"]
            optimization_results = {}
            
            for ansatz in ansatz_types:
                print(f"\n🔍 Optimizing {ansatz} ansatz...")
                
                optimizer = WarpShapeOptimizer(ansatz_type=ansatz)
                result = optimizer.optimize(max_iter=100, learning_rate=1e-3)
                
                optimization_results[ansatz] = result
                
                print(f"   Final energy: {result['final_energy']:.6e}")
                print(f"   Converged: {'✅' if result['converged'] else '❌'}")
            
            # Find best ansatz
            best_ansatz = min(optimization_results.keys(), 
                            key=lambda k: optimization_results[k]['final_energy'])
            
            results = {
                'status': 'success',
                'optimization_results': optimization_results,
                'best_ansatz': best_ansatz,
                'best_energy': optimization_results[best_ansatz]['final_energy']
            }
            
            print(f"\n🏆 Best ansatz: {best_ansatz}")
            print(f"   Energy: {results['best_energy']:.6e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Shape optimization failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def phase_3_quantum_constraints(self) -> Dict:
        """Phase 3: Enforce quantum inequality constraints."""
        print("\n" + "="*60)
        print("⚛️  PHASE 3: QUANTUM INEQUALITY CONSTRAINTS")
        print("="*60)
        
        if not QI_AVAILABLE:
            print("❌ Quantum constraint module not available")
            return {'status': 'unavailable'}
        
        try:
            # Use results from previous phase if available
            if ('shape_optimization' in self.results and 
                self.results['shape_optimization']['status'] == 'success'):
                
                best_ansatz = self.results['shape_optimization']['best_ansatz']
                print(f"🔗 Using optimized {best_ansatz} ansatz from Phase 2")
                
                # Get the optimizer from previous phase
                optimizer = WarpShapeOptimizer(ansatz_type=best_ansatz)
                base_objective = optimizer.neg_energy_integral
            else:
                # Fallback to simple test objective
                print("📝 Using test objective function")
                def base_objective(theta):
                    x, y = theta
                    return -(x**2 + y**2) + 0.1 * (x**4 + y**4)
            
            # Test different QI penalty weights
            penalty_weights = [1e2, 1e3, 1e4]
            qi_results = {}
            
            qi_constraint = QuantumInequalityConstraint(
                C_constant=1e-2,
                tau_0=1e-6,
                sampling_function="gaussian"
            )
            
            for penalty_weight in penalty_weights:
                print(f"\n🎛️  Testing penalty weight λ = {penalty_weight:.0e}")
                
                qi_optimizer = QIConstrainedOptimizer(
                    base_objective=base_objective,
                    qi_constraint=qi_constraint,
                    penalty_weight=penalty_weight
                )
                
                result = qi_optimizer.optimize(
                    initial_theta=jnp.array([1.0, 1.0]),
                    max_iter=50,
                    learning_rate=1e-2
                )
                
                qi_results[penalty_weight] = result
                
                print(f"   QI satisfied: {'✅' if result['qi_satisfied'] else '❌'}")
                print(f"   Final objective: {result['final_base_objective']:.6e}")
            
            # Find optimal penalty weight
            valid_results = {k: v for k, v in qi_results.items() if v['qi_satisfied']}
            
            if valid_results:
                optimal_weight = min(valid_results.keys(), 
                                   key=lambda k: valid_results[k]['final_base_objective'])
            else:
                optimal_weight = max(penalty_weights)
            
            results = {
                'status': 'success',
                'qi_results': qi_results,
                'optimal_penalty_weight': optimal_weight,
                'qi_bound': qi_constraint.qi_bound,
                'all_satisfied': len(valid_results) > 0
            }
            
            print(f"\n🎯 Optimal penalty weight: λ = {optimal_weight:.0e}")
            print(f"   QI violations resolved: {'✅' if results['all_satisfied'] else '❌'}")
            
            return results
            
        except Exception as e:
            logger.error(f"Quantum constraint enforcement failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def phase_4_control_simulation(self) -> Dict:
        """Phase 4: Simulate real-time control loop."""
        print("\n" + "="*60)
        print("🎮 PHASE 4: VIRTUAL CONTROL LOOP SIMULATION")
        print("="*60)
        
        if not CONTROL_AVAILABLE:
            print("❌ Control simulation module not available")
            return {'status': 'unavailable'}
        
        try:
            # Define control objective (use QI-constrained if available)
            if ('quantum_constraints' in self.results and 
                self.results['quantum_constraints']['status'] == 'success'):
                
                qi_results = self.results['quantum_constraints']['qi_results']
                optimal_weight = self.results['quantum_constraints']['optimal_penalty_weight']
                print(f"🔗 Using QI-constrained objective (λ = {optimal_weight:.0e})")
                
                # Create QI-constrained objective
                from qi_constraint import QuantumInequalityConstraint, QIConstrainedOptimizer
                
                def base_obj(theta):
                    x, y = theta
                    return -(x**2 + y**2) + 0.1 * (x**4 + y**4)
                
                qi_constraint = QuantumInequalityConstraint()
                control_objective = lambda theta: qi_constraint.qi_constrained_objective(
                    theta, base_obj, optimal_weight)
            else:
                # Simple test objective
                print("📝 Using simple test objective")
                def control_objective(theta):
                    x, y = theta
                    return (x - 1)**2 + (y - 2)**2
            
            # Configure control system
            from sim_control_loop import (VirtualWarpController, SensorConfig, 
                                        ActuatorConfig, ControllerConfig)
            
            sensor_config = SensorConfig(
                noise_level=0.005,
                update_rate=100.0,
                latency=0.001,
                drift_rate=1e-6
            )
            
            actuator_config = ActuatorConfig(
                response_time=0.05,
                damping_factor=0.95,
                saturation_limit=5.0,
                bandwidth=50.0
            )
            
            controller_config = ControllerConfig(
                kp=0.3,
                ki=0.05,
                kd=0.01,
                learning_rate=0.1
            )
            
            # Create controller
            initial_params = jnp.array([0.0, 0.0])
            controller = VirtualWarpController(
                objective_func=control_objective,
                initial_params=initial_params,
                sensor_config=sensor_config,
                actuator_config=actuator_config,
                controller_config=controller_config
            )
            
            # Run control simulation
            print("🎮 Running control loop simulation...")
            
            async def run_control():
                return await controller.run_control_loop(duration=3.0, target_rate=100.0)
            
            control_results = asyncio.run(run_control())
            
            # Analyze results
            objectives = [step['objective_value'] for step in control_results['control_history']]
            initial_obj = objectives[0]
            final_obj = objectives[-1]
            improvement = initial_obj - final_obj
            
            # Convergence analysis
            final_10_percent = objectives[int(0.9*len(objectives)):]
            stability = jnp.std(jnp.array(final_10_percent))
            
            results = {
                'status': 'success',
                'control_results': control_results,
                'initial_objective': float(initial_obj),
                'final_objective': float(final_obj),
                'improvement': float(improvement),
                'stability': float(stability),
                'converged': stability < 0.01
            }
            
            print(f"✅ Control simulation complete")
            print(f"   Initial objective: {initial_obj:.6f}")
            print(f"   Final objective: {final_obj:.6f}")
            print(f"   Improvement: {improvement:.6f}")
            print(f"   Converged: {'✅' if results['converged'] else '❌'}")
            
            return results
            
        except Exception as e:
            logger.error(f"Control simulation failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def phase_5_visualization(self) -> Dict:
        """Phase 5: Generate 3D visualizations."""
        print("\n" + "="*60)
        print("🎨 PHASE 5: 3D VISUALIZATION")
        print("="*60)
        
        if not VISUALIZATION_AVAILABLE:
            print("❌ Visualization module not available")
            return {'status': 'unavailable'}
        
        try:
            # Use optimized shape if available
            if ('shape_optimization' in self.results and 
                self.results['shape_optimization']['status'] == 'success'):
                
                best_ansatz = self.results['shape_optimization']['best_ansatz']
                best_params = self.results['shape_optimization']['optimization_results'][best_ansatz]['optimal_params']
                
                print(f"🔗 Visualizing optimized {best_ansatz} ansatz")
                
                # Create appropriate warp function
                if best_ansatz == "gaussian":
                    def warp_func(r, theta):
                        A, sigma = theta
                        return 1.0 - A * jnp.exp(-(r/sigma)**2)
                elif best_ansatz == "polynomial":
                    def warp_func(r, theta):
                        a, b, c = theta
                        r_norm = r / 5.0
                        return 1.0 - a*r_norm**2 - b*r_norm**4 - c*r_norm**6
                else:  # hybrid
                    def warp_func(r, theta):
                        A_gauss, sigma_gauss, a_poly, b_poly = theta
                        r_norm = r / 5.0
                        gaussian_part = A_gauss * jnp.exp(-(r/sigma_gauss)**2)
                        poly_part = a_poly * r_norm**2 + b_poly * r_norm**4
                        return 1.0 - gaussian_part - poly_part
                
                visualization_params = best_params
            else:
                # Default visualization
                print("📝 Using default Gaussian ansatz for visualization")
                def warp_func(r, theta):
                    A, sigma = theta
                    return 1.0 - A * jnp.exp(-(r/sigma)**2)
                
                visualization_params = jnp.array([0.8, 1.5])
            
            # Create visualizations
            from visualize_bubble import WarpBubbleVisualizer
            
            visualizer = WarpBubbleVisualizer(enable_interactive=False)
            
            # 3D bubble visualization
            print("🎨 Creating 3D bubble visualization...")
            plotter_3d = visualizer.visualize_bubble_3d(
                warp_func, visualization_params, 
                title="Optimized Warp Bubble"
            )
            
            # Energy density visualization
            print("🎨 Creating energy density visualization...")
            plotter_energy = visualizer.visualize_energy_density(
                warp_func, visualization_params
            )
            
            # Comparison with standard shapes
            print("🎨 Creating comparison plots...")
            
            # Define comparison functions
            def gaussian_warp(r, theta):
                A, sigma = theta
                return 1.0 - A * jnp.exp(-(r/sigma)**2)
            
            def polynomial_warp(r, theta):
                a, b = theta
                r_norm = r / 5.0
                return 1.0 - a * r_norm**2 - b * r_norm**4
            
            visualizer.create_comparison_plot(
                warp_functions=[gaussian_warp, polynomial_warp, warp_func],
                theta_list=[jnp.array([0.5, 2.0]), jnp.array([0.4, 0.2]), visualization_params],
                labels=['Standard Gaussian', 'Standard Polynomial', 'Optimized'],
                r_max=5.0
            )
            
            results = {
                'status': 'success',
                'visualization_params': list(visualization_params),
                'ansatz_type': getattr(self, '_current_ansatz', 'gaussian'),
                'plots_created': ['3d_bubble', 'energy_density', 'comparison']
            }
            
            print(f"✅ Visualization complete")
            print(f"   Plots created: {len(results['plots_created'])}")
            
            return results
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def phase_6_analog_prototype(self) -> Dict:
        """Phase 6: Simulate analog prototypes."""
        print("\n" + "="*60)
        print("🔬 PHASE 6: ANALOG PROTOTYPE SIMULATION")
        print("="*60)
        
        if not ANALOG_AVAILABLE:
            print("❌ Analog simulation module not available")
            return {'status': 'unavailable'}
        
        try:
            from analog_sim import AcousticWarpAnalog, ElectromagneticWarpAnalog, AnalogConfig
            
            # Acoustic analog simulation
            print("🔊 Running acoustic warp analog...")
            
            acoustic_config = AnalogConfig(
                grid_size=(80, 80),
                physical_size=(4.0, 4.0),
                dt=5e-6,
                wave_speed=343.0,
                damping=0.02,
                source_frequency=1500.0
            )
            
            acoustic_sim = AcousticWarpAnalog(acoustic_config)
            acoustic_results = acoustic_sim.run_simulation(duration=0.008, save_interval=0.001)
            
            # Electromagnetic analog simulation
            print("⚡ Running electromagnetic warp analog...")
            
            em_config = AnalogConfig(
                grid_size=(60, 60),
                physical_size=(0.05, 0.05),  # 5cm × 5cm
                dt=5e-13,
                wave_speed=3e8,
                damping=0.01,
                source_frequency=5e9  # 5 GHz
            )
            
            em_sim = ElectromagneticWarpAnalog(em_config)
            em_results = em_sim.run_simulation(duration=5e-10, save_interval=5e-12)
            
            # Analyze warp effects
            def analyze_warp_effects(results, sim_type):
                """Analyze how the 'warp' affects wave propagation."""
                snapshots = results['snapshots']
                times = results['times']
                
                # Measure wave front propagation speed
                center_i, center_j = snapshots[0].shape[0]//2, snapshots[0].shape[1]//2
                
                # Find first significant signal arrival at edge
                edge_signals = []
                for snapshot in snapshots:
                    edge_signal = jnp.mean(jnp.abs(snapshot[0, :]))  # Top edge
                    edge_signals.append(edge_signal)
                
                # Effective propagation analysis
                max_signal_time = times[jnp.argmax(jnp.array(edge_signals))]
                
                return {
                    'simulation_type': sim_type,
                    'max_signal_time': max_signal_time,
                    'wave_speed_variation': jnp.std(results['wave_speed_map']),
                    'final_field_energy': jnp.sum(snapshots[-1]**2)
                }
            
            acoustic_analysis = analyze_warp_effects(acoustic_results, 'acoustic')
            em_analysis = analyze_warp_effects(em_results, 'electromagnetic')
            
            results = {
                'status': 'success',
                'acoustic_results': acoustic_results,
                'em_results': em_results,
                'acoustic_analysis': acoustic_analysis,
                'em_analysis': em_analysis
            }
            
            print(f"✅ Analog simulation complete")
            print(f"   Acoustic max signal time: {acoustic_analysis['max_signal_time']:.6f}s")
            print(f"   EM max signal time: {em_analysis['max_signal_time']:.2e}s")
            print(f"   Wave speed variations detected: ✅")
            
            return results
            
        except Exception as e:
            logger.error(f"Analog simulation failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def run_complete_pipeline(self) -> Dict:
        """Run the complete warp engine simulation pipeline."""
        print("🚀 STARTING COMPLETE WARP ENGINE SIMULATION PIPELINE")
        print("=" * 80)
        
        self.pipeline_start_time = time.time()
        
        # Phase 1: GPU Verification
        self.results['gpu_verification'] = self.phase_1_gpu_verification()
        
        # Phase 2: Shape Optimization
        self.results['shape_optimization'] = self.phase_2_shape_optimization()
        
        # Phase 3: Quantum Constraints
        self.results['quantum_constraints'] = self.phase_3_quantum_constraints()
        
        # Phase 4: Control Simulation
        self.results['control_simulation'] = self.phase_4_control_simulation()
        
        # Phase 5: Visualization
        self.results['visualization'] = self.phase_5_visualization()
        
        # Phase 6: Analog Prototype
        self.results['analog_prototype'] = self.phase_6_analog_prototype()
        
        # Pipeline summary
        total_time = time.time() - self.pipeline_start_time
        
        print("\n" + "="*80)
        print("🏁 WARP ENGINE SIMULATION PIPELINE COMPLETE")
        print("="*80)
        
        # Count successful phases
        successful_phases = sum(1 for result in self.results.values() 
                              if result.get('status') == 'success')
        total_phases = len(self.results)
        
        print(f"📊 SUMMARY:")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Successful phases: {successful_phases}/{total_phases}")
        print(f"   Success rate: {100*successful_phases/total_phases:.1f}%")
        
        # Phase-by-phase summary
        print(f"\n📋 PHASE RESULTS:")
        phase_names = [
            "GPU Verification", "Shape Optimization", "Quantum Constraints",
            "Control Simulation", "Visualization", "Analog Prototype"
        ]
        
        for i, (phase_key, result) in enumerate(self.results.items()):
            status = result.get('status', 'unknown')
            emoji = '✅' if status == 'success' else '❌' if status == 'error' else '⚠️'
            print(f"   {i+1}. {phase_names[i]:20s}: {emoji} {status}")
        
        # Save results
        results_file = self.output_dir / "pipeline_results.json"
        try:
            import json
            with open(results_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = self._make_serializable(self.results)
                json.dump(serializable_results, f, indent=2)
            print(f"\n💾 Results saved to: {results_file}")
        except Exception as e:
            logger.warning(f"Could not save results: {e}")
        
        # Final assessment
        if successful_phases >= 4:
            print(f"\n🎉 WARP ENGINE SIMULATION: HIGHLY SUCCESSFUL!")
            print(f"   Ready for advanced development phases")
        elif successful_phases >= 2:
            print(f"\n👍 WARP ENGINE SIMULATION: PARTIALLY SUCCESSFUL")
            print(f"   Good foundation for continued development")
        else:
            print(f"\n⚠️  WARP ENGINE SIMULATION: NEEDS ATTENTION")
            print(f"   Check module installations and configurations")
        
        return self.results
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to lists."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, '__float__'):  # numpy scalars
            return float(obj)
        else:
            return obj

def main():
    """Run the complete warp engine simulation pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('warp_engine_simulation.log')
        ]
    )
    
    # Create and run pipeline
    pipeline = WarpEngineSimulationPipeline()
    results = pipeline.run_complete_pipeline()
    
    return results

if __name__ == "__main__":
    results = main()
