#!/usr/bin/env python3
"""
ADVANCED WARP OPTIMIZATION DEMONSTRATION
========================================

This script demonstrates the implementation and execution of all six advanced
optimization strategies for pushing E₋ to maximally negative values:

1. 🌊 Mixed-basis ansatz (Gaussians + Fourier modes)
2. 🧠 Surrogate-assisted Bayesian optimization (scikit-optimize GP)
3. 🎯 Multi-objective search (NSGA-II: energy vs. stability)
4. 🌍 High-dimensional global search (CMA-ES with 12-16 Gaussians)
5. ⚡ Gradient-enhanced local descent (JAX L-BFGS-B)
6. 🚀 Parallel evaluation & vectorization (batch processing)

This demonstration runs all strategies and compares their performance,
showcasing the cutting-edge optimization techniques for warp bubble physics.

Usage:
    python demo_advanced_optimization.py [--quick] [--strategies LIST]

Examples:
    python demo_advanced_optimization.py --quick
    python demo_advanced_optimization.py --strategies bayesian,jax,cma
"""

import argparse
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Check for required modules
modules_status = {}

try:
    from advanced_multi_strategy_optimizer import AdvancedWarpOptimizer, OptimizationResult
    modules_status['advanced_optimizer'] = True
except ImportError:
    modules_status['advanced_optimizer'] = False

try:
    from bayes_opt_and_refine import MinimalWarpOptimizer
    modules_status['minimal_optimizer'] = True
except ImportError:
    modules_status['minimal_optimizer'] = False

try:
    import jax
    modules_status['jax'] = True
except ImportError:
    modules_status['jax'] = False

try:
    from skopt import gp_minimize
    modules_status['skopt'] = True
except ImportError:
    modules_status['skopt'] = False

try:
    import cma
    modules_status['cma'] = True
except ImportError:
    modules_status['cma'] = False

try:
    from deap import base, creator, tools
    modules_status['deap'] = True
except ImportError:
    modules_status['deap'] = False

def print_module_status():
    """Print status of all required modules"""
    print("📦 MODULE AVAILABILITY STATUS")
    print("=" * 50)
    
    status_map = {
        'advanced_optimizer': 'Advanced Multi-Strategy Optimizer',
        'minimal_optimizer': 'Minimal Bayesian+JAX Optimizer',
        'jax': 'JAX (Automatic Differentiation)',
        'skopt': 'scikit-optimize (Bayesian Optimization)',
        'cma': 'CMA-ES (Evolution Strategy)',
        'deap': 'DEAP (Multi-objective Optimization)'
    }
    
    for module, available in modules_status.items():
        status = "✅ Available" if available else "❌ Missing"
        name = status_map.get(module, module)
        print(f"   {name:35s} | {status}")
    
    print()
    
    # Installation recommendations
    missing_modules = [k for k, v in modules_status.items() if not v]
    if missing_modules:
        print("📋 TO INSTALL MISSING MODULES:")
        if 'jax' in missing_modules:
            print("   pip install jax jaxlib")
        if 'skopt' in missing_modules:
            print("   pip install scikit-optimize")
        if 'cma' in missing_modules:
            print("   pip install cma")
        if 'deap' in missing_modules:
            print("   pip install deap")
        print()

class AdvancedOptimizationDemo:
    """Demonstration class for advanced optimization strategies"""
    
    def __init__(self, n_gaussians: int = 6, n_fourier: int = 4, quick_mode: bool = False):
        """
        Initialize demonstration
        
        Args:
            n_gaussians: Number of Gaussian components
            n_fourier: Number of Fourier modes
            quick_mode: Use reduced parameters for faster execution
        """
        self.n_gaussians = n_gaussians
        self.n_fourier = n_fourier
        self.quick_mode = quick_mode
        
        # Initialize optimizers if available
        self.advanced_optimizer = None
        self.minimal_optimizer = None
        
        if modules_status['advanced_optimizer']:
            self.advanced_optimizer = AdvancedWarpOptimizer(
                n_gaussians=n_gaussians,
                n_fourier_modes=n_fourier,
                verbose=True
            )
        
        if modules_status['minimal_optimizer']:
            self.minimal_optimizer = MinimalWarpOptimizer(
                n_gaussians=n_gaussians,
                n_fourier=n_fourier
            )
        
        # Strategy parameters
        if quick_mode:
            self.strategy_params = {
                'bayes_calls': 30,
                'bayes_initial': 10,
                'nsga_pop': 20,
                'nsga_gen': 20,
                'cma_gaussians': 8,
                'cma_pop': 32,
                'cma_evals': 1000,
                'parallel_pop': 30,
                'parallel_gen': 15,
                'jax_iterations': 100
            }
        else:
            self.strategy_params = {
                'bayes_calls': 100,
                'bayes_initial': 20,
                'nsga_pop': 40,
                'nsga_gen': 50,
                'cma_gaussians': 12,
                'cma_pop': 128,
                'cma_evals': 5000,
                'parallel_pop': 60,
                'parallel_gen': 30,
                'jax_iterations': 300
            }
        
        self.results = []
        
        print(f"🚀 Advanced Optimization Demo Initialized")
        print(f"   Gaussians: {n_gaussians}, Fourier modes: {n_fourier}")
        print(f"   Quick mode: {quick_mode}")
    
    def demonstrate_strategy_1_mixed_basis(self):
        """Demonstrate Strategy 1: Mixed-basis ansatz"""
        print("\n" + "="*70)
        print("🌊 STRATEGY 1: MIXED-BASIS ANSATZ (Gaussians + Fourier)")
        print("="*70)
        
        if not self.minimal_optimizer:
            print("❌ Minimal optimizer not available. Skipping demonstration.")
            return None
        
        print("🔬 Demonstrating enhanced profile flexibility through mixed basis...")
        
        # Create sample parameters for demonstration
        demo_params = np.zeros(2 + 3*self.n_gaussians + self.n_fourier)
        demo_params[0] = 5e-6  # mu
        demo_params[1] = 2e-5  # G_geo
        
        # Gaussian components with varied characteristics
        for i in range(self.n_gaussians):
            idx = 2 + 3*i
            demo_params[idx] = 1.0 / (i + 1)  # Decreasing amplitudes
            demo_params[idx + 1] = (i + 0.5) * (100 * 299792458) / self.n_gaussians  # Positions
            demo_params[idx + 2] = 0.1 * (100 * 299792458)  # Widths
        
        # Fourier coefficients for oscillatory structure
        fourier_start = 2 + 3*self.n_gaussians
        for k in range(self.n_fourier):
            demo_params[fourier_start + k] = 0.1 * (-1)**k  # Alternating coefficients
        
        # Compute and display energy
        energy = self.minimal_optimizer.compute_negative_energy(demo_params)
        
        print(f"📊 Mixed-basis profile characteristics:")
        print(f"   Gaussian components: {self.n_gaussians}")
        print(f"   Fourier modes: {self.n_fourier}")
        print(f"   Total parameters: {len(demo_params)}")
        print(f"   Demo energy E₋: {energy:.6e} J")
        
        # Visualize profile components if matplotlib available
        try:
            r_grid = np.linspace(0, 100 * 299792458, 1000)
            f_total = self.minimal_optimizer.mixed_profile(r_grid, demo_params)
            
            plt.figure(figsize=(12, 8))
            
            # Total profile
            plt.subplot(2, 2, 1)
            plt.plot(r_grid / (100 * 299792458), f_total, 'b-', linewidth=2)
            plt.title('Mixed-Basis Warp Profile f(r)')
            plt.xlabel('r/R')
            plt.ylabel('f(r)')
            plt.grid(True, alpha=0.3)
            
            # Gaussian components only
            f_gaussian = np.zeros_like(r_grid)
            for i in range(self.n_gaussians):
                idx = 2 + 3*i
                A_i = demo_params[idx]
                r_i = demo_params[idx + 1]
                sigma_i = demo_params[idx + 2]
                gaussian_i = A_i * np.exp(-0.5 * ((r_grid - r_i) / sigma_i)**2)
                f_gaussian += gaussian_i
                plt.subplot(2, 2, 2)
                plt.plot(r_grid / (100 * 299792458), gaussian_i, '--', alpha=0.7, label=f'G{i+1}')
            
            plt.subplot(2, 2, 2)
            plt.plot(r_grid / (100 * 299792458), f_gaussian, 'r-', linewidth=2, label='Total Gaussian')
            plt.title('Individual Gaussian Components')
            plt.xlabel('r/R')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Fourier components
            f_fourier = np.zeros_like(r_grid)
            for k in range(self.n_fourier):
                B_k = demo_params[fourier_start + k]
                fourier_k = B_k * np.cos(np.pi * (k + 1) * r_grid / (100 * 299792458))
                f_fourier += fourier_k
                plt.subplot(2, 2, 3)
                plt.plot(r_grid / (100 * 299792458), fourier_k, '--', alpha=0.7, label=f'F{k+1}')
            
            plt.subplot(2, 2, 3)
            plt.plot(r_grid / (100 * 299792458), f_fourier, 'g-', linewidth=2, label='Total Fourier')
            plt.title('Fourier Mode Components')
            plt.xlabel('r/R')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Profile derivative
            df_dr = self.minimal_optimizer.profile_derivative(r_grid, demo_params)
            plt.subplot(2, 2, 4)
            plt.plot(r_grid / (100 * 299792458), df_dr, 'purple', linewidth=2)
            plt.title("Profile Derivative f'(r)")
            plt.xlabel('r/R')
            plt.ylabel("f'(r)")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('strategy1_mixed_basis_demo.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("📊 Mixed-basis visualization saved as 'strategy1_mixed_basis_demo.png'")
            
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")
        
        return {
            'strategy': 'Mixed-Basis Ansatz',
            'energy': energy,
            'parameters': demo_params,
            'success': True
        }
    
    def demonstrate_strategy_2_bayesian(self):
        """Demonstrate Strategy 2: Bayesian optimization"""
        print("\n" + "="*70)
        print("🧠 STRATEGY 2: SURROGATE-ASSISTED BAYESIAN OPTIMIZATION")
        print("="*70)
        
        if not modules_status['skopt']:
            print("❌ scikit-optimize not available. Skipping Bayesian optimization.")
            return None
        
        if not self.minimal_optimizer:
            print("❌ Minimal optimizer not available. Skipping demonstration.")
            return None
        
        print("🔬 Demonstrating Gaussian Process surrogate-assisted optimization...")
        
        try:
            start_time = time.time()
            
            # Run Bayesian optimization
            best_params, best_energy, metadata = self.minimal_optimizer.run_bayesian_optimization(
                n_calls=self.strategy_params['bayes_calls'],
                n_initial_points=self.strategy_params['bayes_initial']
            )
            
            elapsed = time.time() - start_time
            
            print(f"✅ Bayesian optimization demonstration completed")
            print(f"   Best E₋: {best_energy:.6e} J")
            print(f"   Time: {elapsed:.1f}s")
            print(f"   Function evaluations: {self.strategy_params['bayes_calls']}")
            
            # Convergence visualization
            if 'convergence' in metadata:
                plt.figure(figsize=(10, 6))
                convergence = metadata['convergence']
                plt.plot(convergence, 'b-', linewidth=2)
                plt.axhline(y=best_energy, color='r', linestyle='--', label=f'Best: {best_energy:.2e} J')
                plt.xlabel('Iteration')
                plt.ylabel('Energy E₋ (J)')
                plt.title('Bayesian Optimization Convergence')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig('strategy2_bayesian_convergence.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                print("📊 Convergence plot saved as 'strategy2_bayesian_convergence.png'")
            
            return {
                'strategy': 'Bayesian Optimization',
                'energy': best_energy,
                'parameters': best_params,
                'time': elapsed,
                'evaluations': self.strategy_params['bayes_calls'],
                'success': True,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"❌ Bayesian optimization demonstration failed: {e}")
            return None
    
    def demonstrate_strategy_3_multiobjective(self):
        """Demonstrate Strategy 3: Multi-objective NSGA-II"""
        print("\n" + "="*70)
        print("🎯 STRATEGY 3: MULTI-OBJECTIVE NSGA-II (Energy vs. Stability)")
        print("="*70)
        
        if not modules_status['deap']:
            print("❌ DEAP not available. Skipping multi-objective optimization.")
            return None
        
        if not self.advanced_optimizer:
            print("❌ Advanced optimizer not available. Skipping demonstration.")
            return None
        
        print("🔬 Demonstrating Pareto-optimal trade-offs between energy and stability...")
        
        try:
            result = self.advanced_optimizer.run_multiobjective_nsga2(
                population_size=self.strategy_params['nsga_pop'],
                generations=self.strategy_params['nsga_gen']
            )
            
            if result.success:
                print(f"✅ Multi-objective optimization demonstration completed")
                print(f"   Best E₋: {result.energy:.6e} J")
                print(f"   Time: {result.time:.1f}s")
                print(f"   Pareto front size: {result.metadata.get('pareto_front_size', 'N/A')}")
                
                return {
                    'strategy': 'Multi-Objective NSGA-II',
                    'energy': result.energy,
                    'parameters': result.params,
                    'time': result.time,
                    'evaluations': result.evaluations,
                    'success': True,
                    'metadata': result.metadata
                }
            else:
                print("❌ Multi-objective optimization failed")
                return None
                
        except Exception as e:
            print(f"❌ Multi-objective demonstration failed: {e}")
            return None
    
    def demonstrate_strategy_4_cma_es(self):
        """Demonstrate Strategy 4: High-dimensional CMA-ES"""
        print("\n" + "="*70)
        print("🌍 STRATEGY 4: HIGH-DIMENSIONAL CMA-ES GLOBAL SEARCH")
        print("="*70)
        
        if not modules_status['cma']:
            print("❌ CMA-ES not available. Skipping high-dimensional optimization.")
            return None
        
        if not self.advanced_optimizer:
            print("❌ Advanced optimizer not available. Skipping demonstration.")
            return None
        
        print(f"🔬 Demonstrating global search with {self.strategy_params['cma_gaussians']} Gaussians...")
        
        try:
            result = self.advanced_optimizer.run_high_dimensional_cma_es(
                n_gaussians_hd=self.strategy_params['cma_gaussians'],
                population_size=self.strategy_params['cma_pop'],
                max_evaluations=self.strategy_params['cma_evals']
            )
            
            if result.success:
                print(f"✅ High-dimensional CMA-ES demonstration completed")
                print(f"   Best E₋: {result.energy:.6e} J")
                print(f"   Time: {result.time:.1f}s")
                print(f"   Function evaluations: {result.evaluations}")
                
                return {
                    'strategy': f'CMA-ES ({self.strategy_params[\"cma_gaussians\"]} Gaussians)',
                    'energy': result.energy,
                    'parameters': result.params,
                    'time': result.time,
                    'evaluations': result.evaluations,
                    'success': True
                }
            else:
                print("❌ High-dimensional CMA-ES failed")
                return None
                
        except Exception as e:
            print(f"❌ CMA-ES demonstration failed: {e}")
            return None
    
    def demonstrate_strategy_5_jax_gradient(self, initial_params=None):
        """Demonstrate Strategy 5: JAX gradient-enhanced descent"""
        print("\n" + "="*70)
        print("⚡ STRATEGY 5: JAX GRADIENT-ENHANCED LOCAL DESCENT")
        print("="*70)
        
        if not modules_status['jax']:
            print("❌ JAX not available. Skipping gradient-enhanced optimization.")
            return None
        
        if not self.minimal_optimizer:
            print("❌ Minimal optimizer not available. Skipping demonstration.")
            return None
        
        print("🔬 Demonstrating automatic differentiation + L-BFGS refinement...")
        
        # Use provided initial parameters or generate random ones
        if initial_params is None:
            print("⚠️  No initial parameters provided. Generating random starting point...")
            initial_params = np.random.randn(2 + 3*self.n_gaussians + self.n_fourier) * 0.1
            initial_params[0] = abs(initial_params[0]) * 1e-5  # Ensure positive mu
            initial_params[1] = abs(initial_params[1]) * 1e-5  # Ensure positive G_geo
        
        try:
            refined_params, refined_energy, metadata = self.minimal_optimizer.run_jax_refinement(
                initial_params=initial_params,
                max_iterations=self.strategy_params['jax_iterations']
            )
            
            initial_energy = self.minimal_optimizer.compute_negative_energy(initial_params)
            improvement = initial_energy - refined_energy
            
            print(f"✅ JAX gradient descent demonstration completed")
            print(f"   Initial E₋: {initial_energy:.6e} J")
            print(f"   Final E₋: {refined_energy:.6e} J")
            print(f"   Improvement: {improvement:.2e} J")
            print(f"   Time: {metadata['time']:.1f}s")
            
            return {
                'strategy': 'JAX Gradient Descent',
                'energy': refined_energy,
                'parameters': refined_params,
                'time': metadata['time'],
                'iterations': metadata.get('iterations', 'N/A'),
                'improvement': improvement,
                'success': True,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"❌ JAX gradient descent demonstration failed: {e}")
            return None
    
    def demonstrate_strategy_6_parallel(self):
        """Demonstrate Strategy 6: Parallel batch optimization"""
        print("\n" + "="*70)
        print("🚀 STRATEGY 6: PARALLEL EVALUATION & VECTORIZATION")
        print("="*70)
        
        if not self.advanced_optimizer:
            print("❌ Advanced optimizer not available. Skipping demonstration.")
            return None
        
        print("🔬 Demonstrating parallel batch evaluation for computational efficiency...")
        
        try:
            result = self.advanced_optimizer.run_parallel_batch_optimization(
                population_size=self.strategy_params['parallel_pop'],
                n_generations=self.strategy_params['parallel_gen'],
                n_workers=2  # Conservative for demonstration
            )
            
            if result.success:
                efficiency = abs(result.energy) / result.time if result.time > 0 else 0
                
                print(f"✅ Parallel batch optimization demonstration completed")
                print(f"   Best E₋: {result.energy:.6e} J")
                print(f"   Time: {result.time:.1f}s")
                print(f"   Function evaluations: {result.evaluations}")
                print(f"   Efficiency: {efficiency:.2e} J/s")
                
                return {
                    'strategy': 'Parallel Batch Optimization',
                    'energy': result.energy,
                    'parameters': result.params,
                    'time': result.time,
                    'evaluations': result.evaluations,
                    'efficiency': efficiency,
                    'success': True,
                    'metadata': result.metadata
                }
            else:
                print("❌ Parallel batch optimization failed")
                return None
                
        except Exception as e:
            print(f"❌ Parallel optimization demonstration failed: {e}")
            return None
    
    def run_comprehensive_demonstration(self, strategies: List[str] = None):
        """Run comprehensive demonstration of all available strategies"""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE ADVANCED OPTIMIZATION DEMONSTRATION")
        print("="*80)
        
        # Default to all strategies if none specified
        if strategies is None:
            strategies = ['mixed', 'bayesian', 'multiobjective', 'cma', 'jax', 'parallel']
        
        print(f"📋 Running strategies: {', '.join(strategies)}")
        print(f"⚡ Quick mode: {self.quick_mode}")
        
        demo_start = time.time()
        
        # Strategy 1: Mixed-basis ansatz
        if 'mixed' in strategies:
            result1 = self.demonstrate_strategy_1_mixed_basis()
            if result1:
                self.results.append(result1)
        
        # Strategy 2: Bayesian optimization
        if 'bayesian' in strategies:
            result2 = self.demonstrate_strategy_2_bayesian()
            if result2:
                self.results.append(result2)
        
        # Strategy 3: Multi-objective NSGA-II
        if 'multiobjective' in strategies:
            result3 = self.demonstrate_strategy_3_multiobjective()
            if result3:
                self.results.append(result3)
        
        # Strategy 4: High-dimensional CMA-ES
        if 'cma' in strategies:
            result4 = self.demonstrate_strategy_4_cma_es()
            if result4:
                self.results.append(result4)
        
        # Strategy 5: JAX gradient descent (use best result so far as starting point)
        if 'jax' in strategies:
            initial_params = None
            if self.results:
                # Use best result as starting point for JAX refinement
                best_result = min(self.results, key=lambda x: x.get('energy', float('inf')))
                if 'parameters' in best_result:
                    initial_params = best_result['parameters']
                    print(f"🔥 Using {best_result['strategy']} result as JAX starting point")
            
            result5 = self.demonstrate_strategy_5_jax_gradient(initial_params)
            if result5:
                self.results.append(result5)
        
        # Strategy 6: Parallel batch optimization
        if 'parallel' in strategies:
            result6 = self.demonstrate_strategy_6_parallel()
            if result6:
                self.results.append(result6)
        
        demo_elapsed = time.time() - demo_start
        
        # Generate summary
        self.generate_demonstration_summary(demo_elapsed)
        
        return self.results
    
    def generate_demonstration_summary(self, total_time: float):
        """Generate comprehensive demonstration summary"""
        print("\n" + "="*80)
        print("📊 ADVANCED OPTIMIZATION DEMONSTRATION SUMMARY")
        print("="*80)
        
        if not self.results:
            print("❌ No successful strategy demonstrations completed")
            return
        
        # Sort results by energy
        energy_results = [r for r in self.results if 'energy' in r]
        energy_results.sort(key=lambda x: x['energy'])
        
        print(f"\n🏆 STRATEGY PERFORMANCE RANKING:")
        print("-" * 60)
        for i, result in enumerate(energy_results):
            strategy = result['strategy']
            energy = result['energy']
            time_taken = result.get('time', 0)
            evaluations = result.get('evaluations', 'N/A')
            
            print(f"{i+1:2d}. {strategy:30s} | E₋: {energy:.3e} J | "
                  f"Time: {time_taken:6.1f}s | Evals: {evaluations}")
        
        # Best result details
        if energy_results:
            best = energy_results[0]
            print(f"\n🎯 BEST RESULT: {best['strategy']}")
            print(f"   Energy E₋: {best['energy']:.6e} J")
            print(f"   Time: {best.get('time', 0):.1f} seconds")
            
            if 'parameters' in best and best['parameters'] is not None:
                params = best['parameters']
                print(f"   μ (coupling): {params[0]:.6e}")
                print(f"   G_geo (geometry): {params[1]:.6e}")
        
        # Efficiency analysis
        print(f"\n⚡ EFFICIENCY ANALYSIS:")
        for result in energy_results:
            time_taken = result.get('time', 1)
            efficiency = abs(result['energy']) / time_taken if time_taken > 0 else 0
            print(f"   {result['strategy']:30s}: {efficiency:.2e} J/s")
        
        print(f"\n⏱️  Total demonstration time: {total_time:.1f} seconds")
        print(f"📋 Strategies demonstrated: {len(self.results)}")
        
        # Save summary to file
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'n_gaussians': self.n_gaussians,
                'n_fourier': self.n_fourier,
                'quick_mode': self.quick_mode
            },
            'results': self.results,
            'total_time': total_time,
            'best_result': energy_results[0] if energy_results else None
        }
        
        with open('advanced_optimization_demo_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"\n💾 Demonstration summary saved to 'advanced_optimization_demo_summary.json'")

def main():
    """Main demonstration execution"""
    parser = argparse.ArgumentParser(
        description='Demonstrate advanced warp bubble optimization strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Use reduced parameters for faster demonstration')
    parser.add_argument('--strategies', type=str, default='all',
                       help='Comma-separated list of strategies to demonstrate '
                            '(mixed,bayesian,multiobjective,cma,jax,parallel) or "all"')
    parser.add_argument('--gaussians', type=int, default=6,
                       help='Number of Gaussian components (default: 6)')
    parser.add_argument('--fourier', type=int, default=4,
                       help='Number of Fourier modes (default: 4)')
    
    args = parser.parse_args()
    
    # Print module status
    print_module_status()
    
    # Parse strategy list
    if args.strategies.lower() == 'all':
        strategies = ['mixed', 'bayesian', 'multiobjective', 'cma', 'jax', 'parallel']
    else:
        strategies = [s.strip() for s in args.strategies.split(',')]
    
    # Validate strategies
    valid_strategies = ['mixed', 'bayesian', 'multiobjective', 'cma', 'jax', 'parallel']
    invalid_strategies = [s for s in strategies if s not in valid_strategies]
    if invalid_strategies:
        print(f"❌ Invalid strategies: {invalid_strategies}")
        print(f"   Valid options: {valid_strategies}")
        sys.exit(1)
    
    # Initialize and run demonstration
    try:
        demo = AdvancedOptimizationDemo(
            n_gaussians=args.gaussians,
            n_fourier=args.fourier,
            quick_mode=args.quick
        )
        
        results = demo.run_comprehensive_demonstration(strategies)
        
        print(f"\n✅ Advanced optimization demonstration completed!")
        print(f"📈 {len(results)} strategies successfully demonstrated")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
