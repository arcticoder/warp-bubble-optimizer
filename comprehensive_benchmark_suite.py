#!/usr/bin/env python3
"""
COMPREHENSIVE BENCHMARKING SUITE

Compare performance of different ansatz types and optimization strategies:
1. ✅ 4-Gaussian CMA-ES baseline
2. ✅ 6-Gaussian enhanced
3. ✅ 8-Gaussian two-stage (CMA-ES + JAX)
4. ✅ Hybrid spline-Gaussian
5. ✅ Performance metrics and analysis
6. ✅ Cost-benefit analysis

This script runs all optimization methods and generates a comprehensive
comparison report for determining the optimal strategy.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import optimization modules (with graceful fallbacks)
try:
    from gaussian_optimize_cma_M4 import run_cma_optimization as run_M4_optimization
    from gaussian_optimize_M6_enhanced import optimize_enhanced_M6
    from gaussian_optimize_cma_M8 import run_two_stage_optimization_M8
    from hybrid_spline_gaussian_optimizer import run_hybrid_two_stage_optimization
    print("✅ All optimization modules imported successfully")
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("   Some optimization methods may not be available")

# ── 1. BENCHMARKING CONFIGURATION ─────────────────────────────────────────────
class BenchmarkConfig:
    """Configuration for comprehensive benchmarking"""
    
    def __init__(self):
        # Test parameters
        self.run_M4 = True
        self.run_M6 = True 
        self.run_M8 = True
        self.run_hybrid = True
        
        # Evaluation budgets (scaled for fair comparison)
        self.M4_evals = 2000
        self.M6_evals = 2500
        self.M8_cma_evals = 3000
        self.M8_jax_iters = 500
        self.hybrid_cma_evals = 2500
        self.hybrid_jax_iters = 400
        
        # Analysis options
        self.save_results = True
        self.create_plots = True
        self.verbose = True
        
        # Cost analysis ($/kWh)
        self.electricity_cost = 0.001  # $0.001 per kWh
        self.compute_power_kw = 0.5    # Estimated power consumption

config = BenchmarkConfig()

# ── 2. BENCHMARK RESULT CLASS ─────────────────────────────────────────────────
class BenchmarkResult:
    """Container for individual optimization results"""
    
    def __init__(self, method_name, params, energy, time_s, evaluations, 
                 success=True, error_msg=None, metadata=None):
        self.method_name = method_name
        self.params = params
        self.energy = energy
        self.time_s = time_s
        self.evaluations = evaluations
        self.success = success
        self.error_msg = error_msg
        self.metadata = metadata or {}
        
        # Computed metrics
        self.cost_usd = self.compute_cost()
        self.energy_per_eval = energy / max(evaluations, 1) if success else float('inf')
        self.time_per_eval = time_s / max(evaluations, 1) if success else float('inf')
    
    def compute_cost(self):
        """Estimate computational cost in USD"""
        energy_kwh = (self.time_s / 3600) * config.compute_power_kw
        return energy_kwh * config.electricity_cost
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'method_name': self.method_name,
            'energy_J': float(self.energy) if self.success else None,
            'time_seconds': self.time_s,
            'evaluations': self.evaluations,
            'cost_usd': self.cost_usd,
            'success': self.success,
            'error_message': self.error_msg,
            'energy_per_eval': self.energy_per_eval if self.success else None,
            'time_per_eval': self.time_per_eval,
            'metadata': self.metadata
        }

# ── 3. INDIVIDUAL OPTIMIZATION RUNNERS ────────────────────────────────────────
def run_M4_benchmark():
    """Run 4-Gaussian CMA-ES optimization"""
    print("\n" + "="*60)
    print("🔵 BENCHMARKING: 4-Gaussian CMA-ES")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Run M4 optimization (assuming this function exists)
        result = run_M4_optimization(verbose=config.verbose)
        
        if result and 'params' in result:
            elapsed = time.time() - start_time
            
            return BenchmarkResult(
                method_name="4-Gaussian CMA-ES",
                params=result['params'],
                energy=result['energy'],
                time_s=elapsed,
                evaluations=result.get('evaluations', config.M4_evals),
                success=True,
                metadata={'method': 'CMA-ES', 'gaussians': 4}
            )
        else:
            return BenchmarkResult(
                method_name="4-Gaussian CMA-ES",
                params=None,
                energy=float('inf'),
                time_s=time.time() - start_time,
                evaluations=0,
                success=False,
                error_msg="Optimization returned no result"
            )
            
    except Exception as e:
        return BenchmarkResult(
            method_name="4-Gaussian CMA-ES",
            params=None,
            energy=float('inf'),
            time_s=time.time() - start_time,
            evaluations=0,
            success=False,
            error_msg=str(e)
        )

def run_M6_benchmark():
    """Run 6-Gaussian enhanced optimization"""
    print("\n" + "="*60)
    print("🔵 BENCHMARKING: 6-Gaussian Enhanced")
    print("="*60)
    
    start_time = time.time()
    
    try:
        result = optimize_enhanced_M6(verbose=config.verbose)
        
        if result and 'params' in result:
            elapsed = time.time() - start_time
            
            return BenchmarkResult(
                method_name="6-Gaussian Enhanced",
                params=result['params'],
                energy=result['energy'],
                time_s=elapsed,
                evaluations=result.get('evaluations', config.M6_evals),
                success=True,
                metadata={'method': 'Enhanced-DE', 'gaussians': 6}
            )
        else:
            return BenchmarkResult(
                method_name="6-Gaussian Enhanced",
                params=None,
                energy=float('inf'),
                time_s=time.time() - start_time,
                evaluations=0,
                success=False,
                error_msg="Optimization returned no result"
            )
            
    except Exception as e:
        return BenchmarkResult(
            method_name="6-Gaussian Enhanced",
            params=None,
            energy=float('inf'),
            time_s=time.time() - start_time,
            evaluations=0,
            success=False,
            error_msg=str(e)
        )

def run_M8_benchmark():
    """Run 8-Gaussian two-stage optimization"""
    print("\n" + "="*60)
    print("🔵 BENCHMARKING: 8-Gaussian Two-Stage")
    print("="*60)
    
    start_time = time.time()
    
    try:
        result = run_two_stage_optimization_M8(
            cma_evals=config.M8_cma_evals,
            jax_iters=config.M8_jax_iters,
            verbose=config.verbose
        )
        
        if result and 'final_params' in result:
            elapsed = time.time() - start_time
            
            return BenchmarkResult(
                method_name="8-Gaussian Two-Stage",
                params=result['final_params'],
                energy=result['final_energy'],
                time_s=elapsed,
                evaluations=result.get('total_evaluations', config.M8_cma_evals + config.M8_jax_iters),
                success=True,
                metadata={
                    'method': 'CMA-ES + JAX',
                    'gaussians': 8,
                    'cma_evals': result.get('stage1_cma', {}).get('evaluations', 0),
                    'jax_iters': result.get('stage2_jax', {}).get('iterations', 0)
                }
            )
        else:
            return BenchmarkResult(
                method_name="8-Gaussian Two-Stage",
                params=None,
                energy=float('inf'),
                time_s=time.time() - start_time,
                evaluations=0,
                success=False,
                error_msg="Two-stage optimization failed"
            )
            
    except Exception as e:
        return BenchmarkResult(
            method_name="8-Gaussian Two-Stage",
            params=None,
            energy=float('inf'),
            time_s=time.time() - start_time,
            evaluations=0,
            success=False,
            error_msg=str(e)
        )

def run_hybrid_benchmark():
    """Run hybrid spline-Gaussian optimization"""
    print("\n" + "="*60)
    print("🔵 BENCHMARKING: Hybrid Spline-Gaussian")
    print("="*60)
    
    start_time = time.time()
    
    try:
        result = run_hybrid_two_stage_optimization(
            cma_evals=config.hybrid_cma_evals,
            jax_iters=config.hybrid_jax_iters,
            verbose=config.verbose
        )
        
        if result and 'final_params' in result:
            elapsed = time.time() - start_time
            
            return BenchmarkResult(
                method_name="Hybrid Spline-Gaussian",
                params=result['final_params'],
                energy=result['final_energy'],
                time_s=elapsed,
                evaluations=result.get('total_evaluations', config.hybrid_cma_evals + config.hybrid_jax_iters),
                success=True,
                metadata={
                    'method': 'Hybrid CMA-ES + JAX',
                    'r_knot': float(result['final_params'][2]),
                    'spline_coeffs': result['final_params'][3:7].tolist(),
                    'gaussians': 6
                }
            )
        else:
            return BenchmarkResult(
                method_name="Hybrid Spline-Gaussian",
                params=None,
                energy=float('inf'),
                time_s=time.time() - start_time,
                evaluations=0,
                success=False,
                error_msg="Hybrid optimization failed"
            )
            
    except Exception as e:
        return BenchmarkResult(
            method_name="Hybrid Spline-Gaussian",
            params=None,
            energy=float('inf'),
            time_s=time.time() - start_time,
            evaluations=0,
            success=False,
            error_msg=str(e)
        )

# ── 4. COMPREHENSIVE BENCHMARKING SUITE ───────────────────────────────────────
def run_comprehensive_benchmark():
    """Run all optimization methods and collect results"""
    print("🌟 COMPREHENSIVE WARP BUBBLE OPTIMIZATION BENCHMARK")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Electricity cost: ${config.electricity_cost}/kWh")
    print(f"  Estimated power: {config.compute_power_kw} kW")
    print(f"  Save results: {config.save_results}")
    print("=" * 80)
    
    results = []
    total_start_time = time.time()
    
    # Run 4-Gaussian baseline
    if config.run_M4:
        try:
            result_M4 = run_M4_benchmark()
            results.append(result_M4)
        except Exception as e:
            print(f"❌ M4 benchmark failed: {e}")
            results.append(BenchmarkResult(
                "4-Gaussian CMA-ES", None, float('inf'), 0, 0, 
                success=False, error_msg=f"Benchmark failed: {e}"
            ))
    
    # Run 6-Gaussian enhanced
    if config.run_M6:
        try:
            result_M6 = run_M6_benchmark()
            results.append(result_M6)
        except Exception as e:
            print(f"❌ M6 benchmark failed: {e}")
            results.append(BenchmarkResult(
                "6-Gaussian Enhanced", None, float('inf'), 0, 0,
                success=False, error_msg=f"Benchmark failed: {e}"
            ))
    
    # Run 8-Gaussian two-stage
    if config.run_M8:
        try:
            result_M8 = run_M8_benchmark()
            results.append(result_M8)
        except Exception as e:
            print(f"❌ M8 benchmark failed: {e}")
            results.append(BenchmarkResult(
                "8-Gaussian Two-Stage", None, float('inf'), 0, 0,
                success=False, error_msg=f"Benchmark failed: {e}"
            ))
    
    # Run hybrid spline-Gaussian
    if config.run_hybrid:
        try:
            result_hybrid = run_hybrid_benchmark()
            results.append(result_hybrid)
        except Exception as e:
            print(f"❌ Hybrid benchmark failed: {e}")
            results.append(BenchmarkResult(
                "Hybrid Spline-Gaussian", None, float('inf'), 0, 0,
                success=False, error_msg=f"Benchmark failed: {e}"
            ))
    
    total_time = time.time() - total_start_time
    
    print(f"\n🏁 BENCHMARKING COMPLETE")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Methods tested: {len(results)}")
    
    return results, total_time

# ── 5. ANALYSIS AND REPORTING ─────────────────────────────────────────────────
def analyze_benchmark_results(results, total_time):
    """Comprehensive analysis of benchmark results"""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE BENCHMARK ANALYSIS")
    print("="*80)
    
    # Filter successful results
    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    
    if not successful_results:
        print("❌ No successful optimizations to analyze")
        return None
    
    # Sort by energy (best first)
    successful_results.sort(key=lambda x: x.energy)
    
    print(f"\n🏆 PERFORMANCE RANKING")
    print("-" * 40)
    for i, result in enumerate(successful_results):
        print(f"{i+1:2d}. {result.method_name:25s} E = {result.energy:.4e} J")
    
    if failed_results:
        print(f"\n❌ FAILED OPTIMIZATIONS ({len(failed_results)})")
        print("-" * 40)
        for result in failed_results:
            print(f"   {result.method_name:25s} Error: {result.error_msg}")
    
    # Detailed comparison table
    print(f"\n📋 DETAILED COMPARISON")
    print("-" * 100)
    print(f"{'Method':<25} {'Energy (J)':<15} {'Time (s)':<10} {'Evals':<8} {'Cost ($)':<10} {'Efficiency':<12}")
    print("-" * 100)
    
    for result in successful_results:
        efficiency = abs(result.energy) / result.cost_usd if result.cost_usd > 0 else 0
        print(f"{result.method_name:<25} {result.energy:<15.4e} {result.time_s:<10.1f} "
              f"{result.evaluations:<8d} {result.cost_usd:<10.6f} {efficiency:<12.2e}")
    
    # Statistical analysis
    if len(successful_results) > 1:
        best_result = successful_results[0]
        energies = [r.energy for r in successful_results]
        times = [r.time_s for r in successful_results]
        costs = [r.cost_usd for r in successful_results]
        
        print(f"\n📈 STATISTICAL SUMMARY")
        print("-" * 40)
        print(f"Best energy:      {min(energies):.4e} J ({best_result.method_name})")
        print(f"Energy range:     {max(energies) - min(energies):.4e} J")
        print(f"Average time:     {np.mean(times):.1f} ± {np.std(times):.1f} s")
        print(f"Total cost:       ${sum(costs):.4f}")
        print(f"Best efficiency:  {abs(best_result.energy) / best_result.cost_usd:.2e} J/$")
    
    return successful_results

def create_benchmark_plots(results):
    """Create comprehensive visualization of benchmark results"""
    successful_results = [r for r in results if r.success]
    
    if len(successful_results) < 2:
        print("⚠️  Insufficient data for plotting")
        return
    
    # Prepare data
    methods = [r.method_name for r in successful_results]
    energies = [abs(r.energy) for r in successful_results]  # Use absolute values for log scale
    times = [r.time_s for r in successful_results]
    costs = [r.cost_usd for r in successful_results]
    evaluations = [r.evaluations for r in successful_results]
    
    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Energy comparison (log scale)
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    bars1 = ax1.bar(range(len(methods)), energies, color=colors)
    ax1.set_yscale('log')
    ax1.set_ylabel('|Energy| (J)')
    ax1.set_title('Energy Comparison (Lower is Better)')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, energy) in enumerate(zip(bars1, energies)):
        ax1.text(bar.get_x() + bar.get_width()/2, energy * 1.1, 
                f'{energy:.2e}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    # Plot 2: Time vs Evaluations
    scatter = ax2.scatter(evaluations, times, c=energies, cmap='viridis', 
                         s=100, alpha=0.7, norm=plt.matplotlib.colors.LogNorm())
    ax2.set_xlabel('Evaluations')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Time vs Evaluations (colored by energy)')
    ax2.grid(True, alpha=0.3)
    
    # Add method labels
    for i, method in enumerate(methods):
        ax2.annotate(method, (evaluations[i], times[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('|Energy| (J)')
    
    # Plot 3: Cost efficiency (Energy per dollar)
    efficiencies = [abs(e) / c if c > 0 else 0 for e, c in zip(energies, costs)]
    bars3 = ax3.bar(range(len(methods)), efficiencies, color=colors)
    ax3.set_yscale('log')
    ax3.set_ylabel('|Energy| per $ (J/$)')
    ax3.set_title('Cost Efficiency (Higher is Better)')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time per evaluation
    time_per_eval = [t / e if e > 0 else 0 for t, e in zip(times, evaluations)]
    bars4 = ax4.bar(range(len(methods)), time_per_eval, color=colors)
    ax4.set_ylabel('Time per Evaluation (s)')
    ax4.set_title('Computational Efficiency')
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if config.save_results:
        plt.savefig('comprehensive_benchmark_analysis.png', dpi=300, bbox_inches='tight')
        print(f"📈 Benchmark plots saved to: comprehensive_benchmark_analysis.png")
    
    plt.close()  # Close instead of show to prevent blocking

def save_benchmark_results(results, total_time):
    """Save comprehensive benchmark results to JSON"""
    if not config.save_results:
        return
    
    # Prepare data for JSON serialization
    benchmark_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_benchmark_time_s': total_time,
        'configuration': {
            'electricity_cost_per_kwh': config.electricity_cost,
            'compute_power_kw': config.compute_power_kw,
            'evaluation_budgets': {
                'M4_evals': config.M4_evals,
                'M6_evals': config.M6_evals,
                'M8_cma_evals': config.M8_cma_evals,
                'M8_jax_iters': config.M8_jax_iters,
                'hybrid_cma_evals': config.hybrid_cma_evals,
                'hybrid_jax_iters': config.hybrid_jax_iters
            }
        },
        'results': [result.to_dict() for result in results],
        'summary': {
            'total_methods': len(results),
            'successful_methods': len([r for r in results if r.success]),
            'failed_methods': len([r for r in results if not r.success])
        }
    }
    
    # Add best result summary
    successful_results = [r for r in results if r.success]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x.energy)
        benchmark_data['best_result'] = {
            'method': best_result.method_name,
            'energy_J': best_result.energy,
            'time_s': best_result.time_s,
            'cost_usd': best_result.cost_usd,
            'evaluations': best_result.evaluations
        }
    
    # Save to file
    with open('comprehensive_benchmark_results.json', 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"💾 Benchmark results saved to: comprehensive_benchmark_results.json")

# ── 6. MAIN EXECUTION ─────────────────────────────────────────────────────────
def main():
    """Main benchmarking execution"""
    print("🎯 WARP BUBBLE OPTIMIZATION COMPREHENSIVE BENCHMARK")
    print("=" * 80)
    
    # Run comprehensive benchmark
    results, total_time = run_comprehensive_benchmark()
    
    # Analyze results
    successful_results = analyze_benchmark_results(results, total_time)
    
    # Create visualizations
    if config.create_plots and successful_results:
        create_benchmark_plots(results)
    
    # Save results
    if config.save_results:
        save_benchmark_results(results, total_time)
    
    # Final summary
    print(f"\n🏁 BENCHMARK COMPLETE")
    print("=" * 80)
    
    if successful_results:
        best = successful_results[0]
        print(f"🏆 WINNER: {best.method_name}")
        print(f"   Energy: {best.energy:.6e} J")
        print(f"   Time: {best.time_s:.1f}s")
        print(f"   Cost: ${best.cost_usd:.6f}")
        print(f"   Efficiency: {abs(best.energy) / best.cost_usd:.2e} J/$")
        
        # Provide recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   For maximum energy reduction: {best.method_name}")
        
        if len(successful_results) > 1:
            fastest = min(successful_results, key=lambda x: x.time_s)
            cheapest = min(successful_results, key=lambda x: x.cost_usd)
            most_efficient = max(successful_results, key=lambda x: abs(x.energy) / x.cost_usd if x.cost_usd > 0 else 0)
            
            print(f"   For fastest computation: {fastest.method_name} ({fastest.time_s:.1f}s)")
            print(f"   For lowest cost: {cheapest.method_name} (${cheapest.cost_usd:.6f})")
            print(f"   For best efficiency: {most_efficient.method_name}")
    else:
        print("❌ No successful optimizations completed")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
