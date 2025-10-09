#!/usr/bin/env python3
"""
ULTIMATE OPTIMIZATION BENCHMARKING SUITE
=========================================

Advanced benchmarking system for all optimization approaches:
1. Ultimate B-spline optimizer
2. Advanced B-spline optimizer  
3. 8-Gaussian two-stage optimizer
4. Joint optimization methods
5. Hybrid approaches

Provides comprehensive performance analysis and determines
the best approach for achieving minimum E_-.

Authors: Research Team
Date: 2024-12-20
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
import subprocess
import sys
warnings.filterwarnings('ignore')

class UltimateOptimizationBenchmarker:
    """
    Ultimate benchmarking suite for all optimization methods
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
        self.benchmark_start_time = None
        
        if self.verbose:
            print("🧪 Ultimate Optimization Benchmarking Suite Initialized")
            print("=" * 50)
    
    def check_dependencies(self):
        """Check if all required optimizers are available"""
        optimizers = {
            'ultimate_bspline_optimizer.py': 'Ultimate B-Spline Optimizer',
            'advanced_bspline_optimizer.py': 'Advanced B-Spline',
            'gaussian_optimize_cma_M8.py': '8-Gaussian Two-Stage',
            'jax_joint_stability_optimizer.py': 'JAX Joint Optimization',
            'simple_joint_optimizer.py': 'Simple Joint Optimization',
            'hybrid_spline_gaussian_optimizer.py': 'Hybrid Spline-Gaussian',
            'spline_refine_jax.py': 'JAX B-Spline Refiner'
        }
        
        available = {}
        for script, name in optimizers.items():
            if Path(script).exists():
                available[script] = name
                if self.verbose:
                    print(f"✅ {name}: {script}")
            else:
                if self.verbose:
                    print(f"❌ {name}: {script} (not found)")
        
        return available
    
    def run_optimizer_with_monitoring(self, script_path, timeout=2400):  # 40 minute timeout
        """
        Run optimizer with enhanced monitoring and result extraction
        
        Parameters:
        -----------
        script_path : str
            Path to optimizer script
        timeout : int
            Timeout in seconds
            
        Returns:
        --------
        dict : Execution results with enhanced metrics
        """
        if self.verbose:
            print(f"\n🚀 Running {script_path}...")
            print(f"   Timeout: {timeout/60:.1f} minutes")
        
        start_time = time.time()
        
        try:
            # Run the optimizer with real-time output capture
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path.cwd()
            )
            
            # Monitor progress
            stdout_lines = []
            stderr_lines = []
            
            while True:
                # Check if process is done
                if process.poll() is not None:
                    break
                
                # Check timeout
                if time.time() - start_time > timeout:
                    process.kill()
                    raise subprocess.TimeoutExpired(script_path, timeout)
                
                # Read available output
                try:
                    line = process.stdout.readline()
                    if line:
                        stdout_lines.append(line.strip())
                        if self.verbose and ('✅' in line or '🏆' in line or 'Best energy' in line):
                            print(f"   {line.strip()}")
                except:
                    pass
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            
            # Get remaining output
            remaining_out, remaining_err = process.communicate()
            if remaining_out:
                stdout_lines.extend(remaining_out.strip().split('\n'))
            if remaining_err:
                stderr_lines.extend(remaining_err.strip().split('\n'))
            
            duration = time.time() - start_time
            
            execution_result = {
                'script': script_path,
                'success': process.returncode == 0,
                'duration': duration,
                'return_code': process.returncode,
                'stdout_lines': stdout_lines,
                'stderr_lines': stderr_lines
            }
            
            # Enhanced result extraction
            energy = self.extract_energy_from_lines(stdout_lines)
            parameters = self.extract_parameters_from_lines(stdout_lines)
            performance_metrics = self.extract_performance_metrics(stdout_lines)
            
            if energy is not None:
                execution_result['energy_J'] = energy
            if parameters:
                execution_result['final_parameters'] = parameters
            if performance_metrics:
                execution_result['performance_metrics'] = performance_metrics
            
            # Find and load result files
            result_files = self.find_recent_result_files(script_path, start_time)
            execution_result['result_files'] = result_files
            
            if result_files:
                detailed_results = self.load_detailed_results(result_files)
                execution_result['detailed_results'] = detailed_results
                
                # Extract best energy from detailed results
                best_detailed_energy = self.extract_best_energy_from_detailed(detailed_results)
                if best_detailed_energy is not None and (energy is None or abs(best_detailed_energy) > abs(energy)):
                    execution_result['energy_J'] = best_detailed_energy
            
            if self.verbose:
                status = "✅" if execution_result['success'] else "❌"
                print(f"{status} {script_path} completed in {duration:.1f}s")
                if execution_result.get('energy_J') is not None:
                    print(f"   Best Energy: {execution_result['energy_J']:.3e} J")
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            if self.verbose:
                print(f"⏰ {script_path} timed out after {timeout}s")
            
            return {
                'script': script_path,
                'success': False,
                'duration': duration,
                'error': 'timeout',
                'timeout': timeout
            }
            
        except Exception as e:
            duration = time.time() - start_time
            if self.verbose:
                print(f"❌ {script_path} failed: {e}")
            
            return {
                'script': script_path,
                'success': False,
                'duration': duration,
                'error': str(e)
            }
    
    def extract_energy_from_lines(self, lines):
        """Extract energy value from output lines"""
        try:
            energy_patterns = [
                ('Best energy achieved:', r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'),
                ('Final energy:', r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'),
                ('E_- =', r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'),
                ('Energy:', r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'),
                ('🎯 Best negative energy achieved:', r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?')
            ]
            
            import re
            
            for line in reversed(lines):  # Start from end to get most recent
                for pattern_desc, number_regex in energy_patterns:
                    if pattern_desc in line:
                        numbers = re.findall(number_regex, line)
                        if numbers:
                            try:
                                energy = float(numbers[-1])
                                if abs(energy) > 1e10:  # Reasonable energy scale
                                    return energy
                            except:
                                continue
            
            return None
            
        except Exception:
            return None
    
    def extract_parameters_from_lines(self, lines):
        """Extract final parameters from output"""
        params = {}
        
        try:
            import re
            
            for line in lines:
                # Look for μ parameter
                if 'μ =' in line or 'mu =' in line:
                    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                    if numbers:
                        params['mu'] = float(numbers[-1])
                
                # Look for G_geo parameter
                if 'G_geo =' in line or 'G =' in line:
                    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                    if numbers:
                        params['G_geo'] = float(numbers[-1])
                
                # Look for control points
                if 'Control points:' in line:
                    numbers = re.findall(r'[0-9]+', line)
                    if numbers:
                        params['n_control_points'] = int(numbers[-1])
        
        except Exception:
            pass
        
        return params if params else None
    
    def extract_performance_metrics(self, lines):
        """Extract performance metrics from output"""
        metrics = {}
        
        try:
            import re
            
            for line in lines:
                # Function evaluations
                if 'Function Evals:' in line or 'evaluations:' in line:
                    numbers = re.findall(r'[0-9]+', line)
                    if numbers:
                        metrics['function_evaluations'] = int(numbers[-1])
                
                # Eval per second
                if 'eval/sec' in line or 'Eval/sec:' in line:
                    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', line)
                    if numbers:
                        metrics['evaluations_per_second'] = float(numbers[-1])
                
                # Success rate
                if 'Success Rate:' in line:
                    numbers = re.findall(r'[0-9]+', line)
                    if len(numbers) >= 2:
                        metrics['successful_attempts'] = int(numbers[0])
                        metrics['total_attempts'] = int(numbers[1])
        
        except Exception:
            pass
        
        return metrics if metrics else None
    
    def find_recent_result_files(self, script_path, start_time):
        """Find result files created after script started"""
        script_name = Path(script_path).stem
        
        # Common result file patterns
        patterns = [
            f'{script_name}_results*.json',
            f'*{script_name}*.json',
            f'ultimate_bspline_results*.json',
            f'advanced_bspline_results*.json',
            '*results*.json'
        ]
        
        result_files = []
        for pattern in patterns:
            files = list(Path.cwd().glob(pattern))
            # Filter to files modified after start_time
            recent_files = [f for f in files if f.stat().st_mtime > start_time]
            result_files.extend([str(f) for f in recent_files])
        
        # Remove duplicates and sort by modification time
        result_files = list(set(result_files))
        result_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        
        return result_files[:3]  # Return up to 3 most recent
    
    def load_detailed_results(self, result_files):
        """Load detailed results from JSON files"""
        detailed_results = []
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    detailed_results.append({
                        'file': file_path,
                        'data': data,
                        'timestamp': Path(file_path).stat().st_mtime
                    })
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Could not load {file_path}: {e}")
        
        return detailed_results
    
    def extract_best_energy_from_detailed(self, detailed_results):
        """Extract best energy from detailed JSON results"""
        best_energy = None
        
        for result in detailed_results:
            try:
                data = result['data']
                
                # Common energy field locations
                energy_paths = [
                    ['results', 'best_energy_J'],
                    ['performance', 'new_results', 'after_lbfgs'],
                    ['performance', 'new_results', 'cma_es_stage'],
                    ['best_energy_J'],
                    ['energy_J'],
                    ['final_energy']
                ]
                
                for path in energy_paths:
                    try:
                        value = data
                        for key in path:
                            value = value[key]
                        
                        if isinstance(value, (int, float)) and abs(value) > 1e10:
                            if best_energy is None or abs(value) > abs(best_energy):
                                best_energy = value
                    except (KeyError, TypeError):
                        continue
                        
            except Exception:
                continue
        
        return best_energy
    
    def run_ultimate_benchmark(self, priority_optimizers=None, timeout=2400):
        """
        Run ultimate benchmark focusing on the most advanced optimizers
        
        Parameters:
        -----------
        priority_optimizers : list or None
            List of priority optimizer scripts (None = use default priority)
        timeout : int
            Timeout per optimizer in seconds (40 minutes default)
            
        Returns:
        --------
        dict : Complete benchmark results
        """
        if self.verbose:
            print("\n🏁 STARTING ULTIMATE OPTIMIZATION BENCHMARK")
            print("=" * 60)
        
        self.benchmark_start_time = time.time()
        
        # Check available optimizers
        available_optimizers = self.check_dependencies()
        
        if priority_optimizers is None:
            # Default priority order (most advanced first)
            priority_optimizers = [
                'ultimate_bspline_optimizer.py',        # Ultimate B-spline
                'advanced_bspline_optimizer.py',        # Advanced B-spline
                'gaussian_optimize_cma_M8.py',          # 8-Gaussian two-stage
                'hybrid_spline_gaussian_optimizer.py',  # Hybrid approach
                'jax_joint_stability_optimizer.py',     # JAX joint with stability
                'spline_refine_jax.py',                 # JAX B-spline refiner
                'simple_joint_optimizer.py'             # Simple joint optimization
            ]
        
        # Filter to only available optimizers
        optimizers_to_test = [opt for opt in priority_optimizers if opt in available_optimizers]
        
        if not optimizers_to_test:
            print("❌ No priority optimizers available to test!")
            return {'error': 'No optimizers available'}
        
        if self.verbose:
            print(f"\n📋 Testing {len(optimizers_to_test)} priority optimizers:")
            for i, opt in enumerate(optimizers_to_test, 1):
                print(f"   {i}. {available_optimizers.get(opt, opt)}")
        
        # Run each optimizer with enhanced monitoring
        benchmark_results = {}
        
        for i, optimizer_script in enumerate(optimizers_to_test, 1):
            if self.verbose:
                print(f"\n[{i}/{len(optimizers_to_test)}] " + "="*50)
                print(f"🔧 Starting: {available_optimizers.get(optimizer_script, optimizer_script)}")
            
            result = self.run_optimizer_with_monitoring(optimizer_script, timeout)
            benchmark_results[optimizer_script] = result
            
            # Print intermediate summary
            if result.get('success') and result.get('energy_J'):
                if self.verbose:
                    print(f"✅ Completed successfully!")
                    print(f"   Energy: {result['energy_J']:.3e} J")
                    print(f"   Runtime: {result['duration']:.1f}s")
            else:
                if self.verbose:
                    print(f"❌ Failed or incomplete")
        
        # Compile comprehensive summary
        total_duration = time.time() - self.benchmark_start_time
        
        summary = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'benchmark_type': 'Ultimate Optimization Benchmark',
                'total_duration_seconds': total_duration,
                'optimizers_tested': len(optimizers_to_test),
                'timeout_per_optimizer': timeout
            },
            'results': benchmark_results,
            'performance_ranking': self.rank_ultimate_performance(benchmark_results),
            'summary_statistics': self.compute_enhanced_statistics(benchmark_results),
            'comparative_analysis': self.compare_with_historical_records(benchmark_results)
        }
        
        self.results = summary
        
        if self.verbose:
            print("\n" + "="*60)
            print("🏆 ULTIMATE BENCHMARK COMPLETE")
            print("="*60)
            self.print_ultimate_summary(summary)
        
        return summary
    
    def rank_ultimate_performance(self, benchmark_results):
        """Enhanced performance ranking with multiple metrics"""
        rankings = []
        
        for script, result in benchmark_results.items():
            if result.get('success') and result.get('energy_J') is not None:
                entry = {
                    'optimizer': script,
                    'optimizer_name': Path(script).stem,
                    'energy_J': result['energy_J'],
                    'abs_energy_J': abs(result['energy_J']),
                    'duration_s': result['duration'],
                    'energy_per_second': abs(result['energy_J']) / result['duration'],
                    'success': True
                }
                
                # Add performance metrics if available
                if result.get('performance_metrics'):
                    entry.update(result['performance_metrics'])
                
                # Add parameter info
                if result.get('final_parameters'):
                    entry['final_parameters'] = result['final_parameters']
                
                rankings.append(entry)
            else:
                # Include failed attempts for completeness
                rankings.append({
                    'optimizer': script,
                    'optimizer_name': Path(script).stem,
                    'energy_J': None,
                    'duration_s': result.get('duration', 0),
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                })
        
        # Sort successful runs by best energy (most negative)
        successful_rankings = [r for r in rankings if r['success']]
        failed_rankings = [r for r in rankings if not r['success']]
        
        successful_rankings.sort(key=lambda x: x['energy_J'])
        
        # Add rankings
        for i, entry in enumerate(successful_rankings):
            entry['rank'] = i + 1
        
        return successful_rankings + failed_rankings
    
    def compute_enhanced_statistics(self, benchmark_results):
        """Compute enhanced summary statistics"""
        successful_runs = [r for r in benchmark_results.values() if r.get('success')]
        energies = [r['energy_J'] for r in successful_runs if r.get('energy_J') is not None]
        durations = [r['duration'] for r in successful_runs]
        
        if not energies:
            return {'error': 'No successful runs with energy data'}
        
        stats = {
            'total_runs': len(benchmark_results),
            'successful_runs': len(successful_runs),
            'failed_runs': len(benchmark_results) - len(successful_runs),
            'success_rate': len(successful_runs) / len(benchmark_results),
            'best_energy_J': min(energies),
            'worst_energy_J': max(energies),
            'mean_energy_J': np.mean(energies),
            'median_energy_J': np.median(energies),
            'std_energy_J': np.std(energies),
            'energy_improvement_range': abs(max(energies)) / abs(min(energies)) if min(energies) != 0 else float('inf'),
            'average_duration_s': np.mean(durations),
            'median_duration_s': np.median(durations),
            'total_compute_time_s': sum(durations),
            'efficiency_best': abs(min(energies)) / min([r['duration'] for r in successful_runs if r.get('energy_J') == min(energies)]),
            'efficiency_mean': np.mean([abs(r['energy_J']) / r['duration'] for r in successful_runs if r.get('energy_J')])
        }
        
        return stats
    
    def compare_with_historical_records(self, benchmark_results):
        """Compare current results with historical records"""
        historical_records = [
            {
                'name': '4-Gaussian CMA-ES',
                'energy_J': -6.3e50,
                'source': 'cma_4gaussian_results.json',
                'date': '2024-12-19'
            },
            {
                'name': '8-Gaussian Two-Stage',
                'energy_J': -1.48e53,
                'source': 'M8_RECORD_BREAKING_RESULTS.json',
                'date': '2024-12-19'
            }
        ]
        
        current_best = None
        successful_results = [r for r in benchmark_results.values() if r.get('success') and r.get('energy_J')]
        
        if successful_results:
            current_best = min(successful_results, key=lambda x: x['energy_J'])
        
        comparisons = []
        
        if current_best:
            for record in historical_records:
                improvement_factor = abs(current_best['energy_J']) / abs(record['energy_J'])
                comparisons.append({
                    'historical_method': record['name'],
                    'historical_energy_J': record['energy_J'],
                    'current_energy_J': current_best['energy_J'],
                    'improvement_factor': improvement_factor,
                    'is_improvement': improvement_factor > 1.0,
                    'improvement_percentage': (improvement_factor - 1.0) * 100
                })
        
        return {
            'current_best': current_best,
            'historical_records': historical_records,
            'comparisons': comparisons
        }
    
    def print_ultimate_summary(self, summary):
        """Print comprehensive ultimate summary"""
        stats = summary['summary_statistics']
        rankings = summary['performance_ranking']
        comparison = summary['comparative_analysis']
        
        print(f"📊 BENCHMARK STATISTICS:")
        print(f"   Total runs: {stats['total_runs']}")
        print(f"   Successful: {stats['successful_runs']} ({stats['success_rate']:.1%})")
        print(f"   Failed: {stats['failed_runs']}")
        print(f"   Total compute time: {stats['total_compute_time_s']:.1f} seconds")
        
        successful_rankings = [r for r in rankings if r['success']]
        
        if successful_rankings:
            print(f"\n🏆 ULTIMATE PERFORMANCE RANKING:")
            print("-" * 50)
            for i, entry in enumerate(successful_rankings[:5], 1):  # Top 5
                print(f"{i}. {entry['optimizer_name']}")
                print(f"   Energy: {entry['energy_J']:.3e} J")
                print(f"   Duration: {entry['duration_s']:.1f}s")
                print(f"   Efficiency: {entry['energy_per_second']:.2e} J/s")
                if entry.get('final_parameters'):
                    params = entry['final_parameters']
                    if 'mu' in params:
                        print(f"   μ = {params['mu']:.4f}")
                    if 'G_geo' in params:
                        print(f"   G_geo = {params['G_geo']:.3e}")
                print()
            
            print(f"🎯 ULTIMATE ACHIEVEMENT:")
            print(f"   Best energy: {stats['best_energy_J']:.3e} J")
            print(f"   Energy range: {stats['energy_improvement_range']:.1f}× variation")
            print(f"   Best efficiency: {stats['efficiency_best']:.2e} J/s")
        
        # Historical comparison
        if comparison['comparisons']:
            print(f"\n📈 HISTORICAL COMPARISON:")
            print("-" * 40)
            for comp in comparison['comparisons']:
                improvement_status = "🚀 IMPROVEMENT" if comp['is_improvement'] else "📉 Degradation"
                print(f"vs {comp['historical_method']}: {comp['improvement_factor']:.1f}× {improvement_status}")
                print(f"   Historical: {comp['historical_energy_J']:.2e} J")
                print(f"   Current:    {comp['current_energy_J']:.2e} J")
                print(f"   Change:     {comp['improvement_percentage']:+.1f}%")
                print()

def main():
    """Main ultimate benchmarking execution"""
    print("🚀 ULTIMATE OPTIMIZATION BENCHMARKING SUITE")
    print("=" * 70)
    
    # Initialize ultimate benchmarker
    benchmarker = UltimateOptimizationBenchmarker(verbose=True)
    
    # Run ultimate benchmark with extended timeout
    results = benchmarker.run_ultimate_benchmark(
        timeout=3000  # 50 minutes per optimizer for thorough testing
    )
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'ultimate_benchmark_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Ultimate benchmark results saved to: {results_file}")
    
    # Create summary report
    print("\n" + "="*70)
    print("🏁 ULTIMATE BENCHMARK FINAL REPORT")
    print("="*70)
    
    if results.get('performance_ranking'):
        successful_rankings = [r for r in results['performance_ranking'] if r['success']]
        
        if successful_rankings:
            winner = successful_rankings[0]
            print(f"🏆 ULTIMATE WINNER: {winner['optimizer_name']}")
            print(f"   Record Energy: {winner['energy_J']:.3e} J")
            print(f"   Runtime: {winner['duration_s']:.1f} seconds")
            print(f"   Efficiency: {winner['energy_per_second']:.2e} J/s")
            
            # Calculate total improvement over historical
            comparison = results.get('comparative_analysis', {})
            if comparison.get('comparisons'):
                max_improvement = max([c['improvement_factor'] for c in comparison['comparisons']])
                print(f"   Maximum historical improvement: {max_improvement:.1f}×")
        
        total_time = results['benchmark_info']['total_duration_seconds']
        print(f"\n📊 BENCHMARK SUMMARY:")
        print(f"   Optimizers tested: {results['benchmark_info']['optimizers_tested']}")
        print(f"   Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"   Success rate: {results['summary_statistics']['success_rate']:.1%}")
    
    print("\n🎉 Ultimate optimization benchmarking complete!")

if __name__ == "__main__":
    main()
