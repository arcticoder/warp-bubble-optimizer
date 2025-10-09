#!/usr/bin/env python3
"""
COMPREHENSIVE PARAMETER SCAN MODULE

Joint scan over μ₀ and G_geo (R_ratio) parameters to find the global minimum.
All previous optimizations assumed μ₀=1e-6, G_geo=1e-5, but the true global 
minimum might occur at different parameter values.

This module implements:
1. 2D grid search over (μ₀, G_geo) space
2. Multiple ansatz types: Gaussian (4-5 lumps), Hybrid, Enhanced
3. Heatmap visualization of energy landscape
4. Adaptive refinement around promising regions
5. Parallel processing for efficiency
6. Comprehensive result logging and analysis

Target: Find parameter combinations that push E₋ below -1.2×10³¹ J
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import optimization modules
try:
    from enhanced_gaussian_optimizer import optimize_ansatz as optimize_gaussian
    from hybrid_polynomial_gaussian_optimizer import run_hybrid_optimization
    OPTIMIZERS_AVAILABLE = True
except ImportError:
    print("⚠️  Optimization modules not found. Running in analysis mode only.")
    OPTIMIZERS_AVAILABLE = False

# ── 1. SCANNING CONFIGURATION ─────────────────────────────────────────────────
class ScanConfig:
    """Configuration for parameter scanning"""
    def __init__(self):
        # Parameter ranges (log space)
        self.mu_range = (1e-8, 1e-4)       # Polymer length range
        self.G_geo_range = (1e-6, 1e-3)    # Van den Broeck ratio range
        
        # Grid resolution
        self.mu_points = 6                  # Points along μ axis
        self.G_geo_points = 5               # Points along G_geo axis
        
        # Ansatz types to test
        self.ansatz_types = ['gaussian', 'hybrid']
        self.M_gauss_values = [4, 5]        # Gaussian lump counts to test
        
        # Optimization settings
        self.max_iterations = 200           # Reduced for scanning efficiency
        self.tolerance = 1e-7
        self.enable_physics_constraints = True
        
        # Parallel processing
        self.use_parallel = True
        self.max_workers = 4                # Adjust based on system
        
        # Result filtering
        self.energy_threshold = -1e30       # Only keep results better than this
        self.success_only = True            # Only include successful optimizations

config = ScanConfig()

# ── 2. PARAMETER GRID GENERATION ──────────────────────────────────────────────
def generate_parameter_grid():
    """
    Generate logarithmic parameter grids for μ₀ and G_geo.
    
    Returns:
        tuple: (mu_values, G_geo_values, parameter_combinations)
    """
    # Logarithmic grids
    mu_values = np.logspace(
        np.log10(config.mu_range[0]), 
        np.log10(config.mu_range[1]), 
        config.mu_points
    )
    
    G_geo_values = np.logspace(
        np.log10(config.G_geo_range[0]), 
        np.log10(config.G_geo_range[1]), 
        config.G_geo_points
    )
    
    # Generate all combinations
    combinations = []
    for ansatz_type in config.ansatz_types:
        for M_gauss in config.M_gauss_values:
            # Skip M_gauss for hybrid (uses fixed structure)
            if ansatz_type == 'hybrid' and M_gauss != config.M_gauss_values[0]:
                continue
                
            for mu0 in mu_values:
                for G_geo in G_geo_values:
                    combinations.append({
                        'ansatz_type': ansatz_type,
                        'M_gauss': M_gauss,
                        'mu0': mu0,
                        'G_geo': G_geo
                    })
    
    return mu_values, G_geo_values, combinations

def generate_adaptive_grid(best_results, refinement_factor=0.3):
    """
    Generate refined grid around the best results for adaptive scanning.
    
    Args:
        best_results: List of best optimization results
        refinement_factor: Fraction of original range for refinement
    
    Returns:
        list: Refined parameter combinations
    """
    if not best_results:
        return []
    
    # Find best parameter region
    best_mu = [r['mu0'] for r in best_results[:3]]  # Top 3 results
    best_G_geo = [r['G_geo'] for r in best_results[:3]]
    
    mu_center = np.mean(best_mu)
    G_geo_center = np.mean(best_G_geo)
    
    # Define refined ranges
    mu_width = (config.mu_range[1] - config.mu_range[0]) * refinement_factor
    G_geo_width = (config.G_geo_range[1] - config.G_geo_range[0]) * refinement_factor
    
    mu_refined = np.linspace(
        max(config.mu_range[0], mu_center - mu_width/2),
        min(config.mu_range[1], mu_center + mu_width/2),
        config.mu_points
    )
    
    G_geo_refined = np.linspace(
        max(config.G_geo_range[0], G_geo_center - G_geo_width/2),
        min(config.G_geo_range[1], G_geo_center + G_geo_width/2),
        config.G_geo_points
    )
    
    # Generate refined combinations (focus on best ansatz types)
    best_ansatz = best_results[0]['ansatz_type']
    best_M_gauss = best_results[0].get('M_gauss', 4)
    
    refined_combinations = []
    for mu0 in mu_refined:
        for G_geo in G_geo_refined:
            refined_combinations.append({
                'ansatz_type': best_ansatz,
                'M_gauss': best_M_gauss,
                'mu0': mu0,
                'G_geo': G_geo
            })
    
    return refined_combinations

# ── 3. SINGLE OPTIMIZATION WRAPPER ────────────────────────────────────────────
def run_single_optimization(params_dict):
    """
    Run a single optimization with given parameters.
    
    Args:
        params_dict: Dictionary with 'ansatz_type', 'M_gauss', 'mu0', 'G_geo'
    
    Returns:
        dict: Optimization result with metadata
    """
    if not OPTIMIZERS_AVAILABLE:
        # Mock result for testing
        return {
            'success': False,
            'message': 'Optimizers not available'
        }
    
    ansatz_type = params_dict['ansatz_type']
    M_gauss = params_dict['M_gauss']
    mu0 = params_dict['mu0']
    G_geo = params_dict['G_geo']
    
    start_time = time.time()
    
    try:
        if ansatz_type == 'gaussian':
            result = optimize_gaussian(
                ansatz_type='gaussian',
                mu0=mu0,
                G_geo=G_geo,
                M_gauss=M_gauss,
                verbose=False
            )
        elif ansatz_type == 'hybrid':
            result = run_hybrid_optimization(
                mu0=mu0,
                G_geo=G_geo,
                use_cma=True,
                enable_curvature=config.enable_physics_constraints,
                enable_monotonicity=config.enable_physics_constraints,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")
        
        # Add metadata
        if result.get('success', False):
            result.update({
                'scan_time': time.time() - start_time,
                'scan_params': params_dict.copy()
            })
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'message': f"Optimization failed: {str(e)}",
            'scan_time': time.time() - start_time,
            'scan_params': params_dict.copy()
        }

# ── 4. PARALLEL SCANNING ENGINE ───────────────────────────────────────────────
def run_parallel_scan(combinations, max_workers=None):
    """
    Run parameter scan in parallel using ProcessPoolExecutor.
    
    Args:
        combinations: List of parameter combinations to test
        max_workers: Maximum number of parallel workers
    
    Returns:
        list: All optimization results
    """
    if max_workers is None:
        max_workers = config.max_workers
    
    results = []
    successful_results = []
    failed_count = 0
    
    print(f"🚀 Starting parallel scan with {max_workers} workers")
    print(f"📊 Total combinations: {len(combinations)}")
    
    if config.use_parallel and max_workers > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_params = {
                executor.submit(run_single_optimization, params): params 
                for params in combinations
            }
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_params)):
                params = future_to_params[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.get('success', False):
                        successful_results.append(result)
                        energy = result.get('energy_J', 0)
                        print(f"✅ [{i+1:3d}/{len(combinations)}] "
                              f"{params['ansatz_type']:<8} M={params.get('M_gauss', 'N/A'):<2} "
                              f"μ={params['mu0']:.1e} G={params['G_geo']:.1e} "
                              f"E={energy:.2e} J")
                    else:
                        failed_count += 1
                        if failed_count <= 5:  # Only show first few failures
                            print(f"❌ [{i+1:3d}/{len(combinations)}] "
                                  f"{params['ansatz_type']:<8} - {result.get('message', 'Failed')}")
                
                except Exception as e:
                    failed_count += 1
                    print(f"💥 [{i+1:3d}/{len(combinations)}] Exception: {str(e)}")
                    results.append({
                        'success': False,
                        'message': f"Exception: {str(e)}",
                        'scan_params': params
                    })
    
    else:
        # Sequential execution
        print("🔄 Running sequential scan...")
        for i, params in enumerate(combinations):
            result = run_single_optimization(params)
            results.append(result)
            
            if result.get('success', False):
                successful_results.append(result)
                energy = result.get('energy_J', 0)
                print(f"✅ [{i+1:3d}/{len(combinations)}] "
                      f"{params['ansatz_type']:<8} E={energy:.2e} J")
            else:
                failed_count += 1
                print(f"❌ [{i+1:3d}/{len(combinations)}] Failed")
    
    print(f"\n📊 Scan completed: {len(successful_results)} successful, {failed_count} failed")
    return results

# ── 5. RESULT ANALYSIS AND RANKING ────────────────────────────────────────────
def analyze_scan_results(results):
    """
    Analyze and rank parameter scan results.
    
    Args:
        results: List of optimization results
    
    Returns:
        dict: Analysis summary with rankings and statistics
    """
    # Filter successful results
    successful = [r for r in results if r.get('success', False)]
    
    if not successful:
        return {'success': False, 'message': 'No successful optimizations'}
    
    # Filter by energy threshold
    good_results = [r for r in successful 
                   if r.get('energy_J', 0) < config.energy_threshold]
    
    # Sort by energy (most negative first)
    good_results.sort(key=lambda x: x.get('energy_J', 0))
    
    # Calculate statistics
    energies = [r['energy_J'] for r in good_results]
    mu_values = [r['mu0'] for r in good_results]
    G_geo_values = [r['G_geo'] for r in good_results]
    
    analysis = {
        'success': True,
        'total_combinations': len(results),
        'successful_optimizations': len(successful),
        'good_results': len(good_results),
        'best_results': good_results[:10],  # Top 10
        'statistics': {
            'energy_range': (min(energies), max(energies)) if energies else (0, 0),
            'energy_mean': np.mean(energies) if energies else 0,
            'energy_std': np.std(energies) if energies else 0,
            'mu_range': (min(mu_values), max(mu_values)) if mu_values else (0, 0),
            'G_geo_range': (min(G_geo_values), max(G_geo_values)) if G_geo_values else (0, 0)
        }
    }
    
    # Ansatz type performance
    ansatz_performance = {}
    for ansatz_type in config.ansatz_types:
        ansatz_results = [r for r in good_results 
                         if r.get('ansatz_type') == ansatz_type]
        if ansatz_results:
            ansatz_energies = [r['energy_J'] for r in ansatz_results]
            ansatz_performance[ansatz_type] = {
                'count': len(ansatz_results),
                'best_energy': min(ansatz_energies),
                'mean_energy': np.mean(ansatz_energies),
                'std_energy': np.std(ansatz_energies)
            }
    
    analysis['ansatz_performance'] = ansatz_performance
    
    return analysis

def print_scan_summary(analysis):
    """Print comprehensive scan summary"""
    if not analysis.get('success', False):
        print("❌ Scan analysis failed")
        return
    
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE PARAMETER SCAN RESULTS")
    print("="*60)
    
    # Overall statistics
    print(f"\n🔢 OVERALL STATISTICS:")
    print(f"   Total combinations tested: {analysis['total_combinations']}")
    print(f"   Successful optimizations: {analysis['successful_optimizations']}")
    print(f"   Results above threshold: {analysis['good_results']}")
    
    stats = analysis['statistics']
    if analysis['good_results'] > 0:
        print(f"   Energy range: {stats['energy_range'][0]:.2e} to {stats['energy_range'][1]:.2e} J")
        print(f"   Mean energy: {stats['energy_mean']:.2e} ± {stats['energy_std']:.2e} J")
        print(f"   μ₀ range: {stats['mu_range'][0]:.1e} to {stats['mu_range'][1]:.1e}")
        print(f"   G_geo range: {stats['G_geo_range'][0]:.1e} to {stats['G_geo_range'][1]:.1e}")
    
    # Ansatz performance comparison
    print(f"\n🏆 ANSATZ PERFORMANCE:")
    for ansatz_type, perf in analysis['ansatz_performance'].items():
        print(f"   {ansatz_type.title()}:")
        print(f"     Count: {perf['count']}")
        print(f"     Best: {perf['best_energy']:.2e} J")
        print(f"     Mean: {perf['mean_energy']:.2e} ± {perf['std_energy']:.2e} J")
    
    # Top results
    print(f"\n🥇 TOP 5 RESULTS:")
    print("-" * 50)
    for i, result in enumerate(analysis['best_results'][:5], 1):
        ansatz = result.get('ansatz_type', 'unknown')
        M_gauss = result.get('M_gauss', 'N/A')
        energy = result.get('energy_J', 0)
        mu0 = result.get('mu0', 0)
        G_geo = result.get('G_geo', 0)
        opt_time = result.get('optimization_time', 0)
        
        print(f"   {i}. {ansatz.title()} (M={M_gauss})")
        print(f"      E₋ = {energy:.3e} J")
        print(f"      μ₀ = {mu0:.1e}, G_geo = {G_geo:.1e}")
        print(f"      Time: {opt_time:.1f}s")
        print()
    
    # Baseline comparison
    if analysis['best_results']:
        best = analysis['best_results'][0]
        soliton_baseline = -1.584e31
        improvement = abs(best['energy_J'] / soliton_baseline)
        
        print(f"🎯 BASELINE COMPARISON:")
        print(f"   Soliton baseline: {soliton_baseline:.2e} J")
        print(f"   Best scan result: {best['energy_J']:.2e} J")
        print(f"   Improvement: {improvement:.3f}×")
        
        if improvement > 1.0:
            print(f"   ✅ BREAKTHROUGH: {(improvement-1)*100:.1f}% better!")
        else:
            print(f"   📊 Performance: {improvement*100:.1f}% of baseline")

# ── 6. VISUALIZATION ───────────────────────────────────────────────────────────
def create_energy_heatmap(results, save_plot=True):
    """
    Create heatmap visualization of energy landscape in (μ₀, G_geo) space.
    """
    # Filter successful results
    successful = [r for r in results if r.get('success', False)]
    
    if len(successful) < 4:
        print("⚠️  Insufficient data for heatmap (need at least 4 points)")
        return None
    
    # Extract data
    mu_vals = np.array([r['mu0'] for r in successful])
    G_geo_vals = np.array([r['G_geo'] for r in successful])
    energies = np.array([r['energy_J'] for r in successful])
    ansatz_types = [r.get('ansatz_type', 'unknown') for r in successful]
    
    # Create separate plots for each ansatz type
    unique_ansatz = list(set(ansatz_types))
    n_plots = len(unique_ansatz)
    
    if n_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 8))
        if n_plots == 1:
            axes = [axes]
    
    for i, ansatz_type in enumerate(unique_ansatz):
        # Filter data for this ansatz type
        mask = np.array([a == ansatz_type for a in ansatz_types])
        mu_filtered = mu_vals[mask]
        G_geo_filtered = G_geo_vals[mask]
        energy_filtered = energies[mask]
        
        if len(energy_filtered) < 3:
            axes[i].text(0.5, 0.5, f'Insufficient {ansatz_type} data', 
                        transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f'{ansatz_type.title()} Ansatz')
            continue
        
        # Create scatter plot with color-coded energies
        scatter = axes[i].scatter(
            mu_filtered, G_geo_filtered, c=energy_filtered,
            cmap='viridis', s=100, alpha=0.8, edgecolors='black'
        )
        
        # Formatting
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlabel('μ₀ (polymer length)')
        axes[i].set_ylabel('G_geo (VdB ratio)')
        axes[i].set_title(f'{ansatz_type.title()} Ansatz Energy Landscape')
        axes[i].grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=axes[i])
        cbar.set_label('E₋ (J)', rotation=270, labelpad=15)
        
        # Mark best point
        if len(energy_filtered) > 0:
            best_idx = np.argmin(energy_filtered)
            axes[i].scatter(
                mu_filtered[best_idx], G_geo_filtered[best_idx],
                marker='*', s=200, color='red', edgecolors='white', linewidth=2,
                label=f'Best: {energy_filtered[best_idx]:.2e} J'
            )
            axes[i].legend()
    
    plt.tight_layout()
    
    if save_plot:
        filename = 'parameter_scan_heatmap.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Energy heatmap saved as '{filename}'")
    
    return fig

def plot_parameter_distributions(analysis, save_plot=True):
    """
    Plot distributions of optimal parameters.
    """
    if not analysis.get('success', False) or not analysis['best_results']:
        print("⚠️  No data for parameter distribution plots")
        return None
    
    results = analysis['best_results'][:20]  # Top 20 results
    
    mu_vals = [r['mu0'] for r in results]
    G_geo_vals = [r['G_geo'] for r in results]
    energies = [r['energy_J'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # μ₀ distribution
    axes[0, 0].hist(mu_vals, bins=10, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('μ₀ (polymer length)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Optimal μ₀ Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # G_geo distribution
    axes[0, 1].hist(G_geo_vals, bins=10, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('G_geo (VdB ratio)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Optimal G_geo Values')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy distribution
    axes[1, 0].hist(energies, bins=15, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].set_xlabel('E₋ (J)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Optimal Energies')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Parameter correlation
    scatter = axes[1, 1].scatter(mu_vals, G_geo_vals, c=energies, 
                                cmap='viridis', s=80, alpha=0.8, edgecolors='black')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel('μ₀ (polymer length)')
    axes[1, 1].set_ylabel('G_geo (VdB ratio)')
    axes[1, 1].set_title('Parameter Correlation (colored by energy)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Colorbar for correlation plot
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('E₋ (J)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_plot:
        filename = 'parameter_distributions.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Parameter distributions saved as '{filename}'")
    
    return fig

# ── 7. RESULT PERSISTENCE ─────────────────────────────────────────────────────
def save_scan_results(results, analysis, filename='parameter_scan_results.json'):
    """Save complete scan results to JSON file"""
    output_data = {
        'config': {
            'mu_range': config.mu_range,
            'G_geo_range': config.G_geo_range,
            'mu_points': config.mu_points,
            'G_geo_points': config.G_geo_points,
            'ansatz_types': config.ansatz_types,
            'M_gauss_values': config.M_gauss_values,
            'enable_physics_constraints': config.enable_physics_constraints
        },
        'timestamp': time.time(),
        'results': results,
        'analysis': analysis
    }
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Complete scan results saved to '{filename}'")

def load_scan_results(filename='parameter_scan_results.json'):
    """Load scan results from JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data['results'], data['analysis']
    except FileNotFoundError:
        print(f"❌ File '{filename}' not found")
        return None, None
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None, None

# ── 8. MAIN SCANNING PIPELINE ─────────────────────────────────────────────────
def run_comprehensive_scan(adaptive_refinement=True, save_results=True):
    """
    Run comprehensive parameter scan with optional adaptive refinement.
    
    Args:
        adaptive_refinement: Whether to perform adaptive refinement around best results
        save_results: Whether to save results to file
    
    Returns:
        tuple: (results, analysis)
    """
    print("🚀 COMPREHENSIVE PARAMETER SCAN")
    print("=" * 60)
    print(f"📊 Scanning {config.mu_points}×{config.G_geo_points} parameter grid")
    print(f"🔬 Ansatz types: {config.ansatz_types}")
    print(f"🧮 M_gauss values: {config.M_gauss_values}")
    print(f"⚡ Physics constraints: {config.enable_physics_constraints}")
    print()
    
    # Generate initial parameter grid
    mu_values, G_geo_values, combinations = generate_parameter_grid()
    
    print(f"🎯 Parameter ranges:")
    print(f"   μ₀: {config.mu_range[0]:.1e} to {config.mu_range[1]:.1e}")
    print(f"   G_geo: {config.G_geo_range[0]:.1e} to {config.G_geo_range[1]:.1e}")
    print(f"   Total combinations: {len(combinations)}")
    print()
    
    # Run initial scan
    start_time = time.time()
    results = run_parallel_scan(combinations)
    initial_time = time.time() - start_time
    
    # Analyze initial results
    analysis = analyze_scan_results(results)
    
    # Adaptive refinement
    if adaptive_refinement and analysis.get('success', False) and analysis['best_results']:
        print(f"\n🔍 ADAPTIVE REFINEMENT")
        print("-" * 30)
        
        refined_combinations = generate_adaptive_grid(analysis['best_results'])
        
        if refined_combinations:
            print(f"🎯 Refining around {len(refined_combinations)} promising points...")
            
            refined_start = time.time()
            refined_results = run_parallel_scan(refined_combinations)
            refined_time = time.time() - refined_start
            
            # Combine results
            all_results = results + refined_results
            analysis = analyze_scan_results(all_results)
            
            print(f"✅ Refinement completed in {refined_time:.1f}s")
        else:
            print("⚠️  No refinement needed")
            all_results = results
    else:
        all_results = results
    
    total_time = time.time() - start_time
    
    # Print comprehensive summary
    print_scan_summary(analysis)
    
    print(f"\n⏱️  TIMING SUMMARY:")
    print(f"   Initial scan: {initial_time:.1f}s")
    if adaptive_refinement and 'refined_time' in locals():
        print(f"   Refinement: {refined_time:.1f}s")
    print(f"   Total time: {total_time:.1f}s")
    
    # Visualization
    if analysis.get('success', False):
        print(f"\n📊 Generating visualizations...")
        create_energy_heatmap(all_results)
        plot_parameter_distributions(analysis)
    
    # Save results
    if save_results:
        save_scan_results(all_results, analysis)
    
    print(f"\n🎉 Comprehensive parameter scan completed!")
    
    return all_results, analysis

# ── 9. COMMAND LINE INTERFACE ─────────────────────────────────────────────────
def main():
    """Main execution function"""
    print("🌟 COMPREHENSIVE PARAMETER SCAN")
    print("🎯 Finding optimal (μ₀, G_geo) combinations for minimal E₋")
    print()
    
    if not OPTIMIZERS_AVAILABLE:
        print("❌ Optimization modules not available")
        print("   Please ensure enhanced_gaussian_optimizer.py and")
        print("   hybrid_polynomial_gaussian_optimizer.py are in the same directory")
        return
    
    # Configuration summary
    print("⚙️  SCAN CONFIGURATION:")
    print(f"   μ₀ range: {config.mu_range[0]:.1e} to {config.mu_range[1]:.1e} ({config.mu_points} points)")
    print(f"   G_geo range: {config.G_geo_range[0]:.1e} to {config.G_geo_range[1]:.1e} ({config.G_geo_points} points)")
    print(f"   Ansatz types: {config.ansatz_types}")
    print(f"   Parallel workers: {config.max_workers}")
    print()
    
    # Run scan
    results, analysis = run_comprehensive_scan(
        adaptive_refinement=True,
        save_results=True
    )
    
    # Final summary
    if analysis.get('success', False) and analysis['best_results']:
        best = analysis['best_results'][0]
        print(f"\n🏆 BEST OVERALL RESULT:")
        print(f"   Ansatz: {best['ansatz_type'].title()}")
        print(f"   Parameters: μ₀={best['mu0']:.1e}, G_geo={best['G_geo']:.1e}")
        print(f"   Energy: E₋ = {best['energy_J']:.3e} J")
        
        # Improvement calculation
        soliton_baseline = -1.584e31
        improvement = abs(best['energy_J'] / soliton_baseline)
        if improvement > 1.0:
            print(f"   🎉 BREAKTHROUGH: {(improvement-1)*100:.1f}% better than soliton!")
        else:
            print(f"   📊 Performance: {improvement*100:.1f}% of soliton baseline")
    
    return results, analysis

if __name__ == "__main__":
    main()
