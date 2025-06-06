#!/usr/bin/env python3
"""
Joint Parameter Scan for (μ, G_geo) Optimization

Systematically scans the parameter space of μ (polymer length) and G_geo 
(Van den Broeck–Natário geometric factor) to find the global minimum energy.

Previously we fixed μ=1e-6 and G_geo=1e-5. This scan explores:
- μ ∈ [1e-8, 1e-3] (6 points logarithmically spaced)
- G_geo ∈ [1e-7, 1e-3] (5 points logarithmically spaced)

Tests both 6-Gaussian and Cubic Hybrid ansätze on each (μ, G_geo) pair.
Target: Find parameter combination that pushes E₋ below -2.0×10³¹ J
"""
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Import our optimizers
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gaussian_optimize_M6 import optimize_M6, E_negative_M6
    from hybrid_cubic_optimizer import optimize_hybrid_cubic, E_negative_hybrid_cubic
    OPTIMIZERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import optimizers: {e}")
    print("   Some functionality will be limited")
    OPTIMIZERS_AVAILABLE = False

# ── 1. Parameter Scan Configuration ───────────────────────────────────────────
# Logarithmic grids for parameter exploration
mu_values = np.logspace(-8, -3, 6)      # [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
G_geo_values = np.logspace(-7, -3, 5)   # [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

print(f"Parameter scan grid:")
print(f"  μ values: {mu_values}")
print(f"  G_geo values: {G_geo_values}")
print(f"  Total combinations: {len(mu_values) * len(G_geo_values)}")

# ── 2. Individual Parameter Point Optimization ────────────────────────────────
def optimize_single_point(params_tuple):
    """
    Optimize a single (μ, G_geo) parameter point
    
    Args:
        params_tuple: (mu_val, G_geo_val, ansatz_type)
    
    Returns:
        Dictionary with optimization results
    """
    mu_val, G_geo_val, ansatz_type = params_tuple
    
    result = {
        'mu': mu_val,
        'G_geo': G_geo_val,
        'ansatz_type': ansatz_type,
        'success': False,
        'energy_J': 0.0,
        'params': None,
        'optimization_time': 0.0,
        'error_message': None
    }
    
    try:
        start_time = time.time()
        
        if ansatz_type == '6_gaussian':
            if not OPTIMIZERS_AVAILABLE:
                raise ImportError("6-Gaussian optimizer not available")
            
            opt_result = optimize_M6(mu_val=mu_val, G_geo_val=G_geo_val, verbose=False)
            
            if opt_result is not None:
                result.update({
                    'success': True,
                    'energy_J': opt_result['energy_J'],
                    'params': opt_result['params'].tolist() if hasattr(opt_result['params'], 'tolist') else opt_result['params'],
                    'optimization_time': time.time() - start_time
                })
            else:
                result['error_message'] = "6-Gaussian optimization returned None"
                
        elif ansatz_type == 'cubic_hybrid':
            if not OPTIMIZERS_AVAILABLE:
                raise ImportError("Cubic hybrid optimizer not available")
                
            opt_result = optimize_hybrid_cubic(mu_val=mu_val, G_geo_val=G_geo_val, verbose=False)
            
            if opt_result is not None:
                result.update({
                    'success': True,
                    'energy_J': opt_result['energy_J'],
                    'params': opt_result['params'].tolist() if hasattr(opt_result['params'], 'tolist') else opt_result['params'],
                    'optimization_time': time.time() - start_time
                })
            else:
                result['error_message'] = "Cubic hybrid optimization returned None"
        
        else:
            result['error_message'] = f"Unknown ansatz type: {ansatz_type}"
            
    except Exception as e:
        result['error_message'] = str(e)
        result['optimization_time'] = time.time() - start_time
    
    return result

# ── 3. Parallel Parameter Scan ────────────────────────────────────────────────
def run_parameter_scan(use_parallel=True, max_workers=None):
    """
    Run complete parameter scan across (μ, G_geo) space
    """
    print("🚀 JOINT PARAMETER SCAN")
    print("=" * 60)
    print(f"Scanning {len(mu_values)} × {len(G_geo_values)} = {len(mu_values) * len(G_geo_values)} parameter combinations")
    print("Testing both 6-Gaussian and Cubic Hybrid ansätze")
    
    # Prepare parameter combinations
    param_combinations = []
    for mu_val in mu_values:
        for G_geo_val in G_geo_values:
            param_combinations.append((mu_val, G_geo_val, '6_gaussian'))
            param_combinations.append((mu_val, G_geo_val, 'cubic_hybrid'))
    
    total_combinations = len(param_combinations)
    print(f"Total optimization runs: {total_combinations}")
    
    start_time = time.time()
    results = []
    
    if use_parallel and OPTIMIZERS_AVAILABLE:
        # Parallel execution
        if max_workers is None:
            max_workers = min(cpu_count(), 8)  # Don't overwhelm the system
            
        print(f"🔄 Running parallel optimization with {max_workers} workers...")
        
        with Pool(processes=max_workers) as pool:
            results = pool.map(optimize_single_point, param_combinations)
            
    else:
        # Sequential execution (for debugging or if parallel fails)
        print("🔄 Running sequential optimization...")
        
        for i, param_combo in enumerate(param_combinations):
            mu_val, G_geo_val, ansatz_type = param_combo
            print(f"   [{i+1:2d}/{total_combinations}] μ={mu_val:.1e}, G_geo={G_geo_val:.1e}, {ansatz_type}")
            
            result = optimize_single_point(param_combo)
            results.append(result)
            
            if result['success']:
                print(f"      ✅ E₋ = {result['energy_J']:.3e} J")
            else:
                print(f"      ❌ Failed: {result['error_message']}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total scan time: {total_time:.1f}s")
    
    return results

# ── 4. Results Analysis ────────────────────────────────────────────────────────
def analyze_scan_results(results):
    """
    Analyze parameter scan results and find optimal parameters
    """
    print("\n📊 PARAMETER SCAN ANALYSIS")
    print("=" * 60)
    
    # Separate successful and failed results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"Successful optimizations: {len(successful_results)}/{len(results)}")
    print(f"Failed optimizations: {len(failed_results)}")
    
    if len(successful_results) == 0:
        print("❌ No successful optimizations found!")
        return None
    
    # Find best overall result
    best_result = min(successful_results, key=lambda x: x['energy_J'])
    
    print(f"\n🏆 BEST OVERALL RESULT:")
    print(f"   Ansatz: {best_result['ansatz_type']}")
    print(f"   μ = {best_result['mu']:.1e}")
    print(f"   G_geo = {best_result['G_geo']:.1e}")
    print(f"   E₋ = {best_result['energy_J']:.6e} J")
    print(f"   Optimization time: {best_result['optimization_time']:.1f}s")
    
    # Find best result for each ansatz type
    gaussian_results = [r for r in successful_results if r['ansatz_type'] == '6_gaussian']
    hybrid_results = [r for r in successful_results if r['ansatz_type'] == 'cubic_hybrid']
    
    if len(gaussian_results) > 0:
        best_gaussian = min(gaussian_results, key=lambda x: x['energy_J'])
        print(f"\n🎯 BEST 6-GAUSSIAN RESULT:")
        print(f"   μ = {best_gaussian['mu']:.1e}, G_geo = {best_gaussian['G_geo']:.1e}")
        print(f"   E₋ = {best_gaussian['energy_J']:.6e} J")
    
    if len(hybrid_results) > 0:
        best_hybrid = min(hybrid_results, key=lambda x: x['energy_J'])
        print(f"\n🎯 BEST CUBIC HYBRID RESULT:")
        print(f"   μ = {best_hybrid['mu']:.1e}, G_geo = {best_hybrid['G_geo']:.1e}")
        print(f"   E₋ = {best_hybrid['energy_J']:.6e} J")
    
    # Performance comparison with baselines
    print(f"\n📈 PERFORMANCE COMPARISON:")
    baselines = {
        "2-Lump Soliton": -1.584e31,
        "5-Gaussian (fixed params)": -1.90e31,
        "Linear Hybrid (fixed params)": -1.86e31
    }
    
    best_energy = best_result['energy_J']
    for name, baseline in baselines.items():
        improvement = abs(best_energy / baseline)
        print(f"   {name:25s}: {improvement:.3f}× improvement")
    
    # Cost estimate
    cost_per_joule = 2.78e-10  # $/J at $0.001/kWh
    total_cost = abs(best_energy) * cost_per_joule
    print(f"\n💰 Estimated cost: ${total_cost:.2e} at $0.001/kWh")
    
    return best_result

# ── 5. Visualization ───────────────────────────────────────────────────────────
def plot_parameter_scan_heatmap(results, save_fig=True):
    """
    Create heatmap visualization of parameter scan results
    """
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) == 0:
        print("❌ No successful results to plot")
        return
    
    # Separate by ansatz type
    gaussian_results = [r for r in successful_results if r['ansatz_type'] == '6_gaussian']
    hybrid_results = [r for r in successful_results if r['ansatz_type'] == 'cubic_hybrid']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (results_subset, title, ax) in enumerate([
        (gaussian_results, '6-Gaussian Results', axes[0]),
        (hybrid_results, 'Cubic Hybrid Results', axes[1])
    ]):
        
        if len(results_subset) == 0:
            ax.text(0.5, 0.5, 'No successful results', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        # Create energy matrix
        energy_matrix = np.full((len(G_geo_values), len(mu_values)), np.nan)
        
        for result in results_subset:
            mu_idx = np.where(np.isclose(mu_values, result['mu']))[0][0]
            G_geo_idx = np.where(np.isclose(G_geo_values, result['G_geo']))[0][0]
            energy_matrix[G_geo_idx, mu_idx] = result['energy_J']
        
        # Plot heatmap
        im = ax.imshow(energy_matrix, aspect='auto', origin='lower', 
                      cmap='viridis', interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(range(len(mu_values)))
        ax.set_xticklabels([f'{mu:.0e}' for mu in mu_values], rotation=45)
        ax.set_yticks(range(len(G_geo_values)))
        ax.set_yticklabels([f'{G:.0e}' for G in G_geo_values])
        
        ax.set_xlabel('μ (polymer length)')
        ax.set_ylabel('G_geo (VdB-Natário factor)')
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('E₋ (J)')
        
        # Mark the best point
        best_in_subset = min(results_subset, key=lambda x: x['energy_J'])
        mu_idx = np.where(np.isclose(mu_values, best_in_subset['mu']))[0][0]
        G_geo_idx = np.where(np.isclose(G_geo_values, best_in_subset['G_geo']))[0][0]
        ax.plot(mu_idx, G_geo_idx, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1)
    
    plt.suptitle('Parameter Scan Results: Energy vs (μ, G_geo)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('parameter_scan_heatmap.png', dpi=300, bbox_inches='tight')
        print("📊 Heatmap saved as 'parameter_scan_heatmap.png'")
    
    plt.close()  # Close instead of show to prevent blocking

def plot_energy_trends(results, save_fig=True):
    """
    Plot energy trends vs individual parameters
    """
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Energy vs μ
    mu_energies = {}
    for result in successful_results:
        mu = result['mu']
        energy = result['energy_J']
        ansatz = result['ansatz_type']
        
        if mu not in mu_energies:
            mu_energies[mu] = {'6_gaussian': [], 'cubic_hybrid': []}
        mu_energies[mu][ansatz].append(energy)
    
    mu_sorted = sorted(mu_energies.keys())
    for ansatz_type, color, marker in [('6_gaussian', 'blue', 'o'), ('cubic_hybrid', 'red', 's')]:
        mu_vals = []
        energy_means = []
        energy_stds = []
        
        for mu in mu_sorted:
            energies = mu_energies[mu][ansatz_type]
            if len(energies) > 0:
                mu_vals.append(mu)
                energy_means.append(np.mean(energies))
                energy_stds.append(np.std(energies) if len(energies) > 1 else 0)
        
        if len(mu_vals) > 0:
            axes[0].errorbar(mu_vals, energy_means, yerr=energy_stds, 
                           color=color, marker=marker, label=ansatz_type.replace('_', ' ').title(),
                           capsize=3, capthick=1)
    
    axes[0].set_xscale('log')
    axes[0].set_xlabel('μ (polymer length)')
    axes[0].set_ylabel('E₋ (J)')
    axes[0].set_title('Energy vs Polymer Length')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Energy vs G_geo
    G_geo_energies = {}
    for result in successful_results:
        G_geo = result['G_geo']
        energy = result['energy_J']
        ansatz = result['ansatz_type']
        
        if G_geo not in G_geo_energies:
            G_geo_energies[G_geo] = {'6_gaussian': [], 'cubic_hybrid': []}
        G_geo_energies[G_geo][ansatz].append(energy)
    
    G_geo_sorted = sorted(G_geo_energies.keys())
    for ansatz_type, color, marker in [('6_gaussian', 'blue', 'o'), ('cubic_hybrid', 'red', 's')]:
        G_geo_vals = []
        energy_means = []
        energy_stds = []
        
        for G_geo in G_geo_sorted:
            energies = G_geo_energies[G_geo][ansatz_type]
            if len(energies) > 0:
                G_geo_vals.append(G_geo)
                energy_means.append(np.mean(energies))
                energy_stds.append(np.std(energies) if len(energies) > 1 else 0)
        
        if len(G_geo_vals) > 0:
            axes[1].errorbar(G_geo_vals, energy_means, yerr=energy_stds,
                           color=color, marker=marker, label=ansatz_type.replace('_', ' ').title(),
                           capsize=3, capthick=1)
    
    axes[1].set_xscale('log')
    axes[1].set_xlabel('G_geo (VdB-Natário factor)')
    axes[1].set_ylabel('E₋ (J)')
    axes[1].set_title('Energy vs Geometric Factor')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('parameter_trends.png', dpi=300, bbox_inches='tight')
        print("📊 Trends plot saved as 'parameter_trends.png'")
    
    plt.close()  # Close instead of show to prevent blocking

# ── 6. Save and Load Results ───────────────────────────────────────────────────
def save_scan_results(results, filename='parameter_scan_results.json'):
    """
    Save parameter scan results to JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = []
    for result in results:
        result_copy = result.copy()
        if result_copy['params'] is not None and hasattr(result_copy['params'], 'tolist'):
            result_copy['params'] = result_copy['params'].tolist()
        results_serializable.append(result_copy)
    
    with open(filename, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"💾 Results saved to '{filename}'")

def load_scan_results(filename='parameter_scan_results.json'):
    """
    Load parameter scan results from JSON file
    """
    try:
        with open(filename, 'r') as f:
            results = json.load(f)
        print(f"📂 Results loaded from '{filename}'")
        return results
    except FileNotFoundError:
        print(f"❌ File '{filename}' not found")
        return None

# ── 7. Main Execution ──────────────────────────────────────────────────────────
def main():
    """
    Main parameter scan execution
    """
    print("🎯 JOINT (μ, G_geo) PARAMETER OPTIMIZATION")
    print("=" * 80)
    
    if not OPTIMIZERS_AVAILABLE:
        print("❌ Optimizers not available. Please check imports.")
        print("   Make sure gaussian_optimize_M6.py and hybrid_cubic_optimizer.py exist")
        return
    
    # Run parameter scan
    results = run_parameter_scan(use_parallel=True, max_workers=4)
    
    # Analyze results
    best_result = analyze_scan_results(results)
    
    # Save results
    save_scan_results(results)
    
    # Create visualizations
    if len([r for r in results if r['success']]) > 0:
        plot_parameter_scan_heatmap(results)
        plot_energy_trends(results)
    
    # Save best result separately for easy access
    if best_result:
        with open('best_parameter_result.json', 'w') as f:
            json.dump(best_result, f, indent=2)
        print("🏆 Best result saved as 'best_parameter_result.json'")
    
    return results, best_result

if __name__ == "__main__":
    results, best_result = main()
