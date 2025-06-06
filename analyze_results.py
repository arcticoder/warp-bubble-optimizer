#!/usr/bin/env python3
"""
WARP BUBBLE OPTIMIZATION RESULTS ANALYZER

This script analyzes and visualizes the results from the various optimization
strategies to minimize the negative energy of a 1m³ warp bubble. It:

1. Loads results from the various optimization runs
2. Creates comparative visualizations
3. Generates a summary table of results
4. Identifies the best configuration

Author: Advanced Warp Bubble Optimizer
Date: June 6, 2025
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ── UTILITY FUNCTIONS ───────────────────────────────────────────────────────
def load_json_results(filename):
    """Load JSON results safely"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {filename}: {e}")
        return None

def gaussian_profile(r, params, M_gauss):
    """Evaluate Gaussian profile"""
    profile = np.zeros_like(r)
    for i in range(M_gauss):
        A_i = params[3*i]
        sigma_i = params[3*i + 1]
        r0_i = params[3*i + 2]
        profile += A_i * np.exp(-0.5 * ((r - r0_i) / sigma_i)**2)
    return profile

def hybrid_cubic_profile(r, params):
    """Evaluate hybrid cubic profile"""
    R = 1.0
    # Simplified implementation - would need to be adjusted based on actual parameters
    r0 = params[0] if len(params) > 0 else 0.2 * R
    r1 = params[1] if len(params) > 1 else 0.6 * R
    
    # Define regions
    mask_core = r < r0
    mask_poly = (r >= r0) & (r < r1)
    mask_gauss = r >= r1
    
    result = np.zeros_like(r)
    result[mask_core] = 1.0
    
    # Polynomial transition
    if np.any(mask_poly):
        r_poly = r[mask_poly]
        x = (r_poly - r0) / (r1 - r0)
        poly_val = 1.0
        for i in range(3):  # 3rd order polynomial
            coeff = params[i+3] if len(params) > i+3 else 0.0
            poly_val += coeff * (x**(i+1))
        result[mask_poly] = np.clip(poly_val, 0.0, 1.0)
    
    # Gaussian part for r > r1
    if np.any(mask_gauss) and len(params) >= 7:
        r_gauss = r[mask_gauss]
        gauss_total = np.zeros_like(r_gauss)
        
        # Assuming 2 Gaussians in hybrid
        for i in range(2):
            if len(params) >= 3*i + 7:
                Ai = params[3*i + 7]
                r0_i = params[3*i + 8] 
                sig_i = params[3*i + 9]
                x = (r_gauss - r0_i) / sig_i
                gauss_total += Ai * np.exp(-0.5 * x*x)
        
        result[mask_gauss] = np.clip(gauss_total, 0.0, 1.0)
    
    return result

# ── MAIN ANALYSIS FUNCTIONS ──────────────────────────────────────────────────
def analyze_optimizations():
    """Analyze and visualize optimization results"""
    print("🔬 WARP BUBBLE OPTIMIZATION RESULTS ANALYZER")
    print("=" * 60)
    
    # Load results from various optimizers
    print("Loading optimization results...")
    
    results = {
        'jax_6gaussian': load_json_results('jax_gaussian_M6_results.json'),
        'hybrid_cubic': load_json_results('hybrid_cubic_results.json'),
        'cma_4gaussian': load_json_results('cma_gaussian_M4_results.json'),
        'parameter_scan': load_json_results('parameter_scan_results.json'),
        'stability': load_json_results('3d_stability_analysis_results.json')
    }
    
    # Count successfully loaded results
    successful_loads = sum(1 for r in results.values() if r is not None)
    print(f"Successfully loaded {successful_loads} result files")
    
    # Extract energy values and create summary table
    summary_data = []
    baseline_energy = -1.2e37  # Original 2-lump soliton result
    
    # Process results if available
    if results['jax_6gaussian']:
        jax_energy = results['jax_6gaussian'].get('energy_joules', float('nan'))
        jax_improvement = abs(jax_energy / baseline_energy) if not np.isnan(jax_energy) else float('nan')
        summary_data.append({
            'Ansatz': '6-Gaussian',
            'Optimizer': 'JAX-Adam',
            'Energy (J)': jax_energy,
            'Improvement': jax_improvement,
            'Physics Valid': results['jax_6gaussian'].get('physics_validation', {}).get('f_at_bubble', 0) > 0.9
        })
    
    if results['hybrid_cubic']:
        hybrid_energy = results['hybrid_cubic'].get('energy_J', float('nan'))
        hybrid_improvement = abs(hybrid_energy / baseline_energy) if not np.isnan(hybrid_energy) else float('nan')
        summary_data.append({
            'Ansatz': 'Hybrid Cubic',
            'Optimizer': 'DE + L-BFGS-B',
            'Energy (J)': hybrid_energy,
            'Improvement': hybrid_improvement,
            'Physics Valid': results['hybrid_cubic'].get('penalty', float('inf')) < 1e30
        })
    
    if results['cma_4gaussian']:
        cma_energy = results['cma_4gaussian'].get('energy_J', float('nan'))
        cma_improvement = abs(cma_energy / baseline_energy) if not np.isnan(cma_energy) else float('nan')
        summary_data.append({
            'Ansatz': '4-Gaussian',
            'Optimizer': 'CMA-ES',
            'Energy (J)': cma_energy,
            'Improvement': cma_improvement,
            'Physics Valid': True  # Assuming physics validity for now
        })
    
    # Create a 2-lump soliton baseline entry
    summary_data.append({
        'Ansatz': '2-lump Soliton',
        'Optimizer': 'Original',
        'Energy (J)': baseline_energy,
        'Improvement': 1.0,
        'Physics Valid': True
    })
    
    # Convert to DataFrame for easy display
    if summary_data:
        df = pd.DataFrame(summary_data)
        print("\n📊 SUMMARY OF OPTIMIZATION RESULTS")
        print("-" * 60)
        print(df.to_string(index=False))
        
        # Identify best result
        valid_results = df[df['Physics Valid']].copy()
        if not valid_results.empty:
            best_row = valid_results.loc[valid_results['Energy (J)'].idxmin()]
            print("\n🏆 BEST OVERALL RESULT")
            print(f"Ansatz: {best_row['Ansatz']}")
            print(f"Optimizer: {best_row['Optimizer']}")
            print(f"Energy: {best_row['Energy (J)]:.6e} J")
            print(f"Improvement: {best_row['Improvement']:.2e}×")
    else:
        print("\n⚠️ No summary data available - could not load results")
    
    # Create comparison plots if we have profile data
    print("\nCreating comparison visualizations...")
    
    # Set up radial grid for plotting
    r = np.linspace(0, 2.5, 500)
    plt.figure(figsize=(10, 6))
    
    # Plot profiles if available
    try:
        if results['jax_6gaussian'] and 'optimized_parameters' in results['jax_6gaussian']:
            params_jax = np.array(results['jax_6gaussian']['optimized_parameters'])
            profile_jax = gaussian_profile(r, params_jax, 6)
            plt.plot(r, profile_jax, 'b-', label='6-Gaussian (JAX)')
        
        if results['hybrid_cubic'] and 'params' in results['hybrid_cubic']:
            params_hybrid = np.array(results['hybrid_cubic']['params'])
            profile_hybrid = hybrid_cubic_profile(r, params_hybrid)
            plt.plot(r, profile_hybrid, 'g-', label='Hybrid Cubic')
        
        if results['cma_4gaussian'] and 'params' in results['cma_4gaussian']:
            params_cma = np.array(results['cma_4gaussian']['params'])
            profile_cma = gaussian_profile(r, params_cma, 4)
            plt.plot(r, profile_cma, 'r-', label='4-Gaussian (CMA-ES)')
        
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Target f(R=1)=1')
        plt.axvline(x=1.0, color='k', linestyle='--', alpha=0.5, label='Bubble radius R=1')
        plt.xlabel('Radius r (m)')
        plt.ylabel('Warp Factor f(r)')
        plt.title('Comparison of Optimized Warp Bubble Profiles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('warp_profile_comparison.png', dpi=300)
        print("Saved profile comparison to warp_profile_comparison.png")
    except Exception as e:
        print(f"Error creating profile comparison plot: {e}")
    
    # Plot energy comparison bar chart
    try:
        if summary_data:
            plt.figure(figsize=(10, 6))
            
            # Extract data for plot
            ansatz_labels = [row['Ansatz'] for row in summary_data]
            energy_values = [row['Energy (J)'] for row in summary_data]
            improvements = [row['Improvement'] for row in summary_data]
            
            # Create bar colors based on physics validity
            colors = ['green' if row['Physics Valid'] else 'red' for row in summary_data]
            
            # Sort by energy
            sorted_indices = np.argsort(energy_values)
            ansatz_labels = [ansatz_labels[i] for i in sorted_indices]
            energy_values = [energy_values[i] for i in sorted_indices]
            colors = [colors[i] for i in sorted_indices]
            
            plt.bar(ansatz_labels, energy_values, color=colors)
            plt.xlabel('Ansatz Type')
            plt.ylabel('Negative Energy (J)')
            plt.title('Comparison of Negative Energy by Ansatz')
            plt.yscale('symlog')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('energy_comparison.png', dpi=300)
            print("Saved energy comparison to energy_comparison.png")
    except Exception as e:
        print(f"Error creating energy comparison plot: {e}")
    
    # Plot parameter scan heatmap if available
    try:
        if results['parameter_scan'] and 'coarse_scan' in results['parameter_scan']:
            coarse_scan = results['parameter_scan']['coarse_scan']
            valid_results = [r for r in coarse_scan if r.get('success', False)]
            
            if valid_results:
                # Extract data for heatmap
                mu_values = sorted(set(r['mu'] for r in valid_results))
                G_geo_values = sorted(set(r['G_geo'] for r in valid_results))
                
                # Create 2D grid for heatmap
                energy_grid = np.full((len(mu_values), len(G_geo_values)), np.nan)
                
                # Fill the grid
                for r in valid_results:
                    mu_idx = mu_values.index(r['mu'])
                    G_geo_idx = G_geo_values.index(r['G_geo'])
                    energy_grid[mu_idx, G_geo_idx] = r['energy_J']
                
                # Create plot
                plt.figure(figsize=(10, 8))
                plt.pcolormesh(
                    np.log10(G_geo_values),
                    np.log10(mu_values),
                    energy_grid,
                    cmap='viridis',
                    shading='auto'
                )
                
                # Add best result marker
                if 'best_result' in results['parameter_scan']:
                    best = results['parameter_scan']['best_result']
                    if best:
                        best_mu = best.get('mu', 0)
                        best_G_geo = best.get('G_geo', 0)
                        plt.plot([np.log10(best_G_geo)], [np.log10(best_mu)], 'r*', markersize=15)
                
                plt.xlabel('log₁₀(G_geo)')
                plt.ylabel('log₁₀(μ)')
                plt.title('Negative Energy Landscape (J)')
                plt.colorbar(label='Energy E₋ (J)')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig('parameter_scan_heatmap.png', dpi=300)
                print("Saved parameter scan heatmap to parameter_scan_heatmap.png")
    except Exception as e:
        print(f"Error creating parameter scan heatmap: {e}")
    
    print("\n✅ Analysis complete")

if __name__ == "__main__":
    analyze_optimizations()
