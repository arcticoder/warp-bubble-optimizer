#!/usr/bin/env python3
"""
Run warp bubble stability analysis with GUT polymer corrections.

This script applies unified gauge theory polymer corrections to warp bubble metrics
and analyzes the ANEC integral for stability conditions.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add the path to unified_gut_polymerization package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "unified-gut-polymerization")))

from src.warp_qft.gut_polymer_corrections import GUTPolymerMetricCorrections, ANECIntegralCalculator

# Create output directory if needed
output_dir = "gut_polymer_results"
os.makedirs(output_dir, exist_ok=True)

def run_gut_polymer_anec_analysis():
    """
    Run ANEC integral analysis with GUT polymer corrections for each gauge group.
    Compare stability conditions across groups.
    """
    # GUT groups to analyze
    gut_groups = ['SU5', 'SO10', 'E6']
    
    # Parameters
    polymer_scale_mu = 0.2
    energy_scale = 1e12  # GeV
    field_strength_range = np.linspace(0.1, 5.0, 50)
    
    # Store results for comparison
    results = {}
    
    print("Running GUT-Polymer ANEC Analysis")
    print("=================================")
    print(f"Polymer Scale: μ = {polymer_scale_mu}")
    print(f"Energy Scale: {energy_scale/1e9} TeV")
    print(f"Field Strength Range: {field_strength_range[0]} - {field_strength_range[-1]}")
    print()
    
    # Run analysis for each group
    for group in gut_groups:
        print(f"Analyzing {group} gauge group...")
        
        # Create GUT polymer corrections
        gut_polymer = GUTPolymerMetricCorrections(
            group=group,
            polymer_scale_mu=polymer_scale_mu,
            field_strength=1.0,  # Initial value
            energy_scale=energy_scale
        )
        
        # Create ANEC calculator
        anec_calculator = ANECIntegralCalculator(
            gut_polymer=gut_polymer,
            num_points=200,  # Reduced for faster computation
            integration_range=5.0
        )
        
        # Run stability analysis
        stability_results = anec_calculator.stability_condition_h_infinity(field_strength_range)
        
        # Plot and save results
        plt = anec_calculator.plot_stability_margins(stability_results)
        plt.savefig(f"{output_dir}/{group}_stability_margins.png", dpi=300)
        plt.close()
        
        # Store results
        results[group] = stability_results
        
        # Print summary
        print(f"  Optimal Field Strength: {stability_results['optimal_field_strength']:.4f}")
        print(f"  Optimal Stability Margin: {stability_results['optimal_margin']:.4f}")
        print(f"  Stable Configuration: {stability_results['is_stable']}")
        print()
    
    return results

def compare_gut_groups(results, polymer_scale_mu, energy_scale):
    """
    Create comparative plots for different GUT groups.
    
    Args:
        results: Dictionary with analysis results for each GUT group
        polymer_scale_mu: Polymer scale parameter used in analysis
        energy_scale: Energy scale in GeV used in analysis
    """
    gut_groups = list(results.keys())
    
    # Compare stability margins
    plt.figure(figsize=(12, 8))
    
    for group in gut_groups:
        field_values = results[group]['field_strength_values']
        margins = results[group]['stability_margins']
        plt.plot(field_values, margins, linewidth=2, label=group)
    
    plt.axhline(y=1.0, color='r', linestyle='--', label='Stability Threshold')
    plt.xlabel('Field Strength Parameter')
    plt.ylabel('H-infinity Stability Margin')
    plt.title('Comparison of Stability Margins Across GUT Groups')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gut_group_comparison.png", dpi=300)
    
    # Create summary table
    print("\nSummary of Optimal Configurations:")
    print("=================================")
    print(f"{'Group':<6} {'Optimal Field':<15} {'Stability Margin':<20} {'Stable':<10}")
    print("-" * 55)
    
    for group in gut_groups:
        optimal_field = results[group]['optimal_field_strength']
        optimal_margin = results[group]['optimal_margin']
        is_stable = results[group]['is_stable']
        
        print(f"{group:<6} {optimal_field:<15.4f} {optimal_margin:<20.4f} {str(is_stable):<10}")
      # Save summary to file
    with open(f"{output_dir}/gut_polymer_summary.txt", "w") as f:
        f.write("Summary of Warp Bubble Stability with GUT-Polymer Corrections\n")
        f.write("=======================================================\n\n")
        f.write(f"Polymer Scale: mu = {polymer_scale_mu}\n")
        f.write(f"Energy Scale: {energy_scale/1e9} TeV\n\n")
        
        f.write(f"{'Group':<6} {'Optimal Field':<15} {'Stability Margin':<20} {'Stable':<10}\n")
        f.write("-" * 55 + "\n")
        
        for group in gut_groups:
            optimal_field = results[group]['optimal_field_strength']
            optimal_margin = results[group]['optimal_margin']
            is_stable = results[group]['is_stable']
            
            f.write(f"{group:<6} {optimal_field:<15.4f} {optimal_margin:<20.4f} {str(is_stable):<10}\n")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run GUT-polymer warp analysis')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    args = parser.parse_args()
    
    # Set verbosity level
    if not args.verbose:
        # Temporarily redirect stdout to suppress print statements
        import io
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
    
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Run the analysis
        results = run_gut_polymer_anec_analysis()
        
        # Get parameters from the function scope for the comparison
        polymer_scale_mu = 0.2  # Match the value used in run_gut_polymer_anec_analysis
        energy_scale = 1e12     # Match the value used in run_gut_polymer_anec_analysis
        
        # Compare results across GUT groups
        compare_gut_groups(results, polymer_scale_mu, energy_scale)
        
        print(f"\nAnalysis complete. Results saved to {output_dir}/")
    finally:
        # Restore stdout if redirected
        if not args.verbose:
            sys.stdout = original_stdout
