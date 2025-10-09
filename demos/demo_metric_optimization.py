#!/usr/bin/env python3
"""
Warp Bubble Metric Ansatz Optimizer - Quick Demo

This script demonstrates the core capabilities of the optimizer framework
for designing novel warp bubble metric ansatzes that minimize negative energy requirements.

Features:
1. Van den Broeck-Natário geometric baseline (10^5-10^6× reduction)
2. LQG polymer enhancement with corrected sinc(πμ)
3. Exact metric backreaction (β = 1.9443254780147017)
4. Systematic parameter optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from warp_qft.metrics.van_den_broeck_natario import (
    van_den_broeck_shape, 
    natario_shift_vector,
    van_den_broeck_natario_metric,
    energy_requirement_comparison
)
from warp_qft.lqg_profiles import LQGProfileCalculator
from warp_qft.backreaction_solver import BackreactionSolver
from warp_qft.enhancement_pipeline import WarpBubbleEnhancementPipeline

def demonstrate_geometric_baseline():
    """Demonstrate Van den Broeck-Natário geometric baseline."""
    print("🎯 Van den Broeck-Natário Geometric Baseline Demo")
    print("=" * 55)
    
    # Setup parameters
    r_values = np.linspace(0.1, 5.0, 100)
    R_warp = 1.0
    sigma = 0.1
    
    # Calculate shape function
    f_vdb = van_den_broeck_shape(r_values, R_warp, sigma)
    
    # Calculate shift vector
    beta_natario = natario_shift_vector(r_values, R_warp, sigma)
    
    # Energy comparison
    energy_ratios = energy_requirement_comparison(r_values, R_warp, sigma)
    
    print(f"Shape function peak: {np.max(f_vdb):.6f}")
    print(f"Geometric reduction factor: {np.min(energy_ratios):.2e}")
    print(f"Energy reduction: {1/np.min(energy_ratios):.0e}× improvement")
    
    return {
        'r_values': r_values,
        'shape_function': f_vdb,
        'shift_vector': beta_natario,
        'energy_ratios': energy_ratios,
        'reduction_factor': 1/np.min(energy_ratios)
    }

def demonstrate_lqg_enhancement():
    """Demonstrate LQG polymer enhancement with corrected sinc."""
    print("\n🔬 LQG Polymer Enhancement Demo")
    print("=" * 35)
    
    calculator = LQGProfileCalculator()
    
    # Test parameters
    mu_optimal = 0.10
    R_optimal = 2.3
    
    # Calculate corrected sinc factor
    sinc_factor = calculator.corrected_sinc_factor(mu_optimal)
    
    # Profile enhancement comparison
    profiles = calculator.compare_lqg_profiles(mu_optimal, R_optimal)
    
    print(f"Optimal μ parameter: {mu_optimal}")
    print(f"Corrected sinc(πμ): {sinc_factor:.4f}")
    print(f"Bojowald enhancement: {profiles['bojowald_factor']:.1f}×")
    print(f"Ashtekar enhancement: {profiles['ashtekar_factor']:.1f}×")
    print(f"Polymer Field enhancement: {profiles['polymer_factor']:.1f}×")
    
    return {
        'mu_optimal': mu_optimal,
        'sinc_factor': sinc_factor,
        'enhancement_profiles': profiles
    }

def demonstrate_exact_backreaction():
    """Demonstrate exact metric backreaction analysis."""
    print("\n⚛️  Exact Metric Backreaction Demo")
    print("=" * 37)
    
    solver = BackreactionSolver()
    
    # Calculate at optimal parameters
    mu = 0.10
    R = 2.3
    
    # Exact backreaction factor
    beta_exact = solver.exact_backreaction_factor()
    
    # Energy correction
    energy_naive = R * 0.1**2  # Example energy
    energy_corrected = energy_naive / beta_exact
    
    reduction_percent = (1 - 1/beta_exact) * 100
    
    print(f"Exact backreaction factor: {beta_exact:.13f}")
    print(f"Energy correction: {energy_corrected:.4f} (vs {energy_naive:.4f} naive)")
    print(f"Additional reduction: {reduction_percent:.2f}%")
    
    return {
        'beta_exact': beta_exact,
        'energy_reduction_percent': reduction_percent
    }

def demonstrate_combined_optimization():
    """Demonstrate combined enhancement optimization."""
    print("\n🚀 Combined Enhancement Optimization")
    print("=" * 40)
    
    pipeline = WarpBubbleEnhancementPipeline()
    
    # Configure for Van den Broeck-Natário baseline
    config = pipeline.create_default_config()
    config.use_vdb_natario = True
    config.mu = 0.10
    config.R = 2.3
    
    # Run comprehensive enhancement
    results = pipeline.run_comprehensive_enhancement(config)
    
    print(f"Baseline energy ratio: {results['baseline_ratio']:.3f}")
    print(f"Van den Broeck-Natário ratio: {results['vdb_natario_ratio']:.3e}")
    print(f"Final combined ratio: {results['final_ratio']:.3e}")
    print(f"Total enhancement factor: {results['total_enhancement']:.1e}×")
    
    if results['final_ratio'] < 1.0:
        print("✅ FEASIBILITY ACHIEVED!")
    else:
        print("❌ Feasibility gap remains")
    
    return results

def create_optimization_visualization(geometric_results, lqg_results, backreaction_results):
    """Create visualization of optimization results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Van den Broeck shape function
    ax1.plot(geometric_results['r_values'], geometric_results['shape_function'], 'b-', linewidth=2)
    ax1.set_xlabel('Radius r')
    ax1.set_ylabel('Shape Function f(r)')
    ax1.set_title('Van den Broeck-Natário Shape Function')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy reduction factor
    ax2.semilogy(geometric_results['r_values'], geometric_results['energy_ratios'], 'r-', linewidth=2)
    ax2.set_xlabel('Radius r')
    ax2.set_ylabel('Energy Ratio (log scale)')
    ax2.set_title('Geometric Energy Reduction')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: LQG enhancement factors
    profiles = lqg_results['enhancement_profiles']
    methods = ['Bojowald', 'Ashtekar', 'Polymer Field']
    factors = [profiles['bojowald_factor'], profiles['ashtekar_factor'], profiles['polymer_factor']]
    
    bars = ax3.bar(methods, factors, color=['blue', 'green', 'red'], alpha=0.7)
    ax3.set_ylabel('Enhancement Factor')
    ax3.set_title('LQG Profile Enhancement Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, factor in zip(bars, factors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{factor:.1f}×', ha='center', va='bottom')
    
    # Plot 4: Backreaction effect
    beta = backreaction_results['beta_exact']
    reduction = backreaction_results['energy_reduction_percent']
    
    ax4.bar(['Naive Energy', 'Corrected Energy'], [1.0, 1.0/beta], 
            color=['lightcoral', 'lightgreen'], alpha=0.7)
    ax4.set_ylabel('Relative Energy Requirement')
    ax4.set_title('Metric Backreaction Effect')
    ax4.text(1, 1.0/beta + 0.02, f'-{reduction:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('warp_bubble_optimization_demo.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'warp_bubble_optimization_demo.png'")
    
    return fig

def main():
    """Main demonstration of warp bubble metric ansatz optimization."""
    print("🌌 WARP BUBBLE METRIC ANSATZ OPTIMIZER")
    print("=" * 45)
    print("Demonstrating novel metric design for minimized negative energy requirements")
    print()
    
    try:
        # Run demonstrations
        geometric_results = demonstrate_geometric_baseline()
        lqg_results = demonstrate_lqg_enhancement()
        backreaction_results = demonstrate_exact_backreaction()
        combined_results = demonstrate_combined_optimization()
        
        # Create visualization
        create_optimization_visualization(geometric_results, lqg_results, backreaction_results)
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Van den Broeck-Natário reduction: {geometric_results['reduction_factor']:.1e}×")
        print(f"LQG polymer enhancement: {lqg_results['enhancement_profiles']['polymer_factor']:.1f}×")
        print(f"Metric backreaction reduction: {backreaction_results['energy_reduction_percent']:.1f}%")
        print(f"Combined feasibility ratio: {combined_results['final_ratio']:.3e}")
        
        if combined_results['final_ratio'] < 1.0:
            print("\n🎉 WARP DRIVE FEASIBILITY ACHIEVED THROUGH METRIC OPTIMIZATION!")
        else:
            print(f"\n📈 Significant progress: {1/combined_results['final_ratio']:.1f}× closer to feasibility")
        
        print(f"\nTotal enhancement: {combined_results['total_enhancement']:.1e}× reduction in negative energy requirements")
        print("\n🚀 Ready for novel metric ansatz development!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are properly installed.")
        print("Try: pip install -e .")
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("Check that all required files are present in the repository.")

if __name__ == "__main__":
    main()
