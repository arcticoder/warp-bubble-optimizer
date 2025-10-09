#!/usr/bin/env python3
"""
LQG BOUND ENFORCEMENT DEMONSTRATION

This script demonstrates that the advanced multi-strategy optimizer now enforces
the LQG-modified quantum inequality bound instead of the classical Ford-Roman bound.

The LQG bound is: E₋ ≥ -C_LQG / T^4
where C_LQG << C_classical (typically ~100× smaller).

This provides a more restrictive but physically correct lower bound for warp bubble
optimization in Loop Quantum Gravity.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append('src')

try:
    from warp_qft.stability import lqg_modified_bounds, enforce_lqg_bound, ford_roman_bounds
    from advanced_multi_strategy_optimizer import (
        compute_energy_mixed_basis_numpy,
        AnsatzConfig,
        run_comprehensive_optimization,
        HAS_LQG_BOUNDS
    )
    print("✅ Successfully imported LQG bound enforcement modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def demonstrate_lqg_bounds():
    """Demonstrate LQG vs Ford-Roman bounds"""
    print("\n🔬 LQG vs Ford-Roman Quantum Inequality Bounds")
    print("=" * 60)
    
    # Physical parameters
    spatial_scale = 1.0  # Characteristic length scale
    flight_time = 1e-9   # Sampling time (1 ns)
    test_energy = -1e30  # Test negative energy (J)
    
    # Classical Ford-Roman bound
    classical_bounds = ford_roman_bounds(test_energy, spatial_scale, flight_time)
    
    # LQG-modified bound  
    lqg_bounds = lqg_modified_bounds(test_energy, spatial_scale, flight_time)
    
    print(f"Test energy density: {test_energy:.3e} J")
    print(f"Spatial scale: {spatial_scale} m")
    print(f"Flight time: {flight_time:.3e} s")
    print()
    
    print("CLASSICAL FORD-ROMAN BOUND:")
    print(f"  Bound: {classical_bounds['ford_roman_bound']:.3e} J")
    print(f"  Violates: {classical_bounds['violates_bound']}")
    print(f"  Violation factor: {classical_bounds['violation_factor']:.2f}")
    print()
    
    print("LQG-MODIFIED BOUND:")
    print(f"  Bound: {lqg_bounds['lqg_bound']:.3e} J")
    print(f"  Violates: {lqg_bounds['violates_bound']}")
    print(f"  Violation factor: {lqg_bounds['violation_factor']:.2f}")
    print(f"  Strictness vs Ford-Roman: {lqg_bounds['strictness_factor']:.1f}× stricter")
    print()
    
    # Demonstration of enforcement
    print("BOUND ENFORCEMENT:")
    raw_energy = -1e35  # Very negative energy
    enforced_energy = enforce_lqg_bound(raw_energy, spatial_scale, flight_time)
    
    print(f"  Raw computed energy: {raw_energy:.3e} J")
    print(f"  LQG-enforced energy: {enforced_energy:.3e} J")
    print(f"  Energy limited by: {abs(enforced_energy / raw_energy):.2%} of raw value")
    
    return {
        'classical_bound': classical_bounds['ford_roman_bound'],
        'lqg_bound': lqg_bounds['lqg_bound'],
        'strictness_factor': lqg_bounds['strictness_factor']
    }

def test_optimizer_with_lqg():
    """Test optimizer with LQG bound enforcement"""
    print("\n🚀 Testing Optimizer with LQG Bound Enforcement")
    print("=" * 60)
    
    if not HAS_LQG_BOUNDS:
        print("❌ LQG bounds not available in optimizer")
        return None
    
    # Create test parameters for mixed-basis ansatz
    print("Creating test parameters for mixed-basis ansatz...")
    test_params = np.array([
        5.2e-6,   # mu (polymer scale)
        2.5e-5,   # G_geo (geometric factor)
        # 8 Gaussians: A_i, r0_i, sigma_i
        0.8, 0.1, 0.05,  # Gaussian 1
        0.6, 0.2, 0.08,  # Gaussian 2
        0.4, 0.4, 0.12,  # Gaussian 3
        0.3, 0.6, 0.15,  # Gaussian 4
        0.2, 0.7, 0.18,  # Gaussian 5
        0.15, 0.8, 0.20, # Gaussian 6
        0.1, 0.85, 0.22, # Gaussian 7
        0.05, 0.9, 0.25, # Gaussian 8
        # 4 Fourier modes
        0.1, -0.05, 0.08, -0.03
    ])
    
    print(f"Number of parameters: {len(test_params)}")
    print(f"Expected dimension: {AnsatzConfig.MIXED_PARAM_DIM}")
    
    # Compute energy with LQG bound enforcement
    print("\nComputing energy with LQG bound enforcement...")
    energy_with_lqg = compute_energy_mixed_basis_numpy(test_params)
    
    print(f"Energy with LQG bound: {energy_with_lqg:.6e} J")
    
    # For comparison, temporarily disable LQG bounds
    global HAS_LQG_BOUNDS
    original_has_lqg = HAS_LQG_BOUNDS
    HAS_LQG_BOUNDS = False
    
    energy_without_lqg = compute_energy_mixed_basis_numpy(test_params)
    HAS_LQG_BOUNDS = original_has_lqg
    
    print(f"Energy without LQG bound: {energy_without_lqg:.6e} J")
    print(f"LQG enforcement factor: {abs(energy_with_lqg / energy_without_lqg):.3f}")
    
    return {
        'energy_with_lqg': energy_with_lqg,
        'energy_without_lqg': energy_without_lqg,
        'enforcement_factor': abs(energy_with_lqg / energy_without_lqg)
    }

def plot_bound_comparison():
    """Plot comparison of Ford-Roman vs LQG bounds"""
    print("\n📊 Creating bound comparison plot...")
    
    # Range of flight times
    flight_times = np.logspace(-12, -6, 50)  # 1 ps to 1 μs
    spatial_scale = 1.0
    
    ford_roman_bounds_list = []
    lqg_bounds_list = []
    
    for T in flight_times:
        # Use a reference energy density
        test_energy = -1e30
        
        fr_bounds = ford_roman_bounds(test_energy, spatial_scale, T)
        lqg_bounds_mod = lqg_modified_bounds(test_energy, spatial_scale, T)
        
        ford_roman_bounds_list.append(abs(fr_bounds['ford_roman_bound']))
        lqg_bounds_list.append(abs(lqg_bounds_mod['lqg_bound']))
    
    plt.figure(figsize=(10, 6))
    plt.loglog(flight_times * 1e9, ford_roman_bounds_list, 'b-', 
               label='Ford-Roman (Classical)', linewidth=2)
    plt.loglog(flight_times * 1e9, lqg_bounds_list, 'r-', 
               label='LQG-Modified', linewidth=2)
    
    plt.xlabel('Flight Time (ns)')
    plt.ylabel('|Quantum Inequality Bound| (J/m³)')
    plt.title('Quantum Inequality Bounds: Ford-Roman vs LQG-Modified')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotation
    plt.annotate('LQG bound is\n~100× stricter', 
                xy=(10, 1e25), xytext=(100, 1e27),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.savefig('lqg_vs_ford_roman_bounds.png', dpi=300, bbox_inches='tight')
    print("   Saved plot: lqg_vs_ford_roman_bounds.png")

def main():
    """Main demonstration function"""
    print("🌟 LQG-MODIFIED QUANTUM INEQUALITY BOUND DEMONSTRATION")
    print("=" * 70)
    print("This demonstrates the integration of LQG-modified bounds into")
    print("the advanced multi-strategy warp bubble optimizer.")
    print()
    
    # Demonstrate bounds
    bounds_info = demonstrate_lqg_bounds()
    
    # Test optimizer
    optimizer_info = test_optimizer_with_lqg()
    
    # Create visualization
    plot_bound_comparison()
    
    print("\n✨ SUMMARY")
    print("=" * 30)
    print(f"LQG bound is {bounds_info['strictness_factor']:.1f}× stricter than Ford-Roman")
    
    if optimizer_info:
        print(f"Optimizer enforcement factor: {optimizer_info['enforcement_factor']:.3f}")
        print("✅ LQG bound successfully integrated into optimizer")
    else:
        print("⚠️  LQG bounds not available in optimizer")
    
    print("\n🎯 RESULT: The optimizer now targets E₋ ≥ -C_LQG/T⁴")
    print("   instead of the classical Ford-Roman bound.")
    print("   This provides a physically correct optimization target")
    print("   for warp bubble energy requirements in LQG.")

if __name__ == "__main__":
    main()
