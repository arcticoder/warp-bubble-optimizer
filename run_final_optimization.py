#!/usr/bin/env python3
"""
FINAL WARP BUBBLE OPTIMIZATION PIPELINE
======================================

This script runs the final optimization pipeline for warp bubble negative
energy minimization, combining all advanced strategies:

1. 6-Gaussian ansatz with JAX acceleration
2. Hybrid cubic polynomial transition ansatz
3. Optimal (μ, G_geo) parameter scan
4. 3+1D stability testing

Author: Advanced Warp Bubble Optimizer
Date: June 6, 2025
"""

import numpy as np
import time
import json
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

print("🚀 FINAL WARP BUBBLE OPTIMIZATION PIPELINE")
print("=" * 60)

# Define optimal physical parameters from previous parameter scan
best_mu = 5.2e-6
best_G_geo = 2.5e-5

print(f"Using optimal physical parameters from parameter scan:")
print(f"  μ = {best_mu:.2e}")
print(f"  G_geo = {best_G_geo:.2e}")

# ──── 1. JAX Optimization ───────────────────────────────────────────────────
print("\n🔍 STEP 1: Running JAX-based 6-Gaussian optimization")

try:
    # Import custom modules
    print("  Importing JAX optimizer...")
    from gaussian_optimize_jax import JAXWarpBubbleOptimizer
    
    # Initialize optimizer with custom parameters
    print("  Initializing optimizer...")
    optimizer = JAXWarpBubbleOptimizer(use_jax=True)
    optimizer.mu = best_mu
    optimizer.G_geo = best_G_geo
    
    # Run optimization
    print("  Running optimization...")
    params_jax, energy_jax = optimizer.run_optimization(
        strategy='adam',
        init_strategy='smart',
        max_iter=1000,
        lr=0.01
    )
    
    # Analyze solution
    print("  Analyzing solution...")
    jax_results = optimizer.analyze_solution(params_jax, energy_jax)
    
    print(f"\n  ✅ JAX optimization complete!")
    print(f"  📊 Energy: {energy_jax:.6e} J")
    
except Exception as e:
    print(f"  ❌ JAX optimization failed: {str(e)}")
    jax_results = None

# ──── 2. Hybrid Cubic Optimization ──────────────────────────────────────────
print("\n🔍 STEP 2: Running hybrid cubic polynomial ansatz optimization")

try:
    # Import hybrid optimizer
    print("  Importing hybrid optimizer...")
    from hybrid_cubic_optimizer import run_hybrid_cubic_optimization
    
    # Run optimization
    print("  Running hybrid optimization...")
    hybrid_results = run_hybrid_cubic_optimization(
        mu0=best_mu,
        G_geo=best_G_geo,
        verbose=True
    )
    
    if hybrid_results and 'energy_J' in hybrid_results:
        print(f"\n  ✅ Hybrid optimization complete!")
        print(f"  📊 Energy: {hybrid_results['energy_J']:.6e} J")
    else:
        print(f"  ❌ Hybrid optimization failed to return valid results")
        
except Exception as e:
    print(f"  ❌ Hybrid optimization failed: {str(e)}")
    hybrid_results = None

# ──── 3. CMA-ES 4-Gaussian Optimization ───────────────────────────────────
print("\n🔍 STEP 3: Running CMA-ES 4-Gaussian optimization")

try:
    # Import CMA-ES optimizer
    print("  Importing CMA-ES optimizer...")
    from gaussian_optimize_accelerated import optimize_with_cma, objective_gauss, get_optimization_bounds
    
    # Setup globals
    print("  Setting up optimization...")
    import gaussian_optimize_accelerated as goa
    goa.mu0 = best_mu
    goa.G_geo = best_G_geo
    
    # Run optimization
    print("  Running CMA-ES optimization...")
    bounds = get_optimization_bounds()
    cma_result = optimize_with_cma(bounds, objective_gauss, sigma0=0.2, maxiter=500)
    
    if cma_result and cma_result.get('success', False):
        energy_cma = goa.E_negative_gauss_fast(cma_result['x'])
        print(f"\n  ✅ CMA-ES optimization complete!")
        print(f"  📊 Energy: {energy_cma:.6e} J")
        cma_formatted = {
            'params': cma_result['x'].tolist(),
            'energy_J': float(energy_cma),
            'mu': float(best_mu),
            'G_geo': float(best_G_geo),
            'success': True
        }
    else:
        print(f"  ❌ CMA-ES optimization failed to return valid results")
        cma_formatted = None
        
except Exception as e:
    print(f"  ❌ CMA-ES optimization failed: {str(e)}")
    cma_formatted = None

# ──── 4. 3+1D Stability Testing ───────────────────────────────────────────
print("\n🔍 STEP 4: Running 3+1D stability analysis")

try:
    # Import stability analyzer
    print("  Importing stability analyzer...")
    from test_3d_stability import WarpBubble3DStabilityAnalyzer
    
    # Initialize analyzer
    print("  Initializing stability analyzer...")
    analyzer = WarpBubble3DStabilityAnalyzer()
    
    # Test profiles
    print("  Running stability tests...")
    stability_results = {}
    
    # Test JAX profile if available
    if jax_results and 'params' in jax_results:
        print("  Testing JAX 6-Gaussian profile...")
        classification, max_growth = analyzer.analyze_profile_stability('gaussian_6', np.array(jax_results['params']))
        stability_results['jax_6gaussian'] = {
            'classification': classification,
            'max_growth_rate': float(max_growth)
        }
    
    # Test hybrid profile if available
    if hybrid_results and 'params' in hybrid_results:
        print("  Testing hybrid cubic profile...")
        classification, max_growth = analyzer.analyze_profile_stability('hybrid_cubic', np.array(hybrid_results['params']))
        stability_results['hybrid_cubic'] = {
            'classification': classification,
            'max_growth_rate': float(max_growth)
        }
    
    # Test CMA-ES profile if available
    if cma_formatted and 'params' in cma_formatted:
        print("  Testing CMA-ES 4-Gaussian profile...")
        classification, max_growth = analyzer.analyze_profile_stability('gaussian_4', np.array(cma_formatted['params'][:12]))  # First 12 params for 4 Gaussians
        stability_results['cma_4gaussian'] = {
            'classification': classification,
            'max_growth_rate': float(max_growth)
        }
    
    print("\n  ✅ Stability analysis complete!")
    for profile, result in stability_results.items():
        print(f"  📊 {profile}: {result['classification']} (max growth rate: {result['max_growth_rate']:.3e})")
        
except Exception as e:
    print(f"  ❌ 3+1D Stability analysis failed: {str(e)}")
    stability_results = {}

# ──── 5. Final Results Compilation ──────────────────────────────────────────
print("\n🔍 STEP 5: Compiling and analyzing final results")

# Collect all results
all_results = {
    'optimal_parameters': {
        'mu': float(best_mu),
        'G_geo': float(best_G_geo)
    },
    'jax_6gaussian': jax_results,
    'hybrid_cubic': hybrid_results,
    'cma_4gaussian': cma_formatted,
    'stability_analysis': stability_results
}

# Find best overall result
best_energy = float('inf')
best_result_type = None

if jax_results and 'energy' in jax_results and jax_results['energy'] < best_energy:
    best_energy = jax_results['energy']
    best_result_type = 'jax_6gaussian'

if hybrid_results and 'energy_J' in hybrid_results and hybrid_results['energy_J'] < best_energy:
    best_energy = hybrid_results['energy_J']
    best_result_type = 'hybrid_cubic'

if cma_formatted and 'energy_J' in cma_formatted and cma_formatted['energy_J'] < best_energy:
    best_energy = cma_formatted['energy_J']
    best_result_type = 'cma_4gaussian'

# Print best result
if best_result_type:
    print(f"\n🏆 BEST OVERALL RESULT: {best_result_type}")
    print(f"  Energy: {best_energy:.6e} J")
    
    # Compare with baseline
    baseline = -1.9e31  # Previous best result
    improvement = abs(best_energy / baseline) if baseline != 0 else float('inf')
    print(f"  Improvement over baseline: {improvement:.2f}×")
    
    # Get stability classification if available
    if best_result_type in stability_results:
        stability = stability_results[best_result_type]['classification']
        print(f"  3+1D Stability: {stability}")
else:
    print("\n❌ No valid optimization results found")

# Save comprehensive results
with open('final_optimization_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print("\n💾 Comprehensive results saved to final_optimization_results.json")
print("\n🏁 FINAL OPTIMIZATION PIPELINE COMPLETE")
print("=" * 60)
