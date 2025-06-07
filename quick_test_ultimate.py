#!/usr/bin/env python3
"""
Simplified Ultimate B-Spline Test Run
====================================

Quick test of the ultimate B-spline optimizer with reduced parameters
for faster execution and debugging.
"""

import sys
sys.path.append('.')

try:
    from ultimate_bspline_optimizer import UltimateBSplineOptimizer
    import json
    from datetime import datetime
    
    print("🧪 SIMPLIFIED ULTIMATE B-SPLINE TEST")
    print("=" * 40)
    
    # Initialize with reduced parameters
    optimizer = UltimateBSplineOptimizer(
        n_control_points=8,           # Reduced from 15
        R_bubble=100.0,
        stability_penalty_weight=1e4,  # Reduced penalty  
        surrogate_assisted=False,     # Disabled for testing
        verbose=True
    )
    
    print("✅ Optimizer initialized")
    
    # Run quick optimization
    results = optimizer.optimize(
        max_cma_evaluations=100,      # Reduced from 3000
        max_jax_iterations=50,        # Reduced from 800
        n_initialization_attempts=2, # Reduced from 4
        use_surrogate_jumps=False     # Disabled
    )
    
    print("✅ Optimization completed")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'test_ultimate_bspline_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {results_file}")
    
    # Print summary
    if results['results']['best_params']:
        best_energy = results['results']['best_energy_J']
        print(f"🎯 Best energy: {best_energy:.3e} J")
        
        # Compare with historical
        historical_energy = -1.48e53  # M8 record
        improvement = abs(best_energy) / abs(historical_energy)
        print(f"📈 vs M8 record: {improvement:.1f}× {'improvement' if improvement > 1 else 'degradation'}")
    
    print("🎉 Test completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Ensure JAX and CMA are installed:")
    print("pip install jax cma scikit-learn")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
