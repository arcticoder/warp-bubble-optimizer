#!/usr/bin/env python3
"""
Quick test of the solver to debug the pipeline failures.
"""

import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from warp_qft.energy_sources import GhostCondensateEFT
    from warp_qft.enhanced_warp_solver import EnhancedWarpBubbleSolver
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Create Ghost EFT source
ghost_source = GhostCondensateEFT(
    M=1000,           # Mass scale (GeV)
    alpha=0.01,       # EFT coupling  
    beta=0.1,         # Self-interaction strength
    R0=5.0,           # Bubble radius scale (m)
    sigma=0.5,        # Transition width (m)
    mu_polymer=0.1    # LQG polymer parameter
)

print("✅ Ghost EFT source created")

# Create solver
solver = EnhancedWarpBubbleSolver(
    use_polymer_enhancement=True,
    enable_stability_analysis=True
)

print("✅ Solver created")

# Test simulation
try:
    print("Running test simulation...")
    result = solver.simulate(
        energy_source=ghost_source,
        radius=10.0,
        resolution=25    )
    
    print(f"✅ Simulation completed!")
    print(f"Success: {result.success}")
    print(f"Energy: {result.energy_total:.2e} J")
    print(f"Stability: {result.stability:.3f}")
    print(f"Max negative density: {result.max_negative_density:.2e}")
    print(f"QI violation achieved: {result.qi_violation_achieved}")
    print(f"Polymer enhancement: {result.polymer_enhancement_factor:.2f}")
    
    # Check success criteria
    has_negative = result.max_negative_density < -1e-15
    is_stable = result.stability > 0.01
    print(f"\nSuccess criteria:")
    print(f"Has negative energy (< -1e-15): {has_negative} ({result.max_negative_density:.2e})")
    print(f"Is stable (> 0.01): {is_stable} ({result.stability:.3f})")
    print(f"QI violation achieved: {result.qi_violation_achieved}")
    
except Exception as e:
    print(f"❌ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
