#!/usr/bin/env python3
"""
Detailed diagnostic analysis of GUT polymer corrections.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the path to unified_gut_polymerization package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "unified-gut-polymerization")))

from src.warp_qft.gut_polymer_corrections import GUTPolymerMetricCorrections, ANECIntegralCalculator

def detailed_polymer_diagnostic():
    """
    Run detailed diagnostics on polymer corrections.
    """
    print("Detailed GUT-Polymer Diagnostic Analysis")
    print("========================================")
    
    # Test parameters
    field_strengths = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    polymer_scale = 0.2
    energy_scale = 1e12
    
    for group in ['SU5', 'SO10', 'E6']:
        print(f"\n--- {group} Group Analysis ---")
        
        gut_polymer = GUTPolymerMetricCorrections(
            group=group,
            polymer_scale_mu=polymer_scale,
            field_strength=1.0,
            energy_scale=energy_scale
        )
        
        print(f"Polymer scale μ: {gut_polymer.mu}")
        print(f"Energy scale: {gut_polymer.energy_scale/1e9} TeV")
        
        # Test polymer correction factors
        print("\nPolymer correction factors:")
        print("Field Strength | Correction Factor | Modified Curvature")
        print("-" * 55)
        
        base_curvature = 1.0  # Test curvature value
        
        for F in field_strengths:
            gut_polymer.F_strength = F
            correction = gut_polymer.polymer_correction_factor(polymer_scale, F)
            modified_curv = gut_polymer.polymer_modified_curvature(base_curvature)
            print(f"{F:12.2f} | {correction:15.6f} | {modified_curv:15.6f}")
        
        # Test ANEC integral calculation
        print("\nANEC integral analysis:")
        anec_calc = ANECIntegralCalculator(
            gut_polymer=gut_polymer,
            num_points=50,
            integration_range=3.0
        )
        
        print("Field Strength | ANEC Integral")
        print("-" * 30)
        
        anec_values = []
        for F in field_strengths:
            anec_val = anec_calc.compute_anec_integral(F)
            anec_values.append(anec_val)
            print(f"{F:12.2f} | {anec_val:12.6f}")
        
        # Check if values are varying
        anec_std = np.std(anec_values)
        anec_range = np.max(anec_values) - np.min(anec_values)
        print(f"\nANEC variation:")
        print(f"Standard deviation: {anec_std:.6f}")
        print(f"Range: {anec_range:.6f}")
        
        if anec_std < 1e-6:
            print("WARNING: ANEC values show very little variation!")
            print("This may indicate numerical precision issues or constant corrections.")

def test_stress_tensor_modification():
    """
    Test the stress tensor modification directly.
    """
    print("\n" + "="*50)
    print("Stress Tensor Modification Test")
    print("="*50)
    
    # Create test polymer correction
    gut_polymer = GUTPolymerMetricCorrections(
        group='SU5',
        polymer_scale_mu=0.2,
        field_strength=1.0,
        energy_scale=1e12
    )
    
    # Test coordinates
    r_values = np.linspace(-2, 2, 20)
    
    print("\nStress tensor T_μν modifications:")
    print("r coordinate | Original T_00 | Modified T_00 | Difference")
    print("-" * 60)
    
    for r in r_values[::4]:  # Sample every 4th point
        # Original stress tensor component (simplified Alcubierre form)
        T_orig = -0.5 * np.exp(-r**2)  # Example stress component
        
        # Apply polymer modification (simplified)
        phi_orig = np.exp(-r**2)  # Example curvature
        phi_modified = gut_polymer.polymer_modified_curvature(phi_orig, r)
        T_modified = -0.5 * phi_modified
        
        diff = T_modified - T_orig
        print(f"{r:10.2f} | {T_orig:11.6f} | {T_modified:11.6f} | {diff:10.6f}")

if __name__ == "__main__":
    detailed_polymer_diagnostic()
    test_stress_tensor_modification()
