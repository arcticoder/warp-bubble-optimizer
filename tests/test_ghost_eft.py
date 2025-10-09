#!/usr/bin/env python3
"""
Test the Ghost EFT energy source directly.
"""

import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from warp_qft.energy_sources import GhostCondensateEFT

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
print(f"Valid parameters: {ghost_source.validate_parameters()}")

# Test energy density at various points
test_points = [
    (0.0, 0.0, 0.0),    # Center
    (1.0, 0.0, 0.0),    # Close to center
    (5.0, 0.0, 0.0),    # At R0
    (10.0, 0.0, 0.0),   # Outside
]

# Test at t=0 and t=2.5 (quarter period for sin to be non-zero)
for t_test in [0.0, 2.5]:
    print(f"\nAt time t={t_test}:")
    for x, y, z in test_points:
        # Convert to arrays for the energy_density function
        x_arr = np.array([x])
        y_arr = np.array([y])
        z_arr = np.array([z])
        energy = ghost_source.energy_density(x_arr, y_arr, z_arr, t_test)[0]
        print(f"Energy at ({x}, {y}, {z}): {energy:.2e} J/m³")

# Remove stress tensor test as it doesn't exist
# print("\nStress tensor at (1,0,0):")
# stress = ghost_source.stress_tensor(1.0, 0.0, 0.0)
# print(f"T00: {stress[0,0]:.2e}")
# print(f"T11: {stress[1,1]:.2e}")
# print(f"T22: {stress[2,2]:.2e}")
# print(f"T33: {stress[3,3]:.2e}")

# Test total energy
total = ghost_source.total_energy(1000)  # 1000 m³ volume
print(f"\nTotal energy (1000 m³): {total:.2e} J")
