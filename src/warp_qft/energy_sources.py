#!/usr/bin/env python3
"""
Enhanced Energy Source Interface for Warp Bubble Optimizer

This module defines energy sources that integrate with the existing 
warp-bubble-optimizer framework, including Discovery 21 Ghost/Phantom EFT
and metamaterial Casimir sources.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, Callable
import warnings

# Import existing warp_qft components
from .warp_bubble_analysis import (
    squeezed_vacuum_energy,
    energy_density_polymer, 
    polymer_QI_bound
)
from .negative_energy import compute_energy_density

class EnergySource(ABC):
    """
    Abstract base class for negative energy sources compatible with
    the warp-bubble-optimizer framework.
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any]):
        """
        Initialize energy source.
        
        Args:
            name: Human-readable name for the source
            parameters: Configuration parameters for the source
        """
        self.name = name
        self.parameters = parameters
        self._is_initialized = False
        
    @abstractmethod
    def energy_density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                      t: float = 0.0) -> np.ndarray:
        """
        Compute energy density at given spacetime coordinates.
        
        Args:
            x, y, z: Spatial coordinate arrays (same shape)
            t: Time coordinate
            
        Returns:
            Energy density array (same shape as input coordinates)
        """
        pass
    
    @abstractmethod
    def get_warp_profile(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get warp bubble profile functions f(r) and g(r) for the metric:
        ds² = -f(r)dt² + g(r)dr² + r²(dθ² + sin²θ dφ²)
        
        Args:
            r: Radial coordinate array
            
        Returns:
            Tuple of (f(r), g(r)) arrays
        """
        pass
    
    def total_energy(self, volume: float) -> float:
        """
        Compute total integrated energy over given volume.
        
        Args:
            volume: Integration volume (m³)
            
        Returns:
            Total energy (J)
        """
        # Default implementation using typical shell volume
        return -1e-12  # Placeholder
    
    def validate_parameters(self) -> bool:
        """
        Validate source parameters for physical consistency.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        return True  # Default implementation

class GhostCondensateEFT(EnergySource):
    """
    Ghost/Phantom Effective Field Theory negative energy source.
    
    Based on Discovery 21 optimal parameters:
    M=1000, α=0.01, β=0.1, achieving -1.418×10⁻¹² W ANEC violation
    """
    
    def __init__(self, M: float = 1000, alpha: float = 0.01, beta: float = 0.1,
                 R0: float = 5.0, sigma: float = 0.2, mu_polymer: float = 0.1):
        """
        Initialize Ghost EFT source with polymer field theory integration.
        
        Args:
            M: Ghost mass scale (Discovery 21 optimal: 1000)
            alpha: Coupling parameter α (Discovery 21 optimal: 0.01)
            beta: Coupling parameter β (Discovery 21 optimal: 0.1)
            R0: Characteristic bubble radius (m)
            sigma: Gaussian width parameter (m)
            mu_polymer: Polymer scale parameter for QI violations
        """
        parameters = {
            'M': M,
            'alpha': alpha,
            'beta': beta,
            'R0': R0,
            'sigma': sigma,
            'mu_polymer': mu_polymer
        }
        super().__init__("Ghost/Phantom EFT", parameters)
        
        # Store parameters
        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.R0 = R0
        self.sigma = sigma
        self.mu_polymer = mu_polymer
        
        # Discovery 21 peak ANEC violation: -1.418×10⁻¹² W
        self.amplitude = 1.418e-12  # Base amplitude (W)
        
        # Compute polymer-enhanced factors
        self.polymer_enhancement = self._compute_polymer_enhancement()
        
        self._is_initialized = True
    
    def _compute_polymer_enhancement(self) -> float:
        """
        Compute polymer field theory enhancement factor.
        
        Returns:
            Enhancement factor for negative energy generation
        """
        if self.mu_polymer == 0:
            return 1.0
        
        # Use polymer QI bound to determine enhancement
        qi_bound = polymer_QI_bound(self.mu_polymer, tau=1.0)
        
        # Enhancement scales with polymer modification
        # Peak enhancement around μ ≈ 0.5-0.6 from existing analysis
        enhancement = abs(qi_bound) / (1.055e-34 / (12 * np.pi))  # Normalized
        
        return max(1.0, enhancement)
    
    def energy_density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                      t: float = 0.0) -> np.ndarray:
        """
        Compute Ghost EFT energy density with polymer enhancement.
        
        Args:
            x, y, z: Coordinate arrays
            t: Time coordinate
            
        Returns:
            Energy density array (J/m³)
        """
        # Compute radial distance
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Ghost field profile (Gaussian shell around bubble wall)
        spatial_profile = np.exp(-((r - self.R0)**2) / (2 * self.sigma**2))
        
        # Time-dependent oscillation
        omega = 2 * np.pi / 10.0  # 10-second period
        time_profile = np.sin(omega * t)
        
        # Combine spatial and temporal parts
        pi_field = self.amplitude * spatial_profile * time_profile
        
        # Apply polymer field theory modifications
        if self.mu_polymer > 0:
            energy_density = energy_density_polymer(pi_field, self.mu_polymer)
        else:
            energy_density = 0.5 * pi_field**2  # Classical kinetic energy
        
        # Apply EFT corrections based on parameters
        eft_factor = (self.alpha * self.beta) / (1 + (r / self.M)**2)
        
        # Apply polymer enhancement
        energy_density *= -eft_factor * self.polymer_enhancement
        
        return energy_density
    
    def get_warp_profile(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Alcubierre-style warp profile for Ghost EFT.
        
        Args:
            r: Radial coordinate array
            
        Returns:
            Tuple of (f(r), g(r)) for the metric
        """
        # Smooth transition function
        def transition(r, R, sigma):
            return 0.5 * (1 + np.tanh((r - R) / sigma))
        
        # f(r): time component - approaches 1 outside, 0 inside
        f = transition(r, self.R0, self.sigma)
        
        # g(r): radial component - enhanced near bubble wall
        g_enhancement = 1 + self.alpha * np.exp(-((r - self.R0)**2) / (2 * self.sigma**2))
        g = g_enhancement
        
        return f, g
    
    def total_energy(self, volume: float) -> float:
        """
        Compute total Ghost EFT energy including polymer effects.
        
        Args:
            volume: Integration volume (m³)
            
        Returns:
            Total energy (J)
        """
        # Analytical integration for Gaussian shell
        eft_factor = (self.alpha * self.beta) / (1 + (self.R0 / self.M)**2)
        
        # Shell energy: amplitude × volume × enhancement
        shell_volume = 4 * np.pi * self.R0**2 * self.sigma * np.sqrt(2 * np.pi)
        
        total_energy = (-self.amplitude * eft_factor * self.polymer_enhancement * 
                       shell_volume)
        
        return total_energy
    
    def validate_parameters(self) -> bool:
        """
        Validate Ghost EFT parameters.
        
        Returns:
            True if parameters are physically reasonable
        """
        checks = [
            self.M > 0,
            self.alpha > 0,
            self.beta > 0,
            self.R0 > 0,
            self.sigma > 0,
            self.mu_polymer >= 0,
            self.sigma < self.R0,      # Shell width should be < radius
            self.alpha < 1.0,          # Coupling should be perturbative
            self.beta < 1.0,
            self.mu_polymer < 1.0      # Polymer scale should be moderate
        ]
        return all(checks)

class SqueezedVacuumSource(EnergySource):
    """
    Squeezed vacuum state negative energy source using existing framework.
    """
    
    def __init__(self, r_squeeze: float = 1.0, omega: float = 1e12, 
                 cavity_volume: float = 1e-9, R0: float = 5.0):
        """
        Initialize squeezed vacuum source.
        
        Args:
            r_squeeze: Squeezing parameter
            omega: Cavity frequency (rad/s)
            cavity_volume: Cavity volume (m³)
            R0: Bubble radius (m)
        """
        parameters = {
            'r_squeeze': r_squeeze,
            'omega': omega,
            'cavity_volume': cavity_volume,
            'R0': R0
        }
        super().__init__("Squeezed Vacuum", parameters)
        
        self.r_squeeze = r_squeeze
        self.omega = omega
        self.cavity_volume = cavity_volume
        self.R0 = R0
        
        # Compute base negative energy density
        self.base_density = squeezed_vacuum_energy(r_squeeze, omega, cavity_volume)
        
        self._is_initialized = True
    
    def energy_density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                      t: float = 0.0) -> np.ndarray:
        """
        Compute squeezed vacuum energy density.
        
        Args:
            x, y, z: Coordinate arrays
            t: Time coordinate
            
        Returns:
            Energy density array (J/m³)
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Localize negative energy near bubble radius
        profile = np.exp(-((r - self.R0)**2) / (2 * (0.1 * self.R0)**2))
        
        return self.base_density * profile
    
    def get_warp_profile(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate warp profile for squeezed vacuum.
        
        Args:
            r: Radial coordinate array
            
        Returns:
            Tuple of (f(r), g(r)) for the metric
        """
        # Simple profile based on energy localization
        sigma = 0.1 * self.R0
        profile = np.exp(-((r - self.R0)**2) / (2 * sigma**2))
        
        f = 1 - 0.1 * profile  # Small deviation from flat space
        g = 1 + 0.1 * profile
        
        return f, g

class MetamaterialCasimirSource(EnergySource):
    """
    Metamaterial-enhanced Casimir effect negative energy source.
    """
    
    def __init__(self, epsilon: float = -2.0, mu: float = -1.5, 
                 cell_size: float = 50e-9, n_layers: int = 100,
                 R0: float = 5.0, shell_thickness: float = 0.1):
        """
        Initialize metamaterial Casimir source.
        
        Args:
            epsilon: Relative permittivity (negative for metamaterial)
            mu: Relative permeability (negative for metamaterial)
            cell_size: Unit cell size (m)
            n_layers: Number of metamaterial layers
            R0: Shell radius (m)
            shell_thickness: Shell thickness (m)
        """
        parameters = {
            'epsilon': epsilon,
            'mu': mu,
            'cell_size': cell_size,
            'n_layers': n_layers,
            'R0': R0,
            'shell_thickness': shell_thickness
        }
        super().__init__("Metamaterial Casimir", parameters)
        
        # Store parameters
        self.epsilon = epsilon
        self.mu = mu
        self.cell_size = cell_size
        self.n_layers = n_layers
        self.R0 = R0
        self.shell_thickness = shell_thickness
        
        # Enhanced Casimir energy density from vacuum engineering
        self.base_density = 1e-6  # J/m³ from existing analysis
        self.enhancement = abs(epsilon * mu) * np.sqrt(n_layers)
        
        self._is_initialized = True
    
    def energy_density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                      t: float = 0.0) -> np.ndarray:
        """
        Compute metamaterial Casimir energy density.
        
        Args:
            x, y, z: Coordinate arrays
            t: Time coordinate
            
        Returns:
            Energy density array (J/m³)
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Shell profile: negative energy only within shell thickness
        r_shell = np.abs(r - self.R0)
        in_shell = r_shell <= self.shell_thickness
        
        # Parabolic profile within shell
        profile = np.zeros_like(r)
        shell_coord = r_shell[in_shell] / self.shell_thickness
        profile[in_shell] = 1.0 - shell_coord**2
        
        # Apply metamaterial enhancement
        energy_density = -self.base_density * self.enhancement * profile
        
        return energy_density
    
    def get_warp_profile(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate warp profile for metamaterial source.
        
        Args:
            r: Radial coordinate array
            
        Returns:
            Tuple of (f(r), g(r)) for the metric
        """
        # Shell-based profile
        r_shell = np.abs(r - self.R0)
        in_shell = r_shell <= self.shell_thickness
        
        shell_strength = 0.01 * self.enhancement / 10.0  # Normalized
        
        f = np.ones_like(r)
        g = np.ones_like(r)
        
        f[in_shell] = 1 - shell_strength
        g[in_shell] = 1 + shell_strength
        
        return f, g

def create_energy_source(source_type: str, **kwargs) -> EnergySource:
    """
    Factory function to create energy sources compatible with warp-bubble-optimizer.
    
    Args:
        source_type: Type of source ('ghost', 'squeezed', 'metamaterial')
        **kwargs: Parameters for the specific source type
        
    Returns:
        Initialized energy source
    """
    if source_type.lower() in ['ghost', 'ghost_eft', 'phantom']:
        return GhostCondensateEFT(**kwargs)
    elif source_type.lower() in ['squeezed', 'vacuum']:
        return SqueezedVacuumSource(**kwargs)
    elif source_type.lower() in ['metamaterial', 'casimir', 'meta']:
        return MetamaterialCasimirSource(**kwargs)
    else:
        raise ValueError(f"Unknown energy source type: {source_type}")

# Integration with existing warp_qft framework
def integrate_with_existing_solver(energy_source: EnergySource, 
                                 r_array: np.ndarray) -> Dict[str, Any]:
    """
    Integrate energy source with existing warp bubble analysis tools.
    
    Args:
        energy_source: Energy source to analyze
        r_array: Radial coordinate array
        
    Returns:
        Dictionary with analysis results
    """
    # Create coordinate meshes
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 100)
    R, THETA, PHI = np.meshgrid(r_array, theta, phi, indexing='ij')
    
    # Convert to Cartesian
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    
    # Compute energy density
    energy_density = energy_source.energy_density(X, Y, Z, t=0.0)
    
    # Get warp profiles
    f_profile, g_profile = energy_source.get_warp_profile(r_array)
    
    # Integrate total energy
    dr = r_array[1] - r_array[0] if len(r_array) > 1 else 1.0
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    
    # Volume element: r² sin(θ) dr dθ dφ
    volume_element = R**2 * np.sin(THETA) * dr * dtheta * dphi
    total_energy = np.sum(energy_density * volume_element)
    
    return {
        'energy_density': energy_density,
        'f_profile': f_profile,
        'g_profile': g_profile,
        'total_energy': total_energy,
        'coordinates': (X, Y, Z),
        'source_info': energy_source.__dict__
    }

# Example usage and validation
if __name__ == "__main__":
    # Test Discovery 21 Ghost EFT with optimal parameters
    ghost = GhostCondensateEFT(M=1000, alpha=0.01, beta=0.1, mu_polymer=0.1)
    print(f"Ghost EFT Info: {ghost.__dict__}")
    
    # Test integration with existing framework
    r_test = np.linspace(0.1, 10, 100)
    results = integrate_with_existing_solver(ghost, r_test)
    
    print(f"Total Energy: {results['total_energy']:.2e} J")
    print(f"Energy density range: [{results['energy_density'].min():.2e}, {results['energy_density'].max():.2e}] J/m³")
    print(f"Warp profile f range: [{results['f_profile'].min():.3f}, {results['f_profile'].max():.3f}]")
    print(f"Warp profile g range: [{results['g_profile'].min():.3f}, {results['g_profile'].max():.3f}]")
