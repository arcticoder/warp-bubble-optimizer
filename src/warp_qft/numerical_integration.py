#!/usr/bin/env python3
"""
Numerical Integration Utilities for Warp Bubble Calculations

This module provides specialized numerical integration routines for computing
energy integrals, stress-energy tensors, and geometric quantities in warp bubble
spacetimes.
"""

import numpy as np
import time
from scipy.integrate import quad, dblquad, tplquad, simpson
try:
    from scipy.integrate import trapezoid as trapz
except ImportError:
    from scipy.integrate import trapz
from scipy.special import sph_harm, gamma, factorial
from typing import Callable, Tuple, Dict, List, Optional, Any
import warnings

class WarpBubbleIntegrator:
    """
    Specialized numerical integrator for warp bubble calculations.
    """
    
    def __init__(self, 
                 integration_method: str = 'adaptive',
                 tolerance: float = 1e-8,
                 max_subdivisions: int = 50):
        """
        Initialize the integrator.
        
        Args:
            integration_method: 'adaptive', 'simpson', 'trapezoid'
            tolerance: Integration tolerance
            max_subdivisions: Maximum number of subdivisions for adaptive methods
        """
        self.method = integration_method
        self.tolerance = tolerance
        self.max_subdivisions = max_subdivisions
    
    def energy_density_integral(self, 
                               energy_density: Callable,
                               r_range: Tuple[float, float],
                               theta_range: Tuple[float, float] = (0, np.pi),
                               phi_range: Tuple[float, float] = (0, 2*np.pi)) -> float:
        """
        Integrate energy density over spatial volume.
        
        Args:
            energy_density: Function ρ(r, θ, φ) in spherical coordinates
            r_range: Radial integration bounds (r_min, r_max)
            theta_range: Polar angle bounds
            phi_range: Azimuthal angle bounds
            
        Returns:
            Total energy
        """
        def integrand(phi, theta, r):
            """Volume element in spherical coordinates: r² sin(θ) dr dθ dφ"""
            return energy_density(r, theta, phi) * r**2 * np.sin(theta)
        
        if self.method == 'adaptive':
            result, error = tplquad(
                integrand,
                r_range[0], r_range[1],
                lambda r: theta_range[0], lambda r: theta_range[1],
                lambda r, theta: phi_range[0], lambda r, theta: phi_range[1],
                epsabs=self.tolerance
            )
            return result
        else:
            raise NotImplementedError(f"Method {self.method} not implemented for 3D integrals")
    
    def surface_integral(self,
                        surface_density: Callable,
                        radius: float,
                        theta_range: Tuple[float, float] = (0, np.pi),
                        phi_range: Tuple[float, float] = (0, 2*np.pi)) -> float:
        """
        Integrate over spherical surface at fixed radius.
        
        Args:
            surface_density: Function σ(θ, φ) on the sphere
            radius: Sphere radius
            theta_range: Polar angle bounds
            phi_range: Azimuthal angle bounds
            
        Returns:
            Surface integral
        """
        def integrand(phi, theta):
            """Surface element: r² sin(θ) dθ dφ"""
            return surface_density(theta, phi) * radius**2 * np.sin(theta)
        
        result, error = dblquad(
            integrand,
            theta_range[0], theta_range[1],
            lambda theta: phi_range[0], lambda theta: phi_range[1],
            epsabs=self.tolerance
        )
        return result
    
    def radial_integral(self,
                       radial_function: Callable,
                       r_range: Tuple[float, float],
                       weight_function: Optional[Callable] = None) -> float:
        """
        Integrate radial function with optional weight.
        
        Args:
            radial_function: Function f(r)
            r_range: Integration bounds
            weight_function: Optional weight w(r)
            
        Returns:
            Integral value
        """
        if weight_function is None:
            weight_function = lambda r: 1.0
        
        def integrand(r):
            return radial_function(r) * weight_function(r)
        
        if self.method == 'adaptive':
            result, error = quad(
                integrand,
                r_range[0], r_range[1],
                epsabs=self.tolerance,
                limit=self.max_subdivisions
            )
            return result
        elif self.method == 'simpson':
            r_vals = np.linspace(r_range[0], r_range[1], 1001)  # Odd number for Simpson
            y_vals = integrand(r_vals)
            return simpson(y_vals, x=r_vals)
        elif self.method == 'trapezoid':
            r_vals = np.linspace(r_range[0], r_range[1], 1001)
            y_vals = integrand(r_vals)
            return trapz(y_vals, x=r_vals)
        else:
            raise ValueError(f"Unknown integration method: {self.method}")

class EnergyMomentumCalculator:
    """
    Calculate energy-momentum tensor components and derived quantities.
    """
    
    def __init__(self, metric_calculator: Callable):
        """
        Initialize with metric calculation function.
        
        Args:
            metric_calculator: Function that returns metric tensor given coordinates
        """
        self.metric_calculator = metric_calculator
        self.integrator = WarpBubbleIntegrator()
    
    def stress_energy_00(self, coordinates: Tuple) -> float:
        """
        Calculate T^00 component (energy density).
        
        Args:
            coordinates: (t, r, θ, φ) coordinates
            
        Returns:
            Energy density at the point
        """
        # This would use Einstein field equations: G_μν = 8πT_μν
        # Placeholder implementation
        t, r, theta, phi = coordinates
        
        # Get metric at this point
        g = self.metric_calculator(coordinates)
        
        # Calculate Einstein tensor (simplified placeholder)
        # In practice, this requires computing Christoffel symbols, Ricci tensor, etc.
        G_00 = self._calculate_einstein_tensor_component(g, 0, 0)
        
        # T^00 = G^00 / (8π)
        return G_00 / (8 * np.pi)
    
    def _calculate_einstein_tensor_component(self, metric: np.ndarray, mu: int, nu: int) -> float:
        """
        Calculate Einstein tensor component (placeholder).
        
        In a full implementation, this would:
        1. Compute Christoffel symbols from metric derivatives
        2. Calculate Riemann tensor
        3. Contract to get Ricci tensor and scalar
        4. Form Einstein tensor G_μν = R_μν - (1/2)g_μν R
        """
        # Placeholder: return small value for now
        return 1e-10
    
    def total_energy(self, 
                    integration_bounds: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate total energy by integrating T^00.
        
        Args:
            integration_bounds: Dictionary with 'r', 'theta', 'phi' bounds
            
        Returns:
            Total energy
        """
        def energy_density(r, theta, phi):
            coordinates = (0, r, theta, phi)  # t=0 slice
            return self.stress_energy_00(coordinates)
        
        return self.integrator.energy_density_integral(
            energy_density,
            integration_bounds['r'],
            integration_bounds.get('theta', (0, np.pi)),
            integration_bounds.get('phi', (0, 2*np.pi))
        )

class GeometricQuantityCalculator:
    """
    Calculate geometric quantities: curvature, volume, etc.
    """
    
    def __init__(self):
        self.integrator = WarpBubbleIntegrator()
    
    def proper_volume(self,
                     metric_determinant: Callable,
                     bounds: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate proper volume using metric determinant.
        
        Args:
            metric_determinant: Function returning √|g|
            bounds: Integration bounds for each coordinate
            
        Returns:
            Proper volume
        """
        def volume_element(r, theta, phi):
            return metric_determinant(r, theta, phi)
        
        return self.integrator.energy_density_integral(
            volume_element,
            bounds['r'],
            bounds.get('theta', (0, np.pi)),
            bounds.get('phi', (0, 2*np.pi))
        )
    
    def curvature_scalar_integral(self,
                                 ricci_scalar: Callable,
                                 metric_determinant: Callable,
                                 bounds: Dict[str, Tuple[float, float]]) -> float:
        """
        Integrate Ricci scalar over volume.
        
        Args:
            ricci_scalar: Function R(r, θ, φ)
            metric_determinant: Function √|g|(r, θ, φ)
            bounds: Integration bounds
            
        Returns:
            Integrated curvature
        """
        def integrand(r, theta, phi):
            return ricci_scalar(r, theta, phi) * metric_determinant(r, theta, phi)
        
        return self.integrator.energy_density_integral(
            integrand,
            bounds['r'],
            bounds.get('theta', (0, np.pi)),
            bounds.get('phi', (0, 2*np.pi))
        )

class SpecializedIntegrals:
    """
    Specialized integrals for specific warp bubble calculations.
    """
    
    @staticmethod
    def van_den_broeck_energy_integral(R_warp: float, 
                                      sigma: float,
                                      r_max: float = 10.0) -> float:
        """
        Specialized integral for Van den Broeck metric energy.
        
        Args:
            R_warp: Warp bubble radius
            sigma: Wall thickness parameter
            r_max: Maximum integration radius
            
        Returns:
            Energy integral value
        """
        def integrand(r):
            # Van den Broeck shape function derivative
            xs = (r - R_warp) / sigma
            if abs(xs) > 10:  # Avoid overflow
                return 0.0
            
            tanh_xs = np.tanh(xs)
            sech_xs = 1 / np.cosh(xs)
            
            # Energy density ~ (df/dr)²
            df_dr = (1 / sigma) * sech_xs**2
            
            # Volume element in spherical coordinates
            return 4 * np.pi * r**2 * df_dr**2
        
        result, _ = quad(integrand, 0, r_max, epsabs=1e-10)
        return result
    
    @staticmethod
    def lqg_correction_integral(mu_parameter: float,
                               R_warp: float,
                               sigma: float) -> float:
        """
        Integral for LQG polymer corrections.
        
        Args:
            mu_parameter: LQG area parameter
            R_warp: Warp bubble radius
            sigma: Wall thickness
            
        Returns:
            LQG correction factor
        """
        def integrand(r):
            # LQG correction with sinc(πμ) profile
            if mu_parameter * r == 0:
                sinc_factor = 1.0
            else:
                sinc_factor = np.sin(np.pi * mu_parameter * r) / (np.pi * mu_parameter * r)
            
            # Volume element
            return 4 * np.pi * r**2 * sinc_factor**2
        
        result, _ = quad(integrand, 0, 5 * R_warp, epsabs=1e-10)
        return result

def create_energy_calculator(metric_type: str) -> Callable:
    """
    Factory function for creating energy calculators.
    
    Args:
        metric_type: Type of metric ('van_den_broeck', 'natario', 'custom')
        
    Returns:
        Energy calculation function
    """
    if metric_type == 'van_den_broeck':
        def vdb_energy_calculator(params):
            R_warp, sigma = params[:2]
            return SpecializedIntegrals.van_den_broeck_energy_integral(R_warp, sigma)
        return vdb_energy_calculator
        
    elif metric_type == 'custom':
        def custom_energy_calculator(params):
            # Generic energy calculator
            calculator = EnergyMomentumCalculator(lambda coords: np.eye(4))
            bounds = {'r': (0, 10), 'theta': (0, np.pi), 'phi': (0, 2*np.pi)}
            return calculator.total_energy(bounds)
        return custom_energy_calculator
        
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

class SpacetimeIntegrator:
    """
    Specialized 4D spacetime integration for warp bubble calculations.
    
    Provides efficient integration over (t,r,θ,φ) coordinates with
    support for temporal smearing and quantum inequality bounds.
    """
    
    def __init__(self, Nr: int = 128, Nt: int = 128, Ntheta: int = 32, Nphi: int = 64):
        """
        Initialize 4D spacetime integrator.
        
        Args:
            Nr: Radial grid points
            Nt: Temporal grid points  
            Ntheta: Polar angle grid points
            Nphi: Azimuthal angle grid points
        """
        self.Nr = Nr
        self.Nt = Nt
        self.Ntheta = Ntheta
        self.Nphi = Nphi
    
    def integrate_4d_energy(self, energy_density_func: Callable, 
                           R: float, T: float,
                           integration_method: str = "simpson") -> float:
        """
        Integrate energy density over 4D spacetime volume.
        
        ∫∫∫∫ ρ(t,r,θ,φ) * r² * sin(θ) dt dr dθ dφ
        
        Args:
            energy_density_func: Function ρ(t,r,θ,φ)
            R: Maximum radius
            T: Total time duration
            integration_method: "simpson", "trapezoid", or "monte_carlo"
            
        Returns:
            Total integrated energy
        """
        # Create coordinate grids
        t_grid = np.linspace(0, T, self.Nt)
        r_grid = np.linspace(0, R, self.Nr) 
        theta_grid = np.linspace(0, np.pi, self.Ntheta)
        phi_grid = np.linspace(0, 2*np.pi, self.Nphi)
        
        # Grid spacing
        dt = T / (self.Nt - 1) if self.Nt > 1 else T
        dr = R / (self.Nr - 1) if self.Nr > 1 else R
        dtheta = np.pi / (self.Ntheta - 1) if self.Ntheta > 1 else np.pi
        dphi = 2*np.pi / (self.Nphi - 1) if self.Nphi > 1 else 2*np.pi
        
        total_energy = 0.0
        
        if integration_method == "simpson" and all(n >= 3 for n in [self.Nt, self.Nr, self.Ntheta, self.Nphi]):
            # Use Simpson's rule for higher accuracy
            for i, t in enumerate(t_grid):
                for j, r in enumerate(r_grid):
                    # Integrate over angles for fixed (t,r)
                    integrand_2d = np.zeros((self.Ntheta, self.Nphi))
                    
                    for k, theta in enumerate(theta_grid):
                        for l, phi in enumerate(phi_grid):
                            rho = energy_density_func(t, r, theta, phi)
                            jacobian = r**2 * np.sin(theta)
                            integrand_2d[k, l] = rho * jacobian
                    
                    # 2D Simpson integration over angles
                    angular_integral = simpson_2d(integrand_2d, dtheta, dphi)
                    
                    # Weight for Simpson's rule in (t,r)
                    t_weight = simpson_weight(i, self.Nt)
                    r_weight = simpson_weight(j, self.Nr)
                    
                    total_energy += angular_integral * t_weight * r_weight * dt * dr
                    
        else:
            # Fallback to trapezoidal rule
            for t in t_grid:
                for r in r_grid:
                    for theta in theta_grid:
                        for phi in phi_grid:
                            rho = energy_density_func(t, r, theta, phi)
                            jacobian = r**2 * np.sin(theta)
                            total_energy += rho * jacobian * dt * dr * dtheta * dphi
        
        return total_energy
    
    def temporal_averaged_energy(self, energy_density_func: Callable,
                               sampling_func: Callable,
                               R: float, T: float) -> float:
        """
        Compute temporally averaged energy with sampling function.
        
        Used for quantum inequality calculations:
        ∫ ρ(t,r) * f(t) * 4πr² dt dr
        
        Args:
            energy_density_func: ρ(t,r) 
            sampling_func: Temporal sampling f(t)
            R: Maximum radius
            T: Total time
            
        Returns:
            Temporally averaged energy
        """
        t_grid = np.linspace(0, T, self.Nt)
        r_grid = np.linspace(0, R, self.Nr)
        dt = T / (self.Nt - 1) if self.Nt > 1 else T
        dr = R / (self.Nr - 1) if self.Nr > 1 else R
        
        total_energy = 0.0
        
        for t in t_grid:
            f_t = sampling_func(t)
            for r in r_grid:
                rho = energy_density_func(t, r)
                volume_element = 4 * np.pi * r**2
                total_energy += rho * f_t * volume_element * dt * dr
        
        return total_energy
    
    def energy_moments(self, energy_density_func: Callable,
                      R: float, T: float, max_moment: int = 4) -> Dict[int, float]:
        """
        Compute energy moments for stability analysis.
        
        M_n = ∫∫ r^n * |ρ(t,r)| * 4πr² dt dr
        
        Returns:
            Dictionary of moments {n: M_n}
        """
        t_grid = np.linspace(0, T, self.Nt)
        r_grid = np.linspace(0, R, self.Nr)
        dt = T / (self.Nt - 1) if self.Nt > 1 else T
        dr = R / (self.Nr - 1) if self.Nr > 1 else R
        
        moments = {n: 0.0 for n in range(max_moment + 1)}
        
        for t in t_grid:
            for r in r_grid:
                rho = abs(energy_density_func(t, r))
                volume_element = 4 * np.pi * r**2
                
                for n in range(max_moment + 1):
                    moments[n] += (r**n) * rho * volume_element * dt * dr
        
        return moments


def simpson_2d(integrand: np.ndarray, dx: float, dy: float) -> float:
    """
    2D Simpson's rule integration.
    """
    Nx, Ny = integrand.shape
    
    # 1D Simpson weights
    wx = np.ones(Nx)
    wx[1:-1:2] = 4  # Odd indices
    wx[2:-1:2] = 2  # Even indices (except endpoints)
    wx /= 3
    
    wy = np.ones(Ny) 
    wy[1:-1:2] = 4
    wy[2:-1:2] = 2
    wy /= 3
    
    # 2D integration
    result = 0.0
    for i in range(Nx):
        for j in range(Ny):
            result += integrand[i, j] * wx[i] * wy[j]
    
    return result * dx * dy


def simpson_weight(index: int, N: int) -> float:
    """
    Simpson's rule weight for given index.
    """
    if index == 0 or index == N - 1:
        return 1.0 / 3.0
    elif index % 2 == 1:
        return 4.0 / 3.0
    else:
        return 2.0 / 3.0


def quantum_inequality_sampling(t: np.ndarray, tau: float, 
                              sampling_type: str = "gaussian") -> np.ndarray:
    """
    Generate sampling functions for quantum inequality calculations.
    
    Args:
        t: Time array
        tau: Sampling timescale
        sampling_type: "gaussian", "lorentzian", or "exponential"
        
    Returns:
        Sampling function values f(t)
    """
    if sampling_type == "gaussian":
        return np.exp(-t**2 / (2 * tau**2)) / (np.sqrt(2 * np.pi) * tau)
    elif sampling_type == "lorentzian":
        return (tau / np.pi) / (t**2 + tau**2)
    elif sampling_type == "exponential":
        return np.exp(-np.abs(t) / tau) / (2 * tau)
    else:
        raise ValueError(f"Unknown sampling type: {sampling_type}")


def compute_lqg_corrected_energy(energy_classical: float, 
                               mu: float, G_geo: float,
                               field_amplitude: float) -> float:
    """
    Apply LQG corrections to classical energy calculation.
    
    E_LQG = E_classical * (1 + G_geo * sinc(π * μ * φ))
    
    Args:
        energy_classical: Classical energy value
        mu: LQG polymer parameter  
        G_geo: Geometric coupling constant
        field_amplitude: Typical field amplitude
        
    Returns:
        LQG-corrected energy
    """
    sinc_factor = np.sinc(np.pi * mu * field_amplitude)
    correction_factor = 1 + G_geo * sinc_factor
    
    return energy_classical * correction_factor


def benchmark_4d_integration():
    """
    Benchmark 4D spacetime integration performance.
    """
    print("4D Spacetime Integration Benchmark")
    print("=" * 40)
    
    # Test function: Gaussian energy density
    def test_energy_density(t, r, theta=None, phi=None):
        R0, T0 = 1.0, 1000.0
        sigma_r, sigma_t = 0.3, 200.0
        return -np.exp(-(r/sigma_r)**2 - ((t-T0/2)/sigma_t)**2)
    
    # Test different grid sizes
    grid_sizes = [32, 64, 128]
    R, T = 2.0, 2000.0
    
    results = {}
    
    for N in grid_sizes:
        integrator = SpacetimeIntegrator(Nr=N, Nt=N)
        
        start_time = time.time()
        energy = integrator.integrate_4d_energy(test_energy_density, R, T, "trapezoid")
        integration_time = time.time() - start_time
        
        results[N] = {
            'energy': energy,
            'time': integration_time,
            'points': N**4
        }
        
        print(f"Grid {N}×{N}×{N}×{N}: E = {energy:.3e} J, "
              f"Time = {integration_time:.2f} s, "
              f"Rate = {N**4/integration_time:.0f} pts/s")
    
    return results
