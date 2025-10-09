#!/usr/bin/env python3
"""
3D Visualization for Warp Bubble Metrics and Fields
===================================================

This module provides interactive 3D visualization of warp bubble geometries,
stress-energy distributions, and field profiles using PyVista.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import logging

# Visualization imports
try:
    import pyvista as pv
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available - 3D visualization disabled")

# JAX imports with fallback
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp = np
    JAX_AVAILABLE = False

logger = logging.getLogger(__name__)

class WarpBubbleVisualizer:
    """3D visualization of warp bubble metrics and fields."""
    
    def __init__(self, enable_interactive: bool = True):
        """Initialize the visualizer.
        
        Args:
            enable_interactive: Enable interactive plotting
        """
        self.enable_interactive = enable_interactive and PYVISTA_AVAILABLE
        
        if not PYVISTA_AVAILABLE:
            logger.warning("PyVista not available - using fallback 2D plots")
        
        # Custom colormaps
        self.setup_colormaps()
        
    def setup_colormaps(self):
        """Set up custom colormaps for different field types."""
        # Warp factor colormap (blue to red)
        self.warp_cmap = LinearSegmentedColormap.from_list(
            'warp', ['blue', 'cyan', 'white', 'yellow', 'red'])
        
        # Energy density colormap (negative energy in blue/purple)
        self.energy_cmap = LinearSegmentedColormap.from_list(
            'energy', ['darkblue', 'blue', 'white', 'red', 'darkred'])
        
        # Curvature colormap
        self.curvature_cmap = LinearSegmentedColormap.from_list(
            'curvature', ['purple', 'blue', 'white', 'orange', 'red'])
    
    def create_spherical_grid(self, 
                            r_max: float = 10.0, 
                            n_r: int = 50, 
                            n_theta: int = 30, 
                            n_phi: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create spherical coordinate grid.
        
        Args:
            r_max: Maximum radius
            n_r: Number of radial points
            n_theta: Number of polar angle points
            n_phi: Number of azimuthal angle points
            
        Returns:
            Tuple of (r, theta, phi, xyz_points) arrays
        """
        r = np.linspace(0.1, r_max, n_r)
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        
        R, THETA, PHI = np.meshgrid(r, theta, phi, indexing='ij')
        
        # Convert to Cartesian
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)
        
        xyz_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        return R, THETA, PHI, xyz_points
    
    def evaluate_warp_function(self, 
                             r_points: np.ndarray, 
                             warp_func: Callable, 
                             theta: np.ndarray) -> np.ndarray:
        """Evaluate warp function at given points.
        
        Args:
            r_points: Radial coordinates
            warp_func: Warp function f(r, theta)
            theta: Shape parameters
            
        Returns:
            Array of warp function values
        """
        if JAX_AVAILABLE:
            # Use JAX if available
            r_jax = jnp.array(r_points)
            theta_jax = jnp.array(theta)
            f_vals = warp_func(r_jax, theta_jax)
            return np.array(f_vals)
        else:
            # Fallback to NumPy
            return warp_func(r_points, theta)
      def visualize_bubble_3d(self, 
                          warp_func: Callable, 
                          theta: np.ndarray,
                          r_max: float = 5.0,
                          title: str = "Warp Bubble Metric") -> Optional[object]:
        """Create 3D visualization of warp bubble.
        
        Args:
            warp_func: Warp function f(r, theta)
            theta: Shape parameters
            r_max: Maximum visualization radius
            title: Plot title
            
        Returns:
            PyVista plotter object (if available)
        """
        if not PYVISTA_AVAILABLE:
            logger.warning("PyVista not available - cannot create 3D plot")
            return None
        
        # Create spherical surfaces at different radii
        radii = np.linspace(0.5, r_max, 8)
        
        plotter = pv.Plotter()
        plotter.set_background('black')
        
        for i, radius in enumerate(radii):
            # Create sphere
            sphere = pv.Sphere(radius=radius, theta_resolution=60, phi_resolution=60)
            
            # Compute warp function values
            points = sphere.points
            r_vals = np.linalg.norm(points, axis=1)
            f_vals = self.evaluate_warp_function(r_vals, warp_func, theta)
            
            # Add scalar field
            sphere['warp_factor'] = f_vals
            
            # Set opacity based on radius (inner spheres more transparent)
            opacity = 0.3 + 0.7 * (i / len(radii))
            
            # Add to plot
            plotter.add_mesh(sphere, 
                           scalars='warp_factor',
                           cmap=self.warp_cmap,
                           opacity=opacity,
                           show_scalar_bar=(i == len(radii)-1))
        
        # Add coordinate axes
        plotter.add_axes()
        plotter.add_title(title)
        
        if self.enable_interactive:
            plotter.show()
        
        return plotter
    
    def visualize_energy_density(self, 
                                warp_func: Callable, 
                                theta: np.ndarray,
                                r_max: float = 5.0,
                                slice_plane: str = 'xy') -> Optional[pv.Plotter]:
        """Visualize stress-energy density distribution.
        
        Args:
            warp_func: Warp function f(r, theta)
            theta: Shape parameters
            r_max: Maximum visualization radius
            slice_plane: Plane for 2D slice ('xy', 'xz', 'yz')
            
        Returns:
            PyVista plotter object (if available)
        """
        if not PYVISTA_AVAILABLE:
            logger.warning("PyVista not available - cannot create 3D plot")
            return None
        
        # Create 3D grid
        n_points = 100
        x = np.linspace(-r_max, r_max, n_points)
        y = np.linspace(-r_max, r_max, n_points)
        z = np.linspace(-r_max, r_max, n_points)
        
        plotter = pv.Plotter()
        plotter.set_background('white')
        
        # Create slice plane
        if slice_plane == 'xy':
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            slice_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        elif slice_plane == 'xz':
            X, Z = np.meshgrid(x, z)
            Y = np.zeros_like(X)
            slice_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        else:  # yz
            Y, Z = np.meshgrid(y, z)
            X = np.zeros_like(Y)
            slice_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Compute energy density
        r_vals = np.linalg.norm(slice_points, axis=1)
        f_vals = self.evaluate_warp_function(r_vals, warp_func, theta)
        
        # Approximate energy density (simplified)
        # df/dr ≈ (f(r+δr) - f(r-δr)) / (2δr)
        dr = 0.01
        r_plus = r_vals + dr
        r_minus = np.maximum(r_vals - dr, 0.01)
        
        f_plus = self.evaluate_warp_function(r_plus, warp_func, theta)
        f_minus = self.evaluate_warp_function(r_minus, warp_func, theta)
        
        df_dr = (f_plus - f_minus) / (2 * dr)
        
        # Simplified energy density
        energy_density = -(df_dr**2) / (8 * np.pi)
        
        # Create mesh for slice
        if slice_plane == 'xy':
            mesh = pv.StructuredGrid(X, Y, Z)
        elif slice_plane == 'xz':
            mesh = pv.StructuredGrid(X, Y, Z)
        else:
            mesh = pv.StructuredGrid(X, Y, Z)
        
        mesh['energy_density'] = energy_density.reshape(X.shape)
        
        # Add to plot
        plotter.add_mesh(mesh, 
                        scalars='energy_density',
                        cmap=self.energy_cmap,
                        show_scalar_bar=True)
        
        # Add coordinate axes
        plotter.add_axes()
        plotter.add_title(f"Energy Density - {slice_plane.upper()} Plane")
        
        if self.enable_interactive:
            plotter.show()
        
        return plotter
    
    def create_comparison_plot(self, 
                             warp_functions: List[Callable],
                             theta_list: List[np.ndarray],
                             labels: List[str],
                             r_max: float = 5.0) -> None:
        """Create comparison plot of multiple warp functions.
        
        Args:
            warp_functions: List of warp functions
            theta_list: List of parameter arrays
            labels: Labels for each function
            r_max: Maximum radius
        """
        # Radial profile comparison
        r = np.linspace(0.01, r_max, 200)
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Warp factor profiles
        plt.subplot(2, 2, 1)
        for func, theta, label in zip(warp_functions, theta_list, labels):
            f_vals = self.evaluate_warp_function(r, func, theta)
            plt.plot(r, f_vals, label=label, linewidth=2)
        
        plt.xlabel('Radius r')
        plt.ylabel('Warp Factor f(r)')
        plt.title('Warp Factor Profiles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Derivatives (related to energy density)
        plt.subplot(2, 2, 2)
        for func, theta, label in zip(warp_functions, theta_list, labels):
            f_vals = self.evaluate_warp_function(r, func, theta)
            df_dr = np.gradient(f_vals, r[1] - r[0])
            plt.plot(r, df_dr, label=label, linewidth=2)
        
        plt.xlabel('Radius r')
        plt.ylabel('df/dr')
        plt.title('Warp Factor Derivatives')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Energy density
        plt.subplot(2, 2, 3)
        for func, theta, label in zip(warp_functions, theta_list, labels):
            f_vals = self.evaluate_warp_function(r, func, theta)
            df_dr = np.gradient(f_vals, r[1] - r[0])
            energy_density = -(df_dr**2) / (8 * np.pi)
            plt.plot(r, energy_density, label=label, linewidth=2)
        
        plt.xlabel('Radius r')
        plt.ylabel('Energy Density')
        plt.title('Energy Density Profiles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('symlog')
        
        # Plot 4: Integrated negative energy
        plt.subplot(2, 2, 4)
        neg_energies = []
        for func, theta, label in zip(warp_functions, theta_list, labels):
            f_vals = self.evaluate_warp_function(r, func, theta)
            df_dr = np.gradient(f_vals, r[1] - r[0])
            energy_density = -(df_dr**2) / (8 * np.pi)
            
            # Only count negative energy
            neg_density = np.where(energy_density < 0, -energy_density, 0)
            
            # Integrate over spherical shells
            integrand = neg_density * r**2
            integrated = np.cumsum(integrand) * (r[1] - r[0]) * 4 * np.pi
            
            plt.plot(r, integrated, label=label, linewidth=2)
            neg_energies.append(integrated[-1])
        
        plt.xlabel('Radius r')
        plt.ylabel('Cumulative Negative Energy')
        plt.title('Integrated Negative Energy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n📊 WARP FUNCTION COMPARISON:")
        for label, neg_energy in zip(labels, neg_energies):
            print(f"   {label:20s}: Total negative energy = {neg_energy:.3e}")

def demo_visualization():
    """Demonstrate warp bubble visualization capabilities."""
    print("=" * 60)
    print("WARP BUBBLE 3D VISUALIZATION DEMO")
    print("=" * 60)
    
    if not PYVISTA_AVAILABLE:
        print("⚠️  PyVista not available - using 2D plots only")
    
    # Define test warp functions
    def gaussian_warp(r, theta):
        A, sigma = theta
        return 1.0 - A * np.exp(-(r/sigma)**2)
    
    def polynomial_warp(r, theta):
        a, b = theta
        r_norm = r / 5.0
        return 1.0 - a * r_norm**2 - b * r_norm**4
    
    # Parameters for each function
    gaussian_params = np.array([0.8, 1.5])
    polynomial_params = np.array([0.6, 0.3])
    
    # Create visualizer
    visualizer = WarpBubbleVisualizer(enable_interactive=True)
    
    # 3D visualization (if PyVista available)
    if PYVISTA_AVAILABLE:
        print("🎨 Creating 3D bubble visualization...")
        visualizer.visualize_bubble_3d(gaussian_warp, gaussian_params, 
                                     title="Gaussian Warp Bubble")
        
        print("🎨 Creating energy density visualization...")
        visualizer.visualize_energy_density(gaussian_warp, gaussian_params)
    
    # Comparison plots (always available)
    print("📊 Creating comparison plots...")
    visualizer.create_comparison_plot(
        warp_functions=[gaussian_warp, polynomial_warp],
        theta_list=[gaussian_params, polynomial_params],
        labels=['Gaussian', 'Polynomial'],
        r_max=5.0
    )
    
    print("✅ Visualization demo complete!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run demonstration
    demo_visualization()
