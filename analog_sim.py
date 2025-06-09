#!/usr/bin/env python3
"""
Analog Prototype Simulation for Warp Mechanics
==============================================

This module simulates analog systems that exhibit warp-like behavior
using acoustic waves, electromagnetic fields, or fluid dynamics.
These provide testable analogs for warp bubble physics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import logging
from dataclasses import dataclass

# ProgressTracker import with fallback
try:
    from progress_tracker import ProgressTracker
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False
    class ProgressTracker:
        def __init__(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def set_stage(self, *args, **kwargs): pass
        def log_metric(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

class DummyContext:
    """Dummy context manager for fallback."""
    def __enter__(self): return self
    def __exit__(self, *args): pass

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# JAX imports with fallback
try:
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    jnp = np
    JAX_AVAILABLE = False
    def jit(func):
        return func

logger = logging.getLogger(__name__)

@dataclass
class AnalogConfig:
    """Configuration for analog prototype simulation."""
    grid_size: Tuple[int, int] = (200, 200)
    physical_size: Tuple[float, float] = (10.0, 10.0)  # meters
    dt: float = 1e-4  # time step
    wave_speed: float = 343.0  # m/s (sound in air)
    damping: float = 0.01
    source_frequency: float = 1000.0  # Hz

class AcousticWarpAnalog:
    """Acoustic analog of warp bubble using variable wave speed."""
    
    def __init__(self, config: AnalogConfig):
        """Initialize acoustic warp analog.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.nx, self.ny = config.grid_size
        self.dx = config.physical_size[0] / self.nx
        self.dy = config.physical_size[1] / self.ny
        
        # Initialize fields
        self.pressure = np.zeros((self.nx, self.ny))
        self.pressure_prev = np.zeros((self.nx, self.ny))
        self.velocity_x = np.zeros((self.nx, self.ny))
        self.velocity_y = np.zeros((self.nx, self.ny))
        
        # Coordinate grids
        x = np.linspace(0, config.physical_size[0], self.nx)
        y = np.linspace(0, config.physical_size[1], self.ny)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        # Distance from center
        center_x, center_y = config.physical_size[0]/2, config.physical_size[1]/2
        self.R = np.sqrt((self.X - center_x)**2 + (self.Y - center_y)**2)
        
        # Initialize variable wave speed (analog of metric)
        self.setup_variable_wave_speed()
        
        # History for analysis
        self.history = []
        
    def setup_variable_wave_speed(self):
        """Set up spatially varying wave speed (analog of warp metric)."""
        # Base wave speed
        c0 = self.config.wave_speed
        
        # Create "warp bubble" profile
        # Higher speed in center = analog of space contraction
        # Lower speed at edges = analog of space expansion
        r_warp = 2.0  # warp bubble radius
        
        # Gaussian profile for speed variation
        speed_factor = 1.0 + 0.5 * np.exp(-(self.R / r_warp)**2)
        
        # Ensure stability (CFL condition)
        max_speed = np.max(speed_factor * c0)
        cfl_limit = 0.5 * min(self.dx, self.dy) / (max_speed * self.config.dt)
        
        if cfl_limit < 1.0:
            logger.warning(f"CFL condition violated: {cfl_limit:.3f} < 1.0")
            # Reduce time step to maintain stability
            self.config.dt *= cfl_limit * 0.9
            logger.info(f"Adjusted time step to: {self.config.dt:.2e}")
        
        self.wave_speed = speed_factor * c0
        
    def add_source(self, t: float, source_x: float, source_y: float):
        """Add acoustic source at specified location.
        
        Args:
            t: Current time
            source_x: Source x-coordinate
            source_y: Source y-coordinate
        """
        # Find nearest grid point
        i = int(source_x / self.dx)
        j = int(source_y / self.dy)
        
        if 0 <= i < self.nx and 0 <= j < self.ny:
            # Sinusoidal source
            amplitude = 0.1
            source_value = amplitude * np.sin(2 * np.pi * self.config.source_frequency * t)
            self.pressure[i, j] += source_value
    
    def step(self, t: float) -> np.ndarray:
        """Advance simulation by one time step.
        
        Args:
            t: Current time
            
        Returns:
            Current pressure field
        """
        dt = self.config.dt
        dx, dy = self.dx, self.dy
        
        # Compute spatial derivatives using finite differences
        # ∂p/∂x
        dp_dx = np.zeros_like(self.pressure)
        dp_dx[1:-1, :] = (self.pressure[2:, :] - self.pressure[:-2, :]) / (2 * dx)
        
        # ∂p/∂y
        dp_dy = np.zeros_like(self.pressure)
        dp_dy[:, 1:-1] = (self.pressure[:, 2:] - self.pressure[:, :-2]) / (2 * dy)
        
        # Update velocities
        # ∂vx/∂t = -∂p/∂x
        self.velocity_x -= dt * dp_dx
        
        # ∂vy/∂t = -∂p/∂y
        self.velocity_y -= dt * dp_dy
        
        # Compute divergence of velocity
        # ∂vx/∂x
        dvx_dx = np.zeros_like(self.velocity_x)
        dvx_dx[1:-1, :] = (self.velocity_x[2:, :] - self.velocity_x[:-2, :]) / (2 * dx)
        
        # ∂vy/∂y
        dvy_dy = np.zeros_like(self.velocity_y)
        dvy_dy[:, 1:-1] = (self.velocity_y[:, 2:] - self.velocity_y[:, :-2]) / (2 * dy)
        
        # Update pressure using variable wave speed
        # ∂p/∂t = -c²∇·v
        div_v = dvx_dx + dvy_dy
        pressure_new = self.pressure - dt * (self.wave_speed**2) * div_v
        
        # Apply damping
        pressure_new *= (1.0 - self.config.damping * dt)
        
        # Add sources
        self.add_source(t, self.config.physical_size[0]/2, self.config.physical_size[1]/4)
        
        # Apply boundary conditions (absorbing)
        pressure_new[0, :] = 0
        pressure_new[-1, :] = 0
        pressure_new[:, 0] = 0
        pressure_new[:, -1] = 0
        
        # Update fields        self.pressure_prev = self.pressure.copy()
        self.pressure = pressure_new
        
        return self.pressure.copy()
    
    def run_simulation(self, duration: float, save_interval: float = 0.01) -> Dict:
        """Run the complete simulation.
        
        Args:
            duration: Simulation duration in seconds
            save_interval: Time interval for saving snapshots
            
        Returns:
            Dictionary with simulation results
        """
        n_steps = int(duration / self.config.dt)
        save_every = int(save_interval / self.config.dt)
        
        snapshots = []
        times = []
        
        logger.info(f"Running acoustic simulation: {n_steps} steps, {duration:.3f}s")
        
        # Initialize progress tracking
        progress = None
        if PROGRESS_AVAILABLE:
            try:
                progress = ProgressTracker(
                    total_iterations=n_steps,
                    description="Acoustic Warp Analog Simulation",
                    log_level=logging.INFO
                )
                progress.set_stage("acoustic_wave_evolution")
            except Exception as e:
                logger.warning(f"Failed to initialize ProgressTracker: {e}")
                progress = None
        
        with progress if progress else DummyContext():
            for step in range(n_steps):
                t = step * self.config.dt
                pressure_field = self.step(t)
                
                # Update progress
                if progress and step % 100 == 0:  # Update every 100 steps to avoid overhead
                    try:
                        progress.update(100 if step > 0 else 1)
                        if step % (n_steps // 10) == 0:  # Log metrics every 10%
                            max_pressure = np.max(np.abs(pressure_field))
                            progress.log_metric("max_pressure", max_pressure)
                            progress.log_metric("simulation_time", t)
                    except Exception as e:
                        logger.warning(f"Progress update failed: {e}")
                
                # Save snapshots
                if step % save_every == 0:
                    snapshots.append(pressure_field.copy())
                    times.append(t)
                    
                    # Progress report
                    if len(snapshots) % 10 == 0:
                        max_pressure = np.max(np.abs(pressure_field))
                        logger.info(f"t={t:.3f}s, max pressure={max_pressure:.3e}")
        
        results = {
            'snapshots': snapshots,
            'times': times,
            'wave_speed_map': self.wave_speed.copy(),
            'final_pressure': self.pressure.copy(),
            'final_velocity': (self.velocity_x.copy(), self.velocity_y.copy()),
            'grid_x': self.X.copy(),
            'grid_y': self.Y.copy(),
            'config': self.config
        }
        
        logger.info("Acoustic simulation complete")
        return results

class ElectromagneticWarpAnalog:
    """Electromagnetic analog using metamaterial with variable permittivity."""
    
    def __init__(self, config: AnalogConfig):
        """Initialize EM warp analog.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.nx, self.ny = config.grid_size
        self.dx = config.physical_size[0] / self.nx
        self.dy = config.physical_size[1] / self.ny
        
        # EM fields (2D TM mode: Ez, Hx, Hy)
        self.Ez = np.zeros((self.nx, self.ny))
        self.Hx = np.zeros((self.nx, self.ny))
        self.Hy = np.zeros((self.nx, self.ny))
        
        # Coordinate grids
        x = np.linspace(0, config.physical_size[0], self.nx)
        y = np.linspace(0, config.physical_size[1], self.ny)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        # Distance from center
        center_x, center_y = config.physical_size[0]/2, config.physical_size[1]/2
        self.R = np.sqrt((self.X - center_x)**2 + (self.Y - center_y)**2)
        
        # Material properties (vacuum constants with modifications)
        self.setup_metamaterial_properties()
        
    def setup_metamaterial_properties(self):
        """Set up spatially varying material properties."""
        # Vacuum values
        eps0 = 8.854e-12  # F/m
        mu0 = 4e-7 * np.pi  # H/m
        
        # Create "metamaterial" with varying permittivity
        # This creates effective "warp" in EM wave propagation
        r_meta = 1.5  # metamaterial radius
        
        # Variable permittivity (analog of metric)
        eps_factor = 1.0 + 2.0 * np.exp(-(self.R / r_meta)**2)
        
        self.epsilon = eps_factor * eps0
        self.mu = mu0 * np.ones_like(self.epsilon)
        
        # Wave speed in metamaterial
        self.c_local = 1.0 / np.sqrt(self.epsilon * self.mu)
        
    def add_em_source(self, t: float, source_x: float, source_y: float):
        """Add electromagnetic source.
        
        Args:
            t: Current time
            source_x: Source x-coordinate
            source_y: Source y-coordinate
        """
        i = int(source_x / self.dx)
        j = int(source_y / self.dy)
        
        if 0 <= i < self.nx and 0 <= j < self.ny:
            amplitude = 1e-3
            self.Ez[i, j] += amplitude * np.sin(2 * np.pi * self.config.source_frequency * t)
    
    def step(self, t: float) -> np.ndarray:
        """Advance EM simulation by one time step.
        
        Args:
            t: Current time
            
        Returns:
            Current Ez field
        """
        dt = self.config.dt
        dx, dy = self.dx, self.dy
        
        # Update magnetic field components
        # ∂Hx/∂t = -(1/μ) ∂Ez/∂y
        dEz_dy = np.zeros_like(self.Ez)
        dEz_dy[:, 1:-1] = (self.Ez[:, 2:] - self.Ez[:, :-2]) / (2 * dy)
        self.Hx -= dt * dEz_dy / self.mu
        
        # ∂Hy/∂t = (1/μ) ∂Ez/∂x
        dEz_dx = np.zeros_like(self.Ez)
        dEz_dx[1:-1, :] = (self.Ez[2:, :] - self.Ez[:-2, :]) / (2 * dx)
        self.Hy += dt * dEz_dx / self.mu
        
        # Update electric field
        # ∂Ez/∂t = (1/ε) (∂Hy/∂x - ∂Hx/∂y)
        dHy_dx = np.zeros_like(self.Hy)
        dHy_dx[1:-1, :] = (self.Hy[2:, :] - self.Hy[:-2, :]) / (2 * dx)
        
        dHx_dy = np.zeros_like(self.Hx)
        dHx_dy[:, 1:-1] = (self.Hx[:, 2:] - self.Hx[:, :-2]) / (2 * dy)
        
        curl_H = dHy_dx - dHx_dy
        self.Ez += dt * curl_H / self.epsilon
        
        # Apply damping
        self.Ez *= (1.0 - self.config.damping * dt)
        self.Hx *= (1.0 - self.config.damping * dt)
        self.Hy *= (1.0 - self.config.damping * dt)
        
        # Add source
        self.add_em_source(t, self.config.physical_size[0]/2, self.config.physical_size[1]/4)
        
        # Boundary conditions (PEC - perfect electric conductor)
        self.Ez[0, :] = 0
        self.Ez[-1, :] = 0
        self.Ez[:, 0] = 0
        self.Ez[:, -1] = 0
        
        return self.Ez.copy()
    
    def run_simulation(self, duration: float, save_interval: float = 0.01) -> Dict:
        """Run the EM simulation.
        
        Args:
            duration: Simulation duration
            save_interval: Time interval for saving snapshots
            
        Returns:
            Dictionary with simulation results        """
        n_steps = int(duration / self.config.dt)
        save_every = int(save_interval / self.config.dt)
        
        snapshots = []
        times = []
        
        logger.info(f"Running EM simulation: {n_steps} steps, {duration:.3f}s")
        
        # Initialize progress tracking
        progress = None
        if PROGRESS_AVAILABLE:
            try:
                progress = ProgressTracker(
                    total_iterations=n_steps,
                    description="EM Warp Analog Simulation",
                    log_level=logging.INFO
                )
                progress.set_stage("em_field_evolution")
            except Exception as e:
                logger.warning(f"Failed to initialize ProgressTracker: {e}")
                progress = None
        
        with progress if progress else DummyContext():
            for step in range(n_steps):
                t = step * self.config.dt
                ez_field = self.step(t)
                
                # Update progress
                if progress and step % 100 == 0:
                    try:
                        progress.update(100 if step > 0 else 1)
                        if step % (n_steps // 10) == 0:
                            max_field = np.max(np.abs(ez_field))
                            progress.log_metric("max_field", max_field)
                            progress.log_metric("simulation_time", t)
                    except Exception as e:
                        logger.warning(f"Progress update failed: {e}")
                
                if step % save_every == 0:
                    snapshots.append(ez_field.copy())
                    times.append(t)
                    
                    if len(snapshots) % 10 == 0:
                        max_field = np.max(np.abs(ez_field))
                        logger.info(f"t={t:.3f}s, max field={max_field:.3e}")
        
        results = {
            'snapshots': snapshots,
            'times': times,
            'permittivity_map': self.epsilon.copy(),
            'wave_speed_map': self.c_local.copy(),
            'final_Ez': self.Ez.copy(),
            'final_H': (self.Hx.copy(), self.Hy.copy()),
            'grid_x': self.X.copy(),
            'grid_y': self.Y.copy(),
            'config': self.config
        }
        
        logger.info("EM simulation complete")
        return results

class AnalogVisualization:
    """Visualization tools for analog simulations."""
    
    @staticmethod
    def animate_field(results: Dict, 
                     field_name: str = "pressure",
                     save_path: Optional[str] = None) -> None:
        """Create animation of field evolution.
        
        Args:
            results: Simulation results dictionary
            field_name: Name of field to animate
            save_path: Path to save animation (optional)
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available - cannot create animation")
            return
        
        snapshots = results['snapshots']
        times = results['times']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up plot
        vmax = np.max([np.max(np.abs(snap)) for snap in snapshots])
        vmin = -vmax
        
        im = ax.imshow(snapshots[0].T, origin='lower', 
                      vmin=vmin, vmax=vmax, cmap='RdBu_r')
        ax.set_title(f'{field_name.capitalize()} Field Evolution')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(field_name.capitalize())
        
        # Time text
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                           bbox=dict(boxstyle="round", facecolor='wheat'))
        
        def animate(frame):
            im.set_array(snapshots[frame].T)
            time_text.set_text(f'Time: {times[frame]:.4f} s')
            return [im, time_text]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(snapshots),
                                     interval=50, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
            logger.info(f"Animation saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_warp_effect_analysis(results: Dict) -> None:
        """Analyze and plot warp effects in analog simulation.
        
        Args:
            results: Simulation results dictionary
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available - cannot create plots")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Wave speed map
        ax = axes[0, 0]
        if 'wave_speed_map' in results:
            im1 = ax.imshow(results['wave_speed_map'].T, origin='lower', cmap='viridis')
            ax.set_title('Wave Speed Distribution')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im1, ax=ax, label='Wave Speed (m/s)')
        
        # Plot 2: Final field distribution
        ax = axes[0, 1]
        final_field = results['snapshots'][-1]
        im2 = ax.imshow(final_field.T, origin='lower', cmap='RdBu_r')
        ax.set_title('Final Field Distribution')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im2, ax=ax, label='Field Amplitude')
        
        # Plot 3: Radial profile of wave speed
        ax = axes[1, 0]
        if 'wave_speed_map' in results:
            center_x, center_y = results['wave_speed_map'].shape[0]//2, results['wave_speed_map'].shape[1]//2
            speed_profile = results['wave_speed_map'][center_x, :]
            y_coords = np.linspace(0, results['config'].physical_size[1], len(speed_profile))
            ax.plot(y_coords, speed_profile, 'b-', linewidth=2)
            ax.set_xlabel('Distance from Center (m)')
            ax.set_ylabel('Wave Speed (m/s)')
            ax.set_title('Radial Wave Speed Profile')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Energy propagation analysis
        ax = axes[1, 1]
        # Compute energy density over time at center
        center_energies = []
        for snapshot in results['snapshots']:
            center_i, center_j = snapshot.shape[0]//2, snapshot.shape[1]//2
            energy = snapshot[center_i, center_j]**2
            center_energies.append(energy)
        
        ax.plot(results['times'], center_energies, 'r-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy Density')
        ax.set_title('Energy at Warp Center')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def demo_analog_simulation():
    """Demonstrate analog warp simulations."""
    print("=" * 60)
    print("ANALOG WARP PROTOTYPE SIMULATION DEMO")
    print("=" * 60)
    
    # Configuration
    config = AnalogConfig(
        grid_size=(100, 100),
        physical_size=(5.0, 5.0),
        dt=5e-6,
        wave_speed=343.0,
        damping=0.02,
        source_frequency=2000.0
    )
    
    # Run acoustic analog
    print("🔊 Running acoustic warp analog simulation...")
    acoustic_sim = AcousticWarpAnalog(config)
    acoustic_results = acoustic_sim.run_simulation(duration=0.01, save_interval=0.001)
    
    # Run EM analog
    print("⚡ Running electromagnetic warp analog simulation...")
    em_config = AnalogConfig(
        grid_size=(100, 100),
        physical_size=(0.1, 0.1),  # Smaller scale for EM
        dt=1e-12,
        wave_speed=3e8,
        damping=0.01,
        source_frequency=1e10  # 10 GHz
    )
    em_sim = ElectromagneticWarpAnalog(em_config)
    em_results = em_sim.run_simulation(duration=1e-9, save_interval=1e-11)
    
    # Visualization
    if MATPLOTLIB_AVAILABLE:
        print("📊 Creating visualizations...")
        
        viz = AnalogVisualization()
        
        # Analyze acoustic results
        print("   Acoustic warp effects...")
        viz.plot_warp_effect_analysis(acoustic_results)
        
        # Analyze EM results
        print("   EM warp effects...")
        viz.plot_warp_effect_analysis(em_results)
        
        # Create animations (optional)
        # viz.animate_field(acoustic_results, "pressure")
        # viz.animate_field(em_results, "electric field")
    
    print("✅ Analog simulation demo complete!")
    
    return acoustic_results, em_results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run demonstration
    demo_analog_simulation()
