"""
Thermal Stability Modeling Module

Critical for sub-nanometer positioning systems. Models thermal effects on:
- Mechanical structure expansion/contraction
- Optical path length variations  
- Sensor drift and noise characteristics
- Active thermal compensation systems

Mathematical Framework:
- Thermal expansion: ΔL = α·L₀·ΔT
- Heat conduction: ∂T/∂t = α∇²T + Q/(ρc)
- Thermal noise: σ²_thermal = 4kTR/Δf
- PID control: u(t) = Kp·e(t) + Ki∫e(t)dt + Kd·de(t)/dt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize, signal
from scipy.linalg import solve_continuous_are
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThermalSpecifications:
    """Thermal stability requirements for nanopositioning"""
    max_temperature_drift: float = 0.01      # K
    max_thermal_expansion: float = 0.1e-9    # 0.1 nm
    thermal_time_constant: float = 300       # s
    ambient_temperature: float = 293.15      # K (20°C)
    temperature_stability: float = 0.001     # K over 1 hour
    thermal_conductivity: float = 400        # W/(m·K) for aluminum
    specific_heat: float = 900               # J/(kg·K)
    density: float = 2700                    # kg/m³

@dataclass
class MaterialProperties:
    """Material thermal properties"""
    thermal_expansion_coeff: float          # 1/K
    thermal_conductivity: float             # W/(m·K)
    specific_heat: float                    # J/(kg·K)
    density: float                          # kg/m³
    young_modulus: float                    # Pa
    poisson_ratio: float                    # dimensionless

class ThermalStabilityValidator:
    """
    Comprehensive thermal stability modeling and validation
    """
    
    def __init__(self, specs: ThermalSpecifications = None):
        """Initialize with thermal specifications"""
        self.specs = specs or ThermalSpecifications()
        self.validation_results = {}
        self.material_properties = self._get_default_materials()
        
    def _get_default_materials(self) -> Dict[str, MaterialProperties]:
        """Get default material properties for common nanopositioning materials"""
        return {
            'aluminum': MaterialProperties(
                thermal_expansion_coeff=23e-6,    # 1/K
                thermal_conductivity=237,         # W/(m·K)
                specific_heat=897,               # J/(kg·K)
                density=2700,                    # kg/m³
                young_modulus=70e9,              # Pa
                poisson_ratio=0.33
            ),
            'invar': MaterialProperties(
                thermal_expansion_coeff=1.2e-6,   # 1/K (ultra-low expansion)
                thermal_conductivity=13.8,       # W/(m·K)
                specific_heat=515,               # J/(kg·K)
                density=8100,                    # kg/m³
                young_modulus=141e9,             # Pa
                poisson_ratio=0.26
            ),
            'zerodur': MaterialProperties(
                thermal_expansion_coeff=0.05e-6,  # 1/K (near-zero expansion)
                thermal_conductivity=1.46,       # W/(m·K)
                specific_heat=821,               # J/(kg·K)
                density=2530,                    # kg/m³
                young_modulus=90.3e9,            # Pa
                poisson_ratio=0.24
            ),
            'silicon': MaterialProperties(
                thermal_expansion_coeff=2.6e-6,   # 1/K
                thermal_conductivity=148,         # W/(m·K)
                specific_heat=712,               # J/(kg·K)
                density=2329,                    # kg/m³
                young_modulus=170e9,             # Pa
                poisson_ratio=0.22
            )
        }
    
    def analyze_thermal_expansion(self, 
                                structure_length: float = 0.1,
                                material: str = 'aluminum',
                                temperature_range: float = 1.0) -> Dict:
        """
        Analyze thermal expansion effects on structure
        
        Args:
            structure_length: Length of structure (m)
            material: Material type
            temperature_range: Temperature variation range (K)
            
        Returns:
            Thermal expansion analysis results
        """
        logger.info("Analyzing thermal expansion effects...")
        
        if material not in self.material_properties:
            raise ValueError(f"Material {material} not in database")
            
        mat_props = self.material_properties[material]
        
        # Calculate thermal expansion
        # Mathematical Formula: ΔL = α·L₀·ΔT
        thermal_expansion = (mat_props.thermal_expansion_coeff * 
                           structure_length * temperature_range)
        
        # Position error due to thermal expansion
        position_error = thermal_expansion / 2  # Assuming center-mounted sensor
        
        # Temperature stability required for nanometer precision
        max_temp_change = self.specs.max_thermal_expansion / (
            mat_props.thermal_expansion_coeff * structure_length)
        
        # Thermal stress analysis
        # Mathematical Formula: σ = E·α·ΔT (for constrained expansion)
        thermal_stress = (mat_props.young_modulus * 
                         mat_props.thermal_expansion_coeff * temperature_range)
        
        # Check against specifications
        meets_expansion_specs = thermal_expansion < self.specs.max_thermal_expansion
        
        results = {
            'material': material,
            'thermal_expansion': thermal_expansion,
            'position_error': position_error,
            'max_temperature_change': max_temp_change,
            'thermal_stress': thermal_stress,
            'meets_specs': meets_expansion_specs,
            'safety_margin': self.specs.max_thermal_expansion / thermal_expansion
        }
        
        self.validation_results[f'thermal_expansion_{material}'] = results
        logger.info(f"Thermal expansion ({material}): {thermal_expansion*1e9:.3f} nm")
        logger.info(f"Max temperature change: {max_temp_change*1000:.2f} mK")
        logger.info(f"Meets specifications: {meets_expansion_specs}")
        
        return results
    
    def model_heat_conduction(self, 
                            geometry: str = 'plate',
                            dimensions: Tuple[float, ...] = (0.1, 0.05, 0.01),
                            material: str = 'aluminum',
                            heat_source: float = 1.0) -> Dict:
        """
        Model heat conduction and temperature distribution
        
        Args:
            geometry: Geometry type ('plate', 'cylinder', 'sphere')
            dimensions: Geometry dimensions (m)
            material: Material type
            heat_source: Heat source power (W)
            
        Returns:
            Heat conduction analysis results
        """
        logger.info("Modeling heat conduction...")
        
        if material not in self.material_properties:
            raise ValueError(f"Material {material} not in database")
            
        mat_props = self.material_properties[material]
        
        # Thermal diffusivity
        # Mathematical Formula: α = k/(ρ·c)
        thermal_diffusivity = (mat_props.thermal_conductivity / 
                             (mat_props.density * mat_props.specific_heat))
        
        if geometry == 'plate':
            length, width, thickness = dimensions
            volume = length * width * thickness
            surface_area = 2 * (length * width + length * thickness + width * thickness)
            
            # Characteristic length for thermal time constant
            char_length = volume / surface_area
            
        elif geometry == 'cylinder':
            radius, height = dimensions
            volume = np.pi * radius**2 * height
            surface_area = 2 * np.pi * radius * (radius + height)
            char_length = radius
            
        elif geometry == 'sphere':
            radius = dimensions[0]
            volume = (4/3) * np.pi * radius**3
            surface_area = 4 * np.pi * radius**2
            char_length = radius
            
        else:
            raise ValueError(f"Geometry {geometry} not supported")
        
        # Thermal time constant
        # Mathematical Formula: τ = L²/α
        thermal_time_constant = char_length**2 / thermal_diffusivity
        
        # Steady-state temperature rise
        # Assuming convective cooling: Q = h·A·ΔT
        convective_coeff = 10  # W/(m²·K) - natural convection in air
        steady_temp_rise = heat_source / (convective_coeff * surface_area)
        
        # Temperature distribution (simplified 1D analysis)
        x = np.linspace(0, char_length, 100)
        time_points = np.array([0.1, 0.5, 1.0, 2.0, 5.0]) * thermal_time_constant
        
        # Analytical solution for semi-infinite plate with constant heat flux
        temperature_profiles = []
        for t in time_points:
            if t > 0:
                temp_profile = steady_temp_rise * (
                    1 - np.exp(-x**2 / (4 * thermal_diffusivity * t))
                )
            else:
                temp_profile = np.zeros_like(x)
            temperature_profiles.append(temp_profile)
        
        # Maximum temperature gradient
        max_temp_gradient = steady_temp_rise / char_length
        
        results = {
            'geometry': geometry,
            'material': material,
            'thermal_diffusivity': thermal_diffusivity,
            'thermal_time_constant': thermal_time_constant,
            'steady_temp_rise': steady_temp_rise,
            'max_temp_gradient': max_temp_gradient,
            'x_positions': x,
            'time_points': time_points,
            'temperature_profiles': temperature_profiles,
            'meets_time_specs': thermal_time_constant < self.specs.thermal_time_constant,
            'meets_temp_specs': steady_temp_rise < self.specs.max_temperature_drift
        }
        
        self.validation_results[f'heat_conduction_{geometry}_{material}'] = results
        logger.info(f"Thermal time constant: {thermal_time_constant:.1f} s")
        logger.info(f"Steady-state temperature rise: {steady_temp_rise:.3f} K")
        logger.info(f"Thermal diffusivity: {thermal_diffusivity*1e6:.2f} mm²/s")
        
        return results
    
    def design_thermal_compensation(self, 
                                  material: str = 'aluminum',
                                  structure_length: float = 0.1,
                                  target_stability: float = 0.1e-9) -> Dict:
        """
        Design active thermal compensation system
        
        Args:
            material: Structural material
            structure_length: Length requiring compensation (m)
            target_stability: Target position stability (m)
            
        Returns:
            Thermal compensation design results
        """
        logger.info("Designing thermal compensation system...")
        
        if material not in self.material_properties:
            raise ValueError(f"Material {material} not in database")
            
        mat_props = self.material_properties[material]
        
        # Required temperature stability
        # Mathematical Formula: ΔT_max = Δx_target / (α·L)
        required_temp_stability = target_stability / (
            mat_props.thermal_expansion_coeff * structure_length)
        
        # PID controller design for temperature regulation
        # Plant model: first-order thermal system
        # G(s) = K / (τs + 1)
        thermal_gain = 1.0  # K/W
        thermal_time_constant = 100.0  # s
        
        # PID tuning using pole placement
        desired_settling_time = 30.0  # s
        desired_overshoot = 0.05     # 5%
        
        # Calculate PID gains
        zeta = -np.log(desired_overshoot) / np.sqrt(np.pi**2 + np.log(desired_overshoot)**2)
        omega_n = 4 / (zeta * desired_settling_time)
        
        # PID gains for thermal system
        Kp = (2 * zeta * omega_n * thermal_time_constant - 1) / thermal_gain
        Ki = omega_n**2 * thermal_time_constant / thermal_gain
        Kd = (thermal_time_constant) / thermal_gain
        
        # Actuator requirements
        max_power_density = 1000  # W/m² - typical heating element
        required_power = required_temp_stability * mat_props.thermal_conductivity
        heater_area = required_power / max_power_density
        
        # Sensor requirements
        temperature_sensor_resolution = required_temp_stability / 10  # 10× margin
        
        # Closed-loop performance prediction
        # Transfer function of controlled system
        s = 1j * np.logspace(-3, 2, 1000)  # Frequency response
        
        # Plant transfer function
        G_plant = thermal_gain / (thermal_time_constant * s + 1)
        
        # PID controller transfer function
        G_controller = Kp + Ki / s + Kd * s
        
        # Closed-loop transfer function
        G_closed = (G_controller * G_plant) / (1 + G_controller * G_plant)
        
        # Calculate closed-loop properties
        magnitude = np.abs(G_closed)
        phase = np.angle(G_closed)
        
        # Find bandwidth (3dB frequency)
        mag_db = 20 * np.log10(magnitude)
        bandwidth_idx = np.argmax(mag_db < -3)
        bandwidth = np.imag(s[bandwidth_idx]) / (2 * np.pi)
        
        # Disturbance rejection
        disturbance_rejection = np.abs(1 / (1 + G_controller * G_plant))
        
        results = {
            'material': material,
            'required_temp_stability': required_temp_stability,
            'pid_gains': {'Kp': Kp, 'Ki': Ki, 'Kd': Kd},
            'heater_area': heater_area,
            'required_power': required_power,
            'sensor_resolution': temperature_sensor_resolution,
            'bandwidth': bandwidth,
            'frequencies': np.imag(s) / (2 * np.pi),
            'closed_loop_magnitude': magnitude,
            'disturbance_rejection': disturbance_rejection,
            'meets_specs': required_temp_stability < self.specs.temperature_stability
        }
        
        self.validation_results[f'thermal_compensation_{material}'] = results
        logger.info(f"Required temperature stability: {required_temp_stability*1000:.3f} mK")
        logger.info(f"PID gains - Kp: {Kp:.2f}, Ki: {Ki:.4f}, Kd: {Kd:.2f}")
        logger.info(f"System bandwidth: {bandwidth:.2f} Hz")
        logger.info(f"Heater area required: {heater_area*1e4:.2f} cm²")
        
        return results
    
    def validate_environmental_isolation(self, 
                                       ambient_variation: float = 5.0,
                                       isolation_efficiency: float = 0.99) -> Dict:
        """
        Validate environmental thermal isolation
        
        Args:
            ambient_variation: Ambient temperature variation (K)
            isolation_efficiency: Thermal isolation efficiency (0-1)
            
        Returns:
            Environmental isolation validation results
        """
        logger.info("Validating environmental thermal isolation...")
        
        # Transmitted temperature variation
        transmitted_variation = ambient_variation * (1 - isolation_efficiency)
        
        # Time constant of isolation system
        # Assuming multi-layer insulation with thermal mass
        insulation_thermal_mass = 1000  # J/K
        insulation_thermal_resistance = 100  # K/W
        isolation_time_constant = insulation_thermal_mass * insulation_thermal_resistance
        
        # Frequency response of isolation
        frequencies = np.logspace(-6, 0, 1000)  # 1 μHz to 1 Hz
        omega = 2 * np.pi * frequencies
        
        # First-order low-pass filter model
        # Mathematical Formula: H(jω) = 1 / (1 + jωτ)
        isolation_transfer_function = 1 / (1 + 1j * omega * isolation_time_constant)
        isolation_magnitude = np.abs(isolation_transfer_function)
        
        # Effective isolation for different time scales
        daily_variation_freq = 1 / (24 * 3600)  # 1 day
        hourly_variation_freq = 1 / 3600        # 1 hour
        
        daily_isolation = isolation_efficiency * np.abs(
            1 / (1 + 1j * 2 * np.pi * daily_variation_freq * isolation_time_constant))
        hourly_isolation = isolation_efficiency * np.abs(
            1 / (1 + 1j * 2 * np.pi * hourly_variation_freq * isolation_time_constant))
        
        # Resulting temperature variations
        daily_temp_variation = ambient_variation * (1 - daily_isolation)
        hourly_temp_variation = ambient_variation * (1 - hourly_isolation)
        
        # Check against specifications
        meets_daily_specs = daily_temp_variation < self.specs.max_temperature_drift
        meets_hourly_specs = hourly_temp_variation < self.specs.temperature_stability
        
        results = {
            'ambient_variation': ambient_variation,
            'isolation_efficiency': isolation_efficiency,
            'transmitted_variation': transmitted_variation,
            'isolation_time_constant': isolation_time_constant,
            'frequencies': frequencies,
            'isolation_magnitude': isolation_magnitude,
            'daily_temp_variation': daily_temp_variation,
            'hourly_temp_variation': hourly_temp_variation,
            'daily_isolation': daily_isolation,
            'hourly_isolation': hourly_isolation,
            'meets_daily_specs': meets_daily_specs,
            'meets_hourly_specs': meets_hourly_specs
        }
        
        self.validation_results['environmental_isolation'] = results
        logger.info(f"Daily temperature variation: {daily_temp_variation*1000:.2f} mK")
        logger.info(f"Hourly temperature variation: {hourly_temp_variation*1000:.3f} mK")
        logger.info(f"Isolation time constant: {isolation_time_constant/3600:.1f} hours")
        
        return results
    
    def analyze_thermal_noise_limits(self, 
                                   sensor_resistance: float = 1000,
                                   measurement_bandwidth: float = 1000) -> Dict:
        """
        Analyze fundamental thermal noise limits
        
        Args:
            sensor_resistance: Sensor resistance (Ω)
            measurement_bandwidth: Measurement bandwidth (Hz)
            
        Returns:
            Thermal noise analysis results
        """
        logger.info("Analyzing thermal noise limits...")
        
        # Johnson-Nyquist thermal noise
        # Mathematical Formula: V_rms = √(4kTRΔf)
        boltzmann_constant = 1.38e-23  # J/K
        temperature = self.specs.ambient_temperature
        
        thermal_noise_voltage = np.sqrt(
            4 * boltzmann_constant * temperature * sensor_resistance * measurement_bandwidth
        )
        
        # Current noise for photodetectors
        # Mathematical Formula: i_thermal = √(4kT/(R·Δf))
        thermal_noise_current = np.sqrt(
            4 * boltzmann_constant * temperature / (sensor_resistance * measurement_bandwidth)
        )
        
        # Temperature-dependent resistance variation
        # Typical temperature coefficient: 3900 ppm/K for platinum
        temp_coeff_resistance = 3900e-6  # 1/K
        resistance_variation = (sensor_resistance * temp_coeff_resistance * 
                              self.specs.max_temperature_drift)
        
        # Noise equivalent power (NEP) for thermal detectors
        # Mathematical Formula: NEP = √(4kT²G)
        thermal_conductance = 1e-6  # W/K - typical for microbolometer
        nep_thermal = np.sqrt(4 * boltzmann_constant * temperature**2 * thermal_conductance)
        
        # Fundamental thermal position noise
        # For cantilever-based sensors: x_thermal = √(kT/k_spring)
        spring_constant = 1e-3  # N/m - typical nanopositioning actuator
        thermal_position_noise = np.sqrt(
            boltzmann_constant * temperature / spring_constant
        )
        
        # Check against sensor specifications
        voltage_noise_acceptable = thermal_noise_voltage < 1e-6  # 1 μV threshold
        current_noise_acceptable = thermal_noise_current < 1e-12  # 1 pA threshold
        position_noise_acceptable = thermal_position_noise < 1e-12  # 1 pm threshold
        
        results = {
            'thermal_noise_voltage': thermal_noise_voltage,
            'thermal_noise_current': thermal_noise_current,
            'resistance_variation': resistance_variation,
            'nep_thermal': nep_thermal,
            'thermal_position_noise': thermal_position_noise,
            'voltage_noise_acceptable': voltage_noise_acceptable,
            'current_noise_acceptable': current_noise_acceptable,
            'position_noise_acceptable': position_noise_acceptable,
            'fundamental_limit_ratio': thermal_position_noise / self.specs.max_thermal_expansion
        }
        
        self.validation_results['thermal_noise_limits'] = results
        logger.info(f"Thermal voltage noise: {thermal_noise_voltage*1e9:.2f} nV/√Hz")
        logger.info(f"Thermal position noise: {thermal_position_noise*1e12:.3f} pm")
        logger.info(f"Fundamental limit ratio: {results['fundamental_limit_ratio']:.2e}")
        
        return results
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive thermal validation report"""
        
        report = []
        report.append("=" * 60)
        report.append("THERMAL STABILITY MODELING VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: June 30, 2025")
        report.append("")
        
        # Requirements summary
        report.append("THERMAL REQUIREMENTS:")
        report.append(f"  Max Temperature Drift: {self.specs.max_temperature_drift:.3f} K")
        report.append(f"  Max Thermal Expansion: {self.specs.max_thermal_expansion*1e9:.3f} nm")
        report.append(f"  Temperature Stability: {self.specs.temperature_stability*1000:.3f} mK")
        report.append(f"  Thermal Time Constant: {self.specs.thermal_time_constant:.0f} s")
        report.append("")
        
        # Material comparison
        if any('thermal_expansion' in key for key in self.validation_results.keys()):
            report.append("MATERIAL THERMAL EXPANSION ANALYSIS:")
            for key, result in self.validation_results.items():
                if 'thermal_expansion' in key:
                    material = result['material']
                    expansion = result['thermal_expansion']
                    max_temp = result['max_temperature_change']
                    status = "PASS" if result['meets_specs'] else "FAIL"
                    report.append(f"  {material.capitalize()}: {expansion*1e9:.3f} nm, "
                                f"Max ΔT: {max_temp*1000:.2f} mK - {status}")
        
        # Heat conduction analysis
        heat_conduction_keys = [k for k in self.validation_results.keys() if 'heat_conduction' in k]
        if heat_conduction_keys:
            report.append("\nHEAT CONDUCTION ANALYSIS:")
            for key in heat_conduction_keys:
                result = self.validation_results[key]
                time_const = result['thermal_time_constant']
                temp_rise = result['steady_temp_rise']
                status = "PASS" if (result['meets_time_specs'] and result['meets_temp_specs']) else "FAIL"
                report.append(f"  {result['geometry']}-{result['material']}: "
                            f"τ={time_const:.1f}s, ΔT={temp_rise:.3f}K - {status}")
        
        # Thermal compensation
        compensation_keys = [k for k in self.validation_results.keys() if 'thermal_compensation' in k]
        if compensation_keys:
            report.append("\nTHERMAL COMPENSATION DESIGN:")
            for key in compensation_keys:
                result = self.validation_results[key]
                temp_stability = result['required_temp_stability']
                bandwidth = result['bandwidth']
                status = "PASS" if result['meets_specs'] else "FAIL"
                report.append(f"  {result['material']}: Required stability: "
                            f"{temp_stability*1000:.3f} mK, BW: {bandwidth:.2f} Hz - {status}")
        
        # Environmental isolation
        if 'environmental_isolation' in self.validation_results:
            result = self.validation_results['environmental_isolation']
            daily_var = result['daily_temp_variation']
            hourly_var = result['hourly_temp_variation']
            status = "PASS" if (result['meets_daily_specs'] and result['meets_hourly_specs']) else "FAIL"
            report.append(f"\nENVIRONMENTAL ISOLATION:")
            report.append(f"  Daily variation: {daily_var*1000:.2f} mK")
            report.append(f"  Hourly variation: {hourly_var*1000:.3f} mK - {status}")
        
        # Thermal noise limits
        if 'thermal_noise_limits' in self.validation_results:
            result = self.validation_results['thermal_noise_limits']
            pos_noise = result['thermal_position_noise']
            limit_ratio = result['fundamental_limit_ratio']
            report.append(f"\nTHERMAL NOISE LIMITS:")
            report.append(f"  Fundamental position noise: {pos_noise*1e12:.3f} pm")
            report.append(f"  Limit ratio: {limit_ratio:.2e}")
        
        # Overall assessment
        report.append("")
        report.append("=" * 60)
        overall_pass = all(
            result.get('meets_specs', False) 
            for result in self.validation_results.values()
            if 'meets_specs' in result
        )
        report.append(f"OVERALL VALIDATION STATUS: {'PASS' if overall_pass else 'FAIL'}")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_thermal_analysis(self, save_path: str = None):
        """Generate comprehensive thermal analysis plots"""
        
        if not self.validation_results:
            logger.warning("No validation results available. Run validation methods first.")
            return
        
        fig = plt.figure(figsize=(15, 12))
        
        # Create subplot layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Material thermal expansion comparison
        ax1 = fig.add_subplot(gs[0, 0])
        expansion_results = {k: v for k, v in self.validation_results.items() 
                           if 'thermal_expansion' in k}
        if expansion_results:
            materials = [r['material'] for r in expansion_results.values()]
            expansions = [r['thermal_expansion']*1e9 for r in expansion_results.values()]
            colors = ['red' if not r['meets_specs'] else 'green' for r in expansion_results.values()]
            
            bars = ax1.bar(materials, expansions, color=colors, alpha=0.7)
            ax1.axhline(self.specs.max_thermal_expansion*1e9, color='red', 
                       linestyle='--', label='Requirement')
            ax1.set_ylabel('Thermal Expansion (nm)')
            ax1.set_title('Material Thermal Expansion (1K)')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Heat conduction temperature profiles
        ax2 = fig.add_subplot(gs[0, 1])
        heat_results = {k: v for k, v in self.validation_results.items() 
                       if 'heat_conduction' in k}
        if heat_results:
            # Plot temperature profiles for first result
            result = list(heat_results.values())[0]
            x_pos = result['x_positions'] * 1000  # Convert to mm
            for i, profile in enumerate(result['temperature_profiles']):
                time_label = f"t={result['time_points'][i]:.1f}s"
                ax2.plot(x_pos, profile, label=time_label)
            ax2.set_xlabel('Position (mm)')
            ax2.set_ylabel('Temperature Rise (K)')
            ax2.set_title('Thermal Conduction Profiles')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: PID controller frequency response
        ax3 = fig.add_subplot(gs[0, 2])
        compensation_results = {k: v for k, v in self.validation_results.items() 
                              if 'thermal_compensation' in k}
        if compensation_results:
            result = list(compensation_results.values())[0]
            frequencies = result['frequencies']
            magnitude = result['closed_loop_magnitude']
            disturbance = result['disturbance_rejection']
            
            ax3.semilogx(frequencies, 20*np.log10(magnitude), 'b-', label='Closed Loop')
            ax3.semilogx(frequencies, 20*np.log10(disturbance), 'r--', label='Disturbance Rejection')
            ax3.axhline(-3, color='gray', linestyle=':', alpha=0.7, label='-3dB')
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Magnitude (dB)')
            ax3.set_title('Control System Response')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Environmental isolation performance
        ax4 = fig.add_subplot(gs[1, 0])
        if 'environmental_isolation' in self.validation_results:
            result = self.validation_results['environmental_isolation']
            frequencies = result['frequencies']
            isolation = result['isolation_magnitude']
            
            ax4.loglog(frequencies, isolation)
            ax4.axvline(1/(24*3600), color='r', linestyle='--', alpha=0.7, label='Daily')
            ax4.axvline(1/3600, color='orange', linestyle='--', alpha=0.7, label='Hourly')
            ax4.set_xlabel('Frequency (Hz)')
            ax4.set_ylabel('Isolation Transfer Function')
            ax4.set_title('Environmental Isolation')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Temperature requirements vs. achieved
        ax5 = fig.add_subplot(gs[1, 1])
        categories = ['Max Drift', 'Stability', 'Expansion']
        requirements = [
            self.specs.max_temperature_drift * 1000,
            self.specs.temperature_stability * 1000,
            self.specs.max_thermal_expansion * 1e9
        ]
        
        # Gather achieved values from results
        achieved = []
        if 'environmental_isolation' in self.validation_results:
            achieved.append(self.validation_results['environmental_isolation']['daily_temp_variation'] * 1000)
            achieved.append(self.validation_results['environmental_isolation']['hourly_temp_variation'] * 1000)
        else:
            achieved.extend([0, 0])
            
        expansion_results = {k: v for k, v in self.validation_results.items() 
                           if 'thermal_expansion' in k}
        if expansion_results:
            # Use best material result
            best_expansion = min(r['thermal_expansion'] for r in expansion_results.values())
            achieved.append(best_expansion * 1e9)
        else:
            achieved.append(0)
        
        if len(achieved) == 3:
            x = np.arange(len(categories))
            width = 0.35
            
            ax5.bar(x - width/2, requirements, width, label='Requirement', alpha=0.8)
            ax5.bar(x + width/2, achieved, width, label='Achieved', alpha=0.8)
            
            ax5.set_ylabel('Temperature/Position (mK, nm)')
            ax5.set_title('Requirements vs. Performance')
            ax5.set_xticks(x)
            ax5.set_xticklabels(categories)
            ax5.legend()
            ax5.set_yscale('log')
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Thermal noise analysis
        ax6 = fig.add_subplot(gs[1, 2])
        if 'thermal_noise_limits' in self.validation_results:
            result = self.validation_results['thermal_noise_limits']
            
            noise_types = ['Voltage', 'Current', 'Position']
            noise_values = [
                result['thermal_noise_voltage'] * 1e9,  # nV
                result['thermal_noise_current'] * 1e12,  # pA  
                result['thermal_position_noise'] * 1e12  # pm
            ]
            
            bars = ax6.bar(noise_types, noise_values, alpha=0.7)
            ax6.set_ylabel('Noise Level (nV, pA, pm)')
            ax6.set_title('Fundamental Thermal Noise')
            ax6.set_yscale('log')
            ax6.grid(True, alpha=0.3)
        
        # Plot 7: Time constants comparison
        ax7 = fig.add_subplot(gs[2, :])
        time_constants = {}
        
        # Collect time constants from various analyses
        heat_results = {k: v for k, v in self.validation_results.items() 
                       if 'heat_conduction' in k}
        for key, result in heat_results.items():
            material = result['material']
            time_constants[f"Thermal-{material}"] = result['thermal_time_constant']
            
        compensation_results = {k: v for k, v in self.validation_results.items() 
                              if 'thermal_compensation' in k}
        for key, result in compensation_results.items():
            material = result['material']
            # Control bandwidth to time constant
            bandwidth = result['bandwidth']
            control_time_const = 1 / (2 * np.pi * bandwidth)
            time_constants[f"Control-{material}"] = control_time_const
            
        if 'environmental_isolation' in self.validation_results:
            result = self.validation_results['environmental_isolation']
            time_constants['Isolation'] = result['isolation_time_constant']
        
        if time_constants:
            names = list(time_constants.keys())
            values = list(time_constants.values())
            
            bars = ax7.bar(names, values, alpha=0.7)
            ax7.axhline(self.specs.thermal_time_constant, color='red', 
                       linestyle='--', label='Requirement')
            ax7.set_ylabel('Time Constant (s)')
            ax7.set_title('System Time Constants')
            ax7.legend()
            ax7.tick_params(axis='x', rotation=45)
            ax7.set_yscale('log')
            ax7.grid(True, alpha=0.3)
        
        plt.suptitle('Thermal Stability Analysis', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Thermal analysis plots saved to {save_path}")
        else:
            plt.show()

# Example usage and validation
if __name__ == "__main__":
    # Initialize validator with nanopositioning thermal requirements
    specs = ThermalSpecifications(
        max_temperature_drift=0.01,      # 10 mK
        max_thermal_expansion=0.1e-9,    # 0.1 nm
        thermal_time_constant=300,       # 5 minutes
        temperature_stability=0.001      # 1 mK/hour
    )
    
    validator = ThermalStabilityValidator(specs)
    
    print("Starting comprehensive thermal stability validation...")
    
    # 1. Analyze thermal expansion for different materials
    materials = ['aluminum', 'invar', 'zerodur', 'silicon']
    for material in materials:
        validator.analyze_thermal_expansion(material=material)
    
    # 2. Model heat conduction
    validator.model_heat_conduction(geometry='plate', material='aluminum')
    
    # 3. Design thermal compensation
    validator.design_thermal_compensation(material='invar')
    
    # 4. Validate environmental isolation
    validator.validate_environmental_isolation()
    
    # 5. Analyze thermal noise limits
    validator.analyze_thermal_noise_limits()
    
    # Generate and print validation report
    report = validator.generate_validation_report()
    print(report)
    
    # Generate plots
    validator.plot_thermal_analysis('thermal_stability_validation.png')
    
    print("\nThermal stability modeling validation complete!")
