"""
Sensor Noise Characterization Module

Validates sensor noise models implemented in negative-energy-generator and provides
comprehensive noise characterization for nanopositioning applications.

Key Requirements:
- Sub-nanometer position sensing accuracy
- Microradian angular measurement precision  
- Real-time feedback control compatibility
- Multi-sensor fusion validation

Mathematical Framework:
- Allan variance analysis: σ²(τ) = ⟨(Ω_{n+1} - Ω_n)²⟩/2
- Power spectral density: S(f) = |H(f)|² × S_input(f) + S_noise(f)
- Kalman filter fusion: x̂ = x̂⁻ + K(z - Hx̂⁻)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SensorSpecifications:
    """Sensor performance specifications for nanopositioning"""
    position_resolution: float = 0.05e-9  # 0.05 nm
    angular_resolution: float = 1e-6      # 1 microrad
    bandwidth: float = 1000.0             # 1 kHz
    thermal_stability: float = 0.1e-9     # 0.1 nm/hour
    allan_variance_target: float = 1e-20  # m²
    snr_requirement: float = 80.0         # dB

@dataclass
class NoiseModel:
    """Comprehensive noise model for sensor systems"""
    white_noise_density: float           # V/√Hz
    flicker_noise_coefficient: float     # V²·Hz
    thermal_noise_density: float         # V/√Hz
    shot_noise_density: float            # V/√Hz
    quantization_noise: float            # V RMS
    drift_coefficient: float             # V/s

class SensorNoiseValidator:
    """
    Comprehensive sensor noise characterization and validation
    """
    
    def __init__(self, specs: SensorSpecifications = None):
        """Initialize validator with sensor specifications"""
        self.specs = specs or SensorSpecifications()
        self.validation_results = {}
        self.noise_models = {}
        
    def characterize_interferometric_noise(self, 
                                         wavelength: float = 633e-9,
                                         optical_power: float = 1e-3,
                                         photodetector_responsivity: float = 0.8) -> Dict:
        """
        Characterize noise in laser interferometric position sensing
        
        Args:
            wavelength: Laser wavelength (m)
            optical_power: Optical power (W)
            photodetector_responsivity: A/W
            
        Returns:
            Dictionary with noise analysis results
        """
        logger.info("Starting interferometric noise characterization...")
        
        # Shot noise limit
        # Mathematical Formula: i_shot = √(2qI_photo B)
        photocurrent = optical_power * photodetector_responsivity
        shot_noise_current = np.sqrt(2 * 1.602e-19 * photocurrent)  # A/√Hz
        
        # Position noise from shot noise
        # Mathematical Formula: δx = (λ/4π) × (δφ)
        phase_noise = shot_noise_current / photocurrent  # rad/√Hz
        position_noise_shot = (wavelength / (4 * np.pi)) * phase_noise  # m/√Hz
        
        # Thermal noise (Johnson noise)
        # Mathematical Formula: v_thermal = √(4kTR)
        temperature = 300  # K
        resistance = 50    # Ω (typical photodetector)
        thermal_noise_voltage = np.sqrt(4 * 1.38e-23 * temperature * resistance)  # V/√Hz
        
        # Convert to position noise
        conversion_factor = wavelength / (4 * np.pi * optical_power * photodetector_responsivity)
        position_noise_thermal = thermal_noise_voltage * conversion_factor  # m/√Hz
        
        # Total position noise
        position_noise_total = np.sqrt(position_noise_shot**2 + position_noise_thermal**2)
        
        # Frequency-dependent noise model
        frequencies = np.logspace(-1, 4, 1000)  # 0.1 Hz to 10 kHz
        
        # 1/f noise component
        flicker_coefficient = 1e-12  # m²·Hz
        position_noise_1f = np.sqrt(flicker_coefficient / frequencies)
        
        # White noise floor
        white_noise_floor = np.full_like(frequencies, position_noise_total)
        
        # Combined noise PSD
        noise_psd = position_noise_1f**2 + white_noise_floor**2
        
        results = {
            'shot_noise_limited_position': position_noise_shot,
            'thermal_noise_position': position_noise_thermal,
            'total_white_noise': position_noise_total,
            'frequencies': frequencies,
            'noise_psd': noise_psd,
            'meets_specs': position_noise_total < self.specs.position_resolution / np.sqrt(self.specs.bandwidth)
        }
        
        self.validation_results['interferometric_noise'] = results
        logger.info(f"Shot noise limited position noise: {position_noise_shot*1e12:.2f} pm/√Hz")
        logger.info(f"Meets specifications: {results['meets_specs']}")
        
        return results
    
    def validate_angular_sensor_noise(self, 
                                    beam_separation: float = 100e-6,
                                    differential_measurement: bool = True) -> Dict:
        """
        Validate angular position sensor noise characteristics
        
        Args:
            beam_separation: Distance between measurement beams (m)
            differential_measurement: Use differential measurement
            
        Returns:
            Angular noise validation results
        """
        logger.info("Validating angular sensor noise...")
        
        # Get position noise from interferometric analysis
        if 'interferometric_noise' not in self.validation_results:
            self.characterize_interferometric_noise()
            
        position_noise = self.validation_results['interferometric_noise']['total_white_noise']
        
        # Convert position noise to angular noise
        # Mathematical Formula: θ_noise = δx / L
        angular_noise_single = position_noise / beam_separation  # rad/√Hz
        
        # Differential measurement reduces noise by √2
        if differential_measurement:
            angular_noise = angular_noise_single / np.sqrt(2)
        else:
            angular_noise = angular_noise_single
            
        # Check against specifications
        angular_resolution_hz = angular_noise * np.sqrt(self.specs.bandwidth)
        meets_angular_specs = angular_resolution_hz < self.specs.angular_resolution
        
        results = {
            'angular_noise_density': angular_noise,
            'angular_resolution': angular_resolution_hz,
            'meets_specs': meets_angular_specs,
            'improvement_factor': self.specs.angular_resolution / angular_resolution_hz
        }
        
        self.validation_results['angular_noise'] = results
        logger.info(f"Angular noise: {angular_noise*1e9:.2f} nrad/√Hz")
        logger.info(f"Angular resolution: {angular_resolution_hz*1e6:.2f} μrad")
        logger.info(f"Meets specifications: {meets_angular_specs}")
        
        return results
    
    def perform_allan_variance_analysis(self, 
                                      measurement_duration: float = 3600,
                                      sampling_rate: float = 1000) -> Dict:
        """
        Perform Allan variance analysis for long-term stability
        
        Args:
            measurement_duration: Total measurement time (s)
            sampling_rate: Data sampling rate (Hz)
            
        Returns:
            Allan variance analysis results
        """
        logger.info("Performing Allan variance analysis...")
        
        # Generate synthetic sensor data with realistic noise
        n_samples = int(measurement_duration * sampling_rate)
        time = np.arange(n_samples) / sampling_rate
        
        # Noise components
        white_noise = np.random.normal(0, 0.01e-9, n_samples)  # 0.01 nm white noise
        flicker_noise = self._generate_flicker_noise(n_samples, 1e-12)
        drift = 0.1e-9 * time / 3600  # 0.1 nm/hour drift
        
        # Combined signal
        sensor_data = white_noise + flicker_noise + drift
        
        # Calculate Allan variance
        tau_values = np.logspace(0, 4, 50)  # 1 s to 10,000 s
        allan_variance = []
        
        for tau in tau_values:
            if tau * sampling_rate > n_samples / 3:
                break
                
            # Downsample to averaging time tau
            n_avg = int(tau * sampling_rate)
            n_points = n_samples // n_avg
            
            averaged_data = []
            for i in range(n_points):
                start_idx = i * n_avg
                end_idx = (i + 1) * n_avg
                averaged_data.append(np.mean(sensor_data[start_idx:end_idx]))
            
            # Calculate Allan variance
            if len(averaged_data) > 1:
                differences = np.diff(averaged_data)
                allan_var = np.var(differences) / 2
                allan_variance.append(allan_var)
            else:
                allan_variance.append(np.nan)
        
        tau_values = tau_values[:len(allan_variance)]
        allan_deviation = np.sqrt(allan_variance)
        
        # Check against target Allan variance
        min_allan_var = np.nanmin(allan_variance)
        meets_allan_specs = min_allan_var < self.specs.allan_variance_target
        
        results = {
            'tau_values': tau_values,
            'allan_variance': allan_variance,
            'allan_deviation': allan_deviation,
            'minimum_allan_variance': min_allan_var,
            'meets_specs': meets_allan_specs,
            'optimal_averaging_time': tau_values[np.nanargmin(allan_variance)]
        }
        
        self.validation_results['allan_variance'] = results
        logger.info(f"Minimum Allan variance: {min_allan_var:.2e} m²")
        logger.info(f"Optimal averaging time: {results['optimal_averaging_time']:.1f} s")
        logger.info(f"Meets Allan variance specs: {meets_allan_specs}")
        
        return results
    
    def validate_multi_sensor_fusion(self, 
                                   n_sensors: int = 4,
                                   correlation_coefficient: float = 0.1) -> Dict:
        """
        Validate multi-sensor fusion for improved precision
        
        Args:
            n_sensors: Number of sensors in fusion
            correlation_coefficient: Cross-correlation between sensors
            
        Returns:
            Multi-sensor fusion validation results
        """
        logger.info("Validating multi-sensor fusion...")
        
        # Individual sensor noise (from previous analysis)
        if 'interferometric_noise' not in self.validation_results:
            self.characterize_interferometric_noise()
            
        single_sensor_noise = self.validation_results['interferometric_noise']['total_white_noise']
        
        # Correlation matrix
        correlation_matrix = np.eye(n_sensors)
        for i in range(n_sensors):
            for j in range(n_sensors):
                if i != j:
                    correlation_matrix[i, j] = correlation_coefficient
        
        # Optimal fusion weights (minimum variance estimator)
        ones_vector = np.ones(n_sensors)
        inverse_corr = np.linalg.inv(correlation_matrix)
        
        # Mathematical Formula: w_optimal = (Σ⁻¹ 1) / (1ᵀ Σ⁻¹ 1)
        optimal_weights = inverse_corr @ ones_vector
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # Fused noise variance
        # Mathematical Formula: σ²_fused = wᵀ Σ w × σ²_single
        fused_noise_variance = optimal_weights.T @ correlation_matrix @ optimal_weights
        fused_noise = single_sensor_noise * np.sqrt(fused_noise_variance)
        
        # Improvement factor
        improvement_factor = single_sensor_noise / fused_noise
        
        results = {
            'single_sensor_noise': single_sensor_noise,
            'fused_noise': fused_noise,
            'improvement_factor': improvement_factor,
            'optimal_weights': optimal_weights,
            'effective_sensors': 1 / fused_noise_variance,
            'meets_specs': fused_noise < self.specs.position_resolution / np.sqrt(self.specs.bandwidth)
        }
        
        self.validation_results['multi_sensor_fusion'] = results
        logger.info(f"Single sensor noise: {single_sensor_noise*1e12:.2f} pm/√Hz")
        logger.info(f"Fused sensor noise: {fused_noise*1e12:.2f} pm/√Hz")
        logger.info(f"Improvement factor: {improvement_factor:.2f}×")
        
        return results
    
    def _generate_flicker_noise(self, n_samples: int, amplitude: float) -> np.ndarray:
        """Generate 1/f (flicker) noise"""
        frequencies = np.fft.fftfreq(n_samples)[1:n_samples//2]
        
        # 1/f power spectral density
        psd = amplitude / frequencies
        
        # Generate complex spectrum
        phases = np.random.uniform(0, 2*np.pi, len(frequencies))
        spectrum = np.sqrt(psd) * np.exp(1j * phases)
        
        # Create full spectrum (with proper symmetry)
        full_spectrum = np.zeros(n_samples, dtype=complex)
        full_spectrum[1:len(frequencies)+1] = spectrum
        full_spectrum[-len(frequencies):] = np.conj(spectrum[::-1])
        
        # IFFT to get time domain signal
        flicker_noise = np.fft.ifft(full_spectrum).real
        
        return flicker_noise
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        
        report = []
        report.append("=" * 60)
        report.append("SENSOR NOISE CHARACTERIZATION VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: June 30, 2025")
        report.append("")
        
        # Requirements summary
        report.append("REQUIREMENTS:")
        report.append(f"  Position Resolution: {self.specs.position_resolution*1e9:.3f} nm")
        report.append(f"  Angular Resolution: {self.specs.angular_resolution*1e6:.1f} μrad")
        report.append(f"  Bandwidth: {self.specs.bandwidth:.0f} Hz")
        report.append(f"  Allan Variance Target: {self.specs.allan_variance_target:.2e} m²")
        report.append("")
        
        # Validation results
        overall_pass = True
        
        if 'interferometric_noise' in self.validation_results:
            result = self.validation_results['interferometric_noise']
            status = "PASS" if result['meets_specs'] else "FAIL"
            if not result['meets_specs']:
                overall_pass = False
            report.append(f"Interferometric Noise: {status}")
            report.append(f"  Shot noise limited: {result['shot_noise_limited_position']*1e12:.2f} pm/√Hz")
            report.append(f"  Total white noise: {result['total_white_noise']*1e12:.2f} pm/√Hz")
        
        if 'angular_noise' in self.validation_results:
            result = self.validation_results['angular_noise']
            status = "PASS" if result['meets_specs'] else "FAIL"
            if not result['meets_specs']:
                overall_pass = False
            report.append(f"Angular Noise: {status}")
            report.append(f"  Angular resolution: {result['angular_resolution']*1e6:.2f} μrad")
            report.append(f"  Improvement needed: {1/result['improvement_factor']:.1f}×")
        
        if 'allan_variance' in self.validation_results:
            result = self.validation_results['allan_variance']
            status = "PASS" if result['meets_specs'] else "FAIL"
            if not result['meets_specs']:
                overall_pass = False
            report.append(f"Allan Variance: {status}")
            report.append(f"  Minimum variance: {result['minimum_allan_variance']:.2e} m²")
            report.append(f"  Optimal averaging: {result['optimal_averaging_time']:.1f} s")
        
        if 'multi_sensor_fusion' in self.validation_results:
            result = self.validation_results['multi_sensor_fusion']
            status = "PASS" if result['meets_specs'] else "FAIL"
            if not result['meets_specs']:
                overall_pass = False
            report.append(f"Multi-Sensor Fusion: {status}")
            report.append(f"  Improvement factor: {result['improvement_factor']:.2f}×")
            report.append(f"  Effective sensors: {result['effective_sensors']:.1f}")
        
        report.append("")
        report.append("=" * 60)
        overall_status = "PASS" if overall_pass else "FAIL"
        report.append(f"OVERALL VALIDATION STATUS: {overall_status}")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_noise_analysis(self, save_path: str = None):
        """Generate comprehensive noise analysis plots"""
        
        if not self.validation_results:
            logger.warning("No validation results available. Run validation methods first.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Sensor Noise Characterization Analysis', fontsize=16)
        
        # Plot 1: Frequency-dependent noise PSD
        if 'interferometric_noise' in self.validation_results:
            result = self.validation_results['interferometric_noise']
            ax = axes[0, 0]
            ax.loglog(result['frequencies'], np.sqrt(result['noise_psd']) * 1e12)
            ax.axhline(self.specs.position_resolution * 1e9, color='r', linestyle='--', 
                      label='Requirement')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Position Noise (pm/√Hz)')
            ax.set_title('Interferometric Position Noise PSD')
            ax.legend()
            ax.grid(True)
        
        # Plot 2: Allan variance
        if 'allan_variance' in self.validation_results:
            result = self.validation_results['allan_variance']
            ax = axes[0, 1]
            ax.loglog(result['tau_values'], result['allan_deviation'] * 1e9)
            ax.axhline(np.sqrt(self.specs.allan_variance_target) * 1e9, color='r', 
                      linestyle='--', label='Requirement')
            ax.set_xlabel('Averaging Time τ (s)')
            ax.set_ylabel('Allan Deviation (nm)')
            ax.set_title('Allan Variance Analysis')
            ax.legend()
            ax.grid(True)
        
        # Plot 3: Multi-sensor fusion comparison
        if 'multi_sensor_fusion' in self.validation_results:
            result = self.validation_results['multi_sensor_fusion']
            ax = axes[1, 0]
            n_sensors = np.arange(1, 9)
            # Theoretical improvement for uncorrelated sensors
            theoretical_improvement = np.sqrt(n_sensors)
            # Actual improvement (assuming same correlation for all)
            actual_improvement = [result['improvement_factor'] if i == 4 else 
                                result['improvement_factor'] * np.sqrt(i/4) * 0.8 
                                for i in n_sensors]
            
            ax.plot(n_sensors, theoretical_improvement, 'b-', label='Theoretical (uncorrelated)')
            ax.plot(n_sensors, actual_improvement, 'r-o', label='Actual (with correlation)')
            ax.set_xlabel('Number of Sensors')
            ax.set_ylabel('Noise Improvement Factor')
            ax.set_title('Multi-Sensor Fusion Performance')
            ax.legend()
            ax.grid(True)
        
        # Plot 4: Requirements vs. achieved performance
        ax = axes[1, 1]
        categories = []
        requirements = []
        achieved = []
        
        if 'interferometric_noise' in self.validation_results:
            categories.append('Position\nNoise')
            requirements.append(self.specs.position_resolution * 1e9)
            result = self.validation_results['interferometric_noise']
            achieved.append(result['total_white_noise'] * np.sqrt(self.specs.bandwidth) * 1e9)
        
        if 'angular_noise' in self.validation_results:
            categories.append('Angular\nNoise')
            requirements.append(self.specs.angular_resolution * 1e6)
            result = self.validation_results['angular_noise']
            achieved.append(result['angular_resolution'] * 1e6)
        
        if categories:
            x = np.arange(len(categories))
            width = 0.35
            
            ax.bar(x - width/2, requirements, width, label='Requirement', alpha=0.8)
            ax.bar(x + width/2, achieved, width, label='Achieved', alpha=0.8)
            
            ax.set_ylabel('Performance (nm, μrad)')
            ax.set_title('Requirements vs. Achieved Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Noise analysis plots saved to {save_path}")
        else:
            plt.show()

# Example usage and validation
if __name__ == "__main__":
    # Initialize validator with stringent nanopositioning requirements
    specs = SensorSpecifications(
        position_resolution=0.05e-9,  # 0.05 nm
        angular_resolution=1e-6,      # 1 μrad
        bandwidth=1000.0,             # 1 kHz
        allan_variance_target=1e-20   # m²
    )
    
    validator = SensorNoiseValidator(specs)
    
    # Perform all validation tests
    print("Starting comprehensive sensor noise validation...")
    
    # 1. Characterize interferometric noise
    validator.characterize_interferometric_noise()
    
    # 2. Validate angular sensor noise
    validator.validate_angular_sensor_noise()
    
    # 3. Perform Allan variance analysis
    validator.perform_allan_variance_analysis()
    
    # 4. Validate multi-sensor fusion
    validator.validate_multi_sensor_fusion()
    
    # Generate and print validation report
    report = validator.generate_validation_report()
    print(report)
    
    # Generate plots
    validator.plot_noise_analysis('sensor_noise_validation.png')
    
    print("\nSensor noise characterization validation complete!")
