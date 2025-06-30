"""
Vibration Isolation Verification Module

Essential for microradian parallelism in nanopositioning systems. Analyzes:
- Multi-stage passive isolation systems
- Active vibration control feedback
- Ground motion transmission characteristics
- Resonance frequency optimization

Mathematical Framework:
- Transmissibility: T(ω) = |X_out/X_in| = 1/√[(1-r²)² + (2ζr)²]
- Isolation efficiency: η = 1 - T(ω)
- Multi-stage: T_total = ∏T_i
- Active control: H(s) = K(s)/(1 + K(s)G(s))
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate, optimize
from scipy.linalg import solve_continuous_are
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VibrationSpecifications:
    """Vibration isolation requirements for nanopositioning"""
    max_displacement: float = 0.1e-9         # 0.1 nm RMS
    max_angular_displacement: float = 1e-6   # 1 μrad RMS
    isolation_frequency: float = 1.0         # Hz (isolation above this)
    ground_motion_amplitude: float = 1e-6    # 1 μm RMS typical lab
    required_isolation: float = 1e4          # 10,000× at 10 Hz
    resonance_frequency: float = 0.5         # Hz (isolation stage)
    damping_ratio: float = 0.7               # Critical damping ratio

@dataclass
class VibrationType:
    """Classification of vibration sources"""
    frequency_range: Tuple[float, float]     # Hz
    typical_amplitude: float                 # m RMS
    source_description: str
    mitigation_strategy: str

class VibrationIsolationValidator:
    """
    Comprehensive vibration isolation verification and validation
    """
    
    def __init__(self, specs: VibrationSpecifications = None):
        """Initialize with vibration specifications"""
        self.specs = specs or VibrationSpecifications()
        self.validation_results = {}
        self.vibration_sources = self._get_vibration_sources()
        
    def _get_vibration_sources(self) -> Dict[str, VibrationType]:
        """Define common vibration sources in laboratory environments"""
        return {
            'building_sway': VibrationType(
                frequency_range=(0.1, 1.0),
                typical_amplitude=10e-6,  # 10 μm
                source_description="Building structural resonances, wind",
                mitigation_strategy="Long-period passive isolation"
            ),
            'footsteps': VibrationType(
                frequency_range=(1.0, 10.0),
                typical_amplitude=1e-6,   # 1 μm
                source_description="Human activity, walking",
                mitigation_strategy="Multi-stage passive + active control"
            ),
            'hvac_systems': VibrationType(
                frequency_range=(10.0, 100.0),
                typical_amplitude=0.1e-6, # 0.1 μm
                source_description="Air handling, pumps, fans",
                mitigation_strategy="Active isolation + acoustic enclosure"
            ),
            'machinery': VibrationType(
                frequency_range=(50.0, 500.0),
                typical_amplitude=0.05e-6, # 0.05 μm
                source_description="Laboratory equipment, motors",
                mitigation_strategy="Active control + vibration pads"
            ),
            'acoustic': VibrationType(
                frequency_range=(100.0, 10000.0),
                typical_amplitude=0.01e-6, # 0.01 μm
                source_description="Sound waves, acoustic coupling",
                mitigation_strategy="Acoustic isolation enclosure"
            )
        }
    
    def analyze_passive_isolation(self, 
                                 n_stages: int = 3,
                                 stage_params: List[Dict] = None) -> Dict:
        """
        Analyze multi-stage passive vibration isolation
        
        Args:
            n_stages: Number of isolation stages
            stage_params: List of stage parameters [mass, stiffness, damping]
            
        Returns:
            Passive isolation analysis results
        """
        logger.info("Analyzing passive vibration isolation...")
        
        # Default stage parameters if not provided
        if stage_params is None:
            stage_params = []
            for i in range(n_stages):
                # Progressively lighter and softer stages
                mass = 100.0 * (0.5 ** i)      # kg
                stiffness = 1000.0 * (0.3 ** i) # N/m
                damping = 2 * 0.7 * np.sqrt(mass * stiffness)  # Critical damping
                stage_params.append({
                    'mass': mass,
                    'stiffness': stiffness, 
                    'damping': damping
                })
        
        # Calculate natural frequencies and damping ratios
        stage_properties = []
        for params in stage_params:
            natural_freq = np.sqrt(params['stiffness'] / params['mass']) / (2 * np.pi)
            damping_ratio = params['damping'] / (2 * np.sqrt(params['mass'] * params['stiffness']))
            stage_properties.append({
                'natural_frequency': natural_freq,
                'damping_ratio': damping_ratio,
                **params
            })
        
        # Frequency response analysis
        frequencies = np.logspace(-2, 3, 1000)  # 0.01 Hz to 1000 Hz
        omega = 2 * np.pi * frequencies
        
        # Calculate transmissibility for each stage
        total_transmissibility = np.ones_like(omega, dtype=complex)
        stage_transmissibilities = []
        
        for props in stage_properties:
            omega_n = 2 * np.pi * props['natural_frequency']
            zeta = props['damping_ratio']
            
            # Frequency ratio
            r = omega / omega_n
            
            # Transmissibility for single-DOF system
            # Mathematical Formula: T(ω) = 1/√[(1-r²)² + (2ζr)²]
            numerator = 1.0
            denominator = np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
            transmissibility = numerator / denominator
            
            stage_transmissibilities.append(transmissibility)
            total_transmissibility *= transmissibility
        
        # Calculate isolation efficiency
        isolation_efficiency = 1.0 / np.abs(total_transmissibility)
        
        # Find isolation performance at key frequencies
        target_frequencies = [1.0, 10.0, 50.0, 100.0]  # Hz
        isolation_at_targets = {}
        
        for freq in target_frequencies:
            freq_idx = np.argmin(np.abs(frequencies - freq))
            isolation_at_targets[freq] = isolation_efficiency[freq_idx]
        
        # Check against requirements
        isolation_10hz = isolation_at_targets[10.0]
        meets_isolation_specs = isolation_10hz >= self.specs.required_isolation
        
        # Calculate displacement transmission
        input_displacement = self.specs.ground_motion_amplitude
        output_displacement = input_displacement / isolation_10hz
        meets_displacement_specs = output_displacement <= self.specs.max_displacement
        
        results = {
            'n_stages': n_stages,
            'stage_properties': stage_properties,
            'frequencies': frequencies,
            'total_transmissibility': np.abs(total_transmissibility),
            'isolation_efficiency': isolation_efficiency,
            'stage_transmissibilities': [np.abs(t) for t in stage_transmissibilities],
            'isolation_at_targets': isolation_at_targets,
            'output_displacement_10hz': output_displacement,
            'meets_isolation_specs': meets_isolation_specs,
            'meets_displacement_specs': meets_displacement_specs,
            'isolation_margin': isolation_10hz / self.specs.required_isolation
        }
        
        self.validation_results['passive_isolation'] = results
        logger.info(f"Isolation at 10 Hz: {isolation_10hz:.0f}×")
        logger.info(f"Output displacement: {output_displacement*1e9:.3f} nm RMS")
        logger.info(f"Meets isolation specs: {meets_isolation_specs}")
        logger.info(f"Meets displacement specs: {meets_displacement_specs}")
        
        return results
    
    def design_active_control(self, 
                            plant_dynamics: Dict = None,
                            control_bandwidth: float = 100.0) -> Dict:
        """
        Design active vibration control system
        
        Args:
            plant_dynamics: Plant transfer function parameters
            control_bandwidth: Desired control bandwidth (Hz)
            
        Returns:
            Active control system design results
        """
        logger.info("Designing active vibration control system...")
        
        # Default plant dynamics (typical isolation table)
        if plant_dynamics is None:
            plant_dynamics = {
                'mass': 50.0,           # kg
                'stiffness': 2000.0,    # N/m  
                'damping': 100.0        # N·s/m
            }
        
        # Plant transfer function: G(s) = 1/(ms² + cs + k)
        mass = plant_dynamics['mass']
        damping = plant_dynamics['damping']
        stiffness = plant_dynamics['stiffness']
        
        # Natural frequency and damping ratio
        omega_n = np.sqrt(stiffness / mass)
        zeta = damping / (2 * np.sqrt(mass * stiffness))
        natural_freq_hz = omega_n / (2 * np.pi)
        
        # Design PID controller for vibration suppression
        # Target: high gain at low frequencies, rolloff above bandwidth
        
        # Controller parameters
        omega_c = 2 * np.pi * control_bandwidth  # Crossover frequency
        
        # Lead-lag compensator design
        # Mathematical Formula: K(s) = Kp · (s + z1)/(s + p1) · (s + z2)/(s + p2)
        
        # Proportional gain
        Kp = omega_c**2 / omega_n**2
        
        # Lead compensation (phase boost around crossover)
        alpha = 0.1  # Lead ratio
        T_lead = 1 / (omega_c * np.sqrt(alpha))
        z1 = 1 / T_lead
        p1 = 1 / (alpha * T_lead)
        
        # Lag compensation (low-frequency gain boost)
        beta = 10.0  # Lag ratio  
        T_lag = 10 / omega_c
        z2 = 1 / (beta * T_lag)
        p2 = 1 / T_lag
        
        # Frequency response analysis
        frequencies = np.logspace(-1, 4, 1000)  # 0.1 Hz to 10 kHz
        s = 1j * 2 * np.pi * frequencies
        
        # Plant transfer function
        G_plant = 1 / (mass * s**2 + damping * s + stiffness)
        
        # Controller transfer function
        K_controller = Kp * (s + z1) / (s + p1) * (s + z2) / (s + p2)
        
        # Open-loop transfer function
        L = K_controller * G_plant
        
        # Closed-loop transfer functions
        # Sensitivity: S = 1/(1 + L)
        S = 1 / (1 + L)
        
        # Complementary sensitivity: T = L/(1 + L)
        T = L / (1 + L)
        
        # Calculate stability margins
        # Find gain and phase margins
        magnitude = np.abs(L)
        phase = np.angle(L)
        
        # Gain margin (at 180° phase crossover)
        phase_180_idx = np.argmin(np.abs(phase + np.pi))
        gain_margin_db = -20 * np.log10(magnitude[phase_180_idx])
        
        # Phase margin (at 0 dB gain crossover)
        gain_0db_idx = np.argmin(np.abs(20 * np.log10(magnitude)))
        phase_margin_deg = (phase[gain_0db_idx] + np.pi) * 180 / np.pi
        
        # Disturbance rejection analysis
        disturbance_rejection = np.abs(S)
        
        # Performance at target frequencies
        performance_metrics = {}
        target_frequencies = [1.0, 10.0, 50.0, 100.0]
        
        for freq in target_frequencies:
            freq_idx = np.argmin(np.abs(frequencies - freq))
            performance_metrics[freq] = {
                'sensitivity': np.abs(S[freq_idx]),
                'complementary_sensitivity': np.abs(T[freq_idx]),
                'disturbance_rejection': 1 / np.abs(S[freq_idx])
            }
        
        # Check stability and performance
        stable = (gain_margin_db > 6.0) and (phase_margin_deg > 30.0)
        performance_10hz = performance_metrics[10.0]['disturbance_rejection']
        meets_performance = performance_10hz >= 100.0  # 100× rejection at 10 Hz
        
        results = {
            'plant_dynamics': plant_dynamics,
            'natural_frequency': natural_freq_hz,
            'damping_ratio': zeta,
            'controller_params': {
                'Kp': Kp, 
                'z1': z1, 'p1': p1,
                'z2': z2, 'p2': p2
            },
            'frequencies': frequencies,
            'open_loop_magnitude': magnitude,
            'open_loop_phase': phase,
            'sensitivity': np.abs(S),
            'complementary_sensitivity': np.abs(T),
            'disturbance_rejection': 1 / np.abs(S),
            'gain_margin_db': gain_margin_db,
            'phase_margin_deg': phase_margin_deg,
            'performance_metrics': performance_metrics,
            'stable': stable,
            'meets_performance': meets_performance,
            'control_bandwidth': control_bandwidth
        }
        
        self.validation_results['active_control'] = results
        logger.info(f"Gain margin: {gain_margin_db:.1f} dB")
        logger.info(f"Phase margin: {phase_margin_deg:.1f}°")
        logger.info(f"System stable: {stable}")
        logger.info(f"Disturbance rejection at 10 Hz: {performance_10hz:.1f}×")
        
        return results
    
    def validate_angular_stability(self, 
                                 platform_dimensions: Tuple[float, float] = (0.5, 0.5),
                                 mounting_height: float = 0.1) -> Dict:
        """
        Validate angular vibration isolation for microradian stability
        
        Args:
            platform_dimensions: Platform length and width (m)
            mounting_height: Height of sensitive components (m)
            
        Returns:
            Angular stability validation results
        """
        logger.info("Validating angular vibration stability...")
        
        length, width = platform_dimensions
        
        # Calculate moment of inertia for platform
        # Assuming uniform mass distribution
        total_mass = 100.0  # kg - typical optical table mass
        I_pitch = total_mass * (width**2 + mounting_height**2) / 12
        I_roll = total_mass * (length**2 + mounting_height**2) / 12
        I_yaw = total_mass * (length**2 + width**2) / 12
        
        # Angular natural frequencies (assuming spring mounting)
        # For each rotational axis
        k_angular = 5000.0  # N·m/rad - rotational stiffness
        
        omega_pitch = np.sqrt(k_angular / I_pitch)
        omega_roll = np.sqrt(k_angular / I_roll)
        omega_yaw = np.sqrt(k_angular / I_yaw)
        
        angular_frequencies = {
            'pitch': omega_pitch / (2 * np.pi),
            'roll': omega_roll / (2 * np.pi),
            'yaw': omega_yaw / (2 * np.pi)
        }
        
        # Coupling between translational and angular motion
        # Translation-to-rotation coupling factor
        coupling_factor = mounting_height / max(length, width)
        
        # Get passive isolation results
        if 'passive_isolation' not in self.validation_results:
            self.analyze_passive_isolation()
        
        passive_result = self.validation_results['passive_isolation']
        translational_isolation = passive_result['isolation_efficiency']
        frequencies = passive_result['frequencies']
        
        # Angular isolation (accounting for coupling)
        # Mathematical Formula: θ_out = θ_in · T_trans + (x_in/h) · T_coupling
        angular_transmissibility = {}
        
        for axis, freq_nat in angular_frequencies.items():
            # Direct angular transmission (similar to translational)
            omega_n = 2 * np.pi * freq_nat
            omega = 2 * np.pi * frequencies
            zeta = self.specs.damping_ratio
            
            r = omega / omega_n
            direct_transmission = 1 / np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
            
            # Coupling from translational motion
            coupling_transmission = coupling_factor / translational_isolation
            
            # Total angular transmissibility
            total_angular_trans = np.sqrt(direct_transmission**2 + coupling_transmission**2)
            angular_transmissibility[axis] = total_angular_trans
        
        # Calculate angular displacement for typical ground motion
        input_angular_motion = 1e-6  # 1 μrad RMS typical
        
        # Find performance at 10 Hz
        freq_10hz_idx = np.argmin(np.abs(frequencies - 10.0))
        angular_isolation_10hz = {}
        output_angular_motion = {}
        
        for axis in angular_frequencies.keys():
            isolation = 1 / angular_transmissibility[axis][freq_10hz_idx]
            output = input_angular_motion / isolation
            
            angular_isolation_10hz[axis] = isolation
            output_angular_motion[axis] = output
        
        # Check against specifications
        meets_angular_specs = all(
            output <= self.specs.max_angular_displacement 
            for output in output_angular_motion.values()
        )
        
        # Worst-case analysis
        worst_axis = max(output_angular_motion.keys(), 
                        key=lambda k: output_angular_motion[k])
        worst_output = output_angular_motion[worst_axis]
        
        results = {
            'platform_dimensions': platform_dimensions,
            'mounting_height': mounting_height,
            'moments_of_inertia': {'pitch': I_pitch, 'roll': I_roll, 'yaw': I_yaw},
            'angular_frequencies': angular_frequencies,
            'coupling_factor': coupling_factor,
            'frequencies': frequencies,
            'angular_transmissibility': angular_transmissibility,
            'angular_isolation_10hz': angular_isolation_10hz,
            'output_angular_motion': output_angular_motion,
            'worst_axis': worst_axis,
            'worst_output': worst_output,
            'meets_angular_specs': meets_angular_specs,
            'angular_margin': self.specs.max_angular_displacement / worst_output
        }
        
        self.validation_results['angular_stability'] = results
        logger.info(f"Angular natural frequencies: {angular_frequencies}")
        logger.info(f"Worst angular output ({worst_axis}): {worst_output*1e6:.3f} μrad")
        logger.info(f"Meets angular specifications: {meets_angular_specs}")
        
        return results
    
    def analyze_ground_motion_spectrum(self, 
                                     measurement_data: np.ndarray = None,
                                     sampling_rate: float = 1000.0) -> Dict:
        """
        Analyze ground motion power spectral density
        
        Args:
            measurement_data: Measured acceleration data (m/s²)
            sampling_rate: Data sampling rate (Hz)
            
        Returns:
            Ground motion spectrum analysis results
        """
        logger.info("Analyzing ground motion spectrum...")
        
        # If no data provided, generate synthetic realistic ground motion
        if measurement_data is None:
            duration = 3600  # 1 hour of data
            n_samples = int(duration * sampling_rate)
            time = np.arange(n_samples) / sampling_rate
            
            # Synthetic ground motion model
            # Low frequency building sway + mid frequency activity + high frequency noise
            building_sway = 1e-6 * np.sin(2 * np.pi * 0.2 * time + np.random.random() * 2 * np.pi)
            footsteps = 0.5e-6 * np.sin(2 * np.pi * 2.0 * time + np.random.random() * 2 * np.pi)
            machinery = 0.1e-6 * np.sin(2 * np.pi * 50.0 * time + np.random.random() * 2 * np.pi)
            
            # Add random components
            white_noise = 0.01e-6 * np.random.normal(0, 1, n_samples)
            
            # Integrate acceleration to get displacement
            acceleration = building_sway + footsteps + machinery + white_noise
            # Simple integration (in practice would use proper numerical integration)
            velocity = np.cumsum(acceleration) / sampling_rate
            measurement_data = np.cumsum(velocity) / sampling_rate
        
        # Calculate power spectral density
        frequencies, psd = signal.welch(
            measurement_data, 
            fs=sampling_rate,
            nperseg=min(len(measurement_data)//4, 8192),
            window='hann',
            scaling='density'
        )
        
        # Convert to displacement PSD (if input was acceleration, divide by (2πf)⁴)
        # Assuming input is already displacement
        displacement_psd = psd
        
        # Calculate RMS levels in frequency bands
        frequency_bands = {
            'building_sway': (0.1, 1.0),
            'footsteps': (1.0, 10.0),
            'hvac_systems': (10.0, 100.0),
            'machinery': (50.0, 500.0),
            'acoustic': (100.0, 1000.0)
        }
        
        rms_levels = {}
        for band_name, (f_low, f_high) in frequency_bands.items():
            # Find frequency indices
            idx_low = np.argmin(np.abs(frequencies - f_low))
            idx_high = np.argmin(np.abs(frequencies - f_high))
            
            # Integrate PSD over frequency band
            if idx_high > idx_low:
                band_psd = displacement_psd[idx_low:idx_high]
                band_freq = frequencies[idx_low:idx_high]
                rms_level = np.sqrt(np.trapz(band_psd, band_freq))
                rms_levels[band_name] = rms_level
            else:
                rms_levels[band_name] = 0.0
        
        # Overall RMS level
        total_rms = np.sqrt(np.trapz(displacement_psd, frequencies))
        
        # Compare with typical specifications
        typical_ground_motion = self.specs.ground_motion_amplitude
        ground_motion_factor = total_rms / typical_ground_motion
        
        # Identify dominant frequency components
        peak_frequency_idx = np.argmax(displacement_psd)
        dominant_frequency = frequencies[peak_frequency_idx]
        peak_amplitude = np.sqrt(displacement_psd[peak_frequency_idx])
        
        results = {
            'frequencies': frequencies,
            'displacement_psd': displacement_psd,
            'rms_levels': rms_levels,
            'total_rms': total_rms,
            'ground_motion_factor': ground_motion_factor,
            'dominant_frequency': dominant_frequency,
            'peak_amplitude': peak_amplitude,
            'measurement_duration': len(measurement_data) / sampling_rate,
            'meets_assumptions': abs(ground_motion_factor - 1.0) < 2.0  # Within 2× typical
        }
        
        self.validation_results['ground_motion_spectrum'] = results
        logger.info(f"Total ground motion RMS: {total_rms*1e6:.2f} μm")
        logger.info(f"Dominant frequency: {dominant_frequency:.2f} Hz")
        logger.info(f"Ground motion factor vs. typical: {ground_motion_factor:.2f}×")
        
        return results
    
    def optimize_isolation_design(self, 
                                target_isolation: float = None,
                                max_stages: int = 4) -> Dict:
        """
        Optimize isolation system design for target performance
        
        Args:
            target_isolation: Target isolation factor
            max_stages: Maximum number of isolation stages
            
        Returns:
            Optimized isolation design results
        """
        logger.info("Optimizing vibration isolation design...")
        
        if target_isolation is None:
            target_isolation = self.specs.required_isolation
        
        def objective_function(params):
            """Objective function for optimization"""
            try:
                # Unpack parameters: [n_stages, freq1, zeta1, freq2, zeta2, ...]
                n_stages = int(params[0])
                n_stages = max(1, min(n_stages, max_stages))
                
                # Extract stage parameters
                stage_params = []
                for i in range(n_stages):
                    if len(params) > 2*i + 2:
                        freq = max(0.1, min(params[2*i + 1], 10.0))  # 0.1-10 Hz
                        zeta = max(0.1, min(params[2*i + 2], 2.0))   # 0.1-2.0
                    else:
                        freq = 1.0
                        zeta = 0.7
                    
                    # Convert to mass, stiffness, damping
                    mass = 100.0 * (0.5 ** i)  # Decreasing mass
                    stiffness = mass * (2 * np.pi * freq)**2
                    damping = 2 * zeta * np.sqrt(mass * stiffness)
                    
                    stage_params.append({
                        'mass': mass,
                        'stiffness': stiffness,
                        'damping': damping
                    })
                
                # Calculate isolation performance
                frequencies = np.array([10.0])  # Evaluate at 10 Hz
                omega = 2 * np.pi * frequencies
                
                total_transmissibility = 1.0
                for params_stage in stage_params:
                    mass = params_stage['mass']
                    stiffness = params_stage['stiffness']
                    damping = params_stage['damping']
                    
                    omega_n = np.sqrt(stiffness / mass)
                    zeta = damping / (2 * np.sqrt(mass * stiffness))
                    
                    r = omega / omega_n
                    transmissibility = 1 / np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
                    total_transmissibility *= transmissibility[0]
                
                isolation_factor = 1 / total_transmissibility
                
                # Objective: minimize negative log of isolation factor
                # (maximize isolation while penalizing excessive complexity)
                complexity_penalty = 0.1 * n_stages**2
                objective = -np.log10(isolation_factor) + complexity_penalty
                
                return objective
                
            except:
                return 1e6  # Large penalty for invalid parameters
        
        # Initial guess: 3 stages with reasonable parameters
        initial_guess = [3, 0.5, 0.7, 1.0, 0.7, 2.0, 0.7]
        
        # Parameter bounds
        bounds = [(1, max_stages)]  # Number of stages
        for i in range(max_stages):
            bounds.append((0.1, 10.0))  # Natural frequency (Hz)
            bounds.append((0.1, 2.0))   # Damping ratio
        
        # Optimization
        try:
            result = optimize.differential_evolution(
                objective_function,
                bounds=bounds[:len(initial_guess)],
                seed=42,
                maxiter=100,
                atol=1e-6
            )
            
            optimal_params = result.x
            optimal_objective = result.fun
            optimization_success = result.success
            
        except:
            # Fallback to initial guess if optimization fails
            optimal_params = initial_guess
            optimal_objective = objective_function(initial_guess)
            optimization_success = False
        
        # Extract optimal design
        n_stages_opt = int(optimal_params[0])
        optimal_stage_params = []
        
        for i in range(n_stages_opt):
            if len(optimal_params) > 2*i + 2:
                freq = optimal_params[2*i + 1]
                zeta = optimal_params[2*i + 2]
            else:
                freq = 1.0
                zeta = 0.7
            
            mass = 100.0 * (0.5 ** i)
            stiffness = mass * (2 * np.pi * freq)**2
            damping = 2 * zeta * np.sqrt(mass * stiffness)
            
            optimal_stage_params.append({
                'mass': mass,
                'stiffness': stiffness,
                'damping': damping,
                'natural_frequency': freq,
                'damping_ratio': zeta
            })
        
        # Validate optimal design
        validation_result = self.analyze_passive_isolation(
            n_stages=n_stages_opt,
            stage_params=optimal_stage_params
        )
        
        optimal_isolation = validation_result['isolation_at_targets'][10.0]
        meets_target = optimal_isolation >= target_isolation
        
        results = {
            'target_isolation': target_isolation,
            'optimization_success': optimization_success,
            'optimal_stages': n_stages_opt,
            'optimal_stage_params': optimal_stage_params,
            'achieved_isolation': optimal_isolation,
            'meets_target': meets_target,
            'improvement_factor': optimal_isolation / target_isolation,
            'objective_value': optimal_objective
        }
        
        self.validation_results['optimized_design'] = results
        logger.info(f"Optimal number of stages: {n_stages_opt}")
        logger.info(f"Achieved isolation: {optimal_isolation:.0f}×")
        logger.info(f"Meets target: {meets_target}")
        logger.info(f"Optimization success: {optimization_success}")
        
        return results
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive vibration isolation validation report"""
        
        report = []
        report.append("=" * 60)
        report.append("VIBRATION ISOLATION VERIFICATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: June 30, 2025")
        report.append("")
        
        # Requirements summary
        report.append("VIBRATION ISOLATION REQUIREMENTS:")
        report.append(f"  Max Displacement: {self.specs.max_displacement*1e9:.3f} nm RMS")
        report.append(f"  Max Angular Displacement: {self.specs.max_angular_displacement*1e6:.1f} μrad RMS")
        report.append(f"  Required Isolation: {self.specs.required_isolation:.0f}× at 10 Hz")
        report.append(f"  Ground Motion Amplitude: {self.specs.ground_motion_amplitude*1e6:.1f} μm RMS")
        report.append("")
        
        # Passive isolation results
        if 'passive_isolation' in self.validation_results:
            result = self.validation_results['passive_isolation']
            isolation_10hz = result['isolation_at_targets'][10.0]
            output_disp = result['output_displacement_10hz']
            status = "PASS" if (result['meets_isolation_specs'] and result['meets_displacement_specs']) else "FAIL"
            
            report.append(f"PASSIVE ISOLATION: {status}")
            report.append(f"  Number of stages: {result['n_stages']}")
            report.append(f"  Isolation at 10 Hz: {isolation_10hz:.0f}×")
            report.append(f"  Output displacement: {output_disp*1e9:.3f} nm RMS")
            report.append(f"  Safety margin: {result['isolation_margin']:.1f}×")
        
        # Active control results
        if 'active_control' in self.validation_results:
            result = self.validation_results['active_control']
            gain_margin = result['gain_margin_db']
            phase_margin = result['phase_margin_deg']
            performance = result['performance_metrics'][10.0]['disturbance_rejection']
            status = "PASS" if (result['stable'] and result['meets_performance']) else "FAIL"
            
            report.append(f"\nACTIVE CONTROL: {status}")
            report.append(f"  Control bandwidth: {result['control_bandwidth']:.0f} Hz")
            report.append(f"  Gain margin: {gain_margin:.1f} dB")
            report.append(f"  Phase margin: {phase_margin:.1f}°")
            report.append(f"  Disturbance rejection at 10 Hz: {performance:.0f}×")
        
        # Angular stability results
        if 'angular_stability' in self.validation_results:
            result = self.validation_results['angular_stability']
            worst_output = result['worst_output']
            worst_axis = result['worst_axis']
            status = "PASS" if result['meets_angular_specs'] else "FAIL"
            
            report.append(f"\nANGULAR STABILITY: {status}")
            report.append(f"  Worst axis: {worst_axis}")
            report.append(f"  Angular output: {worst_output*1e6:.2f} μrad RMS")
            report.append(f"  Angular margin: {result['angular_margin']:.1f}×")
        
        # Ground motion analysis
        if 'ground_motion_spectrum' in self.validation_results:
            result = self.validation_results['ground_motion_spectrum']
            total_rms = result['total_rms']
            dominant_freq = result['dominant_frequency']
            
            report.append(f"\nGROUND MOTION ANALYSIS:")
            report.append(f"  Total RMS: {total_rms*1e6:.2f} μm")
            report.append(f"  Dominant frequency: {dominant_freq:.2f} Hz")
            report.append(f"  Ground motion factor: {result['ground_motion_factor']:.2f}×")
        
        # Optimization results
        if 'optimized_design' in self.validation_results:
            result = self.validation_results['optimized_design']
            achieved = result['achieved_isolation']
            target = result['target_isolation']
            status = "PASS" if result['meets_target'] else "FAIL"
            
            report.append(f"\nOPTIMIZED DESIGN: {status}")
            report.append(f"  Optimal stages: {result['optimal_stages']}")
            report.append(f"  Target isolation: {target:.0f}×")
            report.append(f"  Achieved isolation: {achieved:.0f}×")
            report.append(f"  Improvement factor: {result['improvement_factor']:.2f}×")
        
        # Overall assessment
        report.append("")
        report.append("=" * 60)
        overall_pass = all(
            result.get('meets_isolation_specs', False) or
            result.get('meets_performance', False) or
            result.get('meets_angular_specs', False) or
            result.get('meets_target', False)
            for result in self.validation_results.values()
            if any(key in result for key in ['meets_isolation_specs', 'meets_performance', 
                                           'meets_angular_specs', 'meets_target'])
        )
        report.append(f"OVERALL VALIDATION STATUS: {'PASS' if overall_pass else 'FAIL'}")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_vibration_analysis(self, save_path: str = None):
        """Generate comprehensive vibration analysis plots"""
        
        if not self.validation_results:
            logger.warning("No validation results available. Run validation methods first.")
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Plot 1: Passive isolation transmissibility
        if 'passive_isolation' in self.validation_results:
            ax1 = fig.add_subplot(gs[0, 0])
            result = self.validation_results['passive_isolation']
            
            frequencies = result['frequencies']
            total_trans = result['total_transmissibility']
            
            ax1.loglog(frequencies, total_trans, 'b-', linewidth=2, label='Total')
            
            # Plot individual stages
            for i, stage_trans in enumerate(result['stage_transmissibilities']):
                ax1.loglog(frequencies, stage_trans, '--', alpha=0.7, 
                          label=f'Stage {i+1}')
            
            ax1.axhline(1/self.specs.required_isolation, color='r', 
                       linestyle=':', label='Requirement')
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Transmissibility')
            ax1.set_title('Passive Isolation Performance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Active control frequency response
        if 'active_control' in self.validation_results:
            ax2 = fig.add_subplot(gs[0, 1])
            result = self.validation_results['active_control']
            
            frequencies = result['frequencies']
            sensitivity = result['sensitivity']
            
            ax2.semilogx(frequencies, 20*np.log10(sensitivity), 'b-', label='Sensitivity')
            ax2.semilogx(frequencies, 20*np.log10(result['complementary_sensitivity']), 
                        'r--', label='Complementary')
            ax2.axhline(-20, color='gray', linestyle=':', alpha=0.7, label='20 dB rejection')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Magnitude (dB)')
            ax2.set_title('Active Control Response')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Angular stability
        if 'angular_stability' in self.validation_results:
            ax3 = fig.add_subplot(gs[0, 2])
            result = self.validation_results['angular_stability']
            
            axes = list(result['angular_transmissibility'].keys())
            frequencies = result['frequencies']
            
            for axis in axes:
                trans = result['angular_transmissibility'][axis]
                ax3.loglog(frequencies, trans, label=f'{axis.capitalize()}')
            
            ax3.axhline(self.specs.max_angular_displacement / 1e-6, color='r', 
                       linestyle='--', label='Requirement')
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Angular Transmissibility')
            ax3.set_title('Angular Stability')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Ground motion spectrum
        if 'ground_motion_spectrum' in self.validation_results:
            ax4 = fig.add_subplot(gs[0, 3])
            result = self.validation_results['ground_motion_spectrum']
            
            frequencies = result['frequencies']
            psd = result['displacement_psd']
            
            ax4.loglog(frequencies, psd*1e12, 'b-')  # Convert to pm²/Hz
            ax4.set_xlabel('Frequency (Hz)')
            ax4.set_ylabel('Displacement PSD (pm²/Hz)')
            ax4.set_title('Ground Motion Spectrum')
            ax4.grid(True, alpha=0.3)
            
            # Highlight dominant frequency
            dom_freq = result['dominant_frequency']
            ax4.axvline(dom_freq, color='r', linestyle='--', 
                       label=f'Dominant: {dom_freq:.1f} Hz')
            ax4.legend()
        
        # Plot 5: Combined isolation performance
        ax5 = fig.add_subplot(gs[1, 0:2])
        
        target_frequencies = [1.0, 10.0, 50.0, 100.0]
        
        # Passive isolation
        if 'passive_isolation' in self.validation_results:
            passive_isolation = []
            result = self.validation_results['passive_isolation']
            for freq in target_frequencies:
                passive_isolation.append(result['isolation_at_targets'][freq])
            
            ax5.bar([f-0.2 for f in range(len(target_frequencies))], passive_isolation, 
                   width=0.4, label='Passive', alpha=0.7)
        
        # Active control performance
        if 'active_control' in self.validation_results:
            active_performance = []
            result = self.validation_results['active_control']
            for freq in target_frequencies:
                if freq in result['performance_metrics']:
                    active_performance.append(result['performance_metrics'][freq]['disturbance_rejection'])
                else:
                    active_performance.append(1.0)
            
            ax5.bar([f+0.2 for f in range(len(target_frequencies))], active_performance,
                   width=0.4, label='Active', alpha=0.7)
        
        ax5.axhline(self.specs.required_isolation, color='r', linestyle='--', 
                   label='Requirement')
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Isolation Factor')
        ax5.set_title('Isolation Performance Comparison')
        ax5.set_xticks(range(len(target_frequencies)))
        ax5.set_xticklabels([f'{f:.0f}' for f in target_frequencies])
        ax5.set_yscale('log')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: RMS displacement vs frequency bands
        if 'ground_motion_spectrum' in self.validation_results:
            ax6 = fig.add_subplot(gs[1, 2])
            result = self.validation_results['ground_motion_spectrum']
            
            rms_levels = result['rms_levels']
            band_names = list(rms_levels.keys())
            rms_values = [rms_levels[name]*1e9 for name in band_names]  # Convert to nm
            
            bars = ax6.bar(range(len(band_names)), rms_values, alpha=0.7)
            ax6.axhline(self.specs.max_displacement*1e9, color='r', 
                       linestyle='--', label='Target')
            ax6.set_ylabel('RMS Displacement (nm)')
            ax6.set_title('Ground Motion by Frequency Band')
            ax6.set_xticks(range(len(band_names)))
            ax6.set_xticklabels([name.replace('_', '\n') for name in band_names], 
                               rotation=45, ha='right')
            ax6.set_yscale('log')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # Plot 7: Optimization results comparison
        if 'optimized_design' in self.validation_results:
            ax7 = fig.add_subplot(gs[1, 3])
            result = self.validation_results['optimized_design']
            
            categories = ['Target', 'Achieved']
            values = [result['target_isolation'], result['achieved_isolation']]
            colors = ['red', 'green' if result['meets_target'] else 'orange']
            
            bars = ax7.bar(categories, values, color=colors, alpha=0.7)
            ax7.set_ylabel('Isolation Factor')
            ax7.set_title('Optimization Results')
            ax7.set_yscale('log')
            ax7.grid(True, alpha=0.3)
            
            # Add text annotations
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.0f}×', ha='center', va='bottom')
        
        # Plot 8: System design parameters
        ax8 = fig.add_subplot(gs[2, :])
        
        # Create table of key parameters
        table_data = []
        
        if 'passive_isolation' in self.validation_results:
            result = self.validation_results['passive_isolation']
            for i, props in enumerate(result['stage_properties']):
                table_data.append([
                    f"Stage {i+1}",
                    f"{props['mass']:.1f} kg",
                    f"{props['natural_frequency']:.2f} Hz",
                    f"{props['damping_ratio']:.2f}",
                    f"{props['stiffness']:.0f} N/m"
                ])
        
        if table_data:
            table = ax8.table(cellText=table_data,
                            colLabels=['Stage', 'Mass', 'Nat. Freq.', 'Damping', 'Stiffness'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax8.axis('off')
            ax8.set_title('Isolation System Design Parameters')
        
        plt.suptitle('Vibration Isolation Analysis', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Vibration analysis plots saved to {save_path}")
        else:
            plt.show()

# Example usage and validation
if __name__ == "__main__":
    # Initialize validator with stringent nanopositioning requirements
    specs = VibrationSpecifications(
        max_displacement=0.1e-9,         # 0.1 nm RMS
        max_angular_displacement=1e-6,   # 1 μrad RMS
        required_isolation=1e4,          # 10,000× at 10 Hz
        ground_motion_amplitude=1e-6     # 1 μm RMS
    )
    
    validator = VibrationIsolationValidator(specs)
    
    print("Starting comprehensive vibration isolation validation...")
    
    # 1. Analyze passive isolation system
    validator.analyze_passive_isolation(n_stages=3)
    
    # 2. Design active control system
    validator.design_active_control(control_bandwidth=100.0)
    
    # 3. Validate angular stability
    validator.validate_angular_stability()
    
    # 4. Analyze ground motion spectrum
    validator.analyze_ground_motion_spectrum()
    
    # 5. Optimize isolation design
    validator.optimize_isolation_design()
    
    # Generate and print validation report
    report = validator.generate_validation_report()
    print(report)
    
    # Generate plots
    validator.plot_vibration_analysis('vibration_isolation_validation.png')
    
    print("\nVibration isolation verification complete!")
