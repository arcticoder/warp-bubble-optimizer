"""
Material Property Uncertainties Module

Analyzes uncertainties in material properties affecting Casimir force calculations
and nanopositioning performance. Key focus areas:
- Surface roughness effects on Casimir forces
- Material conductivity and dielectric function variations
- Temperature-dependent property changes
- Manufacturing tolerances and aging effects

Mathematical Framework:
- Casimir force: F = -ℏcπ²A/(240d⁴) × corrections
- Surface roughness: δF/F ≈ (δ/d)² for δ << d
- Temperature dependence: ε(T) = ε₀[1 + α(T-T₀)]
- Monte Carlo uncertainty propagation: σ²_F = Σ(∂F/∂pᵢ)²σ²_pᵢ
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize, stats, interpolate
from scipy.special import factorial
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MaterialProperties:
    """Material properties with uncertainties"""
    name: str
    conductivity: float                 # S/m
    conductivity_uncertainty: float     # Relative uncertainty
    permittivity_real: float           # Real part of dielectric constant
    permittivity_imag: float           # Imaginary part
    permittivity_uncertainty: float    # Relative uncertainty
    surface_roughness: float           # nm RMS
    roughness_uncertainty: float       # nm
    thermal_expansion: float           # 1/K
    expansion_uncertainty: float       # Relative uncertainty
    work_function: float               # eV
    work_function_uncertainty: float   # eV

@dataclass
class CasimirParameters:
    """Casimir force calculation parameters"""
    plate_separation: float = 10e-9     # m
    plate_area: float = 100e-12         # m² (100 μm²)
    temperature: float = 300            # K
    separation_uncertainty: float = 0.1e-9  # m
    area_uncertainty: float = 1e-12     # m²
    temperature_uncertainty: float = 0.1    # K

class MaterialPropertyValidator:
    """
    Material property uncertainty analysis and validation
    """
    
    def __init__(self, casimir_params: CasimirParameters = None):
        """Initialize with Casimir force parameters"""
        self.casimir_params = casimir_params or CasimirParameters()
        self.validation_results = {}
        self.material_database = self._initialize_material_database()
        
    def _initialize_material_database(self) -> Dict[str, MaterialProperties]:
        """Initialize database of materials with uncertainties"""
        return {
            'gold': MaterialProperties(
                name='Gold',
                conductivity=4.1e7,              # S/m
                conductivity_uncertainty=0.02,   # 2%
                permittivity_real=-25.0,         # At optical frequencies
                permittivity_imag=1.38,
                permittivity_uncertainty=0.05,   # 5%
                surface_roughness=0.5,           # nm RMS
                roughness_uncertainty=0.1,       # nm
                thermal_expansion=14.2e-6,       # 1/K
                expansion_uncertainty=0.01,      # 1%
                work_function=5.1,               # eV
                work_function_uncertainty=0.1    # eV
            ),
            'silicon': MaterialProperties(
                name='Silicon',
                conductivity=1e-12,              # S/m (intrinsic)
                conductivity_uncertainty=0.5,    # 50% (highly variable)
                permittivity_real=11.68,         # Static dielectric constant
                permittivity_imag=0.01,
                permittivity_uncertainty=0.02,   # 2%
                surface_roughness=0.2,           # nm RMS (polished)
                roughness_uncertainty=0.05,      # nm
                thermal_expansion=2.6e-6,        # 1/K
                expansion_uncertainty=0.005,     # 0.5%
                work_function=4.6,               # eV
                work_function_uncertainty=0.1    # eV
            ),
            'aluminum': MaterialProperties(
                name='Aluminum',
                conductivity=3.5e7,              # S/m
                conductivity_uncertainty=0.03,   # 3%
                permittivity_real=-50.0,         # At optical frequencies
                permittivity_imag=5.0,
                permittivity_uncertainty=0.1,    # 10%
                surface_roughness=1.0,           # nm RMS
                roughness_uncertainty=0.2,       # nm
                thermal_expansion=23.1e-6,       # 1/K
                expansion_uncertainty=0.02,      # 2%
                work_function=4.3,               # eV
                work_function_uncertainty=0.1    # eV
            ),
            'sapphire': MaterialProperties(
                name='Sapphire',
                conductivity=1e-18,              # S/m (insulator)
                conductivity_uncertainty=1.0,    # 100% (very uncertain)
                permittivity_real=9.4,           # Along c-axis
                permittivity_imag=0.001,
                permittivity_uncertainty=0.01,   # 1%
                surface_roughness=0.1,           # nm RMS (ultra-polished)
                roughness_uncertainty=0.02,      # nm
                thermal_expansion=5.4e-6,        # 1/K (along c-axis)
                expansion_uncertainty=0.005,     # 0.5%
                work_function=8.5,               # eV (wide bandgap)
                work_function_uncertainty=0.2    # eV
            )
        }
    
    def calculate_casimir_force(self, 
                              material1: str,
                              material2: str,
                              separation: float = None,
                              temperature: float = None) -> Dict:
        """
        Calculate Casimir force between two materials
        
        Args:
            material1: First material name
            material2: Second material name  
            separation: Plate separation (m)
            temperature: Temperature (K)
            
        Returns:
            Casimir force calculation results
        """
        logger.info(f"Calculating Casimir force between {material1} and {material2}...")
        
        if material1 not in self.material_database:
            raise ValueError(f"Material {material1} not in database")
        if material2 not in self.material_database:
            raise ValueError(f"Material {material2} not in database")
            
        mat1 = self.material_database[material1]
        mat2 = self.material_database[material2]
        
        separation = separation or self.casimir_params.plate_separation
        temperature = temperature or self.casimir_params.temperature
        area = self.casimir_params.plate_area
        
        # Fundamental constants
        hbar = 1.055e-34    # J·s
        c = 3e8             # m/s
        k_B = 1.38e-23      # J/K
        
        # Ideal Casimir force (perfect conductors)
        # Mathematical Formula: F = -ℏcπ²A/(240d⁴)
        F_ideal = -hbar * c * np.pi**2 * area / (240 * separation**4)
        
        # Material corrections
        # Proximity force approximation with material properties
        
        # Average conductivity effect
        sigma_avg = np.sqrt(mat1.conductivity * mat2.conductivity)
        
        # Plasma frequency correction
        # For metals: ωₚ = √(ne²/ε₀m)
        # Approximation: conductivity-dependent correction
        plasma_correction = 1.0
        if sigma_avg > 1e6:  # Metallic
            # Finite conductivity reduces force by ~1-5%
            plasma_correction = 0.97  # Empirical correction factor
        elif sigma_avg < 1e-10:  # Dielectric
            # Dielectric materials reduce force significantly
            eps_avg = np.sqrt(mat1.permittivity_real * mat2.permittivity_real)
            plasma_correction = eps_avg / (eps_avg + 1)
        else:  # Semiconductor
            plasma_correction = 0.8
        
        # Surface roughness correction
        # Mathematical Formula: δF/F ≈ -(δ₁² + δ₂²)/d²
        roughness_total = np.sqrt(mat1.surface_roughness**2 + mat2.surface_roughness**2) * 1e-9
        roughness_correction = 1 - (roughness_total / separation)**2
        
        # Temperature correction (thermal wavelength effect)
        # Mathematical Formula: ζ(3) thermal term
        thermal_wavelength = hbar * c / (k_B * temperature)
        if separation > thermal_wavelength:
            # Classical limit
            temperature_correction = 1.0
        else:
            # Quantum limit with thermal corrections
            thermal_ratio = separation / thermal_wavelength
            temperature_correction = 1 - 0.1 * thermal_ratio  # Approximate
        
        # Combined Casimir force
        F_total = F_ideal * plasma_correction * roughness_correction * temperature_correction
        
        # Force gradient (important for AFM measurements)
        dF_dz = -4 * F_total / separation
        
        results = {
            'material1': material1,
            'material2': material2,
            'separation': separation,
            'temperature': temperature,
            'area': area,
            'F_ideal': F_ideal,
            'F_total': F_total,
            'plasma_correction': plasma_correction,
            'roughness_correction': roughness_correction,
            'temperature_correction': temperature_correction,
            'force_gradient': dF_dz,
            'force_per_area': F_total / area,
            'attractive': F_total < 0
        }
        
        logger.info(f"Ideal Casimir force: {F_ideal*1e12:.2f} pN")
        logger.info(f"Corrected force: {F_total*1e12:.2f} pN")
        logger.info(f"Total correction factor: {F_total/F_ideal:.3f}")
        
        return results
    
    def monte_carlo_uncertainty_analysis(self,
                                       material1: str,
                                       material2: str,
                                       n_samples: int = 10000) -> Dict:
        """
        Monte Carlo uncertainty propagation for Casimir force
        
        Args:
            material1: First material name
            material2: Second material name
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Uncertainty analysis results
        """
        logger.info(f"Running Monte Carlo uncertainty analysis ({n_samples} samples)...")
        
        if material1 not in self.material_database:
            raise ValueError(f"Material {material1} not in database")
        if material2 not in self.material_database:
            raise ValueError(f"Material {material2} not in database")
            
        mat1 = self.material_database[material1]
        mat2 = self.material_database[material2]
        
        # Generate random samples for all uncertain parameters
        np.random.seed(42)  # For reproducibility
        
        # Geometric parameters
        separations = np.random.normal(
            self.casimir_params.plate_separation,
            self.casimir_params.separation_uncertainty,
            n_samples
        )
        
        areas = np.random.normal(
            self.casimir_params.plate_area,
            self.casimir_params.area_uncertainty,
            n_samples
        )
        
        temperatures = np.random.normal(
            self.casimir_params.temperature,
            self.casimir_params.temperature_uncertainty,
            n_samples
        )
        
        # Material 1 properties
        sigma1_samples = np.random.lognormal(
            np.log(mat1.conductivity),
            mat1.conductivity_uncertainty,
            n_samples
        )
        
        eps1_real_samples = np.random.normal(
            mat1.permittivity_real,
            abs(mat1.permittivity_real) * mat1.permittivity_uncertainty,
            n_samples
        )
        
        roughness1_samples = np.random.normal(
            mat1.surface_roughness,
            mat1.roughness_uncertainty,
            n_samples
        )
        
        # Material 2 properties
        sigma2_samples = np.random.lognormal(
            np.log(mat2.conductivity),
            mat2.conductivity_uncertainty,
            n_samples
        )
        
        eps2_real_samples = np.random.normal(
            mat2.permittivity_real,
            abs(mat2.permittivity_real) * mat2.permittivity_uncertainty,
            n_samples
        )
        
        roughness2_samples = np.random.normal(
            mat2.surface_roughness,
            mat2.roughness_uncertainty,
            n_samples
        )
        
        # Calculate Casimir force for each sample
        forces = []
        
        # Constants
        hbar = 1.055e-34
        c = 3e8
        k_B = 1.38e-23
        
        for i in range(n_samples):
            # Clip values to physical ranges
            separation = max(separations[i], 1e-9)  # Minimum 1 nm
            area = max(areas[i], 1e-12)             # Minimum area
            temperature = max(temperatures[i], 10)   # Minimum 10 K
            
            # Ideal force
            F_ideal = -hbar * c * np.pi**2 * area / (240 * separation**4)
            
            # Material corrections
            sigma_avg = np.sqrt(sigma1_samples[i] * sigma2_samples[i])
            
            # Plasma correction
            if sigma_avg > 1e6:
                plasma_correction = 0.97
            elif sigma_avg < 1e-10:
                eps_avg = np.sqrt(abs(eps1_real_samples[i] * eps2_real_samples[i]))
                plasma_correction = eps_avg / (eps_avg + 1)
            else:
                plasma_correction = 0.8
            
            # Roughness correction
            roughness_total = np.sqrt(roughness1_samples[i]**2 + roughness2_samples[i]**2) * 1e-9
            roughness_correction = 1 - (roughness_total / separation)**2
            roughness_correction = max(roughness_correction, 0.1)  # Physical limit
            
            # Temperature correction
            thermal_wavelength = hbar * c / (k_B * temperature)
            if separation > thermal_wavelength:
                temperature_correction = 1.0
            else:
                thermal_ratio = separation / thermal_wavelength
                temperature_correction = 1 - 0.1 * thermal_ratio
            
            # Total force
            F_total = F_ideal * plasma_correction * roughness_correction * temperature_correction
            forces.append(F_total)
        
        forces = np.array(forces)
        
        # Statistical analysis
        mean_force = np.mean(forces)
        std_force = np.std(forces)
        relative_uncertainty = std_force / abs(mean_force)
        
        # Percentiles
        percentiles = np.percentile(forces, [5, 25, 50, 75, 95])
        
        # Distribution analysis
        # Test for normality
        _, p_value_normal = stats.shapiro(forces[:5000])  # Limit sample size for test
        is_normal = p_value_normal > 0.05
        
        # Calculate sensitivity indices (Sobol analysis approximation)
        # Variance-based sensitivity analysis
        sensitivity_analysis = self._calculate_sensitivity_indices(
            separations, areas, temperatures,
            sigma1_samples, sigma2_samples,
            eps1_real_samples, eps2_real_samples,
            roughness1_samples, roughness2_samples,
            forces
        )
        
        results = {
            'material1': material1,
            'material2': material2,
            'n_samples': n_samples,
            'mean_force': mean_force,
            'std_force': std_force,
            'relative_uncertainty': relative_uncertainty,
            'percentiles': {
                '5%': percentiles[0],
                '25%': percentiles[1],
                '50%': percentiles[2],
                '75%': percentiles[3],
                '95%': percentiles[4]
            },
            'confidence_interval_95': [percentiles[0], percentiles[4]],
            'force_samples': forces,
            'is_normal_distributed': is_normal,
            'sensitivity_analysis': sensitivity_analysis
        }
        
        self.validation_results[f'monte_carlo_{material1}_{material2}'] = results
        logger.info(f"Mean force: {mean_force*1e12:.2f} ± {std_force*1e12:.2f} pN")
        logger.info(f"Relative uncertainty: {relative_uncertainty*100:.1f}%")
        logger.info(f"95% confidence interval: [{percentiles[0]*1e12:.2f}, {percentiles[4]*1e12:.2f}] pN")
        
        return results
    
    def _calculate_sensitivity_indices(self, 
                                     separations, areas, temperatures,
                                     sigma1, sigma2, eps1, eps2,
                                     roughness1, roughness2, forces) -> Dict:
        """Calculate first-order sensitivity indices"""
        
        # Normalize inputs
        variables = {
            'separation': (separations - np.mean(separations)) / np.std(separations),
            'area': (areas - np.mean(areas)) / np.std(areas),
            'temperature': (temperatures - np.mean(temperatures)) / np.std(temperatures),
            'conductivity1': (sigma1 - np.mean(sigma1)) / np.std(sigma1),
            'conductivity2': (sigma2 - np.mean(sigma2)) / np.std(sigma2),
            'permittivity1': (eps1 - np.mean(eps1)) / np.std(eps1),
            'permittivity2': (eps2 - np.mean(eps2)) / np.std(eps2),
            'roughness1': (roughness1 - np.mean(roughness1)) / np.std(roughness1),
            'roughness2': (roughness2 - np.mean(roughness2)) / np.std(roughness2)
        }
        
        forces_norm = (forces - np.mean(forces)) / np.std(forces)
        
        # Calculate correlation coefficients (proxy for sensitivity)
        sensitivity_indices = {}
        for var_name, var_values in variables.items():
            correlation = np.corrcoef(var_values, forces_norm)[0, 1]
            sensitivity_indices[var_name] = correlation**2  # R² as sensitivity measure
        
        return sensitivity_indices
    
    def analyze_temperature_dependence(self,
                                     material1: str,
                                     material2: str,
                                     temperature_range: Tuple[float, float] = (77, 373)) -> Dict:
        """
        Analyze temperature dependence of Casimir force
        
        Args:
            material1: First material name
            material2: Second material name
            temperature_range: Temperature range (K)
            
        Returns:
            Temperature dependence analysis
        """
        logger.info(f"Analyzing temperature dependence ({temperature_range[0]}-{temperature_range[1]} K)...")
        
        temperatures = np.linspace(temperature_range[0], temperature_range[1], 100)
        forces = []
        force_gradients = []
        
        for temp in temperatures:
            result = self.calculate_casimir_force(material1, material2, temperature=temp)
            forces.append(result['F_total'])
            force_gradients.append(result['force_gradient'])
        
        forces = np.array(forces)
        force_gradients = np.array(force_gradients)
        
        # Calculate temperature coefficients
        # Linear fit: F(T) = F₀ + αT
        T_ref = 300  # Reference temperature
        ref_idx = np.argmin(np.abs(temperatures - T_ref))
        F_ref = forces[ref_idx]
        
        # Temperature coefficient dF/dT
        temp_coefficient = np.gradient(forces, temperatures)
        temp_coefficient_avg = np.mean(temp_coefficient)
        
        # Relative temperature coefficient (1/F)(dF/dT)
        relative_temp_coeff = temp_coefficient / np.abs(forces)
        relative_temp_coeff_avg = np.mean(relative_temp_coeff)
        
        # Find temperature for minimum/maximum force
        min_force_idx = np.argmin(np.abs(forces))
        max_force_idx = np.argmax(np.abs(forces))
        
        min_force_temp = temperatures[min_force_idx]
        max_force_temp = temperatures[max_force_idx]
        
        # Calculate force variation over temperature range
        force_variation = (np.max(np.abs(forces)) - np.min(np.abs(forces))) / np.abs(F_ref)
        
        results = {
            'material1': material1,
            'material2': material2,
            'temperature_range': temperature_range,
            'temperatures': temperatures,
            'forces': forces,
            'force_gradients': force_gradients,
            'reference_force': F_ref,
            'temp_coefficient_avg': temp_coefficient_avg,
            'relative_temp_coeff_avg': relative_temp_coeff_avg,
            'min_force_temperature': min_force_temp,
            'max_force_temperature': max_force_temp,
            'force_variation_relative': force_variation,
            'temperature_stability_requirement': 0.001  # 1 mK for 0.1% force stability
        }
        
        self.validation_results[f'temperature_dependence_{material1}_{material2}'] = results
        logger.info(f"Temperature coefficient: {temp_coefficient_avg*1e15:.2f} fN/K")
        logger.info(f"Relative coefficient: {relative_temp_coeff_avg*1000:.2f} per mK")
        logger.info(f"Force variation: {force_variation*100:.2f}% over range")
        
        return results
    
    def validate_surface_quality_requirements(self,
                                            material1: str,
                                            material2: str,
                                            target_force_accuracy: float = 0.01) -> Dict:
        """
        Validate surface quality requirements for target force accuracy
        
        Args:
            material1: First material name
            material2: Second material name
            target_force_accuracy: Target force accuracy (relative)
            
        Returns:
            Surface quality validation results
        """
        logger.info(f"Validating surface quality requirements for {target_force_accuracy*100:.1f}% accuracy...")
        
        # Calculate reference force
        ref_result = self.calculate_casimir_force(material1, material2)
        F_ref = ref_result['F_total']
        separation = ref_result['separation']
        
        # Analyze surface roughness sensitivity
        roughness_values = np.logspace(-1, 2, 100)  # 0.1 to 100 nm
        force_errors = []
        
        for roughness in roughness_values:
            # Calculate force with modified roughness
            mat1 = self.material_database[material1]
            mat2 = self.material_database[material2]
            
            # Temporarily modify roughness
            original_roughness1 = mat1.surface_roughness
            original_roughness2 = mat2.surface_roughness
            
            mat1.surface_roughness = roughness
            mat2.surface_roughness = roughness
            
            result = self.calculate_casimir_force(material1, material2)
            force_error = abs(result['F_total'] - F_ref) / abs(F_ref)
            force_errors.append(force_error)
            
            # Restore original values
            mat1.surface_roughness = original_roughness1
            mat2.surface_roughness = original_roughness2
        
        force_errors = np.array(force_errors)
        
        # Find roughness limit for target accuracy
        accuracy_mask = force_errors <= target_force_accuracy
        if np.any(accuracy_mask):
            max_roughness = np.max(roughness_values[accuracy_mask])
        else:
            max_roughness = np.min(roughness_values)
        
        # Analytical approximation for roughness effect
        # δF/F ≈ -(δ₁² + δ₂²)/d²
        analytical_roughness_limit = separation * np.sqrt(target_force_accuracy)
        
        # Surface slope requirements (for manufacturing)
        # Approximate slope limit for given roughness
        correlation_length = 100e-9  # 100 nm typical correlation length
        max_slope = max_roughness * 1e-9 / correlation_length  # rad
        max_slope_deg = max_slope * 180 / np.pi
        
        # Manufacturing requirements
        polishing_requirements = self._get_polishing_requirements(max_roughness)
        
        results = {
            'material1': material1,
            'material2': material2,
            'target_accuracy': target_force_accuracy,
            'separation': separation,
            'roughness_values': roughness_values,
            'force_errors': force_errors,
            'max_roughness_numerical': max_roughness,
            'max_roughness_analytical': analytical_roughness_limit * 1e9,  # Convert to nm
            'max_surface_slope': max_slope_deg,
            'polishing_requirements': polishing_requirements,
            'current_roughness_ok': all([
                self.material_database[material1].surface_roughness <= max_roughness,
                self.material_database[material2].surface_roughness <= max_roughness
            ])
        }
        
        self.validation_results[f'surface_quality_{material1}_{material2}'] = results
        logger.info(f"Max roughness (numerical): {max_roughness:.2f} nm")
        logger.info(f"Max roughness (analytical): {analytical_roughness_limit*1e9:.2f} nm")
        logger.info(f"Max surface slope: {max_slope_deg:.3f}°")
        
        return results
    
    def _get_polishing_requirements(self, max_roughness: float) -> Dict:
        """Get manufacturing requirements for surface quality"""
        
        requirements = {}
        
        if max_roughness >= 10:  # nm
            requirements = {
                'polishing_method': 'Mechanical polishing',
                'abrasive_size': '1 μm diamond',
                'time_estimate': '2-4 hours',
                'cost_relative': 1.0,
                'achievable': True
            }
        elif max_roughness >= 1:
            requirements = {
                'polishing_method': 'Chemical-mechanical polishing (CMP)',
                'abrasive_size': '50 nm silica',
                'time_estimate': '4-8 hours',
                'cost_relative': 3.0,
                'achievable': True
            }
        elif max_roughness >= 0.1:
            requirements = {
                'polishing_method': 'Ion beam polishing',
                'abrasive_size': 'Ar+ ions, 500 eV',
                'time_estimate': '8-16 hours',
                'cost_relative': 10.0,
                'achievable': True
            }
        else:
            requirements = {
                'polishing_method': 'Atomic layer deposition + annealing',
                'abrasive_size': 'Atomic-scale',
                'time_estimate': '24+ hours',
                'cost_relative': 50.0,
                'achievable': False  # Beyond current capabilities
            }
        
        return requirements
    
    def compare_material_combinations(self,
                                    materials: List[str] = None) -> Dict:
        """
        Compare Casimir forces for different material combinations
        
        Args:
            materials: List of materials to compare
            
        Returns:
            Material combination comparison results
        """
        logger.info("Comparing material combinations...")
        
        if materials is None:
            materials = list(self.material_database.keys())
        
        combinations = []
        for i, mat1 in enumerate(materials):
            for j, mat2 in enumerate(materials):
                if i <= j:  # Avoid duplicates
                    combinations.append((mat1, mat2))
        
        comparison_results = []
        
        for mat1, mat2 in combinations:
            # Calculate force
            force_result = self.calculate_casimir_force(mat1, mat2)
            
            # Run uncertainty analysis
            uncertainty_result = self.monte_carlo_uncertainty_analysis(mat1, mat2, n_samples=1000)
            
            # Surface quality analysis
            surface_result = self.validate_surface_quality_requirements(mat1, mat2)
            
            combination_result = {
                'material1': mat1,
                'material2': mat2,
                'force_magnitude': abs(force_result['F_total']),
                'force_uncertainty': uncertainty_result['relative_uncertainty'],
                'surface_roughness_limit': surface_result['max_roughness_numerical'],
                'manufacturing_feasible': surface_result['polishing_requirements']['achievable'],
                'cost_factor': surface_result['polishing_requirements']['cost_relative'],
                'force_gradient': abs(force_result['force_gradient']),
                'overall_score': 0  # Will be calculated
            }
            
            # Calculate overall score (higher is better)
            # Normalize factors and combine
            force_score = np.log10(combination_result['force_magnitude'] * 1e12)  # pN scale
            uncertainty_score = 1 / (1 + combination_result['force_uncertainty'])
            surface_score = min(combination_result['surface_roughness_limit'] / 10, 1.0)
            cost_score = 1 / combination_result['cost_factor']
            
            overall_score = force_score * uncertainty_score * surface_score * cost_score
            combination_result['overall_score'] = overall_score
            
            comparison_results.append(combination_result)
        
        # Sort by overall score
        comparison_results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Find best combination
        best_combination = comparison_results[0]
        
        results = {
            'materials_analyzed': materials,
            'combinations': comparison_results,
            'best_combination': best_combination,
            'ranking_criteria': {
                'force_magnitude': 'Higher force magnitude (better signal)',
                'uncertainty': 'Lower relative uncertainty (better precision)',
                'surface_quality': 'Achievable surface roughness (practical)',
                'cost': 'Lower manufacturing cost (economical)'
            }
        }
        
        self.validation_results['material_comparison'] = results
        logger.info(f"Best combination: {best_combination['material1']}-{best_combination['material2']}")
        logger.info(f"Force: {best_combination['force_magnitude']*1e12:.1f} pN")
        logger.info(f"Uncertainty: {best_combination['force_uncertainty']*100:.1f}%")
        
        return results
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive material property validation report"""
        
        report = []
        report.append("=" * 60)
        report.append("MATERIAL PROPERTY UNCERTAINTIES VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: June 30, 2025")
        report.append("")
        
        # Material database summary
        report.append("MATERIAL DATABASE:")
        for name, props in self.material_database.items():
            report.append(f"  {props.name}:")
            report.append(f"    Conductivity: {props.conductivity:.2e} ± {props.conductivity_uncertainty*100:.0f}% S/m")
            report.append(f"    Surface roughness: {props.surface_roughness:.1f} ± {props.roughness_uncertainty:.1f} nm")
            report.append(f"    Work function: {props.work_function:.1f} ± {props.work_function_uncertainty:.1f} eV")
        
        # Monte Carlo results
        mc_keys = [k for k in self.validation_results.keys() if k.startswith('monte_carlo')]
        if mc_keys:
            report.append("\nMONTE CARLO UNCERTAINTY ANALYSIS:")
            for key in mc_keys:
                result = self.validation_results[key]
                materials = f"{result['material1']}-{result['material2']}"
                uncertainty = result['relative_uncertainty']
                status = "ACCEPTABLE" if uncertainty < 0.1 else "HIGH UNCERTAINTY"
                report.append(f"  {materials}: {uncertainty*100:.1f}% uncertainty - {status}")
        
        # Temperature dependence
        temp_keys = [k for k in self.validation_results.keys() if k.startswith('temperature_dependence')]
        if temp_keys:
            report.append("\nTEMPERATURE DEPENDENCE:")
            for key in temp_keys:
                result = self.validation_results[key]
                materials = f"{result['material1']}-{result['material2']}"
                temp_coeff = result['relative_temp_coeff_avg']
                stability_req = result['temperature_stability_requirement']
                force_stability = abs(temp_coeff * stability_req * 1000)  # Per mK
                status = "STABLE" if force_stability < 0.001 else "SENSITIVE"
                report.append(f"  {materials}: {temp_coeff*1000:.3f}/mK, "
                            f"Force stability: {force_stability*100:.2f}% per mK - {status}")
        
        # Surface quality requirements
        surface_keys = [k for k in self.validation_results.keys() if k.startswith('surface_quality')]
        if surface_keys:
            report.append("\nSURFACE QUALITY REQUIREMENTS:")
            for key in surface_keys:
                result = self.validation_results[key]
                materials = f"{result['material1']}-{result['material2']}"
                max_roughness = result['max_roughness_numerical']
                feasible = result['polishing_requirements']['achievable']
                status = "ACHIEVABLE" if feasible else "CHALLENGING"
                report.append(f"  {materials}: Max roughness {max_roughness:.2f} nm - {status}")
        
        # Material comparison
        if 'material_comparison' in self.validation_results:
            result = self.validation_results['material_comparison']
            best = result['best_combination']
            report.append(f"\nMATERIAL COMBINATION RANKING:")
            report.append(f"  Best combination: {best['material1']}-{best['material2']}")
            report.append(f"  Force magnitude: {best['force_magnitude']*1e12:.1f} pN")
            report.append(f"  Uncertainty: {best['force_uncertainty']*100:.1f}%")
            report.append(f"  Surface limit: {best['surface_roughness_limit']:.2f} nm")
            
            report.append("\n  Top 3 combinations:")
            for i, combo in enumerate(result['combinations'][:3]):
                score = combo['overall_score']
                materials = f"{combo['material1']}-{combo['material2']}"
                force = combo['force_magnitude'] * 1e12
                uncertainty = combo['force_uncertainty'] * 100
                report.append(f"    {i+1}. {materials}: {force:.1f} pN, {uncertainty:.1f}% uncertainty")
        
        # Overall assessment
        report.append("")
        report.append("=" * 60)
        report.append("OVERALL ASSESSMENT:")
        
        # Count acceptable results
        acceptable_count = 0
        total_count = 0
        
        for key, result in self.validation_results.items():
            if 'monte_carlo' in key:
                total_count += 1
                if result['relative_uncertainty'] < 0.1:
                    acceptable_count += 1
        
        if total_count > 0:
            success_rate = acceptable_count / total_count
            overall_status = "PASS" if success_rate >= 0.5 else "REVIEW REQUIRED"
            report.append(f"Material combinations with acceptable uncertainty: {acceptable_count}/{total_count}")
            report.append(f"Overall validation status: {overall_status}")
        else:
            report.append("No uncertainty analysis completed")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_material_analysis(self, save_path: str = None):
        """Generate comprehensive material property analysis plots"""
        
        if not self.validation_results:
            logger.warning("No validation results available. Run validation methods first.")
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Plot 1: Casimir force vs separation for different materials
        ax1 = fig.add_subplot(gs[0, 0])
        separations = np.logspace(-8, -6, 100)  # 10 nm to 1 μm
        
        material_pairs = [('gold', 'gold'), ('silicon', 'silicon'), ('gold', 'silicon')]
        colors = ['gold', 'blue', 'red']
        
        for (mat1, mat2), color in zip(material_pairs, colors):
            forces = []
            for sep in separations:
                result = self.calculate_casimir_force(mat1, mat2, separation=sep)
                forces.append(abs(result['F_total']))
            
            ax1.loglog(separations*1e9, np.array(forces)*1e12, 
                      color=color, label=f'{mat1}-{mat2}')
        
        ax1.set_xlabel('Separation (nm)')
        ax1.set_ylabel('Casimir Force (pN)')
        ax1.set_title('Casimir Force vs Separation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Uncertainty analysis histogram
        ax2 = fig.add_subplot(gs[0, 1])
        mc_keys = [k for k in self.validation_results.keys() if k.startswith('monte_carlo')]
        if mc_keys:
            result = self.validation_results[mc_keys[0]]
            forces = result['force_samples'] * 1e12  # Convert to pN
            
            ax2.hist(forces, bins=50, alpha=0.7, density=True)
            ax2.axvline(result['mean_force']*1e12, color='red', 
                       linestyle='--', label='Mean')
            ax2.axvline(result['percentiles']['5%']*1e12, color='orange', 
                       linestyle=':', label='5-95%')
            ax2.axvline(result['percentiles']['95%']*1e12, color='orange', 
                       linestyle=':')
            ax2.set_xlabel('Casimir Force (pN)')
            ax2.set_ylabel('Probability Density')
            ax2.set_title('Force Uncertainty Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Temperature dependence
        ax3 = fig.add_subplot(gs[0, 2])
        temp_keys = [k for k in self.validation_results.keys() if k.startswith('temperature_dependence')]
        if temp_keys:
            result = self.validation_results[temp_keys[0]]
            temperatures = result['temperatures']
            forces = np.array(result['forces']) * 1e12
            
            ax3.plot(temperatures, forces, 'b-', linewidth=2)
            ax3.axhline(result['reference_force']*1e12, color='red', 
                       linestyle='--', label='Reference (300K)')
            ax3.set_xlabel('Temperature (K)')
            ax3.set_ylabel('Casimir Force (pN)')
            ax3.set_title('Temperature Dependence')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Surface roughness sensitivity
        ax4 = fig.add_subplot(gs[0, 3])
        surface_keys = [k for k in self.validation_results.keys() if k.startswith('surface_quality')]
        if surface_keys:
            result = self.validation_results[surface_keys[0]]
            roughness = result['roughness_values']
            errors = result['force_errors'] * 100  # Convert to percentage
            
            ax4.loglog(roughness, errors, 'b-', linewidth=2)
            ax4.axhline(result['target_accuracy']*100, color='red', 
                       linestyle='--', label='Target accuracy')
            ax4.axvline(result['max_roughness_numerical'], color='green', 
                       linestyle=':', label='Max roughness')
            ax4.set_xlabel('Surface Roughness (nm)')
            ax4.set_ylabel('Force Error (%)')
            ax4.set_title('Surface Quality Requirements')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Sensitivity analysis
        ax5 = fig.add_subplot(gs[1, 0:2])
        mc_keys = [k for k in self.validation_results.keys() if k.startswith('monte_carlo')]
        if mc_keys:
            result = self.validation_results[mc_keys[0]]
            sensitivity = result['sensitivity_analysis']
            
            variables = list(sensitivity.keys())
            indices = [sensitivity[var] for var in variables]
            
            bars = ax5.bar(range(len(variables)), indices, alpha=0.7)
            ax5.set_ylabel('Sensitivity Index')
            ax5.set_title('Parameter Sensitivity Analysis')
            ax5.set_xticks(range(len(variables)))
            ax5.set_xticklabels([var.replace('_', '\n') for var in variables], 
                               rotation=45, ha='right')
            ax5.grid(True, alpha=0.3)
            
            # Color bars by sensitivity level
            for bar, index in zip(bars, indices):
                if index > 0.5:
                    bar.set_color('red')     # High sensitivity
                elif index > 0.1:
                    bar.set_color('orange')  # Medium sensitivity
                else:
                    bar.set_color('green')   # Low sensitivity
        
        # Plot 6: Material comparison
        ax6 = fig.add_subplot(gs[1, 2:])
        if 'material_comparison' in self.validation_results:
            result = self.validation_results['material_comparison']
            combinations = result['combinations'][:6]  # Top 6
            
            materials = [f"{c['material1']}-{c['material2']}" for c in combinations]
            forces = [c['force_magnitude']*1e12 for c in combinations]
            uncertainties = [c['force_uncertainty']*100 for c in combinations]
            
            x = np.arange(len(materials))
            width = 0.35
            
            ax6_twin = ax6.twinx()
            bars1 = ax6.bar(x - width/2, forces, width, label='Force (pN)', alpha=0.7)
            bars2 = ax6_twin.bar(x + width/2, uncertainties, width, 
                                label='Uncertainty (%)', alpha=0.7, color='orange')
            
            ax6.set_xlabel('Material Combination')
            ax6.set_ylabel('Casimir Force (pN)')
            ax6_twin.set_ylabel('Uncertainty (%)')
            ax6.set_title('Material Combination Comparison')
            ax6.set_xticks(x)
            ax6.set_xticklabels([m.replace('-', '\n') for m in materials], 
                               rotation=45, ha='right')
            
            # Combined legend
            lines1, labels1 = ax6.get_legend_handles_labels()
            lines2, labels2 = ax6_twin.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            ax6.grid(True, alpha=0.3)
        
        # Plot 7: Material properties correlation matrix
        ax7 = fig.add_subplot(gs[2, 0:2])
        materials = list(self.material_database.keys())
        properties = ['conductivity', 'permittivity_real', 'surface_roughness', 'work_function']
        
        # Create property matrix
        prop_matrix = np.zeros((len(materials), len(properties)))
        for i, mat_name in enumerate(materials):
            mat = self.material_database[mat_name]
            prop_matrix[i, 0] = np.log10(mat.conductivity)
            prop_matrix[i, 1] = mat.permittivity_real
            prop_matrix[i, 2] = mat.surface_roughness
            prop_matrix[i, 3] = mat.work_function
        
        # Normalize for correlation
        from scipy.stats import zscore
        prop_matrix_norm = zscore(prop_matrix, axis=0)
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(prop_matrix_norm.T)
        
        im = ax7.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax7.set_xticks(range(len(properties)))
        ax7.set_yticks(range(len(properties)))
        ax7.set_xticklabels([p.replace('_', '\n') for p in properties])
        ax7.set_yticklabels([p.replace('_', '\n') for p in properties])
        ax7.set_title('Material Property Correlations')
        
        # Add correlation values
        for i in range(len(properties)):
            for j in range(len(properties)):
                text = ax7.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax7, label='Correlation Coefficient')
        
        # Plot 8: Manufacturing requirements
        ax8 = fig.add_subplot(gs[2, 2:])
        if surface_keys:
            result = self.validation_results[surface_keys[0]]
            roughness_limits = []
            costs = []
            materials_analyzed = []
            
            # Get data for all analyzed combinations
            for key in surface_keys:
                res = self.validation_results[key]
                materials_analyzed.append(f"{res['material1']}-{res['material2']}")
                roughness_limits.append(res['max_roughness_numerical'])
                costs.append(res['polishing_requirements']['cost_relative'])
            
            scatter = ax8.scatter(roughness_limits, costs, s=100, alpha=0.7, 
                                c=range(len(materials_analyzed)), cmap='viridis')
            ax8.set_xlabel('Max Roughness Limit (nm)')
            ax8.set_ylabel('Relative Manufacturing Cost')
            ax8.set_title('Manufacturing Requirements')
            ax8.set_xscale('log')
            ax8.set_yscale('log')
            ax8.grid(True, alpha=0.3)
            
            # Add labels
            for i, mat in enumerate(materials_analyzed):
                ax8.annotate(mat.replace('-', '\n'), 
                           (roughness_limits[i], costs[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, ha='left')
        
        plt.suptitle('Material Property Uncertainty Analysis', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Material analysis plots saved to {save_path}")
        else:
            plt.show()

# Example usage and validation
if __name__ == "__main__":
    # Initialize validator
    casimir_params = CasimirParameters(
        plate_separation=10e-9,          # 10 nm
        plate_area=100e-12,              # 100 μm²
        separation_uncertainty=0.1e-9,   # 0.1 nm
        temperature_uncertainty=0.1      # 0.1 K
    )
    
    validator = MaterialPropertyValidator(casimir_params)
    
    print("Starting comprehensive material property uncertainty validation...")
    
    # 1. Calculate Casimir forces for key material combinations
    material_pairs = [('gold', 'gold'), ('silicon', 'silicon'), ('gold', 'silicon')]
    
    for mat1, mat2 in material_pairs:
        print(f"\nAnalyzing {mat1}-{mat2} combination...")
        
        # Basic force calculation
        validator.calculate_casimir_force(mat1, mat2)
        
        # Monte Carlo uncertainty analysis
        validator.monte_carlo_uncertainty_analysis(mat1, mat2, n_samples=5000)
        
        # Temperature dependence
        validator.analyze_temperature_dependence(mat1, mat2)
        
        # Surface quality requirements
        validator.validate_surface_quality_requirements(mat1, mat2)
    
    # 2. Compare all material combinations
    validator.compare_material_combinations()
    
    # Generate and print validation report
    report = validator.generate_validation_report()
    print(report)
    
    # Generate plots
    validator.plot_material_analysis('material_property_validation.png')
    
    print("\nMaterial property uncertainty validation complete!")
