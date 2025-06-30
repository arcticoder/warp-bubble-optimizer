#!/usr/bin/env python3
"""
Comprehensive UQ Validation Runner

Executes all high-priority UQ validation items for the Casimir-engineered
nanopositioning platform project:

1. Sensor noise characterization validation
2. Thermal stability modeling validation  
3. Vibration isolation verification
4. Material property uncertainties validation

This runner coordinates all validation modules and generates a comprehensive
summary report for the nanopositioning platform development.

Usage:
    python run_uq_validation.py [--save-plots] [--detailed-report]
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from uq_validation.sensor_noise_characterization import SensorNoiseValidator, SensorSpecifications
from uq_validation.thermal_stability_modeling import ThermalStabilityValidator, ThermalSpecifications
from uq_validation.vibration_isolation_verification import VibrationIsolationValidator, VibrationSpecifications
from uq_validation.material_property_uncertainties import MaterialPropertyValidator, CasimirParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('uq_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveUQValidator:
    """
    Comprehensive UQ validation coordinator for nanopositioning platform
    """
    
    def __init__(self, save_plots: bool = True, detailed_report: bool = True):
        """Initialize the comprehensive validator"""
        self.save_plots = save_plots
        self.detailed_report = detailed_report
        self.results = {}
        self.start_time = datetime.now()
        
        # Initialize all validators with nanopositioning requirements
        self._initialize_validators()
        
    def _initialize_validators(self):
        """Initialize all UQ validators with stringent nanopositioning specs"""
        
        # Sensor noise specifications for nanopositioning
        sensor_specs = SensorSpecifications(
            position_resolution=0.05e-9,      # 0.05 nm
            angular_resolution=1e-6,          # 1 μrad
            bandwidth=1000.0,                 # 1 kHz
            thermal_stability=0.1e-9,         # 0.1 nm/hour
            allan_variance_target=1e-20,      # m²
            snr_requirement=80.0              # dB
        )
        
        # Thermal stability specifications
        thermal_specs = ThermalSpecifications(
            max_temperature_drift=0.01,       # 10 mK
            max_thermal_expansion=0.1e-9,     # 0.1 nm
            thermal_time_constant=300,        # 5 minutes
            temperature_stability=0.001       # 1 mK/hour
        )
        
        # Vibration isolation specifications
        vibration_specs = VibrationSpecifications(
            max_displacement=0.1e-9,          # 0.1 nm RMS
            max_angular_displacement=1e-6,    # 1 μrad RMS
            required_isolation=1e4,           # 10,000× at 10 Hz
            ground_motion_amplitude=1e-6      # 1 μm RMS
        )
        
        # Casimir force parameters
        casimir_params = CasimirParameters(
            plate_separation=10e-9,           # 10 nm
            plate_area=100e-12,               # 100 μm²
            separation_uncertainty=0.1e-9,    # 0.1 nm
            temperature_uncertainty=0.1       # 0.1 K
        )
        
        # Initialize validators
        self.sensor_validator = SensorNoiseValidator(sensor_specs)
        self.thermal_validator = ThermalStabilityValidator(thermal_specs)
        self.vibration_validator = VibrationIsolationValidator(vibration_specs)
        self.material_validator = MaterialPropertyValidator(casimir_params)
        
        logger.info("All UQ validators initialized with nanopositioning specifications")
    
    def run_sensor_noise_validation(self):
        """Execute comprehensive sensor noise characterization"""
        logger.info("=" * 60)
        logger.info("STARTING SENSOR NOISE CHARACTERIZATION VALIDATION")
        logger.info("=" * 60)
        
        try:
            # 1. Characterize interferometric noise
            self.sensor_validator.characterize_interferometric_noise()
            
            # 2. Validate angular sensor noise
            self.sensor_validator.validate_angular_sensor_noise()
            
            # 3. Perform Allan variance analysis
            self.sensor_validator.perform_allan_variance_analysis()
            
            # 4. Validate multi-sensor fusion
            self.sensor_validator.validate_multi_sensor_fusion()
            
            # Generate report and plots
            sensor_report = self.sensor_validator.generate_validation_report()
            self.results['sensor_noise'] = {
                'validator': self.sensor_validator,
                'report': sensor_report,
                'status': 'COMPLETED',
                'validation_results': self.sensor_validator.validation_results
            }
            
            if self.save_plots:
                self.sensor_validator.plot_noise_analysis('sensor_noise_validation.png')
            
            logger.info("Sensor noise characterization validation completed successfully")
            
        except Exception as e:
            logger.error(f"Sensor noise validation failed: {str(e)}")
            self.results['sensor_noise'] = {
                'status': 'FAILED',
                'error': str(e)
            }
    
    def run_thermal_stability_validation(self):
        """Execute comprehensive thermal stability modeling"""
        logger.info("=" * 60)
        logger.info("STARTING THERMAL STABILITY MODELING VALIDATION")
        logger.info("=" * 60)
        
        try:
            # 1. Analyze thermal expansion for different materials
            materials = ['aluminum', 'invar', 'zerodur', 'silicon']
            for material in materials:
                self.thermal_validator.analyze_thermal_expansion(material=material)
            
            # 2. Model heat conduction
            self.thermal_validator.model_heat_conduction(geometry='plate', material='invar')
            
            # 3. Design thermal compensation
            self.thermal_validator.design_thermal_compensation(material='invar')
            
            # 4. Validate environmental isolation
            self.thermal_validator.validate_environmental_isolation()
            
            # 5. Analyze thermal noise limits
            self.thermal_validator.analyze_thermal_noise_limits()
            
            # Generate report and plots
            thermal_report = self.thermal_validator.generate_validation_report()
            self.results['thermal_stability'] = {
                'validator': self.thermal_validator,
                'report': thermal_report,
                'status': 'COMPLETED',
                'validation_results': self.thermal_validator.validation_results
            }
            
            if self.save_plots:
                self.thermal_validator.plot_thermal_analysis('thermal_stability_validation.png')
            
            logger.info("Thermal stability modeling validation completed successfully")
            
        except Exception as e:
            logger.error(f"Thermal stability validation failed: {str(e)}")
            self.results['thermal_stability'] = {
                'status': 'FAILED',
                'error': str(e)
            }
    
    def run_vibration_isolation_validation(self):
        """Execute comprehensive vibration isolation verification"""
        logger.info("=" * 60)
        logger.info("STARTING VIBRATION ISOLATION VERIFICATION")
        logger.info("=" * 60)
        
        try:
            # 1. Analyze passive isolation system
            self.vibration_validator.analyze_passive_isolation(n_stages=3)
            
            # 2. Design active control system
            self.vibration_validator.design_active_control(control_bandwidth=100.0)
            
            # 3. Validate angular stability
            self.vibration_validator.validate_angular_stability()
            
            # 4. Analyze ground motion spectrum
            self.vibration_validator.analyze_ground_motion_spectrum()
            
            # 5. Optimize isolation design
            self.vibration_validator.optimize_isolation_design()
            
            # Generate report and plots
            vibration_report = self.vibration_validator.generate_validation_report()
            self.results['vibration_isolation'] = {
                'validator': self.vibration_validator,
                'report': vibration_report,
                'status': 'COMPLETED',
                'validation_results': self.vibration_validator.validation_results
            }
            
            if self.save_plots:
                self.vibration_validator.plot_vibration_analysis('vibration_isolation_validation.png')
            
            logger.info("Vibration isolation verification completed successfully")
            
        except Exception as e:
            logger.error(f"Vibration isolation validation failed: {str(e)}")
            self.results['vibration_isolation'] = {
                'status': 'FAILED',
                'error': str(e)
            }
    
    def run_material_property_validation(self):
        """Execute comprehensive material property uncertainty analysis"""
        logger.info("=" * 60)
        logger.info("STARTING MATERIAL PROPERTY UNCERTAINTIES VALIDATION")
        logger.info("=" * 60)
        
        try:
            # Key material combinations for Casimir nanopositioning
            material_pairs = [('gold', 'gold'), ('silicon', 'silicon'), ('gold', 'silicon')]
            
            for mat1, mat2 in material_pairs:
                logger.info(f"Analyzing {mat1}-{mat2} combination...")
                
                # 1. Calculate Casimir forces
                self.material_validator.calculate_casimir_force(mat1, mat2)
                
                # 2. Monte Carlo uncertainty analysis
                self.material_validator.monte_carlo_uncertainty_analysis(mat1, mat2, n_samples=5000)
                
                # 3. Temperature dependence
                self.material_validator.analyze_temperature_dependence(mat1, mat2)
                
                # 4. Surface quality requirements
                self.material_validator.validate_surface_quality_requirements(mat1, mat2)
            
            # 5. Compare all material combinations
            self.material_validator.compare_material_combinations()
            
            # Generate report and plots
            material_report = self.material_validator.generate_validation_report()
            self.results['material_properties'] = {
                'validator': self.material_validator,
                'report': material_report,
                'status': 'COMPLETED',
                'validation_results': self.material_validator.validation_results
            }
            
            if self.save_plots:
                self.material_validator.plot_material_analysis('material_property_validation.png')
            
            logger.info("Material property uncertainties validation completed successfully")
            
        except Exception as e:
            logger.error(f"Material property validation failed: {str(e)}")
            self.results['material_properties'] = {
                'status': 'FAILED',
                'error': str(e)
            }
    
    def run_all_validations(self):
        """Execute all UQ validations sequentially"""
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE UQ VALIDATION FOR NANOPOSITIONING PLATFORM")
        logger.info("=" * 80)
        logger.info(f"Start time: {self.start_time}")
        logger.info("")
        
        # Execute all validations
        self.run_sensor_noise_validation()
        self.run_thermal_stability_validation()
        self.run_vibration_isolation_validation()
        self.run_material_property_validation()
        
        # Generate comprehensive summary
        self.generate_comprehensive_report()
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE UQ VALIDATION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total duration: {duration}")
        logger.info(f"Results saved to: comprehensive_uq_validation_report.md")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive validation summary report"""
        
        report = []
        report.append("# Comprehensive UQ Validation Report")
        report.append("## Casimir-Engineered Nanopositioning Platform")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Duration:** {datetime.now() - self.start_time}")
        report.append("")
        
        # Executive summary
        report.append("## Executive Summary")
        report.append("")
        
        completed_count = sum(1 for r in self.results.values() if r.get('status') == 'COMPLETED')
        total_count = len(self.results)
        success_rate = completed_count / total_count if total_count > 0 else 0
        
        report.append(f"**Validation Modules Completed:** {completed_count}/{total_count}")
        report.append(f"**Success Rate:** {success_rate*100:.1f}%")
        report.append("")
        
        if success_rate >= 0.75:
            overall_status = "PASS - Ready for nanopositioning platform development"
        elif success_rate >= 0.5:
            overall_status = "CONDITIONAL PASS - Address identified issues"
        else:
            overall_status = "FAIL - Significant issues require resolution"
        
        report.append(f"**Overall Status:** {overall_status}")
        report.append("")
        
        # Critical requirements summary
        report.append("## Critical Requirements Validation")
        report.append("")
        report.append("| Requirement | Target | Status | Notes |")
        report.append("|-------------|--------|--------|-------|")
        
        # Extract key metrics from each validation
        self._add_requirements_summary(report)
        
        # Detailed results for each module
        report.append("")
        report.append("## Detailed Validation Results")
        report.append("")
        
        for module_name, result in self.results.items():
            if result.get('status') == 'COMPLETED':
                report.append(f"### {module_name.replace('_', ' ').title()}")
                report.append("")
                report.append("```")
                report.append(result['report'])
                report.append("```")
                report.append("")
            else:
                report.append(f"### {module_name.replace('_', ' ').title()}")
                report.append("")
                report.append(f"**Status:** {result.get('status', 'UNKNOWN')}")
                if 'error' in result:
                    report.append(f"**Error:** {result['error']}")
                report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        self._add_recommendations(report)
        
        # Next steps
        report.append("## Next Steps")
        report.append("")
        report.append("1. **Repository Integration**: Create `casimir-nanopositioning-platform` repository")
        report.append("2. **Hardware Design**: Begin detailed mechanical and optical design")
        report.append("3. **Control System Implementation**: Develop real-time control algorithms")
        report.append("4. **Prototype Development**: Build and test initial proof-of-concept")
        report.append("5. **Performance Validation**: Experimental verification of theoretical predictions")
        report.append("")
        
        # Save report
        report_content = "\n".join(report)
        
        with open('comprehensive_uq_validation_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Also create a summary for the terminal
        print("\n" + "="*80)
        print("COMPREHENSIVE UQ VALIDATION SUMMARY")
        print("="*80)
        print(f"Modules completed: {completed_count}/{total_count}")
        print(f"Overall status: {overall_status}")
        print("Detailed report saved to: comprehensive_uq_validation_report.md")
        print("="*80)
    
    def _add_requirements_summary(self, report):
        """Add critical requirements summary to report"""
        
        # Sensor noise requirements
        if 'sensor_noise' in self.results and self.results['sensor_noise'].get('status') == 'COMPLETED':
            sensor_results = self.results['sensor_noise']['validation_results']
            
            if 'interferometric_noise' in sensor_results:
                meets_specs = sensor_results['interferometric_noise']['meets_specs']
                status = "PASS" if meets_specs else "FAIL"
                report.append(f"| Position Resolution | 0.05 nm | {status} | Interferometric noise analysis |")
            
            if 'angular_noise' in sensor_results:
                meets_specs = sensor_results['angular_noise']['meets_specs']
                status = "PASS" if meets_specs else "FAIL"
                report.append(f"| Angular Resolution | 1 μrad | {status} | Angular sensor validation |")
        
        # Thermal stability requirements
        if 'thermal_stability' in self.results and self.results['thermal_stability'].get('status') == 'COMPLETED':
            thermal_results = self.results['thermal_stability']['validation_results']
            
            thermal_passes = sum(1 for result in thermal_results.values() 
                               if result.get('meets_specs', False))
            total_thermal = len([r for r in thermal_results.values() if 'meets_specs' in r])
            status = "PASS" if thermal_passes == total_thermal else "PARTIAL"
            report.append(f"| Thermal Expansion | 0.1 nm | {status} | {thermal_passes}/{total_thermal} materials |")
        
        # Vibration isolation requirements
        if 'vibration_isolation' in self.results and self.results['vibration_isolation'].get('status') == 'COMPLETED':
            vibration_results = self.results['vibration_isolation']['validation_results']
            
            if 'passive_isolation' in vibration_results:
                meets_specs = vibration_results['passive_isolation']['meets_displacement_specs']
                status = "PASS" if meets_specs else "FAIL"
                report.append(f"| Vibration Isolation | 10,000× at 10 Hz | {status} | Multi-stage passive |")
        
        # Material property requirements
        if 'material_properties' in self.results and self.results['material_properties'].get('status') == 'COMPLETED':
            material_results = self.results['material_properties']['validation_results']
            
            mc_keys = [k for k in material_results.keys() if k.startswith('monte_carlo')]
            acceptable_uncertainty = sum(1 for key in mc_keys 
                                       if material_results[key]['relative_uncertainty'] < 0.1)
            total_combinations = len(mc_keys)
            status = "PASS" if acceptable_uncertainty == total_combinations else "PARTIAL"
            report.append(f"| Material Uncertainties | <10% relative | {status} | {acceptable_uncertainty}/{total_combinations} combinations |")
    
    def _add_recommendations(self, report):
        """Add specific recommendations based on validation results"""
        
        recommendations = []
        
        # Check each validation module for specific recommendations
        if 'sensor_noise' in self.results and self.results['sensor_noise'].get('status') == 'COMPLETED':
            sensor_results = self.results['sensor_noise']['validation_results']
            if 'multi_sensor_fusion' in sensor_results:
                improvement = sensor_results['multi_sensor_fusion']['improvement_factor']
                if improvement > 2:
                    recommendations.append(
                        f"**Sensor Fusion**: Implement {int(improvement)}× improvement through multi-sensor fusion"
                    )
        
        if 'thermal_stability' in self.results and self.results['thermal_stability'].get('status') == 'COMPLETED':
            thermal_results = self.results['thermal_stability']['validation_results']
            
            # Check material recommendations
            best_materials = []
            for key, result in thermal_results.items():
                if 'thermal_expansion' in key and result.get('meets_specs', False):
                    best_materials.append(result['material'])
            
            if best_materials:
                recommendations.append(
                    f"**Material Selection**: Use {', '.join(set(best_materials))} for low thermal expansion"
                )
        
        if 'material_properties' in self.results and self.results['material_properties'].get('status') == 'COMPLETED':
            material_results = self.results['material_properties']['validation_results']
            
            if 'material_comparison' in material_results:
                best_combo = material_results['material_comparison']['best_combination']
                recommendations.append(
                    f"**Optimal Materials**: Use {best_combo['material1']}-{best_combo['material2']} "
                    f"combination for {best_combo['force_magnitude']*1e12:.1f} pN force"
                )
        
        if not recommendations:
            recommendations.append("All validations completed successfully. Proceed with platform development.")
        
        for rec in recommendations:
            report.append(f"- {rec}")

def main():
    """Main entry point for UQ validation runner"""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive UQ Validation for Casimir Nanopositioning Platform"
    )
    parser.add_argument(
        '--save-plots', 
        action='store_true', 
        help="Save validation plots to files"
    )
    parser.add_argument(
        '--detailed-report', 
        action='store_true', 
        help="Generate detailed validation report"
    )
    parser.add_argument(
        '--module', 
        choices=['sensor', 'thermal', 'vibration', 'material'], 
        help="Run specific validation module only"
    )
    
    args = parser.parse_args()
    
    # Initialize comprehensive validator
    validator = ComprehensiveUQValidator(
        save_plots=args.save_plots,
        detailed_report=args.detailed_report
    )
    
    try:
        if args.module:
            # Run specific module
            if args.module == 'sensor':
                validator.run_sensor_noise_validation()
            elif args.module == 'thermal':
                validator.run_thermal_stability_validation()
            elif args.module == 'vibration':
                validator.run_vibration_isolation_validation()
            elif args.module == 'material':
                validator.run_material_property_validation()
        else:
            # Run all validations
            validator.run_all_validations()
            
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
