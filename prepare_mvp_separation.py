#!/usr/bin/env python3
"""
MVP Module Separation Preparation
=================================

This script prepares the separation of the core warp-bubble-optimizer 
framework from the MVP digital-twin simulation suite, facilitating 
the creation of a dedicated MVP module repository.

Repository Structure Planning:
- warp-bubble-optimizer: Core QFT, geometric optimization, atmospheric constraints
- warp-bubble-mvp-simulator: Complete digital-twin hardware suite and MVP simulation

This separation enables:
1. Independent development of core physics vs. hardware simulation
2. Modular deployment for different use cases
3. Simplified maintenance and version control
4. Clear separation of concerns between theory and application
"""

import os
import shutil
from typing import List, Dict
from pathlib import Path

class MVPModuleSeparator:
    """
    Utility class for preparing MVP module separation.
    """
    
    def __init__(self, source_dir: str = ".", target_mvp_dir: str = None):
        self.source_dir = Path(source_dir)
        if target_mvp_dir is None:
            # Use the existing empty repo directory
            self.target_mvp_dir = Path("C:/Users/sherri3/Code/asciimath/warp-bubble-mvp-simulator")
        else:
            self.target_mvp_dir = Path(target_mvp_dir)
    
    def identify_mvp_files(self) -> Dict[str, List[str]]:
        """Identify files that belong to the MVP digital-twin module."""
        
        mvp_files = {
            'core_mvp': [
                'simulate_full_warp_MVP.py',
                'fidelity_runner.py',
                'simulate_power_and_flight_computer.py',
                'simulated_interfaces.py'
            ],
            'demos': [
                'demo_full_warp_simulated_hardware.py',
                'demo_full_warp_pipeline.py'
            ],
            'tests': [
                'test_digital_twins.py',
                'test_mvp_integration.py',
                'MVP_INTEGRATION_TEST.py',
                'SIMPLE_MVP_VALIDATION.py',
                'FINAL_MVP_SUMMARY.py'
            ],
            'documentation': [
                'DIGITAL_TWIN_SUMMARY.py',
                'COMPLETE_DIGITAL_TWIN_FINAL_SUMMARY.py'
            ]
        }
        
        # Check which files actually exist
        existing_files = {}
        for category, files in mvp_files.items():
            existing_files[category] = []
            for file in files:
                if (self.source_dir / file).exists():
                    existing_files[category].append(file)
        
        return existing_files
    
    def identify_core_files(self) -> Dict[str, List[str]]:
        """Identify files that belong to the core optimizer framework."""
        
        core_files = {
            'qft_core': [
                'comprehensive_lqg_framework.py',
                'advanced_energy_analysis.py',
                'polymer_field_quantization.py'
            ],
            'geometric_optimization': [
                'van_den_broeck_optimization.py',
                'advanced_shape_optimizer.py',
                'bspline_control_point_optimizer.py'
            ],
            'atmospheric_physics': [
                'atmospheric_constraints.py',
                'convective_heating_analysis.py'
            ],
            'protection_systems': [
                'integrated_space_protection.py',
                'leo_collision_avoidance.py',
                'micrometeoroid_protection.py'
            ],
            'analysis_tools': [
                'advanced_multi_strategy_optimizer.py',
                'bayes_opt_and_refine.py',
                'analyze_results.py'
            ]
        }
        
        # Check which files actually exist
        existing_files = {}
        for category, files in core_files.items():
            existing_files[category] = []
            for file in files:
                if (self.source_dir / file).exists():
                    existing_files[category].append(file)
        
        return existing_files
    
    def generate_separation_plan(self):
        """Generate a comprehensive separation plan."""
        
        print("🔧 MVP MODULE SEPARATION PLAN")
        print("=" * 50)
        
        mvp_files = self.identify_mvp_files()
        core_files = self.identify_core_files()
        
        print(f"\n📁 Target MVP Repository: {self.target_mvp_dir}")
        print(f"📁 Source Directory: {self.source_dir}")
        
        print(f"\n🎯 MVP MODULE FILES (to be moved):")
        total_mvp_files = 0
        for category, files in mvp_files.items():
            if files:
                print(f"\n   {category.replace('_', ' ').title()}:")
                for file in files:
                    print(f"     ✓ {file}")
                    total_mvp_files += 1
        
        print(f"\n🔬 CORE FRAMEWORK FILES (to remain):")
        total_core_files = 0
        for category, files in core_files.items():
            if files:
                print(f"\n   {category.replace('_', ' ').title()}:")
                for file in files:
                    print(f"     ✓ {file}")
                    total_core_files += 1
        
        print(f"\n📊 SEPARATION STATISTICS:")
        print(f"   MVP Module Files: {total_mvp_files}")
        print(f"   Core Framework Files: {total_core_files}")
        print(f"   Separation Ratio: {total_mvp_files/(total_mvp_files+total_core_files)*100:.1f}% MVP")
        
        return mvp_files, core_files
    
    def prepare_mvp_repository_structure(self):
        """Prepare the MVP repository directory structure."""
        
        print(f"\n🏗️  PREPARING MVP REPOSITORY STRUCTURE")
        print(f"   Target: {self.target_mvp_dir}")
        
        # Create directory structure
        directories = [
            'src',
            'tests', 
            'demos',
            'docs',
            'examples',
            'config'
        ]
        
        for dir_name in directories:
            dir_path = self.target_mvp_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ Created: {dir_name}/")
        
        # Create MVP-specific README
        readme_content = """# Warp Bubble MVP Digital Twin Simulator

Complete digital-twin simulation suite for warp bubble spacecraft development.

## Features

- **Complete Hardware Digital Twins**: Power, flight computer, sensors, exotic generators
- **Adaptive Fidelity Simulation**: Progressive resolution enhancement from coarse to ultra-fine
- **Monte Carlo Reliability Analysis**: Statistical mission success assessment
- **Real-time Performance Monitoring**: >10 Hz control loops with <1% overhead
- **Pure Software Validation**: 100% simulation-based development without hardware

## Quick Start

```bash
# Complete MVP simulation
python src/simulate_full_warp_MVP.py

# Adaptive fidelity progression
python src/fidelity_runner.py

# Hardware validation demo
python demos/demo_full_warp_simulated_hardware.py
```

## Repository Structure

- `src/`: Core MVP simulation modules
- `tests/`: Validation and integration tests
- `demos/`: Demonstration scripts
- `docs/`: Documentation and specifications
- `examples/`: Usage examples and tutorials
- `config/`: Configuration files and parameters

## Requirements

- Python 3.8+
- NumPy, SciPy
- JAX (optional, for acceleration)
- Core warp-bubble-optimizer framework

## Documentation

See `docs/` for comprehensive documentation including:
- MVP architecture overview
- Digital twin specifications
- Performance analysis
- Usage examples
"""
        
        readme_path = self.target_mvp_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"   ✓ Created: README.md")
        
        return True
    
    def generate_migration_script(self):
        """Generate a script to perform the actual file migration."""
        
        mvp_files, _ = self.identify_mvp_files()
        
        migration_script = """#!/usr/bin/env python3
# MVP Module Migration Script
# Auto-generated file migration plan

import shutil
from pathlib import Path

def migrate_mvp_files():
    source_dir = Path(".")
    target_dir = Path("C:/Users/sherri3/Code/asciimath/warp-bubble-mvp-simulator")
    
    file_mappings = {
"""
        
        for category, files in mvp_files.items():
            for file in files:
                if category == 'core_mvp':
                    target_subdir = 'src'
                elif category == 'demos':
                    target_subdir = 'demos'
                elif category in ['tests', 'documentation']:
                    target_subdir = 'tests'
                else:
                    target_subdir = 'src'
                
                migration_script += f'        "{file}": "{target_subdir}/{file}",\n'
        
        migration_script += """    }
    
    for source_file, target_path in file_mappings.items():
        source_path = source_dir / source_file
        full_target_path = target_dir / target_path
        
        if source_path.exists():
            full_target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, full_target_path)
            print(f"Migrated: {source_file} -> {target_path}")
        else:
            print(f"Warning: {source_file} not found")
    
    print("Migration complete!")

if __name__ == "__main__":
    migrate_mvp_files()
"""
        
        script_path = self.source_dir / "migrate_mvp_module.py"
        with open(script_path, 'w') as f:
            f.write(migration_script)
        
        print(f"\n📝 MIGRATION SCRIPT GENERATED:")
        print(f"   Script: migrate_mvp_module.py")
        print(f"   Ready to execute migration when needed")
        
        return script_path

def main():
    """Main execution function."""
    separator = MVPModuleSeparator()
    
    # Generate separation plan
    mvp_files, core_files = separator.generate_separation_plan()
    
    # Prepare MVP repository structure
    separator.prepare_mvp_repository_structure()
    
    # Generate migration script
    separator.generate_migration_script()
    
    print(f"\n🎯 NEXT STEPS FOR MVP MODULE SEPARATION:")
    print(f"   1. Review the separation plan above")
    print(f"   2. Validate all MVP digital twin components are identified")
    print(f"   3. Execute: python migrate_mvp_module.py (when ready)")
    print(f"   4. Update import statements in both repositories")
    print(f"   5. Create separate version control for MVP module")
    print(f"   6. Establish dependency management between repos")
    
    print(f"\n✅ MVP MODULE SEPARATION PREPARATION COMPLETE")

if __name__ == "__main__":
    main()
