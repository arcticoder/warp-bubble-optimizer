#!/usr/bin/env python3
"""
Comprehensive Test Suite for Warp Bubble Optimizer

This test suite validates the core functionality of the warp-bubble-optimizer
repository, including basic imports, numerical calculations, and module integration.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class TestBasicImports:
    """Test that all core modules can be imported successfully."""
    
    def test_core_module_import(self):
        """Test basic warp_qft module import."""
        try:
            import warp_qft
            assert hasattr(warp_qft, '__version__')
        except ImportError as e:
            pytest.fail(f"Failed to import core module: {e}")
    
    def test_field_algebra_import(self):
        """Test field algebra module import."""
        try:
            from warp_qft.field_algebra import PolymerField
            assert PolymerField is not None
        except ImportError as e:
            pytest.skip(f"Field algebra module not available: {e}")
    
    def test_metrics_import(self):
        """Test metrics module import."""
        try:
            from warp_qft.metrics.van_den_broeck_natario import van_den_broeck_shape
            assert callable(van_den_broeck_shape)
        except ImportError as e:
            pytest.skip(f"Metrics module not available: {e}")
    
    def test_optimization_imports(self):
        """Test new optimization module imports."""
        try:
            from warp_qft.variational_optimizer import MetricAnsatzOptimizer
            from warp_qft.numerical_integration import WarpBubbleIntegrator
            from warp_qft.metric_ansatz_development import MetricAnsatzBuilder
            assert all([MetricAnsatzOptimizer, WarpBubbleIntegrator, MetricAnsatzBuilder])
        except ImportError as e:
            pytest.skip(f"New optimization modules not available: {e}")

class TestNumericalCalculations:
    """Test numerical calculations and basic functionality."""
    
    def test_van_den_broeck_calculation(self):
        """Test Van den Broeck shape function calculation."""
        try:
            from warp_qft.metrics.van_den_broeck_natario import van_den_broeck_shape
            
            # Test parameters
            r_values = np.array([0.5, 1.0, 1.5, 2.0])
            R_warp = 1.0
            sigma = 0.1
            
            # Calculate shape function
            f_values = van_den_broeck_shape(r_values, R_warp, sigma)
            
            # Basic validation
            assert len(f_values) == len(r_values)
            assert np.all(np.isfinite(f_values))
            assert np.all(f_values >= 0)  # Shape function should be non-negative
            
        except ImportError:
            pytest.skip("Van den Broeck module not available")
    
    def test_basic_numpy_operations(self):
        """Test that NumPy operations work correctly."""
        # Test array creation and basic operations
        r = np.linspace(0.1, 5.0, 100)
        
        # Test function evaluation
        f = np.exp(-r**2)
        assert len(f) == 100
        assert np.all(f > 0)
        assert f[0] > f[-1]  # Should decay with r
          # Test integration
        integral = np.trapezoid(f, r)  # Updated from deprecated trapz
        assert integral > 0
        assert np.isfinite(integral)

class TestOptimizationTools:
    """Test optimization and variational tools."""
    
    def test_simple_optimization(self):
        """Test basic optimization functionality."""
        from scipy.optimize import minimize
        
        # Simple quadratic function to minimize
        def quadratic(x):
            return (x[0] - 1)**2 + (x[1] + 2)**2
        
        # Optimize
        result = minimize(quadratic, x0=[0, 0], method='L-BFGS-B')
        
        # Check result
        assert result.success
        assert np.allclose(result.x, [1, -2], atol=1e-6)
        assert result.fun < 1e-10
    
    def test_ansatz_optimizer_creation(self):
        """Test creation of ansatz optimizers."""
        try:
            from warp_qft.variational_optimizer import create_ansatz_optimizer
            
            # Create polynomial ansatz
            poly_ansatz = create_ansatz_optimizer('polynomial')
            assert callable(poly_ansatz)
            
            # Test with sample parameters
            test_params = [1.0, -0.1, 0.01]
            ansatz_func = poly_ansatz(test_params)
            assert callable(ansatz_func)
            
        except ImportError:
            pytest.skip("Variational optimizer module not available")

class TestIntegrationUtilities:
    """Test numerical integration utilities."""
    
    def test_basic_integration(self):
        """Test basic numerical integration."""
        from scipy.integrate import quad
        
        # Simple function to integrate
        def integrand(r):
            return r**2 * np.exp(-r)
        
        # Integrate from 0 to infinity (should be 2)
        result, error = quad(integrand, 0, 10)  # Approximate infinity with 10
        
        assert result > 0
        assert error < 1e-6
        assert abs(result - 2.0) < 0.1  # Should be close to analytical result
    
    def test_energy_integration(self):
        """Test energy-type integration."""
        try:
            from warp_qft.numerical_integration import WarpBubbleIntegrator
            
            integrator = WarpBubbleIntegrator()
            
            # Simple radial function
            def radial_func(r):
                return r**2 * np.exp(-r**2)
            
            # Integrate over finite range
            result = integrator.radial_integral(radial_func, (0, 5))
            
            assert result > 0
            assert np.isfinite(result)
            
        except ImportError:
            pytest.skip("Integration utilities not available")

class TestAnsatzDevelopment:
    """Test metric ansatz development tools."""
    
    def test_ansatz_builder(self):
        """Test metric ansatz builder."""
        try:
            from warp_qft.metric_ansatz_development import MetricAnsatzBuilder
            
            builder = MetricAnsatzBuilder()
            assert builder.dim == 4  # Default spacetime dimension
            assert len(builder.coordinate_names) == 4
            
        except ImportError:
            pytest.skip("Ansatz development module not available")
    
    def test_novel_ansatz_creation(self):
        """Test creation of novel ansatzes."""
        try:
            from warp_qft.metric_ansatz_development import create_novel_ansatz
            
            # Create polynomial warp ansatz
            poly_ansatz = create_novel_ansatz('polynomial_warp', degree=2)
            assert callable(poly_ansatz)
            
            # Test evaluation
            test_params = [0.1, -0.05, 1.0]
            metric_func = poly_ansatz(test_params)
            assert callable(metric_func)
            
        except ImportError:
            pytest.skip("Novel ansatz creation not available")

class TestDemoScripts:
    """Test that demo scripts can be loaded without errors."""
    
    def test_demo_metric_optimization_syntax(self):
        """Test that demo_metric_optimization.py has valid syntax."""
        try:
            with open('demo_metric_optimization.py', 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Compile to check syntax
            compile(code, 'demo_metric_optimization.py', 'exec')
            
        except FileNotFoundError:
            pytest.skip("Demo script not found")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in demo script: {e}")
    
    def test_advanced_demo_syntax(self):
        """Test that advanced demo script has valid syntax."""
        try:
            with open('demo_advanced_ansatz_development.py', 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Compile to check syntax
            compile(code, 'demo_advanced_ansatz_development.py', 'exec')
            
        except FileNotFoundError:
            pytest.skip("Advanced demo script not found")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in advanced demo script: {e}")

class TestRepositoryStructure:
    """Test repository structure and file organization."""
    
    def test_src_directory_structure(self):
        """Test that src directory has correct structure."""
        assert os.path.exists('src'), "src directory should exist"
        assert os.path.exists('src/warp_qft'), "warp_qft package should exist"
        assert os.path.exists('src/warp_qft/__init__.py'), "__init__.py should exist"
    
    def test_docs_directory(self):
        """Test that docs directory exists."""
        assert os.path.exists('docs'), "docs directory should exist"
    
    def test_essential_files(self):
        """Test that essential files exist."""
        essential_files = [
            'README.md',
            'requirements.txt',
            'setup.py',
            'pytest.ini',
            '.gitignore'
        ]
        
        for file in essential_files:
            assert os.path.exists(file), f"{file} should exist"
    
    def test_demo_scripts_exist(self):
        """Test that demo scripts exist."""
        demo_scripts = [
            'demo_metric_optimization.py',
            'demo_advanced_ansatz_development.py'
        ]
        
        for script in demo_scripts:
            assert os.path.exists(script), f"{script} should exist"

def test_integration_workflow():
    """Integration test for typical workflow."""
    # This test demonstrates a typical usage workflow
    
    # 1. Basic imports should work
    try:
        import warp_qft
        assert warp_qft.__version__
    except ImportError:
        pytest.skip("Core module not available")
    
    # 2. NumPy calculations should work
    r_values = np.linspace(0.1, 5.0, 50)
    assert len(r_values) == 50
    
    # 3. Basic optimization should work
    from scipy.optimize import minimize_scalar
    
    def simple_func(x):
        return (x - 2)**2
    
    result = minimize_scalar(simple_func, bounds=(0, 5), method='bounded')
    assert abs(result.x - 2.0) < 1e-6
    
    # 4. Integration should work
    from scipy.integrate import quad
    
    result, _ = quad(lambda x: x**2, 0, 1)
    assert abs(result - 1/3) < 1e-6

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
