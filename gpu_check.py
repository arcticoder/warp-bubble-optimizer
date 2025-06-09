#!/usr/bin/env python3
"""
GPU Configuration Check for JAX Warp Engine
===========================================

This script verifies JAX GPU acceleration is properly configured
and provides diagnostics for troubleshooting.
"""

import jax
import os

def check_jax_gpu():
    """Check current JAX configuration and attempt GPU setup."""
    print("=" * 60)
    print("JAX GPU CONFIGURATION CHECK")
    print("=" * 60)
    
    # Check current configuration
    try:
        from jax.lib import xla_bridge
        current_backend = xla_bridge.get_backend().platform
        print(f"Current JAX backend: {current_backend}")
        print(f"Available devices: {jax.devices()}")
        print(f"JAX version: {jax.__version__}")
        
        if current_backend == "gpu":
            print("✅ GPU acceleration is already active!")
            return True
            
    except Exception as e:
        print(f"❌ Error checking JAX backend: {e}")
    
    # Try to force GPU
    print("\n🔄 Attempting to enable GPU acceleration...")
    
    try:
        # Method 1: Environment variable
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        
        # Method 2: Config update
        jax.config.update("jax_platform_name", "gpu")
        
        # Re-check
        from jax.lib import xla_bridge
        new_backend = xla_bridge.get_backend().platform
        print(f"Backend after GPU override: {new_backend}")
        
        if new_backend == "gpu":
            print("✅ GPU acceleration successfully enabled!")
            return True
        else:
            print("⚠️  GPU override failed, still using CPU")
            return False
            
    except Exception as e:
        print(f"❌ Failed to enable GPU: {e}")
        return False

def provide_installation_guidance():
    """Provide installation guidance for JAX GPU support."""
    print("\n" + "=" * 60)
    print("GPU ACCELERATION SETUP GUIDANCE")
    print("=" * 60)
    
    print("If GPU acceleration is not working, try installing JAX with CUDA support:")
    print()
    print("For CUDA 11.x:")
    print('pip install --upgrade "jax[cuda11_local]" \\')
    print('  -f https://storage.googleapis.com/jax-releases/jax_releases.html')
    print()
    print("For CUDA 12.x:")
    print('pip install --upgrade "jax[cuda12_local]" \\')
    print('  -f https://storage.googleapis.com/jax-releases/jax_releases.html')
    print()
    print("Note: GPU acceleration requires:")
    print("  • NVIDIA GPU with CUDA Compute Capability ≥ 3.5")
    print("  • CUDA toolkit installed")
    print("  • Sufficient GPU memory (≥ 4GB recommended)")

def test_basic_jax_performance():
    """Test basic JAX operations to verify functionality."""
    print("\n" + "=" * 60)
    print("BASIC JAX PERFORMANCE TEST")
    print("=" * 60)
    
    import time
    import jax.numpy as jnp
    from jax import jit
    
    # Simple matrix multiplication test
    @jit
    def matrix_multiply(a, b):
        return jnp.dot(a, b)
    
    # Test data
    size = 1000
    a = jnp.ones((size, size))
    b = jnp.ones((size, size))
    
    # Warmup
    _ = matrix_multiply(a, b).block_until_ready()
    
    # Time the operation
    start = time.time()
    for _ in range(10):
        result = matrix_multiply(a, b).block_until_ready()
    elapsed = time.time() - start
    
    print(f"Matrix multiplication test (10 iterations):")
    print(f"  Size: {size}x{size}")
    print(f"  Time: {elapsed:.4f} seconds")
    print(f"  Rate: {10/elapsed:.2f} ops/second")
    
    # Check result correctness
    expected = size * jnp.ones((size, size))
    error = jnp.max(jnp.abs(result - expected))
    print(f"  Max error: {error:.2e}")
    
    if error < 1e-10:
        print("✅ JAX computation is working correctly!")
    else:
        print("❌ JAX computation has errors!")

def main():
    """Run complete GPU check and diagnostics."""
    has_gpu = check_jax_gpu()
    
    if not has_gpu:
        provide_installation_guidance()
    
    test_basic_jax_performance()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if has_gpu:
        print("🚀 Your system is ready for GPU-accelerated warp engine simulations!")
    else:
        print("💻 Running on CPU - JAX still provides significant acceleration")
        print("   Consider installing JAX with CUDA support for maximum performance")

if __name__ == "__main__":
    main()
