#!/usr/bin/env python3
"""
Test JAX GPU availability and configuration
"""

def test_jax_gpu():
    try:
        import jax
        try:
            from jax.extend import backend
            print("JAX backend:", backend.get_backend().platform)
        except ImportError:
            # Fallback for older JAX versions
            from jax.lib import xla_bridge
            print("JAX backend:", xla_bridge.get_backend().platform)
        
        print("Available devices:", jax.devices())
        
        # Test a simple computation
        import jax.numpy as jnp
        from jax import jit, device_put
        
        @jit
        def test_computation(x):
            return jnp.sum(x ** 2)
        
        # Create test array and move to device
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        x_device = device_put(x)
        
        result = test_computation(x_device)
        print(f"Test computation result: {result}")
        print(f"Result device: {result.device_buffer.device()}")
        
        return True
        
    except ImportError:
        print("JAX not available")
        return False
    except Exception as e:
        print(f"JAX GPU test failed: {e}")
        return False

if __name__ == "__main__":
    test_jax_gpu()
