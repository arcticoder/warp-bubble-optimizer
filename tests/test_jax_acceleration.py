#!/usr/bin/env python3
"""
JAX Acceleration Demo (optional)

This test is illustrative. It will be skipped automatically if JAX isn't installed.
"""
import time
import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, device_put, vmap
    from functools import partial
    HAS_JAX = True
except Exception:
    HAS_JAX = False


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed in CI env")
def test_einstein_tensor_demo():
    metric = np.diag([1.0, 1.0, 1.0, 1.0])

    def numpy_einstein(metric):
        metric_inv = np.linalg.inv(metric)
        ricci = np.array([[1.0, 0.1, 0.0, 0.0],
                          [0.1, 2.0, 0.0, 0.0],
                          [0.0, 0.0, 1.5, 0.0],
                          [0.0, 0.0, 0.0, 1.8]])
        rs = np.trace(metric_inv @ ricci)
        return ricci - 0.5 * metric * rs

    @jit
    def jax_einstein(metric):
        metric_inv = jnp.linalg.inv(metric)
        ricci = jnp.array([[1.0, 0.1, 0.0, 0.0],
                           [0.1, 2.0, 0.0, 0.0],
                           [0.0, 0.0, 1.5, 0.0],
                           [0.0, 0.0, 0.0, 1.8]])
        rs = jnp.trace(metric_inv @ ricci)
        return ricci - 0.5 * metric * rs

    # Warmup
    _ = jax_einstein(jnp.array(metric))
    # Compare single run consistency
    np_val = numpy_einstein(metric)
    jax_val = np.array(jax_einstein(jnp.array(metric)))
    assert np.allclose(np_val, jax_val, atol=1e-8)
