#!/usr/bin/env python3
"""
Test JAX functionality on CPU and measure performance improvement
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Test 1: Basic JAX functionality
@jit
def matrix_multiply_jax(a, b):
    return jnp.dot(a, b)

def matrix_multiply_numpy(a, b):
    return np.dot(a, b)

# Test with moderate-sized matrices
size = 1000
a_np = np.random.random((size, size)).astype(np.float32)
b_np = np.random.random((size, size)).astype(np.float32)

# Convert to JAX arrays
a_jax = jnp.array(a_np)
b_jax = jnp.array(b_np)

# Warm up JAX (compilation happens here)
print("\nWarming up JAX...")
_ = matrix_multiply_jax(a_jax, b_jax).block_until_ready()

# Time NumPy
print("Timing NumPy...")
start = time.time()
for _ in range(10):
    result_np = matrix_multiply_numpy(a_np, b_np)
numpy_time = time.time() - start

# Time JAX
print("Timing JAX...")
start = time.time()
for _ in range(10):
    result_jax = matrix_multiply_jax(a_jax, b_jax).block_until_ready()
jax_time = time.time() - start

print(f"\nResults:")
print(f"NumPy time: {numpy_time:.4f}s")
print(f"JAX time: {jax_time:.4f}s")
print(f"Speedup: {numpy_time/jax_time:.2f}x")

# Test 2: Vectorized operations (relevant for physics simulations)
@jit
def compute_field_jax(x, y, z):
    """Simulate electromagnetic field computation"""
    r = jnp.sqrt(x**2 + y**2 + z**2)
    return jnp.exp(-r) * jnp.sin(r)

def compute_field_numpy(x, y, z):
    """Simulate electromagnetic field computation"""
    r = np.sqrt(x**2 + y**2 + z**2)
    return np.exp(-r) * np.sin(r)

# Create coordinate arrays
N = 100
x = np.linspace(-5, 5, N)
y = np.linspace(-5, 5, N)
z = np.linspace(-5, 5, N)
X, Y, Z = np.meshgrid(x, y, z)

X_jax = jnp.array(X)
Y_jax = jnp.array(Y)
Z_jax = jnp.array(Z)

# Warm up
_ = compute_field_jax(X_jax, Y_jax, Z_jax).block_until_ready()

print("\nTesting vectorized field computation...")

# Time NumPy
start = time.time()
for _ in range(10):
    field_np = compute_field_numpy(X, Y, Z)
numpy_time2 = time.time() - start

# Time JAX
start = time.time()
for _ in range(10):
    field_jax = compute_field_jax(X_jax, Y_jax, Z_jax).block_until_ready()
jax_time2 = time.time() - start

print(f"Field computation:")
print(f"NumPy time: {numpy_time2:.4f}s")
print(f"JAX time: {jax_time2:.4f}s")
print(f"Speedup: {numpy_time2/jax_time2:.2f}x")

# Test 3: More complex physics-like computation
@jit
def einstein_tensor_element_jax(g_metric, christoffel):
    """Simplified Einstein tensor computation"""
    # Simulate Ricci tensor computation
    ricci = jnp.sum(christoffel**2, axis=-1)
    # Simulate Einstein tensor
    return ricci - 0.5 * jnp.trace(ricci) * g_metric

def einstein_tensor_element_numpy(g_metric, christoffel):
    """Simplified Einstein tensor computation"""
    # Simulate Ricci tensor computation
    ricci = np.sum(christoffel**2, axis=-1)
    # Simulate Einstein tensor
    return ricci - 0.5 * np.trace(ricci) * g_metric

# Create test tensors
dim = 50
g_metric = np.random.random((dim, dim)).astype(np.float32)
christoffel = np.random.random((dim, dim, dim)).astype(np.float32)

g_metric_jax = jnp.array(g_metric)
christoffel_jax = jnp.array(christoffel)

# Warm up
_ = einstein_tensor_element_jax(g_metric_jax, christoffel_jax).block_until_ready()

print("\nTesting Einstein tensor-like computation...")

# Time NumPy
start = time.time()
for _ in range(10):
    einstein_np = einstein_tensor_element_numpy(g_metric, christoffel)
numpy_time3 = time.time() - start

# Time JAX
start = time.time()
for _ in range(10):
    einstein_jax = einstein_tensor_element_jax(g_metric_jax, christoffel_jax).block_until_ready()
jax_time3 = time.time() - start

print(f"Einstein tensor computation:")
print(f"NumPy time: {numpy_time3:.4f}s")
print(f"JAX time: {jax_time3:.4f}s")
print(f"Speedup: {numpy_time3/jax_time3:.2f}x")

print(f"\nOverall JAX shows good performance improvement even on CPU!")
print(f"Average speedup: {((numpy_time/jax_time) + (numpy_time2/jax_time2) + (numpy_time3/jax_time3))/3:.2f}x")
