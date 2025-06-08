# Warp Bubble Power Pipeline

A comprehensive automated pipeline for warp bubble simulation, optimization, and validation using Discovery 21 Ghost/Phantom EFT parameters.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the Pipeline

```bash
python test_pipeline.py
```

### 3. Run the Complete Pipeline

```bash
# Use default configuration
python warp_bubble_power_pipeline.py

# Use custom configuration
python warp_bubble_power_pipeline.py --config my_config.json --output my_results/
```

## 📋 What the Pipeline Does

The automated pipeline implements the roadmap you specified:

### Step 1: Initialize Components
- Creates optimal Ghost/Phantom EFT energy source with Discovery 21 parameters:
  - M = 1000 (mass scale)
  - α = 0.01 (coupling parameter)
  - β = 0.1 (coupling parameter)
- Integrates with backreaction solver (~15% energy reduction)
- Sets up 4D B-spline metric ansatz

### Step 2: Parameter Sweep
- Sweeps bubble radius (5m, 10m, 20m) and velocity (1000, 5000, 10000 m/s)
- Generates `power_sweep.csv` with quantitative energy requirements
- Creates heatmaps showing energy vs radius/speed

### Step 3: Optimize Metric Shape
- Uses CMA-ES to optimize B-spline control points
- Minimizes total energy while maintaining stability
- Supports parallel evaluation and detailed diagnostics

### Step 4: Validate Optimized Bubble
- Re-runs simulation with optimized parameters
- Applies backreaction corrections
- Validates stability and convergence

### Step 5: Generate Reports
- Creates visualizations (heatmaps, convergence plots)
- Exports comprehensive results (CSV, JSON)
- Generates summary report

## 📊 Expected Outputs

After running the pipeline, you'll find in the results directory:

- `power_sweep.csv` - Energy requirements for different R,v combinations
- `optimization_results.json` - Best control point parameters
- `validation_results.json` - Final optimized bubble configuration
- `parameter_sweep_heatmap.png` - Energy/stability vs R,v visualization
- `optimization_convergence.png` - CMA-ES convergence plot
- `pipeline_summary.txt` - Human-readable summary
- `warp_pipeline.log` - Detailed execution log

## ⚙️ Configuration

Edit `pipeline_config.json` to customize:

```json
{
  "energy_source": {
    "M": 1000,
    "alpha": 0.01,
    "beta": 0.1,
    "R0": 5.0,
    "sigma": 0.2,
    "mu_polymer": 0.1
  },
  "metric_ansatz": "4d",
  "enable_backreaction": true,
  "enable_stability": true,
  "parameter_sweep": {
    "radii": [5.0, 10.0, 20.0],
    "speeds": [1000, 5000, 10000]
  },
  "optimization": {
    "fixed_radius": 10.0,
    "fixed_speed": 5000.0,
    "generations": 30,
    "population_size": 15,
    "initial_step_size": 0.3,
    "energy_weight": 1.0,
    "stability_weight": 0.5,
    "random_seed": 42
  }
}
```

## 🔧 Architecture

### Core Components

1. **`integrated_warp_solver.py`** - Unified solver interface
   - Combines Ghost EFT source, metric ansatz, backreaction, stability
   - Provides single `simulate()` method for parameter sweeps

2. **`cmaes_optimization.py`** - CMA-ES parameter optimization
   - Optimizes metric control points
   - Multi-objective: minimize energy, maximize stability
   - Supports parallel evaluation

3. **`warp_bubble_power_pipeline.py`** - Main automation script
   - Orchestrates all pipeline steps
   - Handles configuration, logging, visualization
   - Exports comprehensive results

### Integration with Existing Framework

The pipeline integrates with your existing components:
- `energy_sources.py` - GhostCondensateEFT with Discovery 21 parameters
- `backreaction_solver.py` - ~15% energy reduction from metric corrections
- `enhanced_warp_solver.py` - 3D mesh validation and stability analysis
- `van_den_broeck_natario.py` - Hybrid metric calculations

## 🎯 Usage Examples

### Quick Test Run
```bash
# Test with minimal parameters
python warp_bubble_power_pipeline.py --config quick_test_config.json
```

### Production Run
```bash
# Full optimization with detailed analysis
python warp_bubble_power_pipeline.py --config production_config.json --output production_results/
```

### Custom Energy Source
```python
from src.warp_qft.energy_sources import GhostCondensateEFT
from src.warp_qft.integrated_warp_solver import WarpBubbleSolver

# Create custom energy source
custom_ghost = GhostCondensateEFT(
    M=2000,      # Different mass scale
    alpha=0.005, # Different coupling
    beta=0.2
)

# Create solver with custom source
solver = WarpBubbleSolver(
    metric_ansatz="4d",
    energy_source=custom_ghost,
    enable_backreaction=True
)

# Run single simulation
result = solver.simulate(radius=15.0, speed=7500.0)
print(f"Energy: {result.energy_total:.2e} J")
```

## 🔍 Debugging

If you encounter issues:

1. **Run the test script first:**
   ```bash
   python test_pipeline.py
   ```

2. **Check the log file:**
   ```bash
   tail -f warp_pipeline.log
   ```

3. **Verify dependencies:**
   ```bash
   pip list | grep -E "(cma|numpy|scipy|matplotlib)"
   ```

4. **Enable debug logging:**
   ```bash
   python warp_bubble_power_pipeline.py --log-level DEBUG
   ```

## 📈 Expected Performance

Based on Discovery 21 parameters and backreaction corrections:

- **Energy Reduction**: ~15% from backreaction effects
- **Optimization**: CMA-ES typically finds good solutions in 20-50 generations  
- **Stability**: Expect stability scores > 0.7 for viable configurations
- **Execution Time**: Full pipeline ~5-30 minutes depending on configuration

## 🔬 Scientific Background

This pipeline implements:

- **Ghost/Phantom EFT**: Effective field theory for negative energy generation
- **Discovery 21 Parameters**: Optimal (M=1000, α=0.01, β=0.1) achieving -1.418×10⁻¹² W ANEC violation
- **Van den Broeck-Natário Metrics**: Volume-optimized warp bubble geometries
- **Backreaction Corrections**: Self-consistent metric evolution reducing energy by ~15%
- **Polymer Field Theory**: Quantum inequality violations for enhanced negative energy

## 🤝 Contributing

To extend the pipeline:

1. **Add new energy sources**: Inherit from `EnergySource` class
2. **Add new metric ansätze**: Extend `WarpBubbleSolver._compute_warp_profile()`
3. **Add new optimizers**: Implement similar interface to `CMAESOptimizer`
4. **Add new analysis**: Extend the validation and visualization steps

## 📚 References

- Discovery 21: Ghost/Phantom EFT optimal parameters
- Van den Broeck (1999): Volume-optimized warp bubbles
- Natário (2002): Warp drive spacetime geometry
- CMA-ES: Covariance Matrix Adaptation Evolution Strategy

---

**Ready to power your warp bubble simulations! 🛸**
