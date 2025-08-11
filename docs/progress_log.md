```file-history
~/Code/asciimath/warp-bubble-optimizer$ find . -path "./.venv" -prune -o -type f -regex '.*\.\(ps1\|py\|sh\|ndjson\|json\|md\|yml\|toml\|h5\|ini\)$' -print | while read file; do stat -c '%Y %n' "$file"; done | sort -nr | while read timestamp file; do echo "$(date -d @$timestamp '+%Y-%m-%d %H:%M:%S') $file"; done | head -n 40
# LATEST-FILES-LIST-BEGIN
2025-08-10 21:37:12 ./gaussian_optimize.py
2025-08-10 21:34:17 ./tests/test_vnv_vector_impulse.py
2025-08-10 21:34:17 ./tests/test_jax_acceleration.py
2025-08-10 21:34:17 ./tests/conftest.py
2025-08-10 21:34:17 ./test_jax_acceleration.py
2025-08-10 21:34:17 ./src/simulation/simulate_vector_impulse.py
2025-08-10 21:34:17 ./simulate_vector_impulse.py
2025-08-10 21:34:17 ./pytest.ini
2025-08-10 21:34:17 ./integrated_impulse_control_clean.py
2025-08-10 21:34:17 ./integrated_impulse_control.py
2025-08-10 21:34:17 ./docs/progress_log.ndjson
2025-08-10 21:34:17 ./docs/progress_log.md
2025-08-10 21:34:17 ./conftest.py
2025-08-10 21:34:17 ./VnV-TODO.ndjson
2025-08-10 21:34:17 ./VnV-TODO-RESOLVED.ndjson
2025-08-10 21:34:17 ./UQ-TODO.ndjson
2025-08-10 21:24:41 ./.vscode/settings.json
2025-08-10 13:00:56 ./docs/progress_log-completed.ndjson
2025-08-10 12:49:58 ./docs/roadmap.ndjson
2025-08-10 12:49:58 ./docs/risk_register.md
2025-08-10 12:49:58 ./docs/requirements.md
2025-08-10 12:49:58 ./docs/power_budget_reconciliation.md
2025-08-10 12:49:58 ./.github/workflows/ci.yml
2025-08-09 22:07:43 ./tests/test_vnv_field_and_plasma.py
2025-08-09 22:07:43 ./tests/test_vnv_coil_driver.py
2025-08-09 22:07:43 ./docs/technical-documentation.md
2025-08-09 19:45:47 ./UQ-TODO-RESOLVED.ndjson
2025-08-09 19:31:25 ./tests/test_vnv_natario_build_metric.py
2025-08-09 19:11:44 ./README.md
2025-08-09 17:25:11 ./ultimate_bspline_optimizer.py
2025-08-09 17:25:11 ./src/optimization/ultimate_bspline_optimizer.py
2025-08-08 22:24:10 ./src/warp_engine/dynamic_sim.py
2025-08-08 22:24:10 ./src/warp_engine/backreaction.py
2025-08-08 22:24:10 ./src/warp_engine/__init__.py
2025-08-08 22:06:27 ./tests/test_power_profile.py
2025-08-08 22:06:27 ./tests/test_field_and_control.py
2025-08-08 22:06:27 ./src/supraluminal_prototype/warp_generator.py
2025-08-08 21:53:44 ./test_3d_stability.py
2025-08-08 21:15:04 ./src/supraluminal_prototype/hardware.py
2025-08-08 21:15:04 ./src/supraluminal_prototype/control.py
# LATEST-FILES-LIST-END

~/Code/asciimath/warp-bubble-optimizer$ ls .. -lt | awk '{print $1, $2, $5, $6, $7, $8, $9}'
# REPO-LIST-BEGIN
total 252     
drwxrwxrwx 30 12288 Aug 10 21:37 warp-bubble-optimizer
drwxrwxrwx 15 12288 Aug 8 07:57 negative-energy-generator
drwxrwxrwx 19 4096 Aug 8 07:02 energy
drwxrwxrwx 8 4096 Aug 1 20:49 casimir-nanopositioning-platform
drwxrwxrwx 22 4096 Aug 1 20:49 enhanced-simulation-hardware-abstraction-framework
drwxrwxrwx 9 4096 Aug 1 20:49 lqg-first-principles-fine-structure-constant
drwxrwxrwx 9 4096 Aug 1 20:49 lqg-positive-matter-assembler
drwxrwxrwx 9 4096 Aug 1 20:49 warp-spacetime-stability-controller
drwxrwxrwx 23 4096 Jul 31 22:38 lqg-ftl-metric-engineering
drwxrwxrwx 7 4096 Jul 31 22:03 lqg-first-principles-gravitational-constant
drwxrwxrwx 7 4096 Jul 31 19:25 warp-solver-equations
drwxrwxrwx 5 4096 Jul 31 19:25 warp-signature-workflow
drwxrwxrwx 9 4096 Jul 31 19:25 warp-sensitivity-analysis
drwxrwxrwx 5 4096 Jul 31 19:25 warp-mock-data-generator
drwxrwxrwx 9 4096 Jul 31 19:25 warp-lqg-midisuperspace
drwxrwxrwx 16 4096 Jul 31 19:25 warp-field-coils
drwxrwxrwx 7 4096 Jul 31 19:25 warp-discretization
drwxrwxrwx 5 4096 Jul 31 19:25 warp-curvature-analysis
drwxrwxrwx 6 4096 Jul 31 19:25 warp-convergence-analysis
drwxrwxrwx 7 4096 Jul 31 19:25 warp-bubble-shape-catalog
drwxrwxrwx 11 4096 Jul 31 19:25 warp-bubble-qft
drwxrwxrwx 5 4096 Jul 31 19:25 warp-bubble-parameter-constraints
drwxrwxrwx 5 4096 Jul 31 19:25 warp-bubble-mvp-simulator
drwxrwxrwx 6 4096 Jul 31 19:25 warp-bubble-metric-ansatz
drwxrwxrwx 5 4096 Jul 31 19:25 warp-bubble-exotic-matter-density
drwxrwxrwx 5 4096 Jul 31 19:25 warp-bubble-einstein-equations
drwxrwxrwx 9 4096 Jul 31 19:25 warp-bubble-coordinate-spec
drwxrwxrwx 5 4096 Jul 31 19:25 warp-bubble-connection-curvature
drwxrwxrwx 5 4096 Jul 31 19:25 warp-bubble-assemble-expressions
drwxrwxrwx 37 12288 Jul 31 19:25 unified-lqg
drwxrwxrwx 8 12288 Jul 31 19:25 unified-lqg-qft
drwxrwxrwx 10 4096 Jul 31 19:25 unified-gut-polymerization
drwxrwxrwx 8 4096 Jul 31 19:25 su2-node-matrix-elements
drwxrwxrwx 11 4096 Jul 31 19:25 su2-3nj-uniform-closed-form
drwxrwxrwx 4 4096 Jul 31 19:25 su2-3nj-recurrences
drwxrwxrwx 10 4096 Jul 31 19:25 su2-3nj-generating-functional
drwxrwxrwx 8 4096 Jul 31 19:25 su2-3nj-closedform
drwxrwxrwx 8 4096 Jul 31 19:25 polymerized-lqg-replicator-recycler
drwxrwxrwx 8 4096 Jul 31 19:25 polymerized-lqg-matter-transporter
drwxrwxrwx 6 4096 Jul 31 19:25 polymer-fusion-framework
drwxrwxrwx 9 4096 Jul 31 19:25 medical-tractor-array
drwxrwxrwx 10 4096 Jul 31 19:25 lqg-volume-quantization-controller
drwxrwxrwx 9 4096 Jul 31 19:25 lqg-volume-kernel-catalog
drwxrwxrwx 10 4096 Jul 31 19:25 lqg-polymer-field-generator
drwxrwxrwx 5 4096 Jul 31 19:25 lqg-cosmological-constant-predictor
drwxrwxrwx 15 12288 Jul 31 19:25 lqg-anec-framework
drwxrwxrwx 12 4096 Jul 31 19:25 lorentz-violation-pipeline
drwxrwxrwx 12 4096 Jul 31 19:25 elemental-transmutator
drwxrwxrwx 6 4096 Jul 31 19:25 casimir-ultra-smooth-fabrication-platform
drwxrwxrwx 7 4096 Jul 31 19:25 casimir-tunable-permittivity-stacks
drwxrwxrwx 7 4096 Jul 31 19:25 casimir-environmental-enclosure-platform
drwxrwxrwx 8 4096 Jul 31 19:25 casimir-anti-stiction-metasurface-coatings
drwxrwxrwx 7 4096 Jul 31 19:25 artificial-gravity-field-generator
# REPO-LIST-END
````

```test-history
(.venv) ~/Code/asciimath/warp-bubble-optimizer$ python3 -m pytest -q
# PYTEST-RESULTS-BEGIN
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.6.0 -- /home/echo_/Code/asciimath/warp-bubble-optimizer/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/echo_/Code/asciimath/warp-bubble-optimizer
configfile: pytest.ini
testpaths: tests
plugins: asyncio-1.1.0
asyncio: mode=Mode.AUTO, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 25 items

tests/test_field_and_control.py::test_zero_expansion_metric PASSED       [  4%]
tests/test_field_and_control.py::test_ring_sync_tolerance PASSED         [  8%]
tests/test_field_and_control.py::test_coil_driver_linearity PASSED       [ 12%]
tests/test_field_and_control.py::test_plasma_density_shell_profile PASSED [ 16%]
tests/test_field_and_control.py::test_field_synthesis_envelope_bounds PASSED [ 20%]
tests/test_field_and_control.py::test_envelope_fit_error_monotonicity_uniform PASSED [ 24%]
tests/test_field_and_control.py::test_tune_ring_amplitudes_uniform_returns_best_controls PASSED [ 28%]
tests/test_field_and_control.py::test_envelope_to_shift_coupling_divergence_small PASSED [ 32%]
tests/test_field_and_control.py::test_optimize_energy_stub_outputs PASSED [ 36%]
tests/test_field_and_control.py::test_battery_feasibility_flag PASSED    [ 40%]
tests/test_field_and_control.py::test_zero_expansion_tolerance_vs_resolution PASSED [ 44%]
tests/test_field_and_control.py::test_discharge_efficiency_affects_feasibility PASSED [ 48%]
tests/test_jax_acceleration.py::test_einstein_tensor_demo SKIPPED (J...) [ 52%]
tests/test_power_profile.py::test_energy_increases_with_ramp_duration PASSED [ 56%]
tests/test_power_profile.py::test_numerical_values_match_analysis PASSED [ 60%]
tests/test_power_profile.py::test_invalid_inputs PASSED                  [ 64%]
tests/test_power_profile.py::test_triangle_shape_equivalence PASSED      [ 68%]
tests/test_vnv_coil_driver.py::test_coil_driver_near_linear_no_hysteresis PASSED [ 72%]
tests/test_vnv_coil_driver.py::test_coil_driver_hysteresis_reduces_gain PASSED [ 76%]
tests/test_vnv_field_and_plasma.py::test_plasma_density_bounds_and_units PASSED [ 80%]
tests/test_vnv_field_and_plasma.py::test_field_synthesis_bench_against_simplified_target PASSED [ 84%]
tests/test_vnv_natario_build_metric.py::test_build_metric_divergence_free PASSED [ 88%]
tests/test_vnv_natario_build_metric.py::test_envelope_coupled_shift_remains_near_div_free PASSED [ 92%]
tests/test_vnv_vector_impulse.py::test_vector_impulse_energy_scales_quadratic PASSED [ 96%]
tests/test_vnv_vector_impulse.py::test_trajectory_accuracy_improves_with_steps PASSED [100%]

======================== 24 passed, 1 skipped in 0.16s =========================
# PYTEST-RESULTS-END
# Never skip a test if an import isn't available. Those tests should fail and the import should be fixed. 
~/Code/asciimath$ grep -r "importerskip" --include="*.py" --exclude="progress_log_processor.py" . | wc -l
# IMPORTERSKIP-RESULTS-BEGIN
0
# IMPORTERSKIP-RESULTS-END
```