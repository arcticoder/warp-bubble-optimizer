```file-history
~/Code/asciimath/warp-bubble-optimizer$ find . -path "./.venv" -prune -o -type f -regex '.*\.\(ps1\|py\|sh\|ndjson\|json\|md\|yml\|toml\|h5\|ini\)$' -print | while read file; do stat -c '%Y %n' "$file"; done | sort -nr | while read timestamp file; do echo "$(date -d @$timestamp '+%Y-%m-%d %H:%M:%S') $file"; done | head -n 40
# LATEST-FILES-LIST-BEGIN
2025-08-09 18:37:16 ./docs/progress_log.ndjson
2025-08-09 18:37:10 ./tests/test_vnv_vector_impulse.py
2025-08-09 18:36:54 ./README.md
2025-08-09 18:36:03 ./.github/workflows/ci.yml
2025-08-09 18:22:30 ./docs/progress_log.md
2025-08-09 18:17:13 ./docs/roadmap.ndjson
2025-08-09 18:17:13 ./docs/VnV-TODO.ndjson
2025-08-09 17:59:30 ./test_jax_acceleration.py
2025-08-09 17:59:30 ./simulate_vector_impulse.py
2025-08-09 17:59:30 ./pytest.ini
2025-08-09 17:59:30 ./gaussian_optimize.py
2025-08-09 17:59:30 ./docs/VnV-TODO-RESOLVED.ndjson
2025-08-09 17:59:30 ./docs/UQ-TODO.ndjson
2025-08-09 17:59:30 ./docs/UQ-TODO-RESOLVED.ndjson
2025-08-09 17:59:30 ./conftest.py
2025-08-09 17:25:11 ./ultimate_bspline_optimizer.py
2025-08-09 17:25:11 ./src/optimization/ultimate_bspline_optimizer.py
2025-08-08 22:24:10 ./src/warp_engine/dynamic_sim.py
2025-08-08 22:24:10 ./src/warp_engine/backreaction.py
2025-08-08 22:24:10 ./src/warp_engine/__init__.py
2025-08-08 22:06:27 ./tests/test_power_profile.py
2025-08-08 22:06:27 ./tests/test_field_and_control.py
2025-08-08 22:06:27 ./src/supraluminal_prototype/warp_generator.py
2025-08-08 21:53:44 ./test_3d_stability.py
2025-08-08 21:53:44 ./VnV-TODO.ndjson
2025-08-08 21:53:44 ./UQ-TODO.ndjson
2025-08-08 21:15:04 ./src/supraluminal_prototype/hardware.py
2025-08-08 21:15:04 ./src/supraluminal_prototype/control.py
2025-08-08 20:48:56 ./src/supraluminal_prototype/power.py
2025-08-08 20:48:56 ./src/supraluminal_prototype/__init__.py
2025-07-31 19:25:46 ./warp_bubble_power_pipeline_automated_clean.py
2025-07-31 19:25:46 ./warp_bubble_power_pipeline_automated.py
2025-07-31 19:25:46 ./warp_bubble_power_pipeline.py
2025-07-31 19:25:46 ./visualize_bubble.py
2025-07-31 19:25:46 ./verify_lqg_enforcement_simple.py
2025-07-31 19:25:46 ./verify_lqg_bound_enforcement.py
2025-07-31 19:25:46 ./ultimate_benchmark_suite.py
2025-07-31 19:25:46 ./time_dependent_optimizer.py
2025-07-31 19:25:46 ./test_ultimate_bspline.py
2025-07-31 19:25:46 ./test_solver_debug.py
# LATEST-FILES-LIST-END

~/Code/asciimath/warp-bubble-optimizer$ ls .. -lt | awk '{print $1, $2, $5, $6, $7, $8, $9}'
# REPO-LIST-BEGIN
total 252     
drwxrwxrwx 30 12288 Aug 9 17:56 warp-bubble-optimizer
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
(.venv) ~/Code/asciimath/warp-bubble-optimizer$ $ python3 -m pytest --maxfail=1
# PYTEST-RESULTS-BEGIN
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.6.0 -- /home/echo_/Code/asciimath/warp-bubble-optimizer/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/echo_/Code/asciimath/warp-bubble-optimizer
configfile: pytest.ini
testpaths: .
plugins: asyncio-1.1.0
asyncio: mode=Mode.AUTO, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 72 items

test_3d_stability.py::test_stability_analyzer_instantiates PASSED        [  1%]
test_accelerated_gaussian.py::test_vectorized_integration PASSED         [  2%]
test_accelerated_gaussian.py::test_parallel_vs_sequential PASSED         [  4%]
test_accelerated_gaussian.py::test_ansatz_comparison PASSED              [  5%]
test_accelerated_gaussian.py::test_hybrid_ansatz PASSED                  [  6%]
test_accelerated_gaussian.py::test_cma_es_availability PASSED            [  8%]
test_accelerated_gaussian.py::test_physics_constraints PASSED            [  9%]
test_all_protection_systems.py::test_all_systems PASSED                  [ 11%]
test_backreaction_timeout.py::test_backreaction_timeout PASSED           [ 12%]
test_digital_twins.py::test_digital_twins PASSED                         [ 13%]
test_final_integration.py::test_imports PASSED                           [ 15%]
test_final_integration.py::test_basic_functionality PASSED               [ 16%]
test_final_integration.py::test_mission_planning PASSED                  [ 18%]
test_final_integration.py::test_mission_execution PASSED                 [ 19%]
test_final_integration.py::test_mission_reporting PASSED                 [ 20%]
test_imports.py::test_imports PASSED                                     [ 22%]
test_integration.py::test_progress_imports PASSED                        [ 23%]
test_integration.py::test_jax_fallback PASSED                            [ 25%]
test_integration.py::test_virtual_control_loop PASSED                    [ 26%]
test_integration.py::test_analog_simulation PASSED                       [ 27%]
test_integration.py::test_jax_optimization PASSED                        [ 29%]
test_integration.py::test_progress_tracker_direct PASSED                 [ 30%]
test_integration.py::test_impulse_engine_simulation PASSED               [ 31%]
test_integration.py::test_vector_impulse_simulation PASSED               [ 33%]
test_integration.py::test_rotation_simulation PASSED                     [ 34%]
test_integration.py::test_integrated_control_system PASSED               [ 36%]
test_integration.py::test_simulation_integration PASSED                  [ 37%]
test_jax_acceleration.py::test_einstein_tensor_computation PASSED        [ 38%]
test_jax_acceleration.py::test_trajectory_simulation PASSED              [ 40%]
test_jax_acceleration.py::test_stress_energy_computation PASSED          [ 41%]
test_jax_gpu.py::test_jax_gpu PASSED                                     [ 43%]
test_mvp_integration.py::test_mvp_components PASSED                      [ 44%]
test_repository.py::TestBasicImports::test_core_module_import PASSED     [ 45%]
test_repository.py::TestBasicImports::test_field_algebra_import PASSED   [ 47%]
test_repository.py::TestBasicImports::test_metrics_import PASSED         [ 48%]
test_repository.py::TestBasicImports::test_optimization_imports PASSED   [ 50%]
test_repository.py::TestNumericalCalculations::test_van_den_broeck_calculation PASSED [ 51%]
test_repository.py::TestNumericalCalculations::test_basic_numpy_operations PASSED [ 52%]
test_repository.py::TestOptimizationTools::test_simple_optimization PASSED [ 54%]
test_repository.py::TestOptimizationTools::test_ansatz_optimizer_creation PASSED [ 55%]
test_repository.py::TestIntegrationUtilities::test_basic_integration PASSED [ 56%]
test_repository.py::TestIntegrationUtilities::test_energy_integration PASSED [ 58%]
test_repository.py::TestAnsatzDevelopment::test_ansatz_builder PASSED    [ 59%]
test_repository.py::TestAnsatzDevelopment::test_novel_ansatz_creation PASSED [ 61%]
test_repository.py::TestDemoScripts::test_demo_metric_optimization_syntax PASSED [ 62%]
test_repository.py::TestDemoScripts::test_advanced_demo_syntax PASSED    [ 63%]
test_repository.py::TestRepositoryStructure::test_src_directory_structure PASSED [ 65%]
test_repository.py::TestRepositoryStructure::test_docs_directory PASSED  [ 66%]
test_repository.py::TestRepositoryStructure::test_essential_files PASSED [ 68%]
test_repository.py::TestRepositoryStructure::test_demo_scripts_exist PASSED [ 69%]
test_repository.py::test_integration_workflow PASSED                     [ 70%]
test_setup.py::test_basic_imports PASSED                                 [ 72%]
test_setup.py::test_basic_functionality PASSED                           [ 73%]
test_simple_integration.py::test_simple PASSED                           [ 75%]
tests/test_field_and_control.py::test_zero_expansion_metric PASSED       [ 76%]
tests/test_field_and_control.py::test_ring_sync_tolerance PASSED         [ 77%]
tests/test_field_and_control.py::test_coil_driver_linearity PASSED       [ 79%]
tests/test_field_and_control.py::test_plasma_density_shell_profile PASSED [ 80%]
tests/test_field_and_control.py::test_field_synthesis_envelope_bounds PASSED [ 81%]
tests/test_field_and_control.py::test_envelope_fit_error_monotonicity_uniform PASSED [ 83%]
tests/test_field_and_control.py::test_tune_ring_amplitudes_uniform_returns_best_controls PASSED [ 84%]
tests/test_field_and_control.py::test_envelope_to_shift_coupling_divergence_small PASSED [ 86%]
tests/test_field_and_control.py::test_optimize_energy_stub_outputs PASSED [ 87%]
tests/test_field_and_control.py::test_battery_feasibility_flag PASSED    [ 88%]
tests/test_field_and_control.py::test_zero_expansion_tolerance_vs_resolution PASSED [ 90%]
tests/test_field_and_control.py::test_discharge_efficiency_affects_feasibility PASSED [ 91%]
tests/test_power_profile.py::test_energy_increases_with_ramp_duration PASSED [ 93%]
tests/test_power_profile.py::test_numerical_values_match_analysis PASSED [ 94%]
tests/test_power_profile.py::test_invalid_inputs PASSED                  [ 95%]
tests/test_power_profile.py::test_triangle_shape_equivalence PASSED      [ 97%]
tests/test_vnv_vector_impulse.py::test_vector_impulse_energy_scales_quadratic PASSED [ 98%]
tests/test_vnv_vector_impulse.py::test_trajectory_accuracy_improves_with_steps PASSED [100%]

============================= 72 passed in 10.93s ==============================
# PYTEST-RESULTS-END
# Never skip a test if an import isn't available. Those tests should fail and the import should be fixed. 
~/Code/asciimath$ grep -r "importerskip" --include="*.py" --exclude="progress_log_processor.py" . | wc -l
# IMPORTERSKIP-RESULTS-BEGIN
0
# IMPORTERSKIP-RESULTS-END
```