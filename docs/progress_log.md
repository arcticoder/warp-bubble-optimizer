```file-history
~/Code/asciimath/warp-bubble-optimizer$ find . -path "./.venv" -prune -o -type f -regex '.*\.\(ps1\|py\|sh\|ndjson\|json\|md\|yml\|toml\|h5\|ini\)$' -print | while read file; do stat -c '%Y %n' "$file"; done | sort -nr | while read timestamp file; do echo "$(date -d @$timestamp '+%Y-%m-%d %H:%M:%S') $file"; done | head -n 40
# LATEST-FILES-LIST-BEGIN
2025-08-09 17:25:11 ./ultimate_bspline_optimizer.py
2025-08-09 17:25:11 ./src/optimization/ultimate_bspline_optimizer.py
2025-08-09 17:25:11 ./docs/roadmap.ndjson
2025-08-09 17:25:11 ./docs/progress_log.ndjson
2025-08-09 17:25:11 ./docs/VnV-TODO.ndjson
2025-08-09 17:25:11 ./docs/UQ-TODO.ndjson
2025-08-08 22:33:12 ./docs/progress_log.md
2025-08-08 22:24:10 ./src/warp_engine/dynamic_sim.py
2025-08-08 22:24:10 ./src/warp_engine/backreaction.py
2025-08-08 22:24:10 ./src/warp_engine/__init__.py
2025-08-08 22:06:27 ./tests/test_power_profile.py
2025-08-08 22:06:27 ./tests/test_field_and_control.py
2025-08-08 22:06:27 ./src/supraluminal_prototype/warp_generator.py
2025-08-08 22:06:27 ./gaussian_optimize.py
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
2025-07-31 19:25:46 ./test_soliton_3d_stability.py
2025-07-31 19:25:46 ./test_simple_integration.py
2025-07-31 19:25:46 ./test_setup.py
2025-07-31 19:25:46 ./test_repository.py
2025-07-31 19:25:46 ./test_pipeline.py
2025-07-31 19:25:46 ./test_mvp_integration.py
2025-07-31 19:25:46 ./test_lqg_bounds_focused.py
2025-07-31 19:25:46 ./test_jax_gpu.py
2025-07-31 19:25:46 ./test_jax_cpu.py
# LATEST-FILES-LIST-END

~/Code/asciimath/warp-bubble-optimizer$ ls .. -lt | awk '{print $1, $2, $5, $6, $7, $8, $9}'
# REPO-LIST-BEGIN
total 252     
drwxrwxrwx 30 12288 Aug 8 22:36 warp-bubble-optimizer
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
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/echo_/Code/asciimath/warp-bubble-optimizer
configfile: pytest.ini
collected 77 items

test_3d_stability.py .                                                   [  1%]
test_accelerated_gaussian.py ......                                      [  9%]
test_all_protection_systems.py .                                         [ 10%]
test_backreaction_timeout.py .                                           [ 11%]
test_digital_twins.py .                                                  [ 12%]
test_final_integration.py ...E

==================================== ERRORS ====================================
___________________ ERROR at setup of test_mission_execution ___________________
file /home/echo_/Code/asciimath/warp-bubble-optimizer/test_final_integration.py, line 111
  async def test_mission_execution(trajectory_plan):
      """Test mission execution."""
      print("\n🔍 Testing mission execution...")

      try:
          from integrated_impulse_control import IntegratedImpulseController, ImpulseEngineConfig

          config = ImpulseEngineConfig(energy_budget=1e12)
          controller = IntegratedImpulseController(config)

          # Execute mission (open-loop mode for testing)
          mission_results = await controller.execute_impulse_mission(
              trajectory_plan, enable_feedback=False
          )

          metrics = mission_results['performance_metrics']
          print(f"   Mission success: {mission_results['mission_success']}")
          print(f"   Success rate: {metrics['overall_success_rate']*100:.1f}%")
          print(f"   Energy used: {metrics['total_energy_used']/1e9:.2f} GJ")
          print(f"   Mission time: {metrics['mission_duration_hours']:.2f} hours")

          print("✅ Mission execution test passed")
          return True, mission_results

      except Exception as e:
          print(f"❌ Mission execution failed: {e}")
          traceback.print_exc()
          return False, None
E       fixture 'trajectory_plan' not found
>       available fixtures: cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, doctest_namespace, monkeypatch, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory
>       use 'pytest --fixtures [testpath]' for help on them.

/home/echo_/Code/asciimath/warp-bubble-optimizer/test_final_integration.py:111
=============================== warnings summary ===============================
test_accelerated_gaussian.py: 656 warnings
  /home/echo_/Code/asciimath/warp-bubble-optimizer/gaussian_optimize.py:62: DeprecationWarning: `trapz` is deprecated. Use `trapezoid` instead, or one of the numerical integration functions in `scipy.integrate`.
    E = np.trapz(integrand, r_grid)

test_accelerated_gaussian.py::test_vectorized_integration
test_accelerated_gaussian.py::test_vectorized_integration
test_accelerated_gaussian.py::test_vectorized_integration
test_accelerated_gaussian.py::test_vectorized_integration
test_accelerated_gaussian.py::test_vectorized_integration
test_accelerated_gaussian.py::test_vectorized_integration
test_accelerated_gaussian.py::test_vectorized_integration
test_accelerated_gaussian.py::test_vectorized_integration
  /home/echo_/Code/asciimath/warp-bubble-optimizer/gaussian_optimize.py:77: DeprecationWarning: `trapz` is deprecated. Use `trapezoid` instead, or one of the numerical integration functions in `scipy.integrate`.
    E_chunk = float(np.trapz(f**2 * (4.0*np.pi*rg**2), rg))

test_accelerated_gaussian.py::test_vectorized_integration
  /home/echo_/Code/asciimath/warp-bubble-optimizer/.venv/lib/python3.13/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but test_accelerated_gaussian.py::test_vectorized_integration returned <class 'dict'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

test_accelerated_gaussian.py::test_parallel_vs_sequential
  /home/echo_/Code/asciimath/warp-bubble-optimizer/.venv/lib/python3.13/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but test_accelerated_gaussian.py::test_parallel_vs_sequential returned <class 'tuple'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

test_accelerated_gaussian.py::test_ansatz_comparison
  /home/echo_/Code/asciimath/warp-bubble-optimizer/.venv/lib/python3.13/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but test_accelerated_gaussian.py::test_ansatz_comparison returned <class 'dict'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

test_accelerated_gaussian.py::test_hybrid_ansatz
  /home/echo_/Code/asciimath/warp-bubble-optimizer/gaussian_optimize.py:159: DeprecationWarning: `trapz` is deprecated. Use `trapezoid` instead, or one of the numerical integration functions in `scipy.integrate`.
    E = float(np.trapz(f**2 * vol_weights, r_grid))

test_accelerated_gaussian.py::test_hybrid_ansatz
  /home/echo_/Code/asciimath/warp-bubble-optimizer/.venv/lib/python3.13/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but test_accelerated_gaussian.py::test_hybrid_ansatz returned <class 'float'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

test_accelerated_gaussian.py::test_cma_es_availability
  /home/echo_/Code/asciimath/warp-bubble-optimizer/.venv/lib/python3.13/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but test_accelerated_gaussian.py::test_cma_es_availability returned <class 'tuple'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

test_digital_twins.py::test_digital_twins
  /home/echo_/Code/asciimath/warp-bubble-optimizer/.venv/lib/python3.13/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but test_digital_twins.py::test_digital_twins returned <class 'bool'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

test_final_integration.py::test_imports
  /home/echo_/Code/asciimath/warp-bubble-optimizer/.venv/lib/python3.13/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but test_final_integration.py::test_imports returned <class 'bool'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

test_final_integration.py::test_basic_functionality
  /home/echo_/Code/asciimath/warp-bubble-optimizer/.venv/lib/python3.13/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but test_final_integration.py::test_basic_functionality returned <class 'bool'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

test_final_integration.py::test_mission_planning
  /home/echo_/Code/asciimath/warp-bubble-optimizer/.venv/lib/python3.13/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but test_final_integration.py::test_mission_planning returned <class 'tuple'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
ERROR test_final_integration.py::test_mission_execution
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
================== 13 passed, 674 warnings, 1 error in 7.03s ===================
# PYTEST-RESULTS-END
# Never skip a test if an import isn't available. Those tests should fail and the import should be fixed. 
~/Code/asciimath$ grep -r "importerskip" --include="*.py" --exclude="progress_log_processor.py" . | wc -l
# IMPORTERSKIP-RESULTS-BEGIN
0
# IMPORTERSKIP-RESULTS-END
```