# Traceability Matrix Updates (Aug 15, 2025)

This document maps new CI/tests/docs to roadmap items:

- Tiny UQ PNG in CI → mission-validate.yml; Pages publishing via traceability-badge.yml
- Timeline-log schema + validator → schemas/timeline.log.schema.json; bin/validate_timeline_log.py; schema-gate.yml
- Perf CSV schema enforcement → bin/validate_perf_csv.py; tests/test_perf_csv_schema_regression.py
- EKF residual test → tests/test_ekf_residual.py
- Thermal regression harness → tests/test_thermal_regression.py
- Params bounds checks → src/supraluminal_prototype/params_loader.py; tests/test_params_loader_bounds.py
- UQ dashboard publishing → bin/publish_uq_dashboard.py; Pages index update
