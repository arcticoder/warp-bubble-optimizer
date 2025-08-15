# Traceability matrix (skeleton)

- Requirement → Test → Artifact
- Example: 30 s smear → tests/test_power_profile.py::test_triangle_equivalence → docs/power_budget_reconciliation.md
- Example: zero-expansion → tests/test_field_and_control.py::test_zero_expansion_tolerance → CI artifacts
 - Coil ramp constraints → tests/test_vnv_coil_ramp.py → driver profile generator
 - Rehearsal mode → tests/test_supraluminal_prototype_new.py::test_mission_cli_rehearsal → CLI JSON output
 - UQ metamaterial & laser coherence → tests/test_uq_new.py → CI gates (future)
 - PNG integrity check → tests/test_png_check.py → tools/png_check.py
 - Profiles schema check → tests/test_profile_validation.py → tools/validate_profiles.py
