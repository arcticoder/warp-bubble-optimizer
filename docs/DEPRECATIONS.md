# Deprecations & Removal Schedule

This file tracks deprecation notices and planned removal versions.
Version source: `src/version.py` (`__version__`).

| Component / Shim | Introduced | Deprecated In | Removal After | Notes |
|------------------|------------|---------------|---------------|-------|
| `integrated_impulse_control.py` (root) | pre-0.1.0 | 0.1.0 | 0.3.0 | Import from `impulse` package instead |
| `integrated_impulse_control_clean.py` (root) | pre-0.1.0 | 0.1.0 | 0.3.0 | Unified implementation in `src/impulse/` |

## Policy
- Deprecations span at least two minor versions (e.g., 0.1.x → removal after 0.3.0).
- Tests may assert that deprecated shims emit `DeprecationWarning`.
- New features should reference V&V / UQ tasks in docstrings for traceability.

## Recently Added Features (0.1.0)
- Pluggable translation energy estimation strategies (`impulse.energy_strategies`).
- Mission JSON export (`execute_impulse_mission(..., json_export_path=...)`).
- Safety margin feasibility check (planned_energy*(1+margin) ≤ budget).
- Budget depletion abort logic.
- Controller config injection for testing/UQ.
