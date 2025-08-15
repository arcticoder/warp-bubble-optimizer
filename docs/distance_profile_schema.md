# Distance Profile Schema (CSV and JSON)

This repository accepts distance profiles to parameterize segment lengths for UQ runs and planning demos. Two formats are supported:

- CSV (recommended for CI and spreadsheets)
- JSON (for programmatic pipelines)

## CSV format

- Single column of non-negative numbers representing distances in meters
- Optional header row: one of `distance`, `dist`, or `d`
- Comment lines starting with `#` are ignored
- Commas are allowed; both of these are valid:
  - One value per line
  - Comma-separated on a single line (or mixed)

Example (header + one per line):

```
# Varied distance profile (m)
distance
8
12
9
```

Example (no header, comma-separated):

```
# Distances (m)
8,12,9
```

Validation rules (enforced by `tools/validate_dist_profile.py`):
- All values must be numeric and ≥ 0
- At least one value present
- Sum must be > 0

## JSON format

Two accepted shapes:

- Array of numbers: `[8, 12, 9]`
- Object with `distances` key: `{ "distances": [8, 12, 9] }`

Validation rules mirror the CSV rules: non-negative, non-empty, sum > 0.

## Where it’s used

- CI varied-profile UQ: `.github/workflows/mission-validate.yml`
- Local UQ runs: `python -m uq_validation.impulse_uq_runner --dist-profile <file>`

## Tips

- Keep segment counts modest (e.g., 10–30) for quick UQ runs
- Put canonical profiles in `data/` and reference from CI

---

Anchors: CSV format, JSON format, Validation rules, Where it’s used, Tips
