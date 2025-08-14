# Impulse CLI and UQ Runner Guide

This guide covers the mission CLI and the impulse UQ runner: quick starts, key flags, example outputs, and how to interpret perf and UQ artifacts.

## Mission CLI (impulse.mission_cli)

Quick start:

```bash
python -m impulse.mission_cli \
  --waypoints examples/waypoints/simple.json \
  --export mission.json \
  --perf-csv perf.csv \
  --hybrid simulate-first \
  --seed 123
```

Key flags:
- --perf-csv: Write per-segment performance CSV.
- --hybrid: Planning mode: off | simulate-first | estimate-first
- --seed: Set deterministic seeds for reproducibility (also sets WARP_SEED and PYTHONHASHSEED).
- --verbose-export: Include a config snapshot and more meta in mission JSON.
- --error-codes: Non-zero exit codes on infeasible plan (2) or budget abort (3).

Outputs:
- mission.json: Includes plan, results, and schema fields; meta.randomization contains env_seed/python_hash_seed when a seed was set.
- perf.csv: segment_index, kind, segment_time, segment_energy, peak_velocity, total_distance, total_rotation_angle.
- perf_summary.png: Quick visual; generate via:

```bash
python bin/plot_perf_csv.py --csv perf.csv --out perf_summary.png
```

Interpreting perf_summary.png:
- Left: Histogram of per-segment execution times (segment_time).
- Right: Bar plot of per-segment energies. Spikes often indicate longer distances or rotational segments.

## UQ Runner (src.uq_validation.impulse_uq_runner)

Quick start:

```bash
python -m src.uq_validation.impulse_uq_runner \
  --samples 50 \
  --seed 123 \
  --out uq_summary.json \
  --jsonl-out uq_records.jsonl \
  --dist-profile data/dist_profile.csv
```

Key flags:
- --jsonl-out: Write per-sample records (one JSON per line), friendly for large runs.
- --dist-profile: Provide a common distance profile for all samples; accepts JSON (array or {"distances": [...]}) or CSV (values separated by comma/newlines).
- --seed: Set deterministic seeds for reproducibility (propagated to RNGs and env vars).

Example outputs:
- uq_summary.json: { samples, infeasible_count, feasible_fraction, energy_mean, energy_std, energy_cv, records: [...] }
- uq_records.jsonl: One record per line: { planned_energy, feasible, hybrid_mode, dwell_time, approach_speed }

Interpreting uq_summary.json:
- feasible_fraction near 1.0 indicates robust planning under sampled uncertainties.
- energy_cv = energy_std / energy_mean; lower is tighter energy variability.

## Tips
- Reproducibility: Use --seed consistently. mission.json will include meta.randomization with the WARP_SEED and PYTHONHASHSEED for traceability.
- Perf analysis: Batch multiple perf.csv files using bin/aggregate_perf_csv.py to summarize multi-mission performance.
- Schema: Programmatically access the mission schema with impulse.utils.get_mission_schema().
