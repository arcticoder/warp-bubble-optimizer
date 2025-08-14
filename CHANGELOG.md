# Changelog

All notable changes to this project will be documented in this file.

## 0.1.0-dev0 - 2025-08-13
- Pluggable energy strategies (translation + rotation)
- Rotational segment support in mission planner / executor
- Hybrid planning mode and simulation result caching
- JSON export with schema ID and compact segment summaries
- Traceability checker emits JSON report
- V&V tests expanded: angular velocity caps, safety margin infeasibility, deprecation import, JSON schema version
- Packaging scaffold with `pyproject.toml`

## 0.1.0-dev2 - 2025-08-14
- Add UQ runner module and smoke tests (sampling dwell/approach speeds)
- Bundle JSON Schema at `schemas/impulse.mission.v1.json` and validation utility `bin/validate_mission_json.py`
- Enhance mission executor with optional per-segment performance CSV logging
- Expose `--perf-csv` in `impulse.mission_cli` and accept argv parameter for programmatic invocation
- Add CLI and CSV logging tests; add VS Code tasks for quick workflows
