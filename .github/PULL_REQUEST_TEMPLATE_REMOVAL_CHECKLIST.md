# Removal PR Checklist (Deprecation Completion)

Use this template when removing deprecated shims after version 0.3.0.

- [ ] Version in `src/version.py` bumped (e.g., 0.3.0 → 0.4.0-dev0)
- [ ] Remove root shim files:
      - `integrated_impulse_control.py`
      - `integrated_impulse_control_clean.py`
- [ ] Update `docs/DEPRECATIONS.md` (move removed items to historical note)
- [ ] Update tests to stop referencing deprecated paths
- [ ] Run traceability check (no stale references)
- [ ] Confirm CI green
- [ ] Add release notes entry highlighting removal
