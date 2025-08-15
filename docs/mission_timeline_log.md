# Mission timeline logging (--timeline-log)

The mission CLI supports an optional `--timeline-log` argument to record planning and execution milestones. The log can be written as CSV (default) or JSONL (if the path ends with `.jsonl`).

- CSV header: `iso_time,t_rel_s,event,segment_id,planned_value,actual_value`
- JSONL fields: `{ iso_time, t_rel_s, event, segment_id, planned_value, actual_value }`

Events currently emitted:
- `plan_created` (t_rel_s = 0, planned_value = total planned energy J)
- `rehearsal_complete` or `dry_run_abort_complete` (non-executing paths)
- `mission_complete` (actual_value = total energy used J)

Example usage:

```bash
PYTHONPATH=src python -m impulse.mission_cli \
  --waypoints examples/waypoints/simple.json \
  --export mission.json \
  --timeline-log mission_timeline.csv \
  --rehearsal --hybrid simulate-first --seed 123
```
