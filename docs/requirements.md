# Mission Requirements Spec (Draft v0.1)

- Mission: Warp 1 Earth–Moon round trip
  - Distance: ~7.68e8 m; Travel time: ~2.56 s at c; 100 s lunar stop
- Spacecraft
  - Geometry: 5 m diameter spherical hull
  - Rings: four 1 m toroidal rings (plasma + metamaterials + 10 lasers total)
  - Volume budget: battery ~7.85 m³; avionics/sensors internal access
- Power & Energy
  - Target energy: ~1.57e10 J (12.5 MW average with 30 s smearing per accel/decel)
  - Profile: 30 s ramp up, 2.56 s cruise, 30 s ramp down
  - Battery discharge efficiency vs C-rate to be verified in V&V/UQ
- Control & Synchronization
  - Ring phase sync: sub-µs jitter target; closed-loop stability margin > 6 dB
  - Abort criteria: ramp overshoot, thermal margins, sensor failures
- Sensors & Telemetry
  - IMU, thermal, coil current, laser phase/intensity, plasma diagnostics
  - Mission telemetry: timestamps for accel/cruise/decel/stop
- Safety
  - Ramp abort & quench procedures; cooldown profile; comms/tracking checklist

---

## Acceptance Gates
- Energy reconciliation vs model (HIL): within 10% over ramps + cruise
- Closed-loop phase stability margin ≥ 6 dB under nominal disturbances
- Traceability: roadmap → tests → artifacts coverage
