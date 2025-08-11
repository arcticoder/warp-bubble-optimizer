# Power Budget Numeric Reconciliation (Draft)

Given P_peak and durations (t_ramp, t_cruise), total energy (ideal linear ramps):

E_total = 2 * (0.5 * t_ramp * P_peak) + (t_cruise * P_peak)

Examples:
- P_peak = 25 MW, t_ramp = 30 s, t_cruise = 2.56 s:
  - E_total = 2*(0.5*30*25e6) + 2.56*25e6 ≈ 812.5 MJ
- Adjust for discharge efficiency η(C-rate): E_effective = E_total / η

Validation plan:
- Integrate measured P(t) over mission segments and compare to E_total within 10%.
- Track η proxy vs ramp profile; reconcile with battery capacity bounds.
