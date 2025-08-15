# Control state estimator outline

- Sensors: ring current, laser phase/intensity, plasma density/temperature
- Estimator: EKF/UKF with process model from warp_generator dynamics
- Outputs: phase sync errors, envelope amplitude, stability margins
- Interfaces: autopilot scheduler and safety watchdogs
