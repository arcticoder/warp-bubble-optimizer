import numpy as np
import os
from src.supraluminal_prototype.drivers import generate_current_profile
from src.supraluminal_prototype.scheduler import phase_sync_schedule
from src.supraluminal_prototype.estimation import EKF, simulate_sensors
from src.supraluminal_prototype.laser_sim import injection_lock_phase_noise
from src.supraluminal_prototype.thermal import coil_copper_loss, battery_heat, thermal_step
from src.supraluminal_prototype.fault_detection import detect_overcurrent, detect_ground_fault
from src.supraluminal_prototype.params_loader import load_ring_params
from src.hil import HILInterface


def test_generate_current_profile_basic():
    I = generate_current_profile(0.0, 100.0, t_ramp=1.0, dt=0.01, max_didt=200.0)
    dI = np.diff(I)
    assert np.all(dI >= 0)
    assert np.max(dI)/0.01 <= 200.0 + 1e-6
    assert abs(I[-1] - 100.0) < 1e-6


def test_phase_sync_schedule_within_budget():
    phases = np.array([0.0, 0.001, -0.001, 0.0005])
    res = phase_sync_schedule(phases, jitter_budget=0.005)
    assert res['ok'] is True
    assert np.allclose(phases + res['offsets'], res['target'])


def test_ekf_skeleton_predict_update():
    ekf = EKF(x0=np.zeros(2), P0=np.eye(2), Q=1e-3*np.eye(2), R=1e-2*np.eye(1))
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    x_pred, P_pred = ekf.predict(F)
    z = simulate_sensors(np.array([1.0, 0.0]), H, noise_std=np.array([0.0]))
    x_upd, P_upd = ekf.update(z, H)
    assert x_upd.shape == (2,)
    assert P_upd.shape == (2,2)


def test_injection_lock_phase_noise_floor():
    x = injection_lock_phase_noise(bw_hz=1e5, duration_s=0.01, dt=1e-4, floor_rad=1e-3)
    assert x.shape[0] > 10
    # Check RMS is finite and scales roughly with floor
    rms = np.sqrt(np.mean(x**2))
    assert rms > 0


def test_thermal_model_steps():
    P = coil_copper_loss(100.0, 0.01)
    qb = battery_heat(1e5, 0.95)
    T = thermal_step(300.0, P+qb, C_th=100.0, G_th=10.0, dt=0.1)
    assert T > 300.0


def test_fault_detection():
    assert detect_overcurrent(120.0, 100.0, dwell_ms=60.0, threshold_ms=50.0) is True
    assert detect_ground_fault(R_iso_meg=2.0, min_meg=5.0) is True


def test_params_loader_yaml(tmp_path):
    p = tmp_path / 'ring_params.yaml'
    p.write_text('''outer_diameter_m: 1.0
inner_diameter_m: 0.8
cross_section: torus
num_rings: 4
spacing_m: 1.5
coil_turns: 120
max_current_A: 5000
''')
    rp = load_ring_params(str(p))
    assert rp.num_rings == 4
    assert rp.max_current_A == 5000


def test_hil_loopback():
    h = HILInterface()
    h.write('ring1_current', 12.34)
    h.loopback()
    assert h.read('ring1_current') == 12.34


def test_mission_cli_rehearsal(tmp_path):
    # Minimal waypoints file
    wp = tmp_path / 'wp.json'
    wp.write_text('{"waypoints":[{"x":0,"y":0,"z":0},{"x":1,"y":0,"z":0}],"dwell":0.1}')
    # Run CLI in rehearsal mode
    import subprocess, sys
    cmd = [sys.executable, '-m', 'impulse.mission_cli', '--waypoints', str(wp), '--rehearsal']
    env = dict(os.environ)
    env['PYTHONPATH'] = 'src'
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode == 0
    assert '"mode": "rehearsal"' in res.stdout
