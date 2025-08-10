import math

from src.supraluminal_prototype.hardware import CoilDriver


def test_coil_driver_near_linear_no_hysteresis():
    drv = CoilDriver(max_current=5000.0, hysteresis=0.0)
    vals = [drv.command(i / 30.0) for i in range(31)]
    incs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    # Increments should be nearly constant
    span = max(incs) - min(incs)
    assert span < 1e-3 * drv.max_current


def test_coil_driver_hysteresis_reduces_gain():
    drv0 = CoilDriver(max_current=5000.0, hysteresis=0.0)
    drvH = CoilDriver(max_current=5000.0, hysteresis=0.1)
    y0 = drv0.command(0.8)
    yH = drvH.command(0.8)
    assert math.isclose(yH, (1.0 - drvH.hysteresis) * 0.8 * drvH.max_current)
    assert yH < y0
