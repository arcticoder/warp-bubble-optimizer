import os
from src.uq_validation.plot_uq_records import main


def test_plot_uq_records():
    os.makedirs("artifacts", exist_ok=True)
    # The helper just lists PNGs; ensure it runs without error and returns 0
    assert main() == 0
