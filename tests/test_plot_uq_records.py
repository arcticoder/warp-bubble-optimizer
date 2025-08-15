import os
from src.uq_validation.plot_uq_records import main


def test_plot_uq_records():
    os.makedirs("artifacts", exist_ok=True)
    # Provide a tiny CSV input to exercise plotting
    with open("dist_profile_40eridani_varied.csv", "w", encoding="utf-8") as f:
        f.write("energy\n1.0\n2.0\n3.0\n")
    assert main() == 0
    assert os.path.exists("artifacts/40eridani_energy.png")
