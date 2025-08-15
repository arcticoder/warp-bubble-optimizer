import os
from src.uq_validation.plot_uq_records import main


def test_plot_uq_records():
    os.makedirs("artifacts", exist_ok=True)
    # Provide a tiny CSV input to exercise plotting with required columns
    with open("dist_profile_40eridani_varied.csv", "w", encoding="utf-8") as f:
        f.write("energy,feasibility\n1.0,0.8\n2.0,0.9\n3.0,0.7\n")
    assert main() == 0
    for fname in [
        "40eridani_energy.png",
        "40eridani_feasibility.png",
        "40eridani_energy_extended.png",
        "40eridani_feasibility_extended.png",
        "40eridani_energy_varied.png",
        "40eridani_feasibility_varied.png",
    ]:
        path = os.path.join("artifacts", fname)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000
