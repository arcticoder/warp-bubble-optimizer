import tempfile
from pathlib import Path

from src.uq_validation.impulse_uq_runner import _load_distance_profile, parse_csv


def test_load_distance_profile_csv_lines():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "varied.csv"
        p.write_text("10\n20\n15\n")
        vals = _load_distance_profile(str(p))
        assert vals == [10.0, 20.0, 15.0]


def test_load_distance_profile_csv_commas():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "varied.csv"
        p.write_text("10, 20, 15")
        vals = _load_distance_profile(str(p))
        assert vals == [10.0, 20.0, 15.0]


    def test_malformed_csv(tmp_path):
        bad = tmp_path / "malformed.csv"
        bad.write_text("# Comment\n,,,\n1,2")
        try:
            parse_csv(str(bad))
            assert False, "Should raise ValueError"
        except ValueError:
            pass
