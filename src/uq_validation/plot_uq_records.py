from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _generate_fallback_png(outdir: Path, fname: str, text: str) -> None:
    plt.figure()
    plt.text(0.5, 0.5, text, ha="center", va="center")
    plt.title(fname)
    plt.savefig(outdir / fname)
    plt.close()


def main() -> int:
    print(f"Starting plot generation: 2025-08-14 21:41 PDT")
    outdir = Path("artifacts")
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = Path("dist_profile_40eridani_varied.csv")
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        for fname in [
            "40eridani_energy.png",
            "40eridani_feasibility.png",
            "40eridani_energy_extended.png",
            "40eridani_feasibility_extended.png",
            "40eridani_energy_varied.png",
            "40eridani_feasibility_varied.png",
        ]:
            _generate_fallback_png(outdir, fname, "No Data")
            print(f"Generated fallback {fname}")
        return 1
    try:
        try:
            import pandas as pd  # type: ignore
        except Exception as e:  # pragma: no cover - missing optional dep
            print(f"Plot error: pandas not available ({e}); generating fallbacks")
            for fname in [
                "40eridani_energy.png",
                "40eridani_feasibility.png",
                "40eridani_energy_extended.png",
                "40eridani_feasibility_extended.png",
                "40eridani_energy_varied.png",
                "40eridani_feasibility_varied.png",
            ]:
                _generate_fallback_png(outdir, fname, "Pandas missing")
                print(f"Generated fallback {fname}")
            return 1

        df = pd.read_csv(csv_path)
        print(f"CSV rows: {len(df)}, columns: {list(df.columns)}")
        if df.empty or not all(col in df.columns for col in ["energy", "feasibility"]):
            raise ValueError("Invalid CSV")
        plots = [
            ("energy", "40eridani_energy.png", "Energy Distribution"),
            ("feasibility", "40eridani_feasibility.png", "Feasibility"),
            ("energy", "40eridani_energy_extended.png", "Extended Energy"),
            ("feasibility", "40eridani_feasibility_extended.png", "Extended Feasibility"),
            ("energy", "40eridani_energy_varied.png", "Varied Energy"),
            ("feasibility", "40eridani_feasibility_varied.png", "Varied Feasibility"),
        ]
        for col, fname, title in plots:
            plt.figure()
            plt.hist(df[col], bins=50, label=title)
            plt.title(title)
            plt.xlabel(col.capitalize())
            plt.ylabel("Count")
            plt.legend()
            plt.savefig(outdir / fname)
            plt.close()
            print(f"Generated {fname}")
        return 0
    except Exception as e:
        print(f"Plot error: {str(e)}")
        for fname in [
            "40eridani_energy.png",
            "40eridani_feasibility.png",
            "40eridani_energy_extended.png",
            "40eridani_feasibility_extended.png",
            "40eridani_energy_varied.png",
            "40eridani_feasibility_varied.png",
        ]:
            _generate_fallback_png(outdir, fname, f"Error: {str(e)}")
            print(f"Generated fallback {fname}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
