from __future__ import annotations
import json, math, random

def run_laser_coherence_uq(n: int = 128, coherence_reduction: float = 100.0, rel_sigma: float = 0.15) -> dict:
    random.seed(1)
    samples = [random.gauss(coherence_reduction, rel_sigma*coherence_reduction) for _ in range(n)]
    m = sum(samples)/n
    var = sum((x-m)**2 for x in samples)/max(1,n-1)
    cv = math.sqrt(var)/m if m != 0 else float('inf')
    feasible = m >= 50.0  # simple gate
    return {"n": n, "mean": m, "cv": cv, "feasible": feasible}

if __name__ == '__main__':
    print(json.dumps(run_laser_coherence_uq(), indent=2))
