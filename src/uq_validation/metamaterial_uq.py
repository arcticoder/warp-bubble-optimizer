from __future__ import annotations
import json, math, random

def run_metamaterial_uq(n: int = 128, mean_amp: float = 1e4, rel_sigma: float = 0.1) -> dict:
    random.seed(0)
    samples = [random.gauss(mean_amp, rel_sigma*mean_amp) for _ in range(n)]
    m = sum(samples)/n
    var = sum((x-m)**2 for x in samples)/max(1,n-1)
    cv = math.sqrt(var)/m if m != 0 else float('inf')
    return {"n": n, "mean": m, "cv": cv}

if __name__ == '__main__':
    print(json.dumps(run_metamaterial_uq(), indent=2))
