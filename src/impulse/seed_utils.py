from __future__ import annotations

import os
import random

try:
    import numpy as np
except Exception:
    np = None  # type: ignore


def set_seed(seed: int) -> None:
    """Set deterministic seeds for random generators used in the project."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
