"""Deterministic seeding for Python, NumPy, and (if available) PyTorch + CUDA."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every RNG we can find. Does not enable deterministic algorithms —
    those can be 2-3x slower and we want this project to actually run on a
    single 4090 in a few hours.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
