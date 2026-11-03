"""Hash parameter arrays so that bitwise equality of two runs can be tested."""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def hash_arrays(arrays: Iterable) -> str:
    """Hash a sequence of arrays after casting each to float64.

    Casting first makes a float32 run and a float64 run comparable: they are
    equal only if every float32 value equals the float64 value exactly.
    """
    h = hashlib.sha256()
    for a in arrays:
        arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()[:20]


def hash_state_dict(state_dict) -> str:
    """Hash a torch state dict, in key order."""
    return hash_arrays(v.detach().cpu().double().numpy() for _, v in sorted(state_dict.items()))
