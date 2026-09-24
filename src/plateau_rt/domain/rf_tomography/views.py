"""View subsets for the tomography pose bank (NumPy only)."""

from __future__ import annotations

import numpy as np


def nested_view_order(n_bank: int, seed: int) -> np.ndarray:
    """Return one fixed permutation of the pose bank for a seed.

    The view subset of size ``V`` is the prefix ``order[:V]``, so subsets are
    nested (``V1 < V2`` implies inclusion) and deterministic; the reference
    capture c0 is ``(order[0], b = 0)``.
    """
    if n_bank < 1:
        raise ValueError("n_bank must be >= 1")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    return rng.permutation(n_bank).astype(np.int64)
