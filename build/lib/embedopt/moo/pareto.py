"""Pareto front extraction (works in any dimension).

The implementation uses an O(n^2) dominance check, which is plenty fast for the
search-space sizes typical of post-hoc compression sweeps. For larger fronts,
swap in Kung's algorithm without changing the public API.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_F64 = NDArray[np.float64]
_Bool = NDArray[np.bool_]


def is_dominated(candidate: _F64, others: _F64) -> bool:
    """Return True iff some row of ``others`` dominates ``candidate``."""
    if others.size == 0:
        return False
    no_worse = (others <= candidate).all(axis=1)
    strictly_better = (others < candidate).any(axis=1)
    return bool((no_worse & strictly_better).any())


def non_dominated_mask(points: _F64) -> _Bool:
    """Boolean mask selecting the non-dominated rows of ``points``.

    Inputs are assumed to be in canonical *minimization* form (see
    :mod:`embedopt.moo.objectives`).
    """
    n = points.shape[0]
    mask: _Bool = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        diffs = points - points[i]
        no_worse = (diffs >= 0).all(axis=1)
        strictly_better_somewhere = (diffs > 0).any(axis=1)
        dominated = no_worse & strictly_better_somewhere
        dominated[i] = False
        mask[dominated] = False
    return mask


def pareto_indices(points: _F64) -> list[int]:
    """Return the indices of the non-dominated rows."""
    return [int(i) for i, keep in enumerate(non_dominated_mask(points)) if keep]
