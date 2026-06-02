"""Scalarization functions for downstream model selection.

Once the Pareto front is known, a deployment decision typically requires
collapsing the multi-objective trade-off into a scalar. We provide two
classical scalarizers:

* :func:`weighted_sum` — fast and intuitive but can miss non-convex regions of
  the front.
* :func:`tchebycheff` — captures non-convex fronts via a max-norm to a utopia
  point. Recommended when the Pareto surface is known to bend.

Both consume *normalized* coordinates so that objectives with different scales
are commensurate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

_F64 = NDArray[np.float64]


def normalize_columns(points: _F64) -> _F64:
    """Min-max normalize each column to ``[0, 1]``. Zero-range columns stay at ``0``."""
    if points.size == 0:
        return points
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = maxs - mins
    span = np.where(span == 0.0, 1.0, span)
    out: _F64 = ((points - mins) / span).astype(np.float64, copy=False)
    return out


def weighted_sum(points: _F64, weights: Sequence[float]) -> _F64:
    """Weighted sum of normalized coordinates (lower is better)."""
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != points.shape[1]:
        raise ValueError("weights length must match number of objectives")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    norm = normalize_columns(points)
    out: _F64 = (norm @ w).astype(np.float64, copy=False)
    return out


def tchebycheff(points: _F64, weights: Sequence[float], *, rho: float = 1e-3) -> _F64:
    """Augmented Tchebycheff scalarizer toward the utopia point ``(0, ..., 0)``.

    ``rho`` adds a small weighted-sum term that breaks ties on the boundary,
    which is the standard recommendation in the MOO literature.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != points.shape[1]:
        raise ValueError("weights length must match number of objectives")
    norm = normalize_columns(points)
    weighted = norm * w
    out: _F64 = (weighted.max(axis=1) + rho * weighted.sum(axis=1)).astype(np.float64, copy=False)
    return out
