"""Exact 2D and 3D hypervolume.

For two- and three-objective problems, exact hypervolume admits closed-form
sweeps that are simple, deterministic, and fast enough for the front sizes we
typically encounter. Higher-dimensional problems would require WFG / HSO; this
module raises :class:`NotImplementedError` in that case rather than silently
returning a bad number, since hypervolume is a load-bearing reporting metric.

Inputs are assumed to be in canonical *minimization* form. The reference point
``ref`` must dominate every input point (i.e. be worse on every objective).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from embedopt.moo.pareto import non_dominated_mask

_F64 = NDArray[np.float64]


def hypervolume(points: _F64, ref: _F64) -> float:
    """Hypervolume dominated by ``points`` with respect to ``ref`` (worst point)."""
    if points.size == 0:
        return 0.0
    if ref.shape[0] != points.shape[1]:
        raise ValueError("ref must have the same dimensionality as points")
    if (points > ref).any():
        raise ValueError("All points must dominate the reference (be <= ref) coord-wise")

    front = points[non_dominated_mask(points)]
    if front.shape[1] == 1:
        return float(ref[0] - front.min())
    if front.shape[1] == 2:
        return _hv_2d(front, ref)
    if front.shape[1] == 3:
        return _hv_3d(front, ref)
    raise NotImplementedError("hypervolume() supports up to 3 objectives")


def _hv_2d(front: _F64, ref: _F64) -> float:
    order = np.argsort(front[:, 0])
    sorted_front = front[order]
    hv = 0.0
    prev_y = float(ref[1])
    for x, y in sorted_front:
        if y < prev_y:
            hv += (float(ref[0]) - float(x)) * (prev_y - float(y))
            prev_y = float(y)
    return float(hv)


def _hv_3d(front: _F64, ref: _F64) -> float:
    # Sweep the third axis; for each unique z-level, compute the area of the
    # 2D Pareto projection of points with z <= current z.
    order = np.argsort(front[:, 2])
    sorted_front = front[order]
    hv = 0.0
    prev_z = float(ref[2])
    for i in range(sorted_front.shape[0] - 1, -1, -1):
        z = float(sorted_front[i, 2])
        slab = sorted_front[i:, :2]
        slab = slab[non_dominated_mask(slab)]
        area = _hv_2d(slab, ref[:2])
        hv += area * (prev_z - z)
        prev_z = z
    return float(hv)
