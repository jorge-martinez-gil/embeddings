"""Objective specs and helpers.

We canonicalize all objectives as *minimization* internally. An objective with
``sense="max"`` (e.g., nDCG) is negated when stored as a Pareto coordinate;
end-user reports always use the user-facing convention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Sense = Literal["min", "max"]
_F64 = NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class Objective:
    """A single optimization objective."""

    name: str
    sense: Sense
    weight: float = 1.0
    """Used by scalarization. Has no effect on Pareto sorting."""


def to_min_matrix(
    objectives: Sequence[Objective],
    rows: Sequence[Mapping[str, float]],
) -> _F64:
    """Build a ``(n, m)`` matrix in canonical *minimization* form.

    Each row is one candidate solution; columns are objectives in the order of
    ``objectives``. ``max``-sense objectives are negated so that lower is
    better in every column.
    """
    if not rows:
        return np.zeros((0, len(objectives)), dtype=np.float64)
    out = np.empty((len(rows), len(objectives)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, obj in enumerate(objectives):
            if obj.name not in row:
                raise KeyError(f"Objective {obj.name!r} missing from row {i}")
            v = float(row[obj.name])
            out[i, j] = -v if obj.sense == "max" else v
    return out
