"""Deterministic seeding helpers.

The framework treats reproducibility as a first-class concern: every randomized
component (synthetic dataset generation, codebook training, NSGA-II) takes a
seed, and :func:`set_global_seed` is provided for one-shot setup.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class SeedSet:
    """Bundle of seeds derived from a single root seed."""

    root: int
    python: int
    numpy: int

    @classmethod
    def from_root(cls, root: int) -> SeedSet:
        rng = np.random.default_rng(root)
        py_seed = int(rng.integers(0, 2**31 - 1))
        np_seed = int(rng.integers(0, 2**31 - 1))
        return cls(root=root, python=py_seed, numpy=np_seed)


def set_global_seed(seed: int) -> SeedSet:
    """Seed Python ``random``, ``numpy`` global RNG, and ``PYTHONHASHSEED``.

    Returns the derived :class:`SeedSet` so callers can persist exactly which
    seeds were used for each subsystem.
    """
    seeds = SeedSet.from_root(seed)
    random.seed(seeds.python)
    np.random.seed(seeds.numpy)
    os.environ.setdefault("PYTHONHASHSEED", str(seeds.python))
    return seeds


def derive_rng(seed: int) -> np.random.Generator:
    """Return an isolated ``numpy`` ``Generator`` (preferred over the global RNG)."""
    return np.random.default_rng(seed)
