from __future__ import annotations

import numpy as np

from embedopt.utils.seeding import SeedSet, derive_rng, set_global_seed


def test_seed_set_is_deterministic_from_root() -> None:
    a = SeedSet.from_root(42)
    b = SeedSet.from_root(42)
    assert a == b
    assert a.python != a.numpy  # extremely high probability


def test_set_global_seed_returns_seed_set() -> None:
    seeds = set_global_seed(123)
    assert seeds.root == 123


def test_derive_rng_is_isolated_and_deterministic() -> None:
    rng_a = derive_rng(7)
    rng_b = derive_rng(7)
    a = rng_a.standard_normal(8)
    b = rng_b.standard_normal(8)
    assert np.array_equal(a, b)
