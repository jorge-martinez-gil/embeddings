from __future__ import annotations

import numpy as np
import pytest

from embedopt.moo import (
    NSGAConfig,
    Objective,
    hypervolume,
    non_dominated_mask,
    normalize_columns,
    pareto_indices,
    run_nsga2,
    tchebycheff,
    to_min_matrix,
    weighted_sum,
)


def test_to_min_matrix_negates_max_objectives() -> None:
    objs = [Objective(name="quality", sense="max"), Objective(name="bytes", sense="min")]
    rows = [{"quality": 0.9, "bytes": 100.0}, {"quality": 0.8, "bytes": 50.0}]
    m = to_min_matrix(objs, rows)
    assert m.shape == (2, 2)
    assert m[0, 0] == pytest.approx(-0.9)
    assert m[0, 1] == pytest.approx(100.0)


def test_pareto_indices_drops_dominated_points() -> None:
    pts = np.array(
        [
            [0.0, 0.0],  # dominates everything (utopia)
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )
    assert pareto_indices(pts) == [0]


def test_pareto_indices_two_incomparable_points_both_kept() -> None:
    pts = np.array([[1.0, 5.0], [5.0, 1.0]], dtype=np.float64)
    mask = non_dominated_mask(pts)
    assert mask.tolist() == [True, True]


def test_hypervolume_2d_unit_square() -> None:
    # One point at (0,0), reference at (1,1) -> hypervolume is 1.0.
    pts = np.array([[0.0, 0.0]], dtype=np.float64)
    ref = np.array([1.0, 1.0], dtype=np.float64)
    assert hypervolume(pts, ref) == pytest.approx(1.0)


def test_hypervolume_2d_staircase() -> None:
    # Three-point Pareto staircase, ref at (4, 4).
    # Points: (0,3), (1,1), (3,0) -> HV = 4*1 + 3*2 + 1*1 = 4 + 6 + 1 = 11
    pts = np.array([[0.0, 3.0], [1.0, 1.0], [3.0, 0.0]], dtype=np.float64)
    ref = np.array([4.0, 4.0], dtype=np.float64)
    assert hypervolume(pts, ref) == pytest.approx(11.0)


def test_hypervolume_3d_single_point_is_box_volume() -> None:
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert hypervolume(pts, ref) == pytest.approx(6.0)


def test_hypervolume_rejects_dim_above_three() -> None:
    pts = np.zeros((1, 4), dtype=np.float64)
    ref = np.ones(4, dtype=np.float64)
    with pytest.raises(NotImplementedError):
        hypervolume(pts, ref)


def test_weighted_sum_and_tchebycheff_select_correctly() -> None:
    pts = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    ws = weighted_sum(pts, [0.5, 0.5])
    # All three lie on the symmetry axis under uniform weights.
    assert ws[2] <= ws[0] + 1e-9
    assert ws[2] <= ws[1] + 1e-9
    tc = tchebycheff(pts, [0.5, 0.5], rho=0.0)
    # Uniform weights -> middle point minimizes the max-norm.
    assert int(np.argmin(tc)) == 2


def test_normalize_columns_handles_zero_range() -> None:
    pts = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float64)
    norm = normalize_columns(pts)
    assert np.allclose(norm, 0.0)


def test_nsga_finds_known_pareto_on_toy_problem() -> None:
    # Decision space: two genes, each in {0,1,2,3,4,5}. Quality maxes at gene_a=0
    # (high), bytes min at gene_b=0 (low). Pareto front is {(a=0, b=anything)}
    # under uniform weights.
    cardinalities = (6, 6)
    objectives = [
        Objective(name="quality", sense="max"),
        Objective(name="bytes", sense="min"),
    ]

    def evaluate(cand: tuple[int, ...]) -> dict[str, float]:
        a, b = cand
        # Quality decreases with a; bytes increase with b (a is irrelevant for size).
        return {"quality": 1.0 - 0.1 * a, "bytes": float(b)}

    cfg = NSGAConfig(population_size=18, n_generations=12, seed=0)
    result = run_nsga2(
        cardinalities=cardinalities,
        objectives=objectives,
        evaluate=evaluate,
        config=cfg,
    )
    # Every Pareto-front candidate must have a == 0 (best quality).
    for idx in result.front_indices:
        a, _ = result.population[idx]
        assert a == 0
