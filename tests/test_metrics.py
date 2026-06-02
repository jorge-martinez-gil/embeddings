from __future__ import annotations

import numpy as np
import pytest

from embedopt.evaluation.metrics import (
    cosine_pairs,
    map_score,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    spearman_correlation,
)


def test_spearman_perfect_ranking() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0]
    assert spearman_correlation(x, y) == pytest.approx(1.0)


def test_spearman_inverse_ranking() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    y = [40.0, 30.0, 20.0, 10.0]
    assert spearman_correlation(x, y) == pytest.approx(-1.0)


def test_spearman_handles_ties() -> None:
    # Two-tie input should not raise and should fall in [-1, 1].
    x = [1.0, 1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0, 7.0]
    rho = spearman_correlation(x, y)
    assert -1.0 <= rho <= 1.0


def test_cosine_pairs_orthogonal_zero() -> None:
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)
    out = cosine_pairs(a, b)
    assert out[0] == pytest.approx(0.0)


def test_recall_at_k_hand_computed() -> None:
    # Two queries, three docs.
    scores = np.array(
        [
            [0.9, 0.5, 0.1],  # ranking: 0,1,2
            [0.1, 0.5, 0.9],  # ranking: 2,1,0
        ],
        dtype=np.float32,
    )
    qrels = {0: {0: 1.0, 2: 1.0}, 1: {0: 1.0}}
    # Q0 relevant {0, 2}, top-2 = {0, 1} -> recall = 1/2.
    # Q1 relevant {0}, top-2 = {2, 1} -> recall = 0.
    assert recall_at_k(scores, qrels, k=2) == pytest.approx(0.25)


def test_mrr_at_k_hand_computed() -> None:
    scores = np.array([[0.9, 0.5, 0.1]], dtype=np.float32)
    qrels = {0: {2: 1.0}}  # relevant doc is at rank 3
    assert mrr_at_k(scores, qrels, k=3) == pytest.approx(1.0 / 3.0)


def test_ndcg_at_k_perfect() -> None:
    scores = np.array([[0.9, 0.5, 0.1]], dtype=np.float32)
    qrels = {0: {0: 1.0}}
    assert ndcg_at_k(scores, qrels, k=3) == pytest.approx(1.0)


def test_map_at_k_hand_computed() -> None:
    scores = np.array([[0.9, 0.5, 0.3, 0.1]], dtype=np.float32)
    qrels = {0: {1: 1.0, 3: 1.0}}
    # Ranking: 0,1,2,3. Hits at rank 2 (1/2) and rank 4 (2/4).
    # AP = (0.5 + 0.5) / 2 = 0.5
    assert map_score(scores, qrels, k=4) == pytest.approx(0.5)


def test_metrics_skip_queries_without_relevance() -> None:
    scores = np.array([[0.5, 0.5]], dtype=np.float32)
    qrels = {0: {0: 0.0, 1: 0.0}}
    assert recall_at_k(scores, qrels, k=2) == 0.0
    assert ndcg_at_k(scores, qrels, k=2) == 0.0
