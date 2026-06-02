"""Per-query metrics and statistical-significance helpers.

The headline-number tables in the paper need (a) per-query metric arrays so we
can compute paired-bootstrap confidence intervals, and (b) a CI estimator that
controls for the same query distribution across two systems. Both live here so
the rest of the evaluation module can stay focused on aggregate metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from embedopt.evaluation.metrics import QrelMap, _topk_indices
from embedopt.utils.types import FloatArray


@dataclass(slots=True, frozen=True)
class BootstrapCI:
    """Paired-bootstrap confidence interval on a per-query metric delta."""

    mean_delta: float
    lower: float
    upper: float
    alpha: float


@dataclass(slots=True, frozen=True)
class PairedSignificanceTest:
    """Two-sided paired randomization test over per-query metric deltas."""

    mean_delta: float
    p_value: float
    n_pairs: int
    n_resamples: int
    seed: int

    @property
    def significant_05(self) -> bool:
        return self.p_value < 0.05


def per_query_ndcg_at_k(scores: FloatArray, qrels: QrelMap, k: int) -> dict[int, float]:
    """Per-query nDCG@k. Returns a dict ``qid -> ndcg`` for queries with relevance."""
    rankings = _topk_indices(scores, k)
    out: dict[int, float] = {}
    for qid, qrel in qrels.items():
        if not any(r > 0 for r in qrel.values()):
            continue
        gains = np.array(
            [(2.0 ** qrel.get(int(d), 0.0)) - 1.0 for d in rankings[qid]],
            dtype=np.float64,
        )
        discounts = 1.0 / np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
        dcg = float((gains * discounts).sum())
        ideal_rels = sorted(qrel.values(), reverse=True)[:k]
        ideal_gains = np.array([(2.0**r) - 1.0 for r in ideal_rels], dtype=np.float64)
        ideal_disc = 1.0 / np.log2(np.arange(2, ideal_gains.size + 2, dtype=np.float64))
        idcg = float((ideal_gains * ideal_disc).sum())
        out[int(qid)] = dcg / idcg if idcg > 0 else 0.0
    return out


def per_query_recall_at_k(scores: FloatArray, qrels: QrelMap, k: int) -> dict[int, float]:
    """Per-query Recall@k."""
    rankings = _topk_indices(scores, k)
    out: dict[int, float] = {}
    for qid, qrel in qrels.items():
        relevant = {doc for doc, rel in qrel.items() if rel > 0}
        if not relevant:
            continue
        topk = rankings[qid]
        hit = sum(1 for d in topk if d in relevant)
        out[int(qid)] = hit / len(relevant)
    return out


def paired_bootstrap_ci(
    a_per_query: Mapping[int, float],
    b_per_query: Mapping[int, float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """Two-sided paired-bootstrap CI on the *mean per-query delta* ``a - b``.

    Only queries present in *both* mappings contribute, which guarantees the
    pairing assumption holds even when one system fails on some queries.
    """
    common = sorted(set(a_per_query) & set(b_per_query))
    if len(common) < 2:
        raise ValueError("Need at least 2 paired queries for a bootstrap CI")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    deltas = np.array([a_per_query[q] - b_per_query[q] for q in common], dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = deltas.size
    means = np.empty(n_resamples, dtype=np.float64)
    chunk_size = min(512, n_resamples)
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = deltas[idx].mean(axis=1)
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return BootstrapCI(
        mean_delta=float(deltas.mean()),
        lower=lower,
        upper=upper,
        alpha=alpha,
    )


def paired_randomization_test(
    a_per_query: Mapping[int, float],
    b_per_query: Mapping[int, float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PairedSignificanceTest:
    """Approximate two-sided paired randomization test for mean metric deltas.

    The null hypothesis is that system labels are exchangeable within each
    query pair. We randomly flip the sign of each per-query delta and estimate
    the probability of observing an absolute mean delta at least as large as
    the measured one.
    """
    common = sorted(set(a_per_query) & set(b_per_query))
    if len(common) < 2:
        raise ValueError("Need at least 2 paired queries for a paired test")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    deltas = np.array([a_per_query[q] - b_per_query[q] for q in common], dtype=np.float64)
    observed = abs(float(deltas.mean()))
    if observed == 0.0:
        return PairedSignificanceTest(
            mean_delta=0.0,
            p_value=1.0,
            n_pairs=len(common),
            n_resamples=n_resamples,
            seed=seed,
        )
    rng = np.random.default_rng(seed)
    exceed = 0
    chunk_size = min(512, n_resamples)
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        signs = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64),
            size=(stop - start, deltas.size),
        )
        null_means = np.abs((signs * deltas).mean(axis=1))
        exceed += int(np.count_nonzero(null_means >= observed))
    # Add-one smoothing avoids reporting impossible zero p-values from finite Monte Carlo.
    p_value = (exceed + 1.0) / (n_resamples + 1.0)
    return PairedSignificanceTest(
        mean_delta=float(deltas.mean()),
        p_value=float(p_value),
        n_pairs=len(common),
        n_resamples=n_resamples,
        seed=seed,
    )
