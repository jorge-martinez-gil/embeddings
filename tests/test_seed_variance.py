"""Smoke tests for the multi-seed codebook-variance pipeline."""

from __future__ import annotations

import numpy as np

from embedopt.evaluation.datasets import make_synthetic_retrieval
from embedopt.models.backbones import HashingEmbedder
from embedopt.pipelines.seed_variance import (
    default_pq_specs,
    evaluate_compressor_seed_variance,
)


def test_default_pq_specs_filters_divisible_M_only() -> None:
    specs = default_pq_specs(dim=128)
    # dim=128 is divisible by 4, 8, 16, 32, 64.
    ms = sorted({s["n_subspaces"] for s in specs})
    assert ms == [4, 8, 16, 32, 64]
    names = {s["name"] for s in specs}
    assert names == {"product_quantize", "opq"}


def test_default_pq_specs_skips_indivisible_dim() -> None:
    specs = default_pq_specs(dim=33)  # divisible by nothing in the grid
    assert specs == []


def test_evaluate_seed_variance_pq_runs_offline() -> None:
    """The PQ branch is offline (no faiss) and must produce mean/std."""
    embedder = HashingEmbedder(dim_=64)
    dataset = make_synthetic_retrieval(n_queries_per_topic=2, seed=0)
    spec = {"name": "product_quantize", "n_subspaces": 8, "n_bits": 4}
    summary = evaluate_compressor_seed_variance(
        spec,
        seeds=[0, 1, 2],
        embedder=embedder,
        dataset=dataset,
    )
    assert summary.n_seeds == 3
    assert len(summary.rows) == 3
    # Different seeds should land at potentially different points; std must be defined.
    assert summary.ndcg_std >= 0.0
    assert all(0.0 <= r.ndcg_at_10 <= 1.0 for r in summary.rows)
    assert summary.spec_label.startswith("product_quantize(")


def test_evaluate_seed_variance_overrides_seed_in_spec() -> None:
    """Calling the helper with N seeds must actually thread N different seeds."""
    embedder = HashingEmbedder(dim_=64)
    dataset = make_synthetic_retrieval(n_queries_per_topic=2, seed=0)
    spec = {"name": "product_quantize", "n_subspaces": 4, "n_bits": 4, "seed": 999}
    summary = evaluate_compressor_seed_variance(
        spec,
        seeds=[7, 8],
        embedder=embedder,
        dataset=dataset,
    )
    assert sorted(r.seed for r in summary.rows) == [7, 8]


def test_evaluate_seed_variance_accepts_preembedded_corpus() -> None:
    rng = np.random.default_rng(0)
    corpus = rng.standard_normal((40, 32)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = corpus[:5]
    qrels = {i: {i: 1.0} for i in range(5)}
    spec = {"name": "product_quantize", "n_subspaces": 4, "n_bits": 4}
    summary = evaluate_compressor_seed_variance(
        spec,
        seeds=[0, 1],
        corpus=corpus,
        queries=queries,
        qrels=qrels,
    )
    assert summary.n_seeds == 2
