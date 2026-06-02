from __future__ import annotations

from embedopt.compression import IdentityCompressor
from embedopt.evaluation.datasets import make_synthetic_retrieval
from embedopt.evaluation.runner import evaluate_retrieval
from embedopt.models.backbones import HashingEmbedder
from embedopt.pipelines.pareto import default_search_space, run_pareto_sweep


def test_retrieval_evaluation_identity_baseline_is_strong() -> None:
    embedder = HashingEmbedder(dim_=128)
    dataset = make_synthetic_retrieval(n_queries_per_topic=2, seed=0)
    result = evaluate_retrieval(embedder, IdentityCompressor(), dataset)
    # Sanity: the deterministic backbone should at least exceed a random baseline.
    assert result.recall_at_10 >= 0.2
    assert 0.0 <= result.ndcg_at_10 <= 1.0


def test_pareto_sweep_runs_end_to_end_and_finds_a_front() -> None:
    embedder = HashingEmbedder(dim_=64)
    dataset = make_synthetic_retrieval(n_queries_per_topic=2, seed=0)
    specs = default_search_space(64)
    result = run_pareto_sweep(embedder, dataset, specs=specs, profile_repeats=3, profile_warmup=1)
    assert len(result.candidates) == len(specs)
    assert len(result.pareto_idx) >= 1
    assert result.hypervolume >= 0.0
    # Every candidate must report all three objective values.
    for c in result.candidates:
        assert "ndcg_at_10" in c.metrics
        assert c.efficiency.bytes_per_vector > 0
        assert c.efficiency.query_latency_ms >= 0.0


def test_default_search_space_filters_truncate_dims_to_backbone() -> None:
    space = default_search_space(64)
    truncate_dims = [int(s["keep_dim"]) for s in space if s["name"] == "truncate"]
    assert all(d <= 64 for d in truncate_dims)
