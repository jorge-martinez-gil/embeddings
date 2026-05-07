from __future__ import annotations

import pytest

from embedopt.pipelines.smoke import (
    SmokePipeline,
    average_best_cosine_similarity,
    cosine_similarity,
)


class DummyEmbedder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._mapping[text] for text in texts]


def test_cosine_similarity_identity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_average_best_cosine_similarity() -> None:
    query_vectors = [[1.0, 0.0], [0.0, 1.0]]
    corpus_vectors = [[1.0, 0.0], [0.7, 0.7]]

    metric = average_best_cosine_similarity(query_vectors, corpus_vectors)
    assert metric == pytest.approx((1.0 + 0.70710678) / 2, abs=1e-6)


def test_smoke_pipeline_returns_metric() -> None:
    embedder = DummyEmbedder(
        {
            "doc_a": [1.0, 0.0],
            "doc_b": [0.0, 1.0],
            "query": [0.9, 0.1],
        }
    )
    pipeline = SmokePipeline(embedder=embedder)

    result = pipeline.run(corpus=["doc_a", "doc_b"], queries=["query"])

    assert "avg_best_cosine" in result
    assert 0.0 <= result["avg_best_cosine"] <= 1.0


def test_pipeline_raises_on_empty_inputs() -> None:
    embedder = DummyEmbedder({})
    pipeline = SmokePipeline(embedder=embedder)
    with pytest.raises(ValueError, match="non-empty"):
        pipeline.run(corpus=[], queries=[])
