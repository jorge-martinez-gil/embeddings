"""Smoke pipeline for a tiny embedding experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from embedopt.models.backbones import SentenceTransformerEmbedder, TextEmbedder


@dataclass(slots=True)
class SmokePipeline:
    """Runs a tiny embedding task and returns one metric."""

    embedder: TextEmbedder

    def run(self, corpus: list[str], queries: list[str]) -> dict[str, float]:
        corpus_embeddings = self.embedder.encode(corpus)
        query_embeddings = self.embedder.encode(queries)

        if not corpus_embeddings or not query_embeddings:
            raise ValueError("Corpus and queries must be non-empty")

        metric = average_best_cosine_similarity(query_embeddings, corpus_embeddings)
        return {"avg_best_cosine": metric}


@dataclass(slots=True)
class SmokePipelineConfig:
    """Configuration for the smoke pipeline."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


def average_best_cosine_similarity(
    query_vectors: list[list[float]], corpus_vectors: list[list[float]]
) -> float:
    """Compute average best cosine similarity for query vectors against corpus vectors."""
    if not query_vectors or not corpus_vectors:
        raise ValueError("Vectors must be non-empty")

    best_scores: list[float] = []
    for qv in query_vectors:
        if len(qv) == 0:
            raise ValueError("Query vector cannot be empty")
        best = max(cosine_similarity(qv, cv) for cv in corpus_vectors)
        best_scores.append(best)

    return sum(best_scores) / len(best_scores)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    if len(a) == 0:
        raise ValueError("Vectors must be non-empty")

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Vectors must be non-zero")

    return cast(float, dot / (norm_a * norm_b))


def run_smoke_pipeline(config: SmokePipelineConfig | None = None) -> dict[str, float]:
    """Run smoke pipeline with default tiny corpus and query set."""
    cfg = config or SmokePipelineConfig()
    embedder = SentenceTransformerEmbedder(model_name=cfg.model_name)
    pipeline = SmokePipeline(embedder=embedder)

    corpus = [
        "Hydra composes hierarchical experiment configurations.",
        "Sentence transformers produce dense vector representations for text.",
        "Pareto optimization balances conflicting objectives.",
    ]
    queries = ["How do we create embeddings for text?"]

    return pipeline.run(corpus=corpus, queries=queries)
