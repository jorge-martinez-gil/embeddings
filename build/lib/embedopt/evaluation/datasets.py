"""Dataset abstractions and synthetic generators.

The framework ships two synthetic datasets with deterministic generators so
that the full pipeline runs without network access:

* :func:`make_synthetic_sts` — a sentence-similarity dataset of paraphrase
  triplets and unrelated pairs, with graded scores in ``[0, 5]``.
* :func:`make_synthetic_retrieval` — a small retrieval corpus with planted
  topical clusters and ``k`` queries per cluster, producing realistic-looking
  ``Recall@10`` and ``nDCG@10`` numbers for our deterministic backbones.

Real benchmarks (MTEB, BEIR) plug in via the same dataclasses and are loaded
from local paths; we deliberately avoid hard-coding HuggingFace pulls in this
core module so the framework can run on air-gapped hardware.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from embedopt.evaluation.metrics import QrelMap


@dataclass(slots=True, frozen=True)
class STSDataset:
    """A semantic textual similarity dataset of (a, b, score) triples."""

    name: str
    sentences_a: Sequence[str]
    sentences_b: Sequence[str]
    scores: Sequence[float]


@dataclass(slots=True, frozen=True)
class RetrievalDataset:
    """A retrieval dataset: corpus, queries, and per-query relevance judgements."""

    name: str
    corpus: Sequence[str]
    queries: Sequence[str]
    qrels: QrelMap


_TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "machine learning",
        (
            "neural networks learn from data",
            "deep learning models require large datasets",
            "gradient descent optimizes model parameters",
            "transformers attend to every token in a sequence",
            "embedding models map text into dense vector spaces",
        ),
    ),
    (
        "cooking",
        (
            "garlic and onion form the base of many sauces",
            "olive oil is the classic medium for sauteing vegetables",
            "fresh herbs brighten finished dishes at the end of cooking",
            "salt is the primary seasoning across cuisines",
            "stocks and broths add depth to soups and stews",
        ),
    ),
    (
        "astronomy",
        (
            "stars form inside collapsing clouds of cold gas",
            "supernovae enrich galaxies with heavy elements",
            "black holes warp the spacetime around them",
            "spectroscopy reveals the chemical makeup of distant stars",
            "the universe expands at an accelerating rate",
        ),
    ),
    (
        "music theory",
        (
            "scales are ordered sequences of notes that share a tonal center",
            "chords stack thirds above a root note to form harmony",
            "rhythm gives music its temporal structure",
            "counterpoint weaves independent melodic lines together",
            "modulation shifts a piece into a new key",
        ),
    ),
)


def make_synthetic_sts(*, n_pairs: int = 200, seed: int = 0) -> STSDataset:
    """Generate a deterministic STS-style dataset.

    Half the pairs are intra-topic (similar, scored 4-5), the rest are
    cross-topic (dissimilar, scored 0-1). A small fraction of pairs reuse the
    same sentence on both sides as gold-positive anchors.
    """
    rng = np.random.default_rng(seed)
    a_list: list[str] = []
    b_list: list[str] = []
    scores: list[float] = []
    topics = [t for _, t in _TOPICS]
    n_topics = len(topics)
    for i in range(n_pairs):
        if i % 2 == 0:
            t = int(rng.integers(0, n_topics))
            sa, sb = rng.choice(topics[t], size=2, replace=True)
            score = float(rng.uniform(4.0, 5.0)) if sa == sb else float(rng.uniform(3.5, 4.8))
        else:
            t1, t2 = rng.choice(n_topics, size=2, replace=False)
            sa = str(rng.choice(topics[int(t1)]))
            sb = str(rng.choice(topics[int(t2)]))
            score = float(rng.uniform(0.0, 1.2))
        a_list.append(str(sa))
        b_list.append(str(sb))
        scores.append(score)
    return STSDataset(
        name=f"sts_synthetic_n{n_pairs}_s{seed}",
        sentences_a=a_list,
        sentences_b=b_list,
        scores=scores,
    )


def make_synthetic_retrieval(*, n_queries_per_topic: int = 4, seed: int = 0) -> RetrievalDataset:
    """Generate a deterministic retrieval dataset.

    Every topic contributes its sentences to the shared corpus. Per-topic
    queries are *paraphrased prompts* (e.g., "tell me about...") whose
    relevance judgements mark all sentences in the same topic as positive.
    """
    rng = np.random.default_rng(seed)
    corpus: list[str] = []
    topic_ranges: list[tuple[int, int]] = []
    cursor = 0
    topic_names: list[str] = []
    for tname, sentences in _TOPICS:
        topic_names.append(tname)
        corpus.extend(sentences)
        topic_ranges.append((cursor, cursor + len(sentences)))
        cursor += len(sentences)

    queries: list[str] = []
    qrels: dict[int, dict[int, float]] = {}
    for ti, (tname, _) in enumerate(_TOPICS):
        for _ in range(n_queries_per_topic):
            templates = (
                f"tell me about {tname}",
                f"information regarding {tname}",
                f"what is {tname}",
                f"explain the basics of {tname}",
            )
            q = str(rng.choice(templates))
            qid = len(queries)
            queries.append(q)
            start, end = topic_ranges[ti]
            qrels[qid] = {i: 1.0 for i in range(start, end)}
    return RetrievalDataset(
        name=f"retrieval_synthetic_q{n_queries_per_topic}_s{seed}",
        corpus=corpus,
        queries=queries,
        qrels=qrels,
    )
