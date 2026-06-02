"""Evaluation metrics, datasets, and end-to-end runners."""

from embedopt.evaluation.beir import (
    DEFAULT_BEIR_NAMES,
    load_beir_dataset_hf,
    load_beir_dataset_local,
)
from embedopt.evaluation.datasets import (
    RetrievalDataset,
    STSDataset,
    make_synthetic_retrieval,
    make_synthetic_sts,
)
from embedopt.evaluation.metrics import (
    QrelMap,
    cosine_pairs,
    map_score,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    spearman_correlation,
)
from embedopt.evaluation.runner import (
    RetrievalResult,
    STSResult,
    evaluate_retrieval,
    evaluate_sts,
    quality_score,
)
from embedopt.evaluation.stats import (
    BootstrapCI,
    PairedSignificanceTest,
    paired_bootstrap_ci,
    paired_randomization_test,
    per_query_ndcg_at_k,
    per_query_recall_at_k,
)

__all__ = [
    "BootstrapCI",
    "DEFAULT_BEIR_NAMES",
    "PairedSignificanceTest",
    "QrelMap",
    "RetrievalDataset",
    "RetrievalResult",
    "STSDataset",
    "STSResult",
    "cosine_pairs",
    "evaluate_retrieval",
    "evaluate_sts",
    "load_beir_dataset_hf",
    "load_beir_dataset_local",
    "make_synthetic_retrieval",
    "make_synthetic_sts",
    "map_score",
    "mrr_at_k",
    "ndcg_at_k",
    "paired_bootstrap_ci",
    "paired_randomization_test",
    "per_query_ndcg_at_k",
    "per_query_recall_at_k",
    "quality_score",
    "recall_at_k",
    "spearman_correlation",
]
