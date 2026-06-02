"""Index-level retrieval backends and evaluation helpers."""

from embedopt.indexing.backends import (
    DenseIndexEval,
    ExactNumpyIndex,
    FaissFlatIndex,
    dense_views_for_compressor,
    evaluate_dense_index,
)

__all__ = [
    "DenseIndexEval",
    "ExactNumpyIndex",
    "FaissFlatIndex",
    "dense_views_for_compressor",
    "evaluate_dense_index",
]
