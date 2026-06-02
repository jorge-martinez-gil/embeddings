"""Index-level retrieval backends and evaluation helpers."""

from embedopt.indexing.backends import (
    KNOWN_BACKENDS,
    DenseIndexEval,
    ExactNumpyIndex,
    FaissFlatIndex,
    FaissHNSWIndex,
    FaissIVFIndex,
    FaissIVFPQIndex,
    FaissOPQIndex,
    dense_views_for_compressor,
    evaluate_dense_index,
    make_dense_index,
)

__all__ = [
    "KNOWN_BACKENDS",
    "DenseIndexEval",
    "ExactNumpyIndex",
    "FaissFlatIndex",
    "FaissHNSWIndex",
    "FaissIVFIndex",
    "FaissIVFPQIndex",
    "FaissOPQIndex",
    "dense_views_for_compressor",
    "evaluate_dense_index",
    "make_dense_index",
]
