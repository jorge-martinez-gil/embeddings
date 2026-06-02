"""Half-precision (``float16``) storage.

Stores each coordinate as IEEE-754 binary16 (1 sign bit, 5 exponent bits, 10
mantissa bits). This is the workhorse "halfvec" / "f16" mode of essentially
every modern vector database — pgvector ``halfvec``, Milvus ``FLOAT16``,
Qdrant ``Float16``, Weaviate quantization, FAISS ``IndexHNSWFlat`` with
``fp16`` storage — because it halves the memory of ``float32`` for almost no
measurable retrieval quality loss on cosine-normalized embeddings.

For scoring we upcast the corpus to ``float32`` once per query batch and run
a standard dense matmul. Because ``float16`` has only ~3 decimal digits of
mantissa precision the dequantize is essentially lossless for the unit-norm
embeddings produced by modern sentence encoders (whose components live in
``[-1, 1]``), and the per-vector byte cost is exactly half of ``identity``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from embedopt.compression.base import CompressedSet
from embedopt.utils.types import FloatArray


@dataclass(slots=True)
class Float16Compressor:
    """Half-precision (``float16``) storage with full-precision scoring."""

    name: str = "float16"

    @property
    def trained(self) -> bool:
        return True

    def fit(self, vectors: FloatArray) -> None:  # noqa: D401 - protocol no-op
        return None

    def transform(self, vectors: FloatArray) -> CompressedSet:
        codes = np.ascontiguousarray(vectors, dtype=np.float16)
        bpv = int(codes.shape[1] * 2)
        return CompressedSet(codes=codes, bytes_per_vector=bpv)

    def score(self, queries: FloatArray, corpus: CompressedSet) -> FloatArray:
        codes = cast(np.ndarray[Any, Any], corpus.codes)
        decoded = codes.astype(np.float32, copy=False)
        sims: FloatArray = (queries @ decoded.T).astype(np.float32, copy=False)
        return sims

    def bytes_per_vector(self, dim: int) -> int:
        return int(dim * 2)
