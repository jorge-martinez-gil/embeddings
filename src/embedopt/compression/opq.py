"""Optimized Product Quantization (OPQ) compressor.

OPQ~\\cite{ge2014opq} adds a learned orthogonal rotation in front of
plain PQ so that residual energy is minimized after assignment. At the
same byte budget OPQ typically matches or beats vanilla
:class:`ProductQuantizeCompressor`; it is the default high-quality PQ
baseline in production stacks (FAISS ``OPQ``, ScaNN, Milvus
``IVF_PQ`` with the rotation flag).

This implementation is a thin wrapper around FAISS's ``OPQMatrix`` +
``ProductQuantizer``: we let FAISS do the alternating SVD that learns
the rotation, then encode/decode through the standard FAISS PQ API.
The compressor honors the same fit / transform / score / bytes
contract as the rest of :mod:`embedopt.compression`, so it slots into
the existing sweep without any special-casing.

The byte budget reported to the optimizer is ``n_subspaces`` bytes per
vector (one ``uint8`` per subspace), matching
:class:`ProductQuantizeCompressor`. This deliberately keeps the
storage axis comparable between PQ and OPQ so the catalog row reads
``OPQ vs PQ at fixed bytes`` rather than ``OPQ at different bytes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from embedopt.compression.base import CompressedSet
from embedopt.utils.types import FloatArray


@dataclass(slots=True)
class OptimizedProductQuantizeCompressor:
    """OPQ wrapped around FAISS for the heavy lifting.

    Parameters
    ----------
    n_subspaces:
        Number of PQ subspaces ``M``. The embedding dim must be
        divisible by ``M``.
    n_bits:
        Bits per subspace code (``2 ** n_bits`` centroids per
        subspace). Supports 1--8.
    seed:
        Seed for FAISS's internal RNG (set via ``faiss.cvar.seed`` when
        available).
    """

    n_subspaces: int = 8
    n_bits: int = 8
    seed: int = 0
    name: str = "opq"

    _wrapper: Any = None
    _code_size: int = 0
    _trained: bool = False
    _dim: int = 0

    @property
    def trained(self) -> bool:
        return self._trained

    def _validate(self, dim: int) -> None:
        if dim % self.n_subspaces != 0:
            raise ValueError(f"Embedding dim {dim} not divisible by n_subspaces={self.n_subspaces}")
        if self.n_bits < 1 or self.n_bits > 8:
            raise ValueError("n_bits must be between 1 and 8")

    def _require_faiss(self) -> Any:
        try:
            import faiss
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OptimizedProductQuantizeCompressor requires faiss. "
                "Install faiss-cpu (or faiss-gpu)."
            ) from exc
        return faiss

    def fit(self, vectors: FloatArray) -> None:
        faiss = self._require_faiss()
        n, d = vectors.shape
        self._validate(d)
        # FAISS's OPQMatrix internal training uses 256 centroids per subspace
        # regardless of the outer PQ n_bits; surface this clearly instead of
        # letting FAISS raise a cryptic Clustering error.
        if n < 256:
            raise ValueError(
                f"OptimizedProductQuantizeCompressor requires at least 256 "
                f"training vectors for the OPQ rotation; got {n}. Use "
                f"ProductQuantizeCompressor for smaller corpora."
            )
        if n < (1 << self.n_bits):
            raise ValueError(
                f"OPQ needs at least {1 << self.n_bits} training vectors "
                f"at n_bits={self.n_bits}, got {n}"
            )
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        # FAISS factory configures the OPQMatrix's internal PQ at the same
        # n_bits as the outer PQ so packed-code accounting is consistent.
        factory = f"OPQ{self.n_subspaces}_{d},PQ{self.n_subspaces}x{self.n_bits}"
        wrapper = faiss.index_factory(d, factory, faiss.METRIC_INNER_PRODUCT)
        wrapper.train(arr)
        # Use the standalone-encoder API (``sa_encode`` / ``sa_decode``) on
        # the wrapper directly. This sidesteps SWIG downcast bugs in some
        # FAISS builds where ``downcast_VectorTransform`` returns garbage.
        self._wrapper = wrapper
        self._code_size = int(wrapper.sa_code_size())
        self._dim = d
        self._trained = True

    def transform(self, vectors: FloatArray) -> CompressedSet:
        if not self._trained:
            self.fit(vectors)
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        n = arr.shape[0]
        # FAISS's Python wrapper accepts 2-D input and returns a (n, code_size)
        # uint8 array. Packed PQ codes when n_bits < 8 (e.g., n_bits=4 yields
        # 2 codes per byte) -- matches Milvus / FAISS production layouts.
        codes = self._wrapper.sa_encode(arr)
        codes = np.ascontiguousarray(codes, dtype=np.uint8).reshape(n, self._code_size)
        return CompressedSet(codes=codes, bytes_per_vector=int(self._code_size))

    def _decode(self, codes: Any) -> FloatArray:
        """Decode codes back to the original (un-rotated) space.

        Used by :func:`embedopt.indexing.dense_views_for_compressor` so
        OPQ can also be probed under dense ANN indexes.
        """
        n = int(codes.shape[0])
        codes_u8 = np.ascontiguousarray(codes, dtype=np.uint8).reshape(n, self._code_size)
        decoded = self._wrapper.sa_decode(codes_u8)
        decoded = np.ascontiguousarray(decoded, dtype=np.float32).reshape(n, self._dim)
        return cast(FloatArray, decoded)

    def score(self, queries: FloatArray, corpus: CompressedSet) -> FloatArray:
        if not self._trained:
            raise RuntimeError("OptimizedProductQuantizeCompressor must be fit before scoring")
        codes = corpus.codes
        decoded = self._decode(codes)
        norms = np.linalg.norm(decoded, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        decoded_norm = (decoded / norms).astype(np.float32, copy=False)
        sims = (queries @ decoded_norm.T).astype(np.float32, copy=False)
        return sims

    def bytes_per_vector(self, dim: int) -> int:
        """Packed PQ-code size in bytes.

        FAISS packs PQ codes at ``n_bits < 8``: ``M = 8`` with
        ``n_bits = 4`` gives ``M * n_bits / 8 = 4`` bytes per vector,
        matching Milvus / FAISS production layouts. At ``n_bits = 8``
        this equals ``n_subspaces``.
        """
        self._validate(dim)
        bits_total = self.n_subspaces * self.n_bits
        return (bits_total + 7) // 8
