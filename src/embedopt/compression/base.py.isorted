"""Compressor protocol and shared types.

A compressor is a *post-hoc* operator on a frozen embedder: ``fit`` may train
codebooks on a sample of vectors, ``transform`` produces compressed codes, and
``score`` computes a ``[n_queries, n_corpus]`` similarity matrix where higher
scores indicate closer matches. Every compressor reports its per-vector byte
footprint via :meth:`bytes_per_vector` so the multi-objective optimizer can
trade off quality against size without method-specific glue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from embedopt.utils.types import FloatArray


@dataclass(slots=True)
class CompressedSet:
    """A bundle of compressed vectors plus the byte cost of one entry."""

    codes: Any
    """Backend-specific code array. Shape and dtype depend on the compressor."""

    bytes_per_vector: int
    """Number of bytes one compressed vector occupies in memory/storage."""


class Compressor(Protocol):
    """Post-hoc compressor over a frozen embedder."""

    name: str

    @property
    def trained(self) -> bool:
        """Whether ``fit`` has been called (codecs that need it)."""

    def fit(self, vectors: FloatArray) -> None:
        """Train codebooks on a sample of vectors. May be a no-op."""

    def transform(self, vectors: FloatArray) -> CompressedSet:
        """Encode ``vectors`` (shape ``(n, d)``) into compressed codes."""

    def score(self, queries: FloatArray, corpus: CompressedSet) -> FloatArray:
        """Return a ``(n_queries, n_corpus)`` similarity matrix (higher = closer)."""

    def bytes_per_vector(self, dim: int) -> int:
        """Per-vector byte cost for an embedding dimension ``dim``."""
