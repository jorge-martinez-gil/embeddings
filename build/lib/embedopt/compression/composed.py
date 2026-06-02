"""Composition of compressors as a single :class:`Compressor`.

Lets the multi-objective optimizer search a strictly larger space than the
union of the base methods. The canonical paper compositions are:

* ``Truncate(keep_dim=k) -> ProductQuantize(M, n_bits)`` — first reduce the
  dimensionality, then quantize each subspace. Often dominates pure PQ at the
  same byte budget when the leading coordinates carry most of the signal.
* ``Truncate(keep_dim=k) -> Binary`` — drastically smaller codes than pure
  binary on full vectors, with the same Hamming surrogate scoring.

A composition is *trained sequentially*: each operator's ``fit`` sees the
output of the previous ``transform``. Storage is determined entirely by the
last operator (earlier ones only act as projections at index time).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from embedopt.compression.base import CompressedSet, Compressor
from embedopt.utils.types import FloatArray


@dataclass(slots=True)
class ComposedCompressor:
    """Chain ``len(stages)`` compressors. The last one owns the storage layout."""

    stages: Sequence[Compressor]
    name: str = "composed"
    _trained: bool = False
    _intermediate_dim: int = field(default=0, init=False)

    @property
    def trained(self) -> bool:
        return self._trained

    def _fit_chain(self, vectors: FloatArray) -> tuple[FloatArray, CompressedSet]:
        """Run ``fit`` then ``transform`` along the chain, returning the final codes."""
        current = vectors
        last_out: CompressedSet | None = None
        for i, stage in enumerate(self.stages):
            if not stage.trained:
                stage.fit(current)
            out = stage.transform(current)
            if i < len(self.stages) - 1:
                # Decode-or-pass-through: for non-terminal stages we expect their
                # codes to be float32 vectors that the next stage can consume
                # directly (which is true for Identity and Truncate).
                if not isinstance(out.codes, np.ndarray) or out.codes.dtype != np.float32:
                    raise TypeError(
                        f"Stage {i} ({stage.name}) does not produce float32 codes; "
                        "only Identity and Truncate are supported as non-terminal stages."
                    )
                current = out.codes
                self._intermediate_dim = int(current.shape[1])
            last_out = out
        assert last_out is not None
        return current, last_out

    def fit(self, vectors: FloatArray) -> None:
        self._fit_chain(vectors)
        self._trained = True

    def transform(self, vectors: FloatArray) -> CompressedSet:
        current = vectors
        last_out: CompressedSet | None = None
        for i, stage in enumerate(self.stages):
            out = stage.transform(current)
            if i < len(self.stages) - 1:
                current = out.codes
            last_out = out
        assert last_out is not None
        return last_out

    def score(self, queries: FloatArray, corpus: CompressedSet) -> FloatArray:
        # Push the queries through every non-terminal stage's projection, then
        # call the terminal stage's score against the prebuilt corpus codes.
        projected = queries
        for stage in self.stages[:-1]:
            projected = stage.transform(projected).codes
        terminal = self.stages[-1]
        return terminal.score(projected, corpus)

    def bytes_per_vector(self, dim: int) -> int:
        # Walk the chain to compute the dim observed by the terminal stage.
        d = dim
        for stage in self.stages[:-1]:
            d = stage.bytes_per_vector(d) // 4  # float32 in non-terminal stages
        return self.stages[-1].bytes_per_vector(d)
