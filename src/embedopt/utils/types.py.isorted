"""Shared type aliases for embedopt.

All array-typed APIs use ``float32`` to keep memory accounting and codec
arithmetic predictable across the framework.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
"""Embedding matrix or vector. Shape: ``(n, d)`` or ``(d,)`` depending on use."""

IntArray = NDArray[np.int64]
"""Integer index/code array."""

ByteArray = NDArray[np.uint8]
"""Packed byte array (binary / quantized codes)."""


def as_float_array(x: Sequence[Sequence[float]] | FloatArray) -> FloatArray:
    """Normalize a list-of-lists or numpy array into a contiguous ``float32`` matrix."""
    if isinstance(x, np.ndarray):
        arr = np.ascontiguousarray(x, dtype=np.float32)
    else:
        arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")
    return arr
