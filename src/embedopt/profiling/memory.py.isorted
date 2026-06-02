"""Peak Python memory profiling via :mod:`tracemalloc`.

Reports the peak heap delta in bytes consumed by an arbitrary callable, which
is sufficient for measuring corpus-size memory blow-up across compressors. For
extension-level profiling (e.g. native BLAS arenas), supplement these numbers
with an external sampling profiler.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class MemoryStats:
    """Peak heap delta in bytes for one invocation of a callable."""

    peak_bytes: int


def measure_peak_memory(fn: Callable[[], T]) -> MemoryStats:
    """Run ``fn`` once and return :class:`MemoryStats` with the peak heap delta."""
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return MemoryStats(peak_bytes=int(peak))
