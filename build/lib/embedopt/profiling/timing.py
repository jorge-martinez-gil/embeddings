"""Latency / throughput timers.

We report median and p95 latency over a configurable number of repeats with
warmup, using ``time.perf_counter_ns`` for nanosecond resolution. The bench
function is closure-friendly so callers can profile arbitrary callables
without monkey-patching.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class TimingStats:
    """Latency summary in milliseconds."""

    median_ms: float
    p95_ms: float
    min_ms: float
    n_repeats: int

    @property
    def throughput_per_sec(self) -> float:
        if self.median_ms <= 0:
            return float("inf")
        return 1000.0 / self.median_ms


def time_callable(
    fn: Callable[[], T],
    *,
    n_repeats: int = 25,
    n_warmup: int = 3,
) -> TimingStats:
    """Time ``fn`` and return :class:`TimingStats` over ``n_repeats`` calls."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    for _ in range(max(n_warmup, 0)):
        fn()
    timings_ns: list[int] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter_ns()
        fn()
        timings_ns.append(time.perf_counter_ns() - t0)
    arr = np.asarray(timings_ns, dtype=np.float64) / 1e6  # to ms
    return TimingStats(
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        min_ms=float(arr.min()),
        n_repeats=n_repeats,
    )
