"""NSGA-II over a discrete configuration space.

For this framework, a *candidate* is an integer vector indexing into a
user-provided list of categorical / ordinal variables (e.g. compression method
choice, ``keep_dim``, ``n_bits``). The fitness of a candidate is supplied by
an evaluator callback that returns a dict of objective values; this lets the
exact same NSGA-II loop drive embedding compression, prompt-template search,
or any other discrete trade-off problem without modification.

The implementation is the classical NSGA-II of Deb et al. (2002): fast
non-dominated sort, crowding-distance secondary sort, binary tournament
selection, single-point crossover, and per-gene uniform-resample mutation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from embedopt.moo.objectives import Objective, to_min_matrix

_F64 = NDArray[np.float64]
CandidateGene = int
Candidate = tuple[CandidateGene, ...]
EvaluateFn = Callable[[Candidate], Mapping[str, float]]


@dataclass(slots=True, frozen=True)
class NSGAConfig:
    """Hyperparameters for the NSGA-II loop."""

    population_size: int = 24
    n_generations: int = 10
    crossover_prob: float = 0.9
    mutation_prob: float = 0.2
    seed: int = 0


@dataclass(slots=True)
class NSGAResult:
    """Output of :func:`run_nsga2`."""

    population: list[Candidate]
    objectives: list[dict[str, float]]
    front_indices: list[int]
    history: list[dict[str, float]] = field(default_factory=list)


def _fast_non_dominated_sort(points: _F64) -> list[list[int]]:
    n = points.shape[0]
    fronts: list[list[int]] = [[]]
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    dom_count = np.zeros(n, dtype=np.int64)
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            diff = points[p] - points[q]
            if (diff <= 0).all() and (diff < 0).any():
                dominated_by[p].append(q)
            elif (diff >= 0).all() and (diff > 0).any():
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return [f for f in fronts if f]


def _crowding_distance(points: _F64, indices: Sequence[int]) -> _F64:
    if not indices:
        return np.zeros(0, dtype=np.float64)
    sub = points[list(indices)]
    n, m = sub.shape
    dist = np.zeros(n, dtype=np.float64)
    for j in range(m):
        order = np.argsort(sub[:, j])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        span = sub[order[-1], j] - sub[order[0], j]
        if span == 0:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (sub[order[k + 1], j] - sub[order[k - 1], j]) / span
    return dist


def _select(
    rng: np.random.Generator,
    points: _F64,
    fronts: list[list[int]],
    crowding: dict[int, float],
) -> int:
    """Binary tournament selection."""
    a, b = (int(x) for x in rng.integers(0, points.shape[0], size=2))
    fa = next(i for i, f in enumerate(fronts) if a in f)
    fb = next(i for i, f in enumerate(fronts) if b in f)
    if fa < fb:
        return a
    if fb < fa:
        return b
    return a if crowding[a] >= crowding[b] else b


def _crossover_and_mutate(
    rng: np.random.Generator,
    parent_a: Candidate,
    parent_b: Candidate,
    cardinalities: Sequence[int],
    crossover_prob: float,
    mutation_prob: float,
) -> Candidate:
    n = len(parent_a)
    if rng.random() < crossover_prob and n >= 2:
        cut = int(rng.integers(1, n))
        child = list(parent_a[:cut]) + list(parent_b[cut:])
    else:
        child = list(parent_a)
    for j in range(n):
        if rng.random() < mutation_prob:
            child[j] = int(rng.integers(0, cardinalities[j]))
    return tuple(child)


def run_nsga2(
    *,
    cardinalities: Sequence[int],
    objectives: Sequence[Objective],
    evaluate: EvaluateFn,
    config: NSGAConfig | None = None,
) -> NSGAResult:
    """Run NSGA-II over a discrete decision space.

    Parameters
    ----------
    cardinalities:
        Per-gene cardinality. A candidate ``c`` satisfies ``0 <= c[j] < cardinalities[j]``.
    objectives:
        Optimization objectives (the order determines coordinate order).
    evaluate:
        Callback that maps a candidate to a dict of objective values.
    """
    cfg = config or NSGAConfig()
    rng = np.random.default_rng(cfg.seed)
    pop_size = cfg.population_size

    # Initial random population (deterministic given seed).
    population: list[Candidate] = []
    seen: set[Candidate] = set()
    while len(population) < pop_size:
        cand = tuple(int(rng.integers(0, c)) for c in cardinalities)
        if cand not in seen:
            seen.add(cand)
            population.append(cand)

    obj_history: list[dict[str, float]] = []
    cache: dict[Candidate, dict[str, float]] = {}

    def _eval(cand: Candidate) -> dict[str, float]:
        if cand not in cache:
            cache[cand] = dict(evaluate(cand))
        return cache[cand]

    for _ in range(cfg.n_generations):
        # Evaluate parents.
        parent_obj = [_eval(c) for c in population]
        parent_pts = to_min_matrix(objectives, parent_obj)
        # Generate offspring via tournament + crossover + mutation.
        fronts = _fast_non_dominated_sort(parent_pts)
        crowding: dict[int, float] = {}
        for f in fronts:
            cd = _crowding_distance(parent_pts, f)
            for idx, val in zip(f, cd, strict=True):
                crowding[idx] = float(val)
        offspring: list[Candidate] = []
        while len(offspring) < pop_size:
            i_a = _select(rng, parent_pts, fronts, crowding)
            i_b = _select(rng, parent_pts, fronts, crowding)
            child = _crossover_and_mutate(
                rng,
                population[i_a],
                population[i_b],
                cardinalities,
                cfg.crossover_prob,
                cfg.mutation_prob,
            )
            offspring.append(child)
        # Combine and select next generation.
        combined = population + offspring
        combined_obj = [_eval(c) for c in combined]
        combined_pts = to_min_matrix(objectives, combined_obj)
        fronts = _fast_non_dominated_sort(combined_pts)
        next_gen: list[Candidate] = []
        for f in fronts:
            if len(next_gen) + len(f) <= pop_size:
                next_gen.extend(combined[i] for i in f)
            else:
                cd = _crowding_distance(combined_pts, f)
                order = sorted(zip(f, cd, strict=True), key=lambda t: -t[1])
                slots = pop_size - len(next_gen)
                next_gen.extend(combined[i] for i, _ in order[:slots])
                break
        population = next_gen
        obj_history.append({"front_size": float(len(fronts[0]))})

    final_obj = [_eval(c) for c in population]
    final_pts = to_min_matrix(objectives, final_obj)
    fronts = _fast_non_dominated_sort(final_pts)
    return NSGAResult(
        population=list(population),
        objectives=final_obj,
        front_indices=[int(i) for i in fronts[0]],
        history=obj_history,
    )
