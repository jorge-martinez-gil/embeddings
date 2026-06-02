"""Multi-objective optimization primitives."""

from embedopt.moo.hypervolume import hypervolume
from embedopt.moo.nsga import NSGAConfig, NSGAResult, run_nsga2
from embedopt.moo.objectives import Objective, Sense, to_min_matrix
from embedopt.moo.pareto import is_dominated, non_dominated_mask, pareto_indices
from embedopt.moo.scalarization import normalize_columns, tchebycheff, weighted_sum

__all__ = [
    "NSGAConfig",
    "NSGAResult",
    "Objective",
    "Sense",
    "hypervolume",
    "is_dominated",
    "non_dominated_mask",
    "normalize_columns",
    "pareto_indices",
    "run_nsga2",
    "tchebycheff",
    "to_min_matrix",
    "weighted_sum",
]
