"""Composed end-to-end experiment pipelines."""

from embedopt.pipelines.pareto import (
    CompressionCandidate,
    ParetoSweepResult,
    default_search_space,
    run_pareto_sweep,
)
from embedopt.pipelines.smoke import (
    SmokePipeline,
    SmokePipelineConfig,
    average_best_cosine_similarity,
    cosine_similarity,
    run_smoke_pipeline,
)
from embedopt.pipelines.seed_variance import (
    SeedVarianceRow,
    SeedVarianceSummary,
    default_pq_specs,
    evaluate_compressor_seed_variance,
)
from embedopt.pipelines.storage_modes import (
    StorageModeComparison,
    StorageModeRow,
    compare_storage_modes,
    default_storage_mode_specs,
    format_comparison_table,
)

__all__ = [
    "CompressionCandidate",
    "ParetoSweepResult",
    "SmokePipeline",
    "SmokePipelineConfig",
    "SeedVarianceRow",
    "SeedVarianceSummary",
    "StorageModeComparison",
    "StorageModeRow",
    "average_best_cosine_similarity",
    "compare_storage_modes",
    "cosine_similarity",
    "default_pq_specs",
    "default_search_space",
    "default_storage_mode_specs",
    "evaluate_compressor_seed_variance",
    "format_comparison_table",
    "run_pareto_sweep",
    "run_smoke_pipeline",
]
