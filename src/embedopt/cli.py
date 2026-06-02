"""CLI entrypoints for embedopt.

The CLI exposes four commands:

* ``embedopt run`` — original Hydra-driven smoke pipeline (kept for backward
  compatibility with Deliverable 1).
* ``embedopt pareto`` — multi-objective compression sweep on the synthetic
  retrieval dataset using the deterministic backbone. Prints a table of
  candidates and flags which ones are on the Pareto front.
* ``embedopt evaluate`` — single-config retrieval evaluation.
* ``embedopt profile`` — efficiency profiling for one compressor spec.

The ``pareto`` / ``evaluate`` / ``profile`` commands are intentionally
network-free (they default to the synthetic dataset and the hashing backbone)
so that ``embedopt --help`` and the smoke commands work out of the box on a
clean install with only the ``[dev]`` extras.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import typer
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from embedopt import __version__
from embedopt.compression.registry import build_compressor, spec_label
from embedopt.evaluation.datasets import make_synthetic_retrieval
from embedopt.evaluation.runner import evaluate_retrieval
from embedopt.models.backbones import HashingEmbedder
from embedopt.pipelines.pareto import default_search_space, run_pareto_sweep
from embedopt.pipelines.smoke import SmokePipelineConfig, run_smoke_pipeline
from embedopt.pipelines.storage_modes import (
    compare_storage_modes,
    default_storage_mode_specs,
    format_comparison_table,
)
from embedopt.profiling.aggregator import profile_compressor
from embedopt.utils.seeding import set_global_seed
from embedopt.utils.types import as_float_array

app = typer.Typer(help="EmbedOpt command line interface")


@dataclass(slots=True)
class ModelConfig:
    """Model settings."""

    name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(slots=True)
class EvalConfig:
    """Evaluation settings."""

    metric: str = "avg_best_cosine"


@dataclass(slots=True)
class AppConfig:
    """Hydra root configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


cs = ConfigStore.instance()
cs.store(name="base_config", node=AppConfig)


@app.command("run")
def run_command() -> None:
    """Run the smoke embedding pipeline using Hydra config files.

    Hydra reads ``sys.argv`` directly, so we hand it a clean argv (the
    Typer subcommand token has already been consumed by Typer itself).
    Pass Hydra-style overrides like ``embedopt run model.name=other-model``.
    """
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]] + saved_argv[2:]  # drop the 'run' token
    try:
        _hydra_entry()
    finally:
        sys.argv = saved_argv


@app.command("version")
def version_command() -> None:
    """Print the embedopt version."""
    typer.echo(__version__)


@app.command("pareto")
def pareto_command(
    dim: int = typer.Option(128, help="Backbone embedding dim (HashingEmbedder)."),
    seed: int = typer.Option(0, help="Root seed."),
    queries_per_topic: int = typer.Option(4, help="Synthetic queries per topic."),
    profile_repeats: int = typer.Option(15, help="Latency repeats per spec."),
) -> None:
    """Run the multi-objective compression sweep on the synthetic retrieval dataset."""
    set_global_seed(seed)
    embedder = HashingEmbedder(dim_=dim)
    dataset = make_synthetic_retrieval(n_queries_per_topic=queries_per_topic, seed=seed)
    result = run_pareto_sweep(
        embedder,
        dataset,
        specs=default_search_space(dim),
        profile_repeats=profile_repeats,
    )
    typer.echo(f"{'spec':<40s}  {'nDCG@10':>8s}  {'B/vec':>6s}  {'lat ms':>7s}  pareto")
    for i, c in enumerate(result.candidates):
        flag = "*" if i in result.pareto_idx else " "
        typer.echo(
            f"{c.label:<40s}  {c.metrics['ndcg_at_10']:>8.4f}  "
            f"{c.efficiency.bytes_per_vector:>6d}  "
            f"{c.efficiency.query_latency_ms:>7.3f}  {flag}"
        )
    typer.echo(f"hypervolume = {result.hypervolume:.6f}")


@app.command("evaluate")
def evaluate_command(
    compressor_name: str = typer.Argument("identity", help="Compressor name."),
    keep_dim: int = typer.Option(0, help="keep_dim for truncate."),
    n_subspaces: int = typer.Option(8, help="n_subspaces for product_quantize."),
    n_bits: int = typer.Option(8, help="n_bits for product_quantize."),
    dim: int = typer.Option(128, help="Backbone dim."),
    seed: int = typer.Option(0, help="Root seed."),
) -> None:
    """Single-config evaluation on the synthetic retrieval dataset."""
    set_global_seed(seed)
    embedder = HashingEmbedder(dim_=dim)
    dataset = make_synthetic_retrieval(n_queries_per_topic=4, seed=seed)
    spec: dict[str, object] = {"name": compressor_name}
    if compressor_name == "truncate":
        spec["keep_dim"] = keep_dim or dim // 2
    elif compressor_name == "product_quantize":
        spec["n_subspaces"] = n_subspaces
        spec["n_bits"] = n_bits
    compressor = build_compressor(spec)
    result = evaluate_retrieval(embedder, compressor, dataset)
    typer.echo(f"spec     = {spec_label(spec)}")
    typer.echo(f"recall@10= {result.recall_at_10:.4f}")
    typer.echo(f"mrr@10   = {result.mrr_at_10:.4f}")
    typer.echo(f"ndcg@10  = {result.ndcg_at_10:.4f}")
    typer.echo(f"map      = {result.map:.4f}")


@app.command("storage-modes")
def storage_modes_command(
    dim: int = typer.Option(128, help="Backbone embedding dim (HashingEmbedder)."),
    seed: int = typer.Option(0, help="Root seed."),
    queries_per_topic: int = typer.Option(4, help="Synthetic queries per topic."),
    profile_repeats: int = typer.Option(20, help="Latency repeats per mode."),
) -> None:
    """Direct comparison: float32 vs. float16 vs. int8 vs. binary vs. PQ.

    Runs all five common vector-DB storage modes on the synthetic retrieval
    dataset and prints a single side-by-side table with bytes/vec,
    compression ratio, nDCG@10 (and Δ vs. float32), recall, and latency.
    """
    set_global_seed(seed)
    embedder = HashingEmbedder(dim_=dim)
    dataset = make_synthetic_retrieval(n_queries_per_topic=queries_per_topic, seed=seed)
    comparison = compare_storage_modes(
        embedder,
        dataset,
        modes=default_storage_mode_specs(dim),
        profile_repeats=profile_repeats,
    )
    typer.echo(format_comparison_table(comparison))


@app.command("profile")
def profile_command(
    compressor_name: str = typer.Argument("identity", help="Compressor name."),
    dim: int = typer.Option(128, help="Backbone dim."),
    seed: int = typer.Option(0, help="Root seed."),
    n_repeats: int = typer.Option(20, help="Latency repeats."),
) -> None:
    """Efficiency profile of one compressor on the synthetic dataset."""
    set_global_seed(seed)
    embedder = HashingEmbedder(dim_=dim)
    dataset = make_synthetic_retrieval(n_queries_per_topic=4, seed=seed)
    corpus = as_float_array(embedder.encode(list(dataset.corpus)))
    queries = as_float_array(embedder.encode(list(dataset.queries)))
    spec: dict[str, object] = {"name": compressor_name}
    compressor = build_compressor(spec)
    stats, _ = profile_compressor(compressor, corpus, queries, n_repeats=n_repeats)
    typer.echo(f"bytes/vec    = {stats.bytes_per_vector}")
    typer.echo(f"corpus bytes = {stats.corpus_bytes}")
    typer.echo(f"latency med  = {stats.query_latency_ms:.3f} ms")
    typer.echo(f"latency p95  = {stats.query_p95_ms:.3f} ms")


def _get_config_path() -> str:
    project_root = Path(__file__).resolve().parents[2]
    return str(project_root / "configs")


@hydra.main(version_base=None, config_path=_get_config_path(), config_name="config")
def _hydra_entry(cfg: DictConfig) -> None:
    _run_from_cfg(cfg)


def _run_from_cfg(cfg: DictConfig) -> None:
    model_name = str(cfg.model.name)
    result = run_smoke_pipeline(SmokePipelineConfig(model_name=model_name))
    typer.echo(OmegaConf.to_yaml(cfg))
    typer.echo(result)


if __name__ == "__main__":
    app()
