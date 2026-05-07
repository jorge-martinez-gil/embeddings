"""CLI entrypoints for embedopt."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import hydra
import typer
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from embedopt.pipelines.smoke import SmokePipelineConfig, run_smoke_pipeline

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
    """Run an experiment using Hydra config files."""
    _hydra_entry()


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
