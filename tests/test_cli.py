from __future__ import annotations

from typer.testing import CliRunner

from embedopt import __version__
from embedopt.cli import app


def test_version_command_prints_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_evaluate_identity_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["evaluate", "identity", "--dim", "64", "--seed", "0"])
    assert result.exit_code == 0
    assert "ndcg@10" in result.stdout


def test_profile_truncate_command() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "binary",
            "--dim",
            "64",
            "--seed",
            "0",
            "--n-repeats",
            "3",
        ],
    )
    assert result.exit_code == 0
    assert "bytes/vec" in result.stdout


def test_storage_modes_command_renders_table() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "storage-modes",
            "--dim",
            "64",
            "--seed",
            "0",
            "--queries-per-topic",
            "2",
            "--profile-repeats",
            "2",
        ],
    )
    assert result.exit_code == 0
    # All five common modes must appear in the rendered table.
    for mode in ("float32", "float16", "int8", "binary", "pq("):
        assert mode in result.stdout


def test_pareto_command_runs_end_to_end() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "pareto",
            "--dim",
            "64",
            "--seed",
            "0",
            "--queries-per-topic",
            "2",
            "--profile-repeats",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "hypervolume" in result.stdout
