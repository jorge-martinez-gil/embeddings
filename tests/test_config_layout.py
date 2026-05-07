from pathlib import Path


def test_hydra_config_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "configs" / "config.yaml").exists()
    assert (root / "configs" / "model" / "sentence_transformer.yaml").exists()
    assert (root / "configs" / "eval" / "smoke_metric.yaml").exists()
