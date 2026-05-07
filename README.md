# embeddings

`embedopt` is a research-oriented framework scaffold for multi-objective text embedding optimization.

## Deliverable 1 status

This repository now includes:

- Python package scaffold under `src/embedopt`
- Hydra config scaffold under `configs/`
- Typer CLI (`embedopt run`) that executes a smoke embedding pipeline
- One smoke pipeline with a Sentence-Transformers backbone and one metric (`avg_best_cosine`)
- Basic CI workflow for lint, type-check, and tests
- Pre-commit hooks for formatting/linting/type checks

## Quickstart

```bash
python -m pip install -e .[dev]
pytest
embedopt run
```

> Note: `embedopt run` requires Sentence-Transformers extras and model download on first run:
> `python -m pip install -e .[models]`.
