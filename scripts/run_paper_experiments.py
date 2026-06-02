"""End-to-end paper experiments runner (Colab A100 friendly).

What it does
------------
For each (backbone, dataset) pair, this script:

1. Encodes the BEIR corpus and queries on whatever device sentence-transformers
   picks up (A100 if visible to torch, otherwise CPU).
2. Runs the multi-objective compression sweep over the framework's default
   space *plus* two-stage compositions (truncate -> PQ, truncate -> binary).
3. Computes per-query nDCG@10 for every spec and reports a paired-bootstrap
   95% CI vs. the identity baseline.
4. Writes a tidy CSV of all candidates and a JSON summary of the Pareto front
   and hypervolume to ``--output-dir`` (default: ``./results``).

How to run on Google Colab (A100 runtime)
-----------------------------------------
Open a fresh Colab, switch the runtime to *GPU* (A100 if available), then:

.. code-block:: python

    !git clone https://github.com/<your-username>/embeddings.git
    %cd embeddings
    !pip install -e .[paper]
    !python scripts/run_paper_experiments.py \
        --backbones e5-base bge-base \
        --datasets beir-local:/content/scifact beir-local:/content/nfcorpus \
        --output-dir results

If you don't have BEIR data locally yet, fetch the official archive (one-time):

.. code-block:: python

    !pip install beir
    from beir.util import download_and_unzip
    for name in ("scifact", "nfcorpus", "arguana", "fiqa", "trec-covid"):
        download_and_unzip(
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip",
            "/content",
        )

Smoke mode
----------
Pass ``--smoke`` to run with the deterministic ``hashing`` backbone on the
synthetic dataset; useful for verifying the script end-to-end without GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from embedopt.compression.base import CompressedSet, Compressor
from embedopt.compression.registry import build_compressor, spec_label
from embedopt.evaluation.beir import load_beir_dataset_local
from embedopt.evaluation.datasets import RetrievalDataset, make_synthetic_retrieval
from embedopt.evaluation.metrics import QrelMap, _topk_indices
from embedopt.evaluation.stats import (
    BootstrapCI,
    paired_bootstrap_ci,
    paired_randomization_test,
)
from embedopt.indexing import dense_views_for_compressor, evaluate_dense_index
from embedopt.models.backbones import TextEmbedder
from embedopt.models.factory import PrefixedEmbedder, make_backbone
from embedopt.moo.hypervolume import hypervolume
from embedopt.moo.objectives import Objective, to_min_matrix
from embedopt.moo.pareto import pareto_indices
from embedopt.profiling.aggregator import profile_compressor
from embedopt.utils.manifest import RunManifest
from embedopt.utils.seeding import set_global_seed
from embedopt.utils.types import FloatArray, as_float_array

_CHECKPOINT_FORMAT_VERSION = 1


def _score_device_available(score_device: str) -> str:
    """Resolve the requested top-k scoring device without making torch required."""
    requested = score_device.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("--score-device must be one of: auto, cpu, cuda")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except Exception:
        if requested == "cuda":
            raise RuntimeError("--score-device cuda requested, but torch is not importable")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise RuntimeError("--score-device cuda requested, but CUDA is not available to torch")
    return "cpu"


def _encode_split(
    embedder: TextEmbedder, texts: list[str], *, kind: str, batch_size: int
) -> np.ndarray:
    """Encode in batches; uses query/passage prefixes if the embedder supports them."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if isinstance(embedder, PrefixedEmbedder):
            out.extend(embedder.encode(batch, kind=cast(Any, kind), batch_size=batch_size))
        else:
            out.extend(embedder.encode(batch, batch_size=batch_size))
    return as_float_array(out)


def _embedding_cache_root(output_dir: Path) -> Path:
    return output_dir / "_embeddings"


def _embedding_cache_path(output_dir: Path, base: str, cache_key: str) -> Path:
    return _embedding_cache_root(output_dir) / f"{base}__{cache_key[:16]}.npz"


def _embedding_cache_key(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_embedding_cache(
    output_dir: Path,
    *,
    base: str,
    cache_key: str,
    expected_dim: int | None = None,
) -> tuple[FloatArray, FloatArray, float] | None:
    path = _embedding_cache_path(output_dir, base, cache_key)
    if not _is_nonempty_file(path):
        return None
    try:
        with np.load(path) as data:
            stored_key = str(data["cache_key"].item())
            corpus = data["corpus"].astype(np.float32, copy=False)
            queries = data["queries"].astype(np.float32, copy=False)
            encode_secs = float(data["encode_secs"].item())
    except Exception:
        return None
    if stored_key != cache_key:
        return None
    if corpus.ndim != 2 or queries.ndim != 2 or corpus.shape[1] != queries.shape[1]:
        return None
    if expected_dim is not None and corpus.shape[1] != expected_dim:
        return None
    return cast(FloatArray, corpus), cast(FloatArray, queries), encode_secs


def _write_embedding_cache(
    output_dir: Path,
    *,
    base: str,
    cache_key: str,
    corpus_vec: FloatArray,
    query_vec: FloatArray,
    encode_secs: float,
) -> None:
    root = _embedding_cache_root(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = _embedding_cache_path(output_dir, base, cache_key)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as f:
        np.savez(
            f,
            cache_key=np.array(cache_key),
            encode_secs=np.array(encode_secs, dtype=np.float64),
            corpus=corpus_vec.astype(np.float32, copy=False),
            queries=query_vec.astype(np.float32, copy=False),
        )
    tmp_path.replace(path)


def _project_queries_for_terminal(
    compressor: Compressor,
    queries: FloatArray,
) -> tuple[Compressor, FloatArray] | None:
    stages = getattr(compressor, "stages", None)
    if not stages:
        return compressor, queries
    projected = queries
    for stage in stages[:-1]:
        out = stage.transform(projected)
        if not isinstance(out.codes, np.ndarray) or out.codes.dtype != np.float32:
            return None
        projected = cast(FloatArray, out.codes)
    return cast(Compressor, stages[-1]), projected


def _gpu_dense_topk(
    terminal: Compressor,
    compressed: CompressedSet,
    queries: FloatArray,
    *,
    k: int,
    score_batch_size: int,
    device: str,
) -> np.ndarray | None:
    try:
        import torch
    except Exception:
        return None

    codes = compressed.codes
    normalize_query = False
    if getattr(terminal, "name", "") == "scalar_int8" and hasattr(terminal, "_decode"):
        decoded = terminal._decode(cast(np.ndarray[Any, Any], codes))
        norms = np.linalg.norm(decoded, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        corpus_np = cast(FloatArray, (decoded / norms).astype(np.float32, copy=False))
    elif isinstance(codes, np.ndarray) and codes.dtype == np.float32 and codes.ndim == 2:
        corpus_np = cast(FloatArray, codes)
        normalize_query = getattr(terminal, "name", "") == "truncate"
    else:
        return None

    if queries.shape[1] < corpus_np.shape[1]:
        return None

    topk_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        corpus_t = torch.as_tensor(corpus_np, dtype=torch.float32, device=device)
        k_eff = min(k, int(corpus_t.shape[0]))
        for start in range(0, queries.shape[0], score_batch_size):
            q_np = queries[start : start + score_batch_size]
            q_t = torch.as_tensor(q_np, dtype=torch.float32, device=device)
            if q_t.shape[1] > corpus_t.shape[1]:
                q_t = q_t[:, : corpus_t.shape[1]]
            if normalize_query:
                norms = torch.linalg.vector_norm(q_t, dim=1, keepdim=True).clamp_min(1e-12)
                q_t = q_t / norms
            scores = q_t @ corpus_t.T
            topk_chunks.append(torch.topk(scores, k=k_eff, dim=1).indices.cpu().numpy())
    if not topk_chunks:
        return np.zeros((0, 0), dtype=np.int64)
    return np.vstack(topk_chunks).astype(np.int64, copy=False)


def _gpu_pq_topk(
    terminal: Compressor,
    compressed: CompressedSet,
    queries: FloatArray,
    *,
    k: int,
    score_batch_size: int,
    device: str,
) -> np.ndarray | None:
    try:
        import torch
    except Exception:
        return None

    if getattr(terminal, "name", "") != "product_quantize":
        return None
    codes = compressed.codes
    codebooks = getattr(terminal, "_codebooks", None)
    n_subspaces = int(getattr(terminal, "n_subspaces", 0))
    sub_dim = int(getattr(terminal, "_sub_dim", 0))
    if (
        not isinstance(codes, np.ndarray)
        or codes.ndim != 2
        or not isinstance(codebooks, np.ndarray)
        or codebooks.ndim != 3
        or n_subspaces <= 0
        or sub_dim <= 0
        or queries.shape[1] != n_subspaces * sub_dim
    ):
        return None

    topk_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        codebooks_t = torch.as_tensor(codebooks, dtype=torch.float32, device=device)
        codes_t = torch.as_tensor(codes, device=device).long()
        cc = (codebooks_t * codebooks_t).sum(dim=2)
        k_eff = min(k, int(codes_t.shape[0]))
        for start in range(0, queries.shape[0], score_batch_size):
            q_np = queries[start : start + score_batch_size]
            q_t = torch.as_tensor(q_np, dtype=torch.float32, device=device)
            q_r = q_t.reshape(q_t.shape[0], n_subspaces, sub_dim)
            qq = (q_r * q_r).sum(dim=2)
            qc = torch.einsum("nmd,mkd->nmk", q_r, codebooks_t)
            tables = qq[:, :, None] + cc[None, :, :] - 2.0 * qc
            sims = -tables[:, 0].index_select(1, codes_t[:, 0])
            for m in range(1, n_subspaces):
                sims.add_(-tables[:, m].index_select(1, codes_t[:, m]))
            topk_chunks.append(torch.topk(sims, k=k_eff, dim=1).indices.cpu().numpy())
    if not topk_chunks:
        return np.zeros((0, 0), dtype=np.int64)
    return np.vstack(topk_chunks).astype(np.int64, copy=False)


def _gpu_topk_for_compressor(
    compressor: Compressor,
    compressed: CompressedSet,
    queries: FloatArray,
    *,
    k: int,
    score_batch_size: int,
    device: str,
) -> np.ndarray | None:
    projected = _project_queries_for_terminal(compressor, queries)
    if projected is None:
        return None
    terminal, projected_queries = projected
    pq_topk = _gpu_pq_topk(
        terminal,
        compressed,
        projected_queries,
        k=k,
        score_batch_size=score_batch_size,
        device=device,
    )
    if pq_topk is not None:
        return pq_topk
    return _gpu_dense_topk(
        terminal,
        compressed,
        projected_queries,
        k=k,
        score_batch_size=score_batch_size,
        device=device,
    )


def _topk_for_compressor(
    compressor: Compressor,
    compressed: CompressedSet,
    queries: FloatArray,
    *,
    k: int,
    score_batch_size: int,
    score_device: str,
) -> np.ndarray:
    if score_batch_size < 1:
        raise ValueError("score_batch_size must be >= 1")
    if score_device == "cuda":
        try:
            gpu_topk = _gpu_topk_for_compressor(
                compressor,
                compressed,
                queries,
                k=k,
                score_batch_size=score_batch_size,
                device=score_device,
            )
            if gpu_topk is not None:
                return gpu_topk
        except RuntimeError as exc:
            print(f"  GPU scoring failed for {getattr(compressor, 'name', 'compressor')}: {exc}")
            print("  falling back to CPU NumPy scoring for this spec")
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
    chunks: list[np.ndarray] = []
    for start in range(0, queries.shape[0], score_batch_size):
        scores = compressor.score(queries[start : start + score_batch_size], compressed)
        chunks.append(_topk_indices(scores, k=k))
    if not chunks:
        return np.zeros((0, 0), dtype=np.int64)
    return np.vstack(chunks).astype(np.int64, copy=False)


def _per_query_ndcg_from_topk(topk: np.ndarray, qrels: QrelMap) -> dict[int, float]:
    out: dict[int, float] = {}
    for qid, qrel in qrels.items():
        qid_i = int(qid)
        if qid_i >= topk.shape[0] or not any(r > 0 for r in qrel.values()):
            continue
        ranking = topk[qid_i]
        gains = np.array(
            [(2.0 ** qrel.get(int(doc), 0.0)) - 1.0 for doc in ranking],
            dtype=np.float64,
        )
        discounts = 1.0 / np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
        dcg = float((gains * discounts).sum())
        ideal_rels = sorted(qrel.values(), reverse=True)[: ranking.size]
        ideal_gains = np.array([(2.0**rel) - 1.0 for rel in ideal_rels], dtype=np.float64)
        ideal_disc = 1.0 / np.log2(np.arange(2, ideal_gains.size + 2, dtype=np.float64))
        idcg = float((ideal_gains * ideal_disc).sum())
        out[qid_i] = dcg / idcg if idcg > 0 else 0.0
    return out


def _parse_int_list(raw: str) -> tuple[int, ...]:
    vals = tuple(int(p.strip()) for p in raw.split(",") if p.strip())
    if not vals:
        raise ValueError(f"Expected at least one integer in {raw!r}")
    return vals


def _composition_specs(
    dim: int,
    *,
    truncate_dims: Sequence[int],
    pq_subspaces: Sequence[int],
    pq_bits: Sequence[int],
) -> list[Mapping[str, Any]]:
    """Two-stage compositions to add on top of the default search space."""
    out: list[Mapping[str, Any]] = []
    for keep in truncate_dims:
        if keep > dim:
            continue
        for m in pq_subspaces:
            if keep % m == 0:
                for bits in pq_bits:
                    out.append(
                        {
                            "name": "composed",
                            "stages": [
                                {"name": "truncate", "keep_dim": keep},
                                {
                                    "name": "product_quantize",
                                    "n_subspaces": m,
                                    "n_bits": bits,
                                },
                            ],
                        }
                    )
        out.append(
            {
                "name": "composed",
                "stages": [
                    {"name": "truncate", "keep_dim": keep},
                    {"name": "binary"},
                ],
            }
        )
    return out


def _default_search_space(
    dim: int,
    *,
    truncate_dims: Sequence[int],
    pq_subspaces: Sequence[int],
    pq_bits: Sequence[int],
    composition_truncate_dims: Sequence[int],
    composition_pq_subspaces: Sequence[int],
    composition_pq_bits: Sequence[int],
) -> list[Mapping[str, Any]]:
    truncate_dims = [d for d in truncate_dims if d <= dim]
    space: list[Mapping[str, Any]] = [{"name": "identity"}]
    for kd in truncate_dims:
        space.append({"name": "truncate", "keep_dim": kd})
    space.append({"name": "scalar_int8"})
    space.append({"name": "binary"})
    for m in pq_subspaces:
        if dim % m == 0:
            for bits in pq_bits:
                space.append({"name": "product_quantize", "n_subspaces": m, "n_bits": bits})
    return space + _composition_specs(
        dim,
        truncate_dims=composition_truncate_dims,
        pq_subspaces=composition_pq_subspaces,
        pq_bits=composition_pq_bits,
    )


def _resolve_dataset(
    spec: str,
    *,
    max_corpus: int | None,
    max_queries: int | None,
) -> RetrievalDataset:
    if spec == "synthetic":
        return make_synthetic_retrieval(n_queries_per_topic=4, seed=0)
    if spec.startswith("beir-local:"):
        root = spec.split(":", 1)[1]
        return load_beir_dataset_local(root, max_corpus=max_corpus, max_queries=max_queries)
    if spec.startswith("beir-hf:"):
        name = spec.split(":", 1)[1]
        from embedopt.evaluation.beir import load_beir_dataset_hf

        return load_beir_dataset_hf(name, max_corpus=max_corpus, max_queries=max_queries)
    raise ValueError(
        f"Unknown dataset spec {spec!r}. Use 'synthetic', 'beir-local:/path', or 'beir-hf:name'."
    )


def _dataset_name_from_spec(spec: str) -> str:
    if spec == "synthetic":
        return "retrieval_synthetic_q4_s0"
    if spec.startswith("beir-local:"):
        return Path(spec.split(":", 1)[1]).name
    if spec.startswith("beir-hf:"):
        return f"beir/{spec.split(':', 1)[1]}"
    raise ValueError(
        f"Unknown dataset spec {spec!r}. Use 'synthetic', 'beir-local:/path', or 'beir-hf:name'."
    )


def _artifact_base(backbone_name: str, dataset_spec: str) -> str:
    dataset_name = _dataset_name_from_spec(dataset_spec)
    return f"{backbone_name}__{dataset_name.replace('/', '_')}"


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _intermediate_root(output_dir: Path) -> Path:
    return output_dir / "_intermediate"


def _candidate_checkpoint_dir(output_dir: Path, base: str) -> Path:
    return _intermediate_root(output_dir) / base


def _checkpoint_path(checkpoint_dir: Path, label: str) -> Path:
    # Keep checkpoint files recognizable on disk:
    #   identity.json
    #   truncate(keep_dim=32).json
    #   composed(truncate(keep_dim=64)+binary).json
    safe = "".join("_" if c in '<>:"/\\|?*' else c for c in label).strip()
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:12]
    if not safe:
        safe = digest
    if len(safe) > 180:
        safe = f"{safe[:160]}__{digest}"
    return checkpoint_dir / f"{safe}.json"


def _checkpoint_run_key(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_candidate_checkpoint(
    checkpoint_dir: Path,
    *,
    label: str,
    run_key: str,
) -> dict[str, Any] | None:
    path = _checkpoint_path(checkpoint_dir, label)
    if not _is_nonempty_file(path):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("format_version") != _CHECKPOINT_FORMAT_VERSION:
        return None
    if payload.get("run_key") != run_key or payload.get("spec") != label:
        return None
    row = payload.get("row")
    objective_row = payload.get("objective_row")
    index_rows = payload.get("index_rows", [])
    if not isinstance(row, dict) or not isinstance(objective_row, dict):
        return None
    if not isinstance(index_rows, list):
        return None
    return {
        "row": row,
        "objective_row": objective_row,
        "index_rows": [r for r in index_rows if isinstance(r, dict)],
    }


def _write_candidate_checkpoint(
    checkpoint_dir: Path,
    *,
    label: str,
    run_key: str,
    row: Mapping[str, Any],
    objective_row: Mapping[str, Any],
    index_rows: Sequence[Mapping[str, Any]],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(checkpoint_dir, label)
    tmp_path = path.with_suffix(".tmp")
    payload = {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "run_key": run_key,
        "spec": label,
        "row": dict(row),
        "objective_row": dict(objective_row),
        "index_rows": [dict(r) for r in index_rows],
    }
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _remove_readonly(func: Any, path: str, _exc_info: Any) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        func(path)
    except OSError:
        pass


def _cleanup_all_checkpoints(output_dir: Path) -> None:
    for transient_root in (_intermediate_root(output_dir), _embedding_cache_root(output_dir)):
        root = transient_root.resolve()
        if not root.exists():
            continue
        try:
            shutil.rmtree(root, onerror=_remove_readonly)
        except OSError as exc:
            print(f"Warning: could not remove transient cache at {root}: {exc}")


def _existing_result_summary(
    *,
    backbone_name: str,
    dataset_spec: str,
    output_dir: Path,
    index_backends: Sequence[str],
    expected_seed: int,
    expected_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    base = _artifact_base(backbone_name, dataset_spec)
    csv_path = output_dir / f"{base}__candidates.csv"
    manifest_path = output_dir / f"{base}__manifest.json"
    index_csv_path = output_dir / f"{base}__index.csv"
    if not (_is_nonempty_file(csv_path) and _is_nonempty_file(manifest_path)):
        return None
    if index_backends and not _is_nonempty_file(index_csv_path):
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("seed") != expected_seed:
        return None
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        return None
    for key, expected_value in expected_config.items():
        if key == "index_backends":
            existing_raw = config.get("index_backends", [])
            if not isinstance(existing_raw, list):
                return None
            existing_backends = set(existing_raw)
            if not set(expected_value).issubset(existing_backends):
                return None
        elif config.get(key) != expected_value:
            return None

    extra = manifest.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
    pareto_specs = extra.get("pareto_specs", [])
    existing_index_path: str | None = None
    if _is_nonempty_file(index_csv_path):
        existing_index_path = str(index_csv_path)
    return {
        "backbone": backbone_name,
        "dataset": _dataset_name_from_spec(dataset_spec),
        "manifest_path": str(manifest_path),
        "csv_path": str(csv_path),
        "index_csv_path": existing_index_path,
        "hypervolume": extra.get("hypervolume"),
        "n_pareto": len(pareto_specs) if isinstance(pareto_specs, list) else None,
        "skipped_existing": True,
    }


def run_one(
    *,
    backbone_name: str,
    dataset_spec: str,
    output_dir: Path,
    seed: int,
    batch_size: int,
    profile_repeats: int,
    bootstrap_resamples: int,
    significance_resamples: int,
    truncate_dims: Sequence[int],
    pq_subspaces: Sequence[int],
    pq_bits: Sequence[int],
    composition_truncate_dims: Sequence[int],
    composition_pq_subspaces: Sequence[int],
    composition_pq_bits: Sequence[int],
    max_corpus: int | None,
    max_queries: int | None,
    index_backends: Sequence[str],
    score_batch_size: int,
    score_device: str,
) -> dict[str, Any]:
    """Run the full sweep for one (backbone, dataset) pair and write artifacts.

    Returns the manifest dict so callers can build a top-level summary.
    """
    set_global_seed(seed)
    embedder = make_backbone(backbone_name)
    dataset = _resolve_dataset(dataset_spec, max_corpus=max_corpus, max_queries=max_queries)
    base = f"{backbone_name}__{dataset.name.replace('/', '_')}"
    print(
        f"[{backbone_name} | {dataset.name}] "
        f"corpus={len(dataset.corpus)} queries={len(dataset.queries)}"
    )
    device = getattr(embedder, "device", "cpu-or-unknown")
    print(f"  encoder device={device}; encode batch_size={batch_size}")
    print(f"  top-k score device={score_device}; score_batch_size={score_batch_size}")

    embedding_cache_config = {
        "backbone": backbone_name,
        "dataset_spec": dataset_spec,
        "dataset_name": dataset.name,
        "n_corpus": int(len(dataset.corpus)),
        "n_queries": int(len(dataset.queries)),
        "max_corpus": max_corpus,
        "max_queries": max_queries,
    }
    embedding_cache_key = _embedding_cache_key(embedding_cache_config)
    cached_embeddings = _load_embedding_cache(
        output_dir,
        base=base,
        cache_key=embedding_cache_key,
    )
    if cached_embeddings is None:
        t0 = time.time()
        corpus_vec = _encode_split(
            embedder, list(dataset.corpus), kind="passage", batch_size=batch_size
        )
        query_vec = _encode_split(
            embedder, list(dataset.queries), kind="query", batch_size=batch_size
        )
        encode_secs = time.time() - t0
        _write_embedding_cache(
            output_dir,
            base=base,
            cache_key=embedding_cache_key,
            corpus_vec=corpus_vec,
            query_vec=query_vec,
            encode_secs=encode_secs,
        )
        encode_source = "encoded"
    else:
        corpus_vec, query_vec, encode_secs = cached_embeddings
        encode_source = "loaded cached embeddings"
    dim = corpus_vec.shape[1]
    print(f"  {encode_source} in {encode_secs:.1f}s; dim={dim}")

    specs = _default_search_space(
        dim,
        truncate_dims=truncate_dims,
        pq_subspaces=pq_subspaces,
        pq_bits=pq_bits,
        composition_truncate_dims=composition_truncate_dims,
        composition_pq_subspaces=composition_pq_subspaces,
        composition_pq_bits=composition_pq_bits,
    )
    checkpoint_config = {
        "backbone": backbone_name,
        "dataset_spec": dataset_spec,
        "dim": int(dim),
        "n_corpus": int(len(dataset.corpus)),
        "n_queries": int(len(dataset.queries)),
        "specs": [spec_label(s) for s in specs],
        "profile_repeats": profile_repeats,
        "bootstrap_resamples": bootstrap_resamples,
        "significance_resamples": significance_resamples,
        "truncate_dims": list(truncate_dims),
        "pq_subspaces": list(pq_subspaces),
        "pq_bits": list(pq_bits),
        "composition_truncate_dims": list(composition_truncate_dims),
        "composition_pq_subspaces": list(composition_pq_subspaces),
        "composition_pq_bits": list(composition_pq_bits),
        "max_corpus": max_corpus,
        "max_queries": max_queries,
        "index_backends": list(index_backends),
    }
    checkpoint_dir = _candidate_checkpoint_dir(output_dir, base)
    run_key = _checkpoint_run_key({"seed": seed, **checkpoint_config})

    # First evaluate identity to anchor the bootstrap deltas.
    identity = build_compressor({"name": "identity"})
    identity_compressed = identity.transform(corpus_vec)
    identity_topk = _topk_for_compressor(
        identity,
        identity_compressed,
        query_vec,
        k=10,
        score_batch_size=score_batch_size,
        score_device=score_device,
    )
    identity_per_q = _per_query_ndcg_from_topk(identity_topk, dataset.qrels)

    rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    objective_rows: list[dict[str, float]] = []
    for spec in specs:
        label = spec_label(spec)
        checkpoint = _load_candidate_checkpoint(
            checkpoint_dir,
            label=label,
            run_key=run_key,
        )
        if checkpoint is not None:
            rows.append(cast(dict[str, Any], checkpoint["row"]))
            objective_rows.append(cast(dict[str, float], checkpoint["objective_row"]))
            index_rows.extend(cast(list[dict[str, Any]], checkpoint["index_rows"]))
            print(f"  {label:<60s} restored from intermediate checkpoint")
            continue

        compressor = build_compressor(spec)
        stats, compressed = profile_compressor(
            compressor,
            corpus_vec,
            query_vec,
            n_repeats=profile_repeats,
            n_warmup=2,
        )
        topk = _topk_for_compressor(
            compressor,
            compressed,
            query_vec,
            k=10,
            score_batch_size=score_batch_size,
            score_device=score_device,
        )
        per_q = _per_query_ndcg_from_topk(topk, dataset.qrels)
        ndcg10 = float(np.mean(list(per_q.values()))) if per_q else 0.0
        ci: BootstrapCI | None
        try:
            ci = paired_bootstrap_ci(
                per_q,
                identity_per_q,
                n_resamples=bootstrap_resamples,
                seed=seed,
            )
        except ValueError:
            ci = None
        try:
            sig = paired_randomization_test(
                per_q,
                identity_per_q,
                n_resamples=significance_resamples,
                seed=seed,
            )
            p_value: float | None = sig.p_value
            significant_05: bool | None = sig.significant_05
        except ValueError:
            p_value = None
            significant_05 = None
        row = {
            "backbone": backbone_name,
            "dataset": dataset.name,
            "spec": label,
            "ndcg_at_10": ndcg10,
            "delta_vs_identity": ci.mean_delta if ci else None,
            "ci_lower": ci.lower if ci else None,
            "ci_upper": ci.upper if ci else None,
            "p_value": p_value,
            "significant_05": significant_05,
            "bytes_per_vector": stats.bytes_per_vector,
            "corpus_bytes": stats.corpus_bytes,
            "fit_ms": stats.fit_ms,
            "encode_ms": stats.encode_ms,
            "query_latency_ms": stats.query_latency_ms,
            "query_p95_ms": stats.query_p95_ms,
        }
        objective_row = {
            "ndcg_at_10": ndcg10,
            "bytes_per_vector": float(stats.bytes_per_vector),
            "query_latency_ms": stats.query_latency_ms,
        }
        rows.append(row)
        objective_rows.append(objective_row)
        ci_str = (
            f"  delta={ci.mean_delta:+.4f} [{ci.lower:+.4f}, {ci.upper:+.4f}]"
            if ci
            else ""
        )
        print(
            f"  {row['spec']:<60s} nDCG@10={ndcg10:.4f}  "
            f"B/v={stats.bytes_per_vector:>5d}  "
            f"lat={stats.query_latency_ms:6.2f}ms{ci_str}"
        )
        dense_views = (
            dense_views_for_compressor(compressor, query_vec, compressed)
            if index_backends
            else None
        )
        candidate_index_rows: list[dict[str, Any]] = []
        for backend in index_backends:
            index_row: dict[str, Any] = {
                "backbone": backbone_name,
                "dataset": dataset.name,
                "spec": row["spec"],
                "index_backend": backend,
                "index_status": "ok",
                "index_build_ms": None,
                "index_bytes": None,
                "index_query_latency_ms": None,
                "index_query_p95_ms": None,
                "index_recall_at_10": None,
                "index_ndcg_at_10": None,
                "index_exact_recall_at_10": None,
                "index_error": None,
            }
            if dense_views is None:
                index_row["index_status"] = "not_dense"
            else:
                index_queries, index_corpus = dense_views
                try:
                    index_eval = evaluate_dense_index(
                        corpus_vectors=index_corpus,
                        query_vectors=index_queries,
                        qrels=dataset.qrels,
                        backend=backend,
                        k=10,
                        n_repeats=max(1, min(profile_repeats, 10)),
                        n_warmup=1,
                        reference_topk=identity_topk,
                        query_batch_size=score_batch_size,
                    )
                    index_row.update(
                        {
                            "index_build_ms": index_eval.build_ms,
                            "index_bytes": index_eval.index_bytes,
                            "index_query_latency_ms": index_eval.search_latency_ms,
                            "index_query_p95_ms": index_eval.search_p95_ms,
                            "index_recall_at_10": index_eval.recall_at_10,
                            "index_ndcg_at_10": index_eval.ndcg_at_10,
                            "index_exact_recall_at_10": index_eval.exact_recall_at_10,
                        }
                    )
                except ModuleNotFoundError:
                    index_row["index_status"] = "missing_backend"
                except ValueError as exc:
                    # Backends like FAISS IVF-PQ refuse to train on
                    # corpora smaller than 2**n_bits per subspace. We
                    # mark the row instead of failing the whole pair so
                    # smoke runs and tiny datasets still produce a CSV.
                    index_row["index_status"] = "training_too_small"
                    index_row["index_error"] = str(exc)
            candidate_index_rows.append(index_row)
        index_rows.extend(candidate_index_rows)
        _write_candidate_checkpoint(
            checkpoint_dir,
            label=label,
            run_key=run_key,
            row=row,
            objective_row=objective_row,
            index_rows=candidate_index_rows,
        )

    objectives = [
        Objective(name="ndcg_at_10", sense="max"),
        Objective(name="bytes_per_vector", sense="min"),
        Objective(name="query_latency_ms", sense="min"),
    ]
    pts = to_min_matrix(objectives, objective_rows)
    pareto = pareto_indices(pts)
    if pts.size:
        ref = pts.max(axis=0) + np.abs(pts.max(axis=0)) * 0.05 + 1e-6
        hv = hypervolume(pts[pareto], ref)
    else:
        hv = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{base}__candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    index_csv_path: Path | None = None
    if index_rows:
        index_csv_path = output_dir / f"{base}__index.csv"
        with index_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)

    manifest = RunManifest(
        name=base,
        config={
            **checkpoint_config,
            "score_device": score_device,
            "encode_secs": encode_secs,
        },
        seed=seed,
        extra={
            "pareto_specs": [rows[i]["spec"] for i in pareto],
            "hypervolume": hv,
            "csv_path": str(csv_path),
            "index_csv_path": str(index_csv_path) if index_csv_path else None,
        },
    )
    manifest_path = manifest.write(output_dir / f"{base}__manifest.json")
    print(f"  HV={hv:.6f}  pareto={len(pareto)}  -> {csv_path.name}")
    return {
        "backbone": backbone_name,
        "dataset": dataset.name,
        "manifest_path": str(manifest_path),
        "csv_path": str(csv_path),
        "index_csv_path": str(index_csv_path) if index_csv_path else None,
        "hypervolume": hv,
        "n_pareto": len(pareto),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["hashing"],
        help="Backbone names (see embedopt.models.factory.list_backbones()).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["synthetic"],
        help=(
            "Dataset specs. 'synthetic' uses the deterministic synthetic suite. "
            "'beir-local:/path/to/<name>' loads a local BEIR-formatted directory."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=32,
        help=(
            "Number of queries scored at once when computing top-k metrics. "
            "Lower this to reduce RAM on large BEIR corpora."
        ),
    )
    parser.add_argument(
        "--score-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "Device for full-corpus top-k quality scoring. 'auto' uses CUDA "
            "when torch can see a GPU, otherwise CPU."
        ),
    )
    parser.add_argument("--profile-repeats", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--significance-resamples", type=int, default=5000)
    parser.add_argument("--truncate-dims", default="32,64,128,256,512")
    parser.add_argument("--pq-subspaces", default="4,8,16,32,64")
    parser.add_argument("--pq-bits", default="4,6,8")
    parser.add_argument("--composition-truncate-dims", default="64,128,256")
    parser.add_argument("--composition-pq-subspaces", default="4,8,16,32")
    parser.add_argument("--composition-pq-bits", default="4,6,8")
    parser.add_argument("--max-corpus", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--index-backends",
        nargs="*",
        default=[],
        help=(
            "Optional dense index backends to evaluate. Available: "
            "exact-numpy, faiss-flat, faiss-ivf, faiss-ivfpq, "
            "faiss-hnsw, faiss-opq."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shortcut: backbone=hashing, dataset=synthetic, tiny budgets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run experiments even when candidate CSVs and manifests already exist.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.backbones = ["hashing"]
        args.datasets = ["synthetic"]
        args.profile_repeats = 3
        args.bootstrap_resamples = 200
        args.significance_resamples = 200
        args.truncate_dims = "32,64,128,256"
        args.pq_subspaces = "8,16"
        args.pq_bits = "4,8"
        args.composition_truncate_dims = "64,128"
        args.composition_pq_subspaces = "8,16"
        args.composition_pq_bits = "4,8"

    truncate_dims = _parse_int_list(args.truncate_dims)
    pq_subspaces = _parse_int_list(args.pq_subspaces)
    pq_bits = _parse_int_list(args.pq_bits)
    composition_truncate_dims = _parse_int_list(args.composition_truncate_dims)
    composition_pq_subspaces = _parse_int_list(args.composition_pq_subspaces)
    composition_pq_bits = _parse_int_list(args.composition_pq_bits)
    score_device = _score_device_available(args.score_device)

    summary: list[dict[str, Any]] = []
    failures: list[tuple[str, str, str]] = []
    for backbone in args.backbones:
        for ds in args.datasets:
            try:
                if not args.force:
                    existing = _existing_result_summary(
                        backbone_name=backbone,
                        dataset_spec=ds,
                        output_dir=args.output_dir,
                        index_backends=args.index_backends,
                        expected_seed=args.seed,
                        expected_config={
                            "backbone": backbone,
                            "dataset_spec": ds,
                            "profile_repeats": args.profile_repeats,
                            "bootstrap_resamples": args.bootstrap_resamples,
                            "significance_resamples": args.significance_resamples,
                            "truncate_dims": list(truncate_dims),
                            "pq_subspaces": list(pq_subspaces),
                            "pq_bits": list(pq_bits),
                            "composition_truncate_dims": list(composition_truncate_dims),
                            "composition_pq_subspaces": list(composition_pq_subspaces),
                            "composition_pq_bits": list(composition_pq_bits),
                            "max_corpus": args.max_corpus,
                            "max_queries": args.max_queries,
                            "index_backends": list(args.index_backends),
                        },
                    )
                    if existing is not None:
                        print(
                            f"[{backbone} | {existing['dataset']}] "
                            f"existing artifacts found; skipping "
                            f"{Path(existing['csv_path']).name}"
                        )
                        summary.append(existing)
                        continue
                summary.append(
                    run_one(
                        backbone_name=backbone,
                        dataset_spec=ds,
                        output_dir=args.output_dir,
                        seed=args.seed,
                        batch_size=args.batch_size,
                        profile_repeats=args.profile_repeats,
                        bootstrap_resamples=args.bootstrap_resamples,
                        significance_resamples=args.significance_resamples,
                        truncate_dims=truncate_dims,
                        pq_subspaces=pq_subspaces,
                        pq_bits=pq_bits,
                        composition_truncate_dims=composition_truncate_dims,
                        composition_pq_subspaces=composition_pq_subspaces,
                        composition_pq_bits=composition_pq_bits,
                        max_corpus=args.max_corpus,
                        max_queries=args.max_queries,
                        index_backends=args.index_backends,
                        score_batch_size=args.score_batch_size,
                        score_device=score_device,
                    )
                )
            except Exception as exc:  # pragma: no cover - paper-script logging
                print(f"!! {backbone} on {ds} failed: {exc}")
                failures.append((backbone, ds, str(exc)))
    summary_path = args.output_dir / "summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        print(
            f"Kept intermediate checkpoints in {args.output_dir / '_intermediate'} "
            f"because {len(failures)} run(s) failed."
        )
    else:
        _cleanup_all_checkpoints(args.output_dir)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
