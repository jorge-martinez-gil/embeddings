"""Generate publication-quality figures from embedopt experiment outputs.

The script reads the result directory produced by ``scripts/run_paper_experiments.py``
and writes paper-ready PDF plus high-DPI PNG figures. It intentionally depends
only on matplotlib and the Python standard library so it can run in Colab,
CI artifacts, or a local workstation without a plotting notebook.

Examples
--------
python scripts/plot_paper_figures.py --results-dir results --output-dir figures
python scripts/plot_paper_figures.py --results-dir outputs/smoke-index --output-dir figures-smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PALETTE = {
    "identity": "#2F3A45",
    "truncate": "#3B82F6",
    "scalar_int8": "#10B981",
    "binary": "#F59E0B",
    "product_quantize": "#8B5CF6",
    "composed": "#EF4444",
    "other": "#64748B",
}


@dataclass(slots=True)
class CandidateRow:
    backbone: str
    dataset: str
    spec: str
    ndcg_at_10: float
    delta_vs_identity: float | None
    ci_lower: float | None
    ci_upper: float | None
    p_value: float | None
    significant_05: bool | None
    bytes_per_vector: float
    fit_ms: float
    encode_ms: float
    query_latency_ms: float
    query_p95_ms: float


@dataclass(slots=True)
class StorageModeFigureRow:
    """One row of ``storage_modes_comparison.csv`` for the grouped bar chart."""

    mode: str
    bytes_per_vector: float
    compression_ratio: float
    ndcg_at_10: float
    recall_at_10: float
    mrr_at_10: float
    delta_ndcg_vs_fp32: float
    query_latency_ms: float


@dataclass(slots=True)
class IndexRow:
    backbone: str
    dataset: str
    spec: str
    index_backend: str
    index_status: str
    index_build_ms: float | None
    index_bytes: float | None
    index_query_latency_ms: float | None
    index_query_p95_ms: float | None
    index_recall_at_10: float | None
    index_ndcg_at_10: float | None
    index_exact_recall_at_10: float | None


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _bool_or_none(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return value.lower() in {"1", "true", "yes"}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_candidates(results_dir: Path) -> list[CandidateRow]:
    rows: list[CandidateRow] = []
    for path in sorted(results_dir.glob("*__candidates.csv")):
        for raw in _load_csv(path):
            rows.append(
                CandidateRow(
                    backbone=raw["backbone"],
                    dataset=raw["dataset"],
                    spec=raw["spec"],
                    ndcg_at_10=float(raw["ndcg_at_10"]),
                    delta_vs_identity=_float_or_none(raw.get("delta_vs_identity")),
                    ci_lower=_float_or_none(raw.get("ci_lower")),
                    ci_upper=_float_or_none(raw.get("ci_upper")),
                    p_value=_float_or_none(raw.get("p_value")),
                    significant_05=_bool_or_none(raw.get("significant_05")),
                    bytes_per_vector=float(raw["bytes_per_vector"]),
                    fit_ms=float(raw.get("fit_ms") or 0.0),
                    encode_ms=float(raw.get("encode_ms") or 0.0),
                    query_latency_ms=float(raw["query_latency_ms"]),
                    query_p95_ms=float(raw["query_p95_ms"]),
                )
            )
    return rows


def load_index_rows(results_dir: Path) -> list[IndexRow]:
    rows: list[IndexRow] = []
    for path in sorted(results_dir.glob("*__index.csv")):
        for raw in _load_csv(path):
            rows.append(
                IndexRow(
                    backbone=raw["backbone"],
                    dataset=raw["dataset"],
                    spec=raw["spec"],
                    index_backend=raw["index_backend"],
                    index_status=raw["index_status"],
                    index_build_ms=_float_or_none(raw.get("index_build_ms")),
                    index_bytes=_float_or_none(raw.get("index_bytes")),
                    index_query_latency_ms=_float_or_none(raw.get("index_query_latency_ms")),
                    index_query_p95_ms=_float_or_none(raw.get("index_query_p95_ms")),
                    index_recall_at_10=_float_or_none(raw.get("index_recall_at_10")),
                    index_ndcg_at_10=_float_or_none(raw.get("index_ndcg_at_10")),
                    index_exact_recall_at_10=_float_or_none(raw.get("index_exact_recall_at_10")),
                )
            )
    return rows


def load_summary(results_dir: Path) -> list[dict[str, Any]]:
    path = results_dir / "summary.json"
    if not path.exists():
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        return []
    return [row for row in loaded if isinstance(row, dict)]


def load_storage_modes(results_dir: Path) -> list[StorageModeFigureRow]:
    """Read ``storage_modes_comparison.csv`` if present.

    Returns an empty list (so the figure is skipped) when the file is
    missing — callers of the figure script don't have to run
    ``scripts/compare_storage_modes.py`` first.
    """
    path = results_dir / "storage_modes_comparison.csv"
    if not path.exists():
        return []
    rows: list[StorageModeFigureRow] = []
    for raw in _load_csv(path):
        rows.append(
            StorageModeFigureRow(
                mode=raw["mode"],
                bytes_per_vector=float(raw["bytes_per_vector"]),
                compression_ratio=float(raw["compression_ratio"]),
                ndcg_at_10=float(raw["ndcg_at_10"]),
                recall_at_10=float(raw["recall_at_10"]),
                mrr_at_10=float(raw["mrr_at_10"]),
                delta_ndcg_vs_fp32=float(raw["delta_ndcg_vs_fp32"]),
                query_latency_ms=float(raw["query_latency_ms"]),
            )
        )
    return rows


def spec_family(spec: str) -> str:
    if spec == "identity":
        return "identity"
    if spec.startswith("truncate("):
        return "truncate"
    if spec == "scalar_int8":
        return "scalar_int8"
    if spec == "binary":
        return "binary"
    if spec.startswith("product_quantize("):
        return "product_quantize"
    if spec.startswith("composed("):
        return "composed"
    return "other"


def short_spec(spec: str) -> str:
    if spec == "identity":
        return "identity"
    if spec.startswith("truncate("):
        keep = _param_int(spec, "keep_dim")
        return f"trunc-{keep}" if keep is not None else "truncate"
    if spec == "scalar_int8":
        return "int8"
    if spec == "binary":
        return "binary"
    if spec.startswith("product_quantize("):
        m = _param_int(spec, "n_subspaces")
        b = _param_int(spec, "n_bits")
        return f"PQ M={m}, b={b}" if m is not None and b is not None else "PQ"
    if spec.startswith("composed("):
        keep = _param_int(spec, "keep_dim")
        m = _param_int(spec, "n_subspaces")
        b = _param_int(spec, "n_bits")
        if m is not None and b is not None:
            return f"trunc-{keep}+PQ M={m}, b={b}"
        if "binary" in spec:
            return f"trunc-{keep}+binary"
        return "composed"
    return spec[:36]


def _param_int(spec: str, name: str) -> int | None:
    match = re.search(rf"{re.escape(name)}=(\d+)", spec)
    return int(match.group(1)) if match else None


def _group_key(row: CandidateRow | IndexRow) -> tuple[str, str]:
    return (row.backbone, row.dataset)


def _dominates(a: CandidateRow, b: CandidateRow) -> bool:
    no_worse = (
        a.ndcg_at_10 >= b.ndcg_at_10
        and a.bytes_per_vector <= b.bytes_per_vector
        and a.query_latency_ms <= b.query_latency_ms
    )
    strictly = (
        a.ndcg_at_10 > b.ndcg_at_10
        or a.bytes_per_vector < b.bytes_per_vector
        or a.query_latency_ms < b.query_latency_ms
    )
    return no_worse and strictly


def pareto_rows(rows: Sequence[CandidateRow]) -> list[CandidateRow]:
    out: list[CandidateRow] = []
    for row in rows:
        if not any(_dominates(other, row) for other in rows):
            out.append(row)
    return out


def _import_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install with: pip install -e .[paper]"
        ) from exc

    rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 350,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _save(fig: Any, output_dir: Path, stem: str, formats: Iterable[str]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))
    return paths


def plot_pareto_front(
    rows: Sequence[CandidateRow],
    output_dir: Path,
    formats: Sequence[str],
) -> list[str]:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    all_pareto: set[tuple[str, str, str]] = set()
    for key in sorted({_group_key(r) for r in rows}):
        group = [r for r in rows if _group_key(r) == key]
        for r in pareto_rows(group):
            all_pareto.add((r.backbone, r.dataset, r.spec))

    for family in ("identity", "truncate", "scalar_int8", "binary", "product_quantize", "composed"):
        fam_rows = [r for r in rows if spec_family(r.spec) == family]
        if not fam_rows:
            continue
        ax.scatter(
            [r.bytes_per_vector for r in fam_rows],
            [r.ndcg_at_10 for r in fam_rows],
            s=[
                72 if (r.backbone, r.dataset, r.spec) in all_pareto else 34
                for r in fam_rows
            ],
            c=PALETTE[family],
            alpha=0.82,
            edgecolors=[
                "#111827" if (r.backbone, r.dataset, r.spec) in all_pareto else "white"
                for r in fam_rows
            ],
            linewidths=[
                1.25 if (r.backbone, r.dataset, r.spec) in all_pareto else 0.45
                for r in fam_rows
            ],
            label=family.replace("_", " "),
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Bytes per vector (log2)")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Quality-storage Pareto frontier")
    ax.legend(frameon=False, ncol=2)
    return _save(fig, output_dir, "fig_pareto_quality_storage", formats)


def _complete_matrix_rows(rows: Sequence[CandidateRow]) -> list[CandidateRow]:
    datasets: dict[str, set[str]] = {}
    for row in rows:
        datasets.setdefault(row.dataset, set()).add(row.backbone)
    if not datasets:
        return []
    max_backbones = max(len(backbones) for backbones in datasets.values())
    if max_backbones <= 1:
        return list(rows)
    keep_datasets = {
        dataset for dataset, backbones in datasets.items() if len(backbones) == max_backbones
    }
    return [row for row in rows if row.dataset in keep_datasets]


def plot_pair_pareto_frontiers(
    rows: Sequence[CandidateRow],
    output_dir: Path,
    formats: Sequence[str],
) -> list[str]:
    matrix_rows = _complete_matrix_rows(rows)
    if not matrix_rows:
        return []
    backbones = sorted({row.backbone for row in matrix_rows})
    datasets = sorted({row.dataset for row in matrix_rows})
    plt = _import_matplotlib()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(
        len(backbones),
        len(datasets),
        figsize=(3.15 * len(datasets), 2.35 * len(backbones)),
        squeeze=False,
        sharex=True,
    )
    families = (
        "identity",
        "truncate",
        "scalar_int8",
        "binary",
        "product_quantize",
        "composed",
    )
    for i, backbone in enumerate(backbones):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]
            group = [r for r in matrix_rows if r.backbone == backbone and r.dataset == dataset]
            if not group:
                ax.axis("off")
                continue
            front = sorted(pareto_rows(group), key=lambda r: r.bytes_per_vector)
            ax.scatter(
                [r.bytes_per_vector for r in group],
                [r.ndcg_at_10 for r in group],
                c="#CBD5E1",
                s=15,
                alpha=0.65,
                edgecolors="none",
                zorder=1,
            )
            ax.plot(
                [r.bytes_per_vector for r in front],
                [r.ndcg_at_10 for r in front],
                color="#111827",
                linewidth=0.9,
                alpha=0.78,
                zorder=2,
            )
            for family in families:
                fam_front = [r for r in front if spec_family(r.spec) == family]
                if not fam_front:
                    continue
                ax.scatter(
                    [r.bytes_per_vector for r in fam_front],
                    [r.ndcg_at_10 for r in fam_front],
                    c=PALETTE[family],
                    s=32,
                    alpha=0.95,
                    edgecolors="#111827",
                    linewidths=0.45,
                    zorder=3,
                )
            ax.set_xscale("log", base=2)
            ax.set_title(f"{backbone} / {dataset}", pad=4)
            ymax = max(r.ndcg_at_10 for r in group)
            ax.set_ylim(bottom=0.0, top=min(1.0, ymax + 0.08))
            if i == len(backbones) - 1:
                ax.set_xlabel("Bytes/vec")
            if j == 0:
                ax.set_ylabel("nDCG@10")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PALETTE[family],
            markeredgecolor="#111827",
            markersize=5,
            label=family.replace("_", " "),
        )
        for family in families
    ]
    handles.append(
        Line2D([0], [0], color="#111827", linewidth=1.0, label="Pareto projection")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Per-pair quality-storage Pareto frontiers", y=1.01)
    return _save(fig, output_dir, "fig_pair_pareto_frontiers", formats)


def plot_latency_storage(
    rows: Sequence[CandidateRow],
    output_dir: Path,
    formats: Sequence[str],
) -> list[str]:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    latencies = [max(r.query_latency_ms, 1e-6) for r in rows]
    sizes = [r.bytes_per_vector for r in rows]
    quality = [r.ndcg_at_10 for r in rows]
    scatter = ax.scatter(
        sizes,
        latencies,
        c=quality,
        s=48,
        cmap="viridis",
        edgecolors="white",
        linewidths=0.5,
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Bytes per vector (log2)")
    ax.set_ylabel("Median query latency (ms, log)")
    ax.set_title("Storage-latency-quality trade-off")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("nDCG@10")
    return _save(fig, output_dir, "fig_latency_storage_quality", formats)


def plot_pq_heatmap(
    rows: Sequence[CandidateRow],
    output_dir: Path,
    formats: Sequence[str],
) -> list[str]:
    pq = [
        r
        for r in rows
        if spec_family(r.spec) == "product_quantize"
        and _param_int(r.spec, "n_subspaces") is not None
        and _param_int(r.spec, "n_bits") is not None
    ]
    if not pq:
        return []
    plt = _import_matplotlib()
    buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in pq:
        key = (_param_int(row.spec, "n_subspaces") or 0, _param_int(row.spec, "n_bits") or 0)
        value = row.delta_vs_identity if row.delta_vs_identity is not None else row.ndcg_at_10
        buckets[key].append(value)
    ms = sorted({k[0] for k in buckets})
    bits = sorted({k[1] for k in buckets})
    matrix = [[math.nan for _ in ms] for _ in bits]
    for i, bit in enumerate(bits):
        for j, m in enumerate(ms):
            vals = buckets.get((m, bit), [])
            if vals:
                matrix[i][j] = sum(vals) / len(vals)
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    im = ax.imshow(matrix, cmap="RdYlBu", aspect="auto")
    ax.set_xticks(range(len(ms)), [str(m) for m in ms])
    ax.set_yticks(range(len(bits)), [str(b) for b in bits])
    ax.set_xlabel("PQ subspaces (M)")
    ax.set_ylabel("Bits per subspace")
    ax.set_title("PQ ablation: mean delta vs. identity")
    for i, _bit in enumerate(bits):
        for j, _m in enumerate(ms):
            val = matrix[i][j]
            if not math.isnan(val):
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean delta nDCG@10")
    return _save(fig, output_dir, "fig_pq_ablation_heatmap", formats)


def plot_ci_forest(
    rows: Sequence[CandidateRow],
    output_dir: Path,
    formats: Sequence[str],
    top_n: int,
) -> list[str]:
    with_ci = [
        r
        for r in rows
        if r.spec != "identity"
        and r.delta_vs_identity is not None
        and r.ci_lower is not None
        and r.ci_upper is not None
    ]
    if not with_ci:
        return []
    selected = sorted(
        with_ci,
        key=lambda r: (
            r.p_value if r.p_value is not None else 1.0,
            abs(r.delta_vs_identity or 0.0),
        ),
        reverse=False,
    )[:top_n]
    selected = list(reversed(selected))
    plt = _import_matplotlib()
    height = max(3.2, 0.34 * len(selected) + 1.2)
    fig, ax = plt.subplots(figsize=(7.0, height))
    y = list(range(len(selected)))
    deltas = [r.delta_vs_identity or 0.0 for r in selected]
    xerr = [
        [abs((r.delta_vs_identity or 0.0) - (r.ci_lower or 0.0)) for r in selected],
        [abs((r.ci_upper or 0.0) - (r.delta_vs_identity or 0.0)) for r in selected],
    ]
    colors = [PALETTE[spec_family(r.spec)] for r in selected]
    ax.errorbar(deltas, y, xerr=xerr, fmt="none", ecolor="#6B7280", elinewidth=1.0, capsize=2)
    ax.scatter(deltas, y, c=colors, s=38, edgecolors="white", linewidths=0.5, zorder=3)
    ax.axvline(0.0, color="#111827", linewidth=1.0)
    labels = [f"{r.backbone}/{r.dataset}: {short_spec(r.spec)}" for r in selected]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Delta nDCG@10 vs. identity")
    ax.set_title("Paired-bootstrap confidence intervals")
    return _save(fig, output_dir, "fig_significance_ci_forest", formats)


def plot_training_cost(
    rows: Sequence[CandidateRow],
    output_dir: Path,
    formats: Sequence[str],
    top_n: int,
) -> list[str]:
    learned = [
        r
        for r in rows
        if r.fit_ms > 0.0
        or spec_family(r.spec) in {"product_quantize", "scalar_int8", "composed"}
    ]
    if not learned:
        return []
    selected = sorted(learned, key=lambda r: r.fit_ms + r.encode_ms, reverse=True)[:top_n]
    selected = list(reversed(selected))
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, max(3.2, 0.34 * len(selected) + 1.2)))
    y = list(range(len(selected)))
    fit = [r.fit_ms for r in selected]
    encode = [r.encode_ms for r in selected]
    ax.barh(y, fit, color="#7C3AED", label="fit / codebook training")
    ax.barh(y, encode, left=fit, color="#A78BFA", label="corpus encoding")
    ax.set_yticks(y, [short_spec(r.spec) for r in selected])
    ax.set_xlabel("Milliseconds")
    ax.set_title("Offline compression cost")
    ax.legend(frameon=False)
    return _save(fig, output_dir, "fig_training_cost", formats)


def plot_index_tradeoff(
    index_rows: Sequence[IndexRow],
    output_dir: Path,
    formats: Sequence[str],
) -> list[str]:
    ok = [
        r
        for r in index_rows
        if r.index_status == "ok"
        and r.index_bytes is not None
        and r.index_ndcg_at_10 is not None
        and r.index_query_latency_ms is not None
    ]
    if not ok:
        return []
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    scatter = ax.scatter(
        [r.index_bytes or 0.0 for r in ok],
        [r.index_ndcg_at_10 or 0.0 for r in ok],
        c=[r.index_query_latency_ms or 0.0 for r in ok],
        s=52,
        cmap="magma_r",
        edgecolors="white",
        linewidths=0.5,
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Index bytes (log2)")
    ax.set_ylabel("Index nDCG@10")
    ax.set_title("Index-level quality-storage-latency")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Median index query latency (ms)")
    return _save(fig, output_dir, "fig_index_tradeoff", formats)


_STORAGE_MODE_FAMILY: dict[str, str] = {
    "float32": "identity",
    "float16": "other",
    "int8": "scalar_int8",
    "binary": "binary",
}


def _storage_mode_color(mode: str) -> str:
    """Map a storage-mode label to the global figure palette."""
    if mode in _STORAGE_MODE_FAMILY:
        return PALETTE[_STORAGE_MODE_FAMILY[mode]]
    if mode.startswith("pq("):
        return PALETTE["product_quantize"]
    return PALETTE["other"]


def plot_storage_modes_bar(
    rows: Sequence[StorageModeFigureRow],
    output_dir: Path,
    formats: Sequence[str],
) -> list[str]:
    """Grouped bar chart: nDCG@10 vs. Recall@10 per storage mode.

    Each storage mode (float32 / float16 / int8 / binary / PQ8 / PQ4) is
    one x-position with two bars (nDCG@10 and Recall@10). Bytes / vec
    and compression ratio are annotated above the nDCG bar so reviewers
    see the storage cost without having to cross-reference the table.
    """
    if not rows:
        return []
    plt = _import_matplotlib()
    n = len(rows)
    width = 0.38
    xs = list(range(n))
    fig, ax = plt.subplots(figsize=(max(6.4, 1.05 * n), 4.2))
    ndcg_vals = [r.ndcg_at_10 for r in rows]
    recall_vals = [r.recall_at_10 for r in rows]
    bar_colors = [_storage_mode_color(r.mode) for r in rows]

    bars_ndcg = ax.bar(
        [x - width / 2 for x in xs],
        ndcg_vals,
        width=width,
        color=bar_colors,
        edgecolor="#111827",
        linewidth=0.6,
        label="nDCG@10",
    )
    ax.bar(
        [x + width / 2 for x in xs],
        recall_vals,
        width=width,
        color=bar_colors,
        alpha=0.55,
        edgecolor="#111827",
        linewidth=0.6,
        hatch="//",
        label="Recall@10",
    )

    # Annotate B/vec and compression ratio above each nDCG bar.
    y_max = max(max(ndcg_vals), max(recall_vals)) if ndcg_vals else 1.0
    headroom = y_max * 0.18 if y_max > 0 else 0.05
    for bar, row in zip(bars_ndcg, rows, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + headroom * 0.05,
            f"{int(row.bytes_per_vector)} B\n{row.compression_ratio:.1f}×",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#111827",
        )

    ax.set_xticks(xs, [r.mode for r in rows], rotation=15, ha="right")
    ax.set_ylabel("Retrieval quality")
    ax.set_ylim(0.0, y_max + headroom)
    ax.set_title("Storage-mode comparison: quality vs. bytes per vector")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    return _save(fig, output_dir, "fig_storage_modes_bar", formats)


def plot_hypervolume(
    summary: Sequence[Mapping[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
) -> list[str]:
    if not summary:
        return []
    plt = _import_matplotlib()
    labels = [f"{row['backbone']}\n{row['dataset']}" for row in summary]
    values = [float(row["hypervolume"]) for row in summary]
    order = sorted(range(len(values)), key=lambda i: values[i])
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(max(5.6, 0.52 * len(values)), 3.8))
    ax.bar(range(len(values)), values, color="#2563EB")
    ax.set_xticks(range(len(values)), labels, rotation=45, ha="right")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Deployment frontier hypervolume")
    return _save(fig, output_dir, "fig_hypervolume", formats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=18,
        help="Rows to show in CI/training-cost plots.",
    )
    args = parser.parse_args()

    candidates = load_candidates(args.results_dir)
    index_rows = load_index_rows(args.results_dir)
    summary = load_summary(args.results_dir)
    storage_rows = load_storage_modes(args.results_dir)
    if not candidates:
        raise SystemExit(f"No *__candidates.csv files found in {args.results_dir}")

    written: list[str] = []
    written += plot_pareto_front(candidates, args.output_dir, args.formats)
    written += plot_pair_pareto_frontiers(candidates, args.output_dir, args.formats)
    written += plot_latency_storage(candidates, args.output_dir, args.formats)
    written += plot_pq_heatmap(candidates, args.output_dir, args.formats)
    written += plot_ci_forest(candidates, args.output_dir, args.formats, args.top_n)
    written += plot_training_cost(candidates, args.output_dir, args.formats, args.top_n)
    written += plot_index_tradeoff(index_rows, args.output_dir, args.formats)
    written += plot_storage_modes_bar(storage_rows, args.output_dir, args.formats)
    written += plot_hypervolume(summary, args.output_dir, args.formats)

    manifest = {
        "results_dir": str(args.results_dir),
        "output_dir": str(args.output_dir),
        "n_candidates": len(candidates),
        "n_index_rows": len(index_rows),
        "n_storage_modes": len(storage_rows),
        "figures": written,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
