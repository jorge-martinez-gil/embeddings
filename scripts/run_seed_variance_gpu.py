#!/usr/bin/env python3
r"""Codebook seed-variance (RQ5) on the GPU (pure torch).

Retrains PQ and OPQ at several seeds on cached embeddings and writes the schema
regen_tables.py expects. Reuses the torch operators in run_catalog_gpu.py.

Usage:
    python scripts/run_seed_variance_gpu.py \
        --backbones e5-base bge-base mxbai-large gte-base \
        --datasets scifact nfcorpus arguana fiqa \
        --seeds 0 1 2 3 4 --cache-dir cache --output-dir results
"""
from __future__ import annotations
import argparse, csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_catalog as rc
import run_catalog_gpu as g

def pqopq_specs(d, n):
    for M in [4, 8, 16, 32, 64]:
        if d % M: continue
        for b in [4, 6, 8]:
            yield (f"product_quantize(n_subspaces={M},n_bits={b})", "pq", {"M": M, "b": b})
    for M in [4, 8, 16, 32, 64]:
        if d % M or n < 256: continue
        for b in [4, 8]:
            yield (f"opq(n_subspaces={M},n_bits={b})", "opq", {"M": M, "b": b})

def run_pair(bb, ds, cache_dir, out_dir, seeds, k=10):
    npz = np.load(os.path.join(cache_dir, f"{bb}__{ds}.npz"))
    meta = json.load(open(os.path.join(cache_dir, f"{bb}__{ds}.json")))
    C = g._norm(g._tn(npz["corpus"])); Q = g._norm(g._tn(npz["queries"]))
    n, d = C.shape; qrels = meta["qrels"]; qids = meta["query_ids"]; row2doc = meta["doc_ids"]
    per_seed = []; summary = []
    for label, kind, params in pqopq_specs(d, n):
        vals = []
        for s in seeds:
            try:
                idx = g.rank_gpu(kind, params, C, Q, k, seed=int(s))
            except Exception as e:
                print(f"  skip {label} seed {s}: {e}"); continue
            nd = float(rc.ndcg_per_query(idx, row2doc, qrels, qids, k).mean())
            vals.append(nd); per_seed.append((label, s, nd))
        if vals:
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary.append((label, len(vals), float(np.mean(vals)), std))
    os.makedirs(out_dir, exist_ok=True); base = os.path.join(out_dir, f"{bb}__{ds}")
    with open(base + "__seed_variance.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["spec_label","seed","ndcg_at_10","recall_at_10","mrr_at_10","map_at_10"])
        for lab, s, nd in per_seed: w.writerow([lab, s, f"{nd:.6f}", 0.0, 0.0, 0.0])
    with open(base + "__seed_variance_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spec_label","n_seeds","ndcg_mean","ndcg_std","recall_mean","recall_std",
                    "mrr_mean","mrr_std","map_mean","map_std"])
        for lab, ns, mean, std in summary:
            w.writerow([lab, ns, f"{mean:.6f}", f"{std:.6f}", 0, 0, 0, 0, 0, 0])
    print(f"  {bb}/{ds} [{g.DEV}]: {len(summary)} specs x {len(seeds)} seeds")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--cache-dir", default="cache"); ap.add_argument("--output-dir", default="results")
    a = ap.parse_args()
    print(f"[seed_variance_gpu] device = {g.DEV}")
    for bb in a.backbones:
        for ds in a.datasets:
            run_pair(bb, ds, a.cache_dir, a.output_dir, a.seeds)

if __name__ == "__main__":
    main()
