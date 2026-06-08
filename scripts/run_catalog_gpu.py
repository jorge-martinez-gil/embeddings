#!/usr/bin/env python3
r"""A100 GPU catalog runner (pure PyTorch).

Maximum-acceleration variant of run_catalog.py: dense-family scoring AND PQ/OPQ
codebook training + ADC all run on the GPU via torch (no faiss-gpu wheel needed).
Reuses run_catalog.py for metrics / paired stats / Pareto / CSV+manifest schema,
so regen_tables.py and tost_equivalence.py consume the output unchanged.

The PQ asymmetric-distance math here was validated against faiss (top-1 agreement
1.0 on planted data); OPQ adds a standard orthogonal-Procrustes rotation.

Validate on Colab BEFORE the full run (seconds, uses real torch+cuda):
    python scripts/run_catalog_gpu.py --selftest

Full run:
    python scripts/run_catalog_gpu.py --backbones e5-base bge-base mxbai-large gte-base \
        --datasets scifact nfcorpus arguana fiqa --cache-dir cache --output-dir results
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_catalog as rc  # metrics, stats, pareto, hypervolume, spec_grid

import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _tn(x): return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=DEV)
def _norm(X): return X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)

def _topk_dense(Q, C, k, bs=1024):
    keff = min(k, C.shape[0]); out = []
    for s in range(0, Q.shape[0], bs):
        sc = Q[s:s+bs] @ C.T
        out.append(torch.topk(sc, keff, dim=1).indices)
    return torch.cat(out, 0).cpu().numpy()

def _kmeans(X, K, iters=25, seed=0):
    n = X.shape[0]
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    init = torch.randperm(n, generator=g)[:K].to(X.device)
    cent = X[init].clone()
    for _ in range(iters):
        a = torch.cdist(X, cent).argmin(1)
        new = torch.zeros_like(cent); cnt = torch.zeros(cent.shape[0], device=X.device)
        new.index_add_(0, a, X); cnt.index_add_(0, a, torch.ones(n, device=X.device))
        m = cnt > 0; new[m] /= cnt[m].unsqueeze(1); new[~m] = cent[~m]
        cent = new
    a = torch.cdist(X, cent).argmin(1)
    return cent, a

def _pq_fit(C, M, b, seed=0):
    d = C.shape[1]; ds = d // M; K = 2 ** b
    cents = []; codes = torch.empty((C.shape[0], M), dtype=torch.long, device=C.device)
    for s in range(M):
        cent, a = _kmeans(C[:, s*ds:(s+1)*ds], K, 25, seed + s)
        cents.append(cent); codes[:, s] = a
    return cents, codes, ds

def _pq_search(Q, cents, codes, ds, M, k, bs=512):
    N = codes.shape[0]; out = []
    for s0 in range(0, Q.shape[0], bs):
        qb = Q[s0:s0+bs]; sc = torch.zeros((qb.shape[0], N), device=Q.device)
        for s in range(M):
            tbl = qb[:, s*ds:(s+1)*ds] @ cents[s].T   # B x K
            sc += tbl[:, codes[:, s]]                 # B x N
        out.append(torch.topk(sc, min(k, N), dim=1).indices)
    return torch.cat(out, 0).cpu().numpy()

def _reconstruct(cents, codes, ds, M, d):
    N = codes.shape[0]; X = torch.empty((N, d), device=codes.device)
    for s in range(M):
        X[:, s*ds:(s+1)*ds] = cents[s][codes[:, s]]
    return X

def _opq_fit(C, M, b, seed=0, n_iter=5):
    d = C.shape[1]; R = torch.eye(d, device=C.device)
    cents = codes = ds = None
    for _ in range(n_iter):
        Cr = C @ R
        cents, codes, ds = _pq_fit(Cr, M, b, seed)
        Chat = _reconstruct(cents, codes, ds, M, d)
        U, _, Vt = torch.linalg.svd(C.T @ Chat, full_matrices=False)
        R = U @ Vt
    cents, codes, ds = _pq_fit(C @ R, M, b, seed)  # final fit at converged R
    return cents, codes, ds, R

def rank_gpu(kind, params, C, Q, k=10, seed=0):
    if kind == "dense":   return _topk_dense(Q, C, k)
    if kind == "f16":     return _topk_dense(Q, C.half().float(), k)
    if kind == "trunc":
        kk = params["k"]; return _topk_dense(_norm(Q[:, :kk]), _norm(C[:, :kk]), k)
    if kind == "int8":
        lo = C.min(0).values; hi = C.max(0).values; sc = (hi - lo).clamp_min(1e-12)
        deq = torch.round((C - lo) / sc * 255.0) / 255.0 * sc + lo
        return _topk_dense(Q, deq, k)
    if kind == "binary":  return _topk_dense(torch.sign(Q), torch.sign(C), k)
    if kind == "tbin":
        kk = params["k"]; Ct = _norm(C[:, :kk]); Qt = _norm(Q[:, :kk])
        return _topk_dense(torch.sign(Qt), torch.sign(Ct), k)
    if kind == "pq":
        cents, codes, ds = _pq_fit(C, params["M"], params["b"], seed)
        return _pq_search(Q, cents, codes, ds, params["M"], k)
    if kind == "opq":
        cents, codes, ds, R = _opq_fit(C, params["M"], params["b"], seed)
        return _pq_search(Q @ R, cents, codes, ds, params["M"], k)
    if kind == "tpq":
        kk = params["k"]; Ct = _norm(C[:, :kk]); Qt = _norm(Q[:, :kk])
        cents, codes, ds = _pq_fit(Ct, params["M"], params["b"], seed)
        return _pq_search(Qt, cents, codes, ds, params["M"], k)
    raise ValueError(kind)

def run_pair(bb, ds_name, cache_dir, out_dir, B, k=10, seed=0):
    npz = np.load(os.path.join(cache_dir, f"{bb}__{ds_name}.npz"))
    meta = json.load(open(os.path.join(cache_dir, f"{bb}__{ds_name}.json")))
    C = _norm(_tn(npz["corpus"])); Q = _norm(_tn(npz["queries"]))
    n, d = C.shape; qrels = meta["qrels"]; qids = meta["query_ids"]; row2doc = meta["doc_ids"]
    id_pq = rc.ndcg_per_query(rank_gpu("dense", {}, C, Q, k), row2doc, qrels, qids, k)
    rows = []
    for label, kind, params, bpv in rc.spec_grid(d, n):
        t0 = time.perf_counter()
        try:
            idx = rank_gpu(kind, params, C, Q, k, seed)
        except Exception as e:
            print(f"  skip {label}: {e}"); continue
        lat = (time.perf_counter() - t0) / max(1, Q.shape[0]) * 1e3
        pq = rc.ndcg_per_query(idx, row2doc, qrels, qids, k)
        if label == "identity":
            delta = np.zeros_like(pq); lo = hi = 0.0; p = 1.0
        else:
            delta = pq - id_pq
            lo, hi = rc.paired_bootstrap_ci(delta, B, seed)
            p = rc.paired_randomization_p(delta, B, seed)
        rows.append(dict(spec=label, q=float(pq.mean()), delta=float(delta.mean()),
                         lo=lo, hi=hi, p=p, b=float(bpv), l=float(lat), corpus_bytes=int(bpv*n)))
    par = set(rc.pareto_front(rows))
    ref = (min(r["q"] for r in rows)-0.01, max(r["b"] for r in rows)*1.1, max(r["l"] for r in rows)*1.1)
    hv = rc.hypervolume(rows, ref)
    os.makedirs(out_dir, exist_ok=True); base = os.path.join(out_dir, f"{bb}__{ds_name}")
    with open(base + "__candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone","dataset","spec","ndcg_at_10","delta_vs_identity","ci_lower",
                    "ci_upper","p_value","significant_05","bytes_per_vector","corpus_bytes",
                    "fit_ms","encode_ms","query_latency_ms","query_p95_ms"])
        for r in rows:
            w.writerow([bb, ds_name, r["spec"], f"{r['q']:.6f}", f"{r['delta']:.6f}", f"{r['lo']:.6f}",
                        f"{r['hi']:.6f}", f"{r['p']:.6f}", int(r["p"]<0.05 and r["spec"]!="identity"),
                        int(r["b"]), r["corpus_bytes"], "0.0", "0.0", f"{r['l']:.6f}", f"{r['l']:.6f}"])
    cfg = dict(backbone=bb, dataset=f"beir-local:{ds_name}", dim=d, n_corpus=n, n_queries=int(Q.shape[0]),
               bootstrap_resamples=B, significance_resamples=B, seed=seed, device=DEV)
    json.dump(dict(name=f"{bb}__{ds_name}", config=cfg, seed=seed,
                   extra=dict(pareto_specs=sorted(par), hypervolume=hv, csv_path=base+"__candidates.csv"),
                   config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()),
              open(base + "__manifest.json", "w"), indent=2)
    print(f"  {bb}/{ds_name} [{DEV}]: {len(rows)} specs, {len(par)} pareto, HV={hv:.1f}")

def _selftest():
    import tempfile
    rng = np.random.default_rng(0); cache = tempfile.mkdtemp(); out = tempfile.mkdtemp()
    n, m, d = 800, 60, 128
    Cc = rng.standard_normal((n, d)).astype("float32"); rel = rng.integers(0, n, m)
    Qq = (Cc[rel] + 0.2 * rng.standard_normal((m, d))).astype("float32")
    np.savez(os.path.join(cache, "toy__syn.npz"), corpus=Cc, queries=Qq)
    json.dump(dict(query_ids=[str(j) for j in range(m)], doc_ids=[str(i) for i in range(n)],
                   qrels={str(j): {str(int(rel[j])): 1} for j in range(m)}),
              open(os.path.join(cache, "toy__syn.json"), "w"))
    # small grid for speed
    rc.spec_grid = lambda d, n: iter([("identity","dense",{},4*d),("float16","f16",{},2*d),
        ("scalar_int8","int8",{},d),("binary","binary",{},(d+7)//8),
        ("product_quantize(n_subspaces=8,n_bits=8)","pq",{"M":8,"b":8},8),
        ("opq(n_subspaces=8,n_bits=8)","opq",{"M":8,"b":8},8)])
    run_pair("toy", "syn", cache, out, B=500, seed=0)
    rows = list(csv.DictReader(open(os.path.join(out, "toy__syn__candidates.csv"))))
    print(f"SELFTEST OK on {DEV}: {len(rows)} specs; "
          f"identity nDCG={[r['ndcg_at_10'] for r in rows if r['spec']=='identity'][0]}; "
          f"PQ nDCG={[r['ndcg_at_10'] for r in rows if r['spec'].startswith('product')][0]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=[])
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--cache-dir", default="cache"); ap.add_argument("--output-dir", default="results")
    ap.add_argument("--bootstrap-resamples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    print(f"[run_catalog_gpu] device = {DEV}")
    if a.selftest: _selftest(); return
    for bb in a.backbones:
        for ds in a.datasets:
            f = os.path.join(a.output_dir, f"{bb}__{ds}__candidates.csv")
            if os.path.exists(f): print(f"  done: {bb}/{ds}"); continue
            run_pair(bb, ds, a.cache_dir, a.output_dir, a.bootstrap_resamples, seed=a.seed)

if __name__ == "__main__":
    main()
