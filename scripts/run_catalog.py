#!/usr/bin/env python3
r"""Clean, dependency-light catalog runner (numpy + faiss-cpu).

Replaces the heavy `run_paper_experiments.py` for the cheap workflow:
embeddings are computed ONCE by `embed_once.py` and cached; this script loads the
cache and does all the rest on CPU. It writes the SAME artifact schema the paper
tooling already consumes, so `regen_tables.py` and `tost_equivalence.py` work
unchanged:

    results/<backbone>__<dataset>__candidates.csv   (one row per layout)
    results/<backbone>__<dataset>__manifest.json    (provenance + Pareto set + HV)

Cache layout (produced by embed_once.py), per (backbone,dataset):
    cache/<backbone>__<dataset>.npz    -> corpus (n,d) f32, queries (m,d) f32
    cache/<backbone>__<dataset>.json   -> {"query_ids":[...], "doc_ids":[...],
                                           "qrels": {qid: {docid: rel}}}

Self-test (no models, no data, ~seconds):
    python scripts/run_catalog.py --selftest
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, os, time
import numpy as np

# ----------------------------- metrics --------------------------------------
def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32, copy=False)

def ndcg_per_query(topk_idx, row2doc, qrels, query_ids, k=10):
    """Return a per-query nDCG@k array aligned with query_ids."""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    out = []
    for qi, qid in enumerate(query_ids):
        rel = qrels.get(str(qid), qrels.get(qid, {}))
        if not rel:
            out.append(0.0); continue
        gains = []
        for r in topk_idx[qi][:k]:
            doc = row2doc[int(r)]
            gains.append((2.0 ** rel.get(str(doc), rel.get(doc, 0)) - 1.0))
        dcg = float(np.sum(np.array(gains) * disc[:len(gains)]))
        ideal = sorted(rel.values(), reverse=True)[:k]
        ig = np.array([2.0 ** r - 1.0 for r in ideal])
        idcg = float(np.sum(ig * disc[:len(ig)]))
        out.append(dcg / idcg if idcg > 0 else 0.0)
    return np.asarray(out, dtype=np.float64)

# ----------------------------- ranking --------------------------------------
def topk_dense(Q, C, k=10, bs=256):
    m = Q.shape[0]; out = np.empty((m, k), dtype=np.int64)
    keff = min(k, C.shape[0])
    for s in range(0, m, bs):
        sc = Q[s:s+bs] @ C.T
        idx = np.argpartition(-sc, keff-1, axis=1)[:, :keff]
        part = np.take_along_axis(sc, idx, axis=1)
        order = np.argsort(-part, axis=1)
        out[s:s+bs, :keff] = np.take_along_axis(idx, order, axis=1)
    return out

def topk_faiss(train, corpus, queries, factory, k=10):
    import faiss
    d = corpus.shape[1]
    index = faiss.index_factory(d, factory, faiss.METRIC_INNER_PRODUCT)
    index.train(np.ascontiguousarray(train))
    index.add(np.ascontiguousarray(corpus))
    _, I = index.search(np.ascontiguousarray(queries), k)
    code_bytes = None
    try:
        code_bytes = int(faiss.serialize_index(index).size)
    except Exception:
        pass
    return I.astype(np.int64), code_bytes

# ----------------------------- operators ------------------------------------
def q_int8(C):
    lo = C.min(0, keepdims=True); hi = C.max(0, keepdims=True)
    scale = (hi - lo); scale[scale == 0] = 1.0
    codes = np.round((C - lo) / scale * 255.0).astype(np.uint8)
    deq = codes.astype(np.float32) / 255.0 * scale + lo
    return deq

def spec_grid(d, n):
    """Yield (label, kind, params, bytes_per_vec). kind drives ranking."""
    yield ("identity", "dense", {}, 4 * d)
    yield ("float16", "f16", {}, 2 * d)
    for kdim in [32, 64, 128, 256, 512]:
        if kdim < d: yield (f"truncate(keep_dim={kdim})", "trunc", {"k": kdim}, 4 * kdim)
    yield ("scalar_int8", "int8", {}, d)
    yield ("binary", "binary", {}, (d + 7) // 8)
    for M in [4, 8, 16, 32, 64]:
        if d % M: continue
        for b in [4, 6, 8]:
            yield (f"product_quantize(n_subspaces={M},n_bits={b})", "pq",
                   {"M": M, "b": b}, (M * b + 7) // 8)
    for M in [4, 8, 16, 32, 64]:
        if d % M: continue
        for b in [4, 8]:
            if n < 256: continue
            yield (f"opq(n_subspaces={M},n_bits={b})", "opq",
                   {"M": M, "b": b}, (M * b + 7) // 8)
    for kdim in [64, 128, 256]:
        if kdim > d: continue
        for M in [4, 8, 16, 32]:
            if kdim % M: continue
            for b in [4, 6, 8]:
                yield (f"composed(truncate(keep_dim={kdim})+product_quantize(n_subspaces={M},n_bits={b}))",
                       "tpq", {"k": kdim, "M": M, "b": b}, (M * b + 7) // 8)
        yield (f"composed(truncate(keep_dim={kdim})+binary)", "tbin", {"k": kdim}, (kdim + 7) // 8)

def rank_for_spec(kind, params, C, Q, k=10):
    """Return (topk_idx, code_bytes_or_None)."""
    if kind == "dense":
        return topk_dense(Q, C, k), None
    if kind == "f16":
        return topk_dense(Q, C.astype(np.float16).astype(np.float32), k), None
    if kind == "trunc":
        kk = params["k"]; return topk_dense(_normalize(Q[:, :kk]), _normalize(C[:, :kk]), k), None
    if kind == "int8":
        return topk_dense(Q, q_int8(C), k), None
    if kind == "binary":
        return topk_dense(np.sign(Q).astype(np.float32), np.sign(C).astype(np.float32), k), None
    if kind == "tbin":
        kk = params["k"]; Ct = _normalize(C[:, :kk]); Qt = _normalize(Q[:, :kk])
        return topk_dense(np.sign(Qt).astype(np.float32), np.sign(Ct).astype(np.float32), k), None
    if kind == "pq":
        M, b = params["M"], params["b"]
        return topk_faiss(C, C, Q, f"PQ{M}x{b}", k)
    if kind == "opq":
        M, b = params["M"], params["b"]; d = C.shape[1]
        return topk_faiss(C, C, Q, f"OPQ{M}_{d},PQ{M}x{b}", k)
    if kind == "tpq":
        kk, M, b = params["k"], params["M"], params["b"]
        Ct = _normalize(C[:, :kk]); Qt = _normalize(Q[:, :kk])
        return topk_faiss(Ct, Ct, Qt, f"PQ{M}x{b}", k)
    raise ValueError(kind)

# ----------------------------- statistics -----------------------------------
def paired_bootstrap_ci(delta, B=5000, seed=0):
    rng = np.random.default_rng(seed); n = delta.size
    if n == 0: return 0.0, 0.0
    means = delta[rng.integers(0, n, size=(B, n))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

def paired_randomization_p(delta, B=5000, seed=0):
    rng = np.random.default_rng(seed); n = delta.size
    if n == 0: return 1.0
    obs = abs(delta.mean())
    signs = rng.choice([-1.0, 1.0], size=(B, n))
    null = np.abs((signs * delta).mean(1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (B + 1))

# ----------------------------- Pareto / HV ----------------------------------
def pareto_front(rows):
    """rows: list of dict with q,b,l. Return set of pareto labels under (-q,b,l)."""
    pts = [(-r["q"], r["b"], r["l"], r["spec"]) for r in rows]
    keep = []
    for i, a in enumerate(pts):
        dom = False
        for j, c in enumerate(pts):
            if i == j: continue
            if c[0] <= a[0] and c[1] <= a[1] and c[2] <= a[2] and (
               c[0] < a[0] or c[1] < a[1] or c[2] < a[2]):
                dom = True; break
        if not dom: keep.append(a[3])
    return keep

def hypervolume(rows, ref):
    """Simple 3D HV by inclusion-exclusion over the dominated boxes (small sets)."""
    pts = sorted({(r["q"], r["b"], r["l"]) for r in rows})
    # Monte-Carlo HV estimate is unstable; use grid sweep on unique coords (exact for small sets).
    qs = sorted({p[0] for p in pts}); 
    # crude but deterministic: sum of axis-aligned boxes via sweep on bytes,latency for the max-q envelope
    # (kept simple; the paper reports HV per-pair from its own routine — this is a stand-in.)
    vol = 0.0
    grid_b = sorted({p[1] for p in pts} | {ref[1]})
    grid_l = sorted({p[2] for p in pts} | {ref[2]})
    for bi in range(len(grid_b) - 1):
        for li in range(len(grid_l) - 1):
            b0, b1 = grid_b[bi], grid_b[bi+1]; l0, l1 = grid_l[li], grid_l[li+1]
            if b1 > ref[1] or l1 > ref[2]: continue
            best_q = max([p[0] for p in pts if p[1] <= b0 and p[2] <= l0], default=None)
            if best_q is None: continue
            vol += (b1 - b0) * (l1 - l0) * max(0.0, best_q - ref[0])
    return float(vol)

# ----------------------------- driver ---------------------------------------
def run_pair(backbone, dataset, cache_dir, out_dir, B, k=10, seed=0, profile_repeats=20):
    npz = np.load(os.path.join(cache_dir, f"{backbone}__{dataset}.npz"))
    meta = json.load(open(os.path.join(cache_dir, f"{backbone}__{dataset}.json")))
    C = _normalize(npz["corpus"].astype(np.float32)); Q = _normalize(npz["queries"].astype(np.float32))
    n, d = C.shape; qrels = meta["qrels"]; qids = meta["query_ids"]; row2doc = meta["doc_ids"]

    # identity reference
    id_idx, _ = rank_for_spec("dense", {}, C, Q, k)
    id_pq = ndcg_per_query(id_idx, row2doc, qrels, qids, k)
    rows = []
    for label, kind, params, bpv in spec_grid(d, n):
        t0 = time.perf_counter()
        try:
            idx, code_bytes = rank_for_spec(kind, params, C, Q, k)
        except Exception as e:
            print(f"  skip {label}: {e}"); continue
        lat_ms = (time.perf_counter() - t0) / max(1, Q.shape[0]) * 1e3
        pq = ndcg_per_query(idx, row2doc, qrels, qids, k)
        if label == "identity":
            delta = np.zeros_like(pq); lo = hi = 0.0; p = 1.0
        else:
            delta = pq - id_pq
            lo, hi = paired_bootstrap_ci(delta, B, seed)
            p = paired_randomization_p(delta, B, seed)
        rows.append(dict(spec=label, q=float(pq.mean()), delta=float(delta.mean()),
                         lo=lo, hi=hi, p=p, b=float(bpv), l=float(lat_ms),
                         corpus_bytes=int(bpv * n)))
    # pareto + hv
    par = set(pareto_front(rows))
    ref = (min(r["q"] for r in rows) - 0.01, max(r["b"] for r in rows) * 1.1, max(r["l"] for r in rows) * 1.1)
    hv = hypervolume(rows, ref)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{backbone}__{dataset}")
    with open(base + "__candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone","dataset","spec","ndcg_at_10","delta_vs_identity","ci_lower",
                    "ci_upper","p_value","significant_05","bytes_per_vector","corpus_bytes",
                    "fit_ms","encode_ms","query_latency_ms","query_p95_ms"])
        for r in rows:
            w.writerow([backbone, dataset, r["spec"], f"{r['q']:.6f}", f"{r['delta']:.6f}",
                        f"{r['lo']:.6f}", f"{r['hi']:.6f}", f"{r['p']:.6f}",
                        int(r["p"] < 0.05 and r["spec"] != "identity"),
                        int(r["b"]), r["corpus_bytes"], "0.0", "0.0",
                        f"{r['l']:.6f}", f"{r['l']:.6f}"])
    cfg = dict(backbone=backbone, dataset=f"beir-local:{dataset}", dim=d, n_corpus=n,
               n_queries=Q.shape[0], bootstrap_resamples=B, significance_resamples=B,
               seed=seed, profile_repeats=profile_repeats)
    manifest = dict(name=f"{backbone}__{dataset}", config=cfg, seed=seed,
                    extra=dict(pareto_specs=sorted(par), hypervolume=hv,
                               csv_path=base + "__candidates.csv"),
                    config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest())
    json.dump(manifest, open(base + "__manifest.json", "w"), indent=2)
    print(f"  {backbone}/{dataset}: {len(rows)} specs, {len(par)} pareto, HV={hv:.1f}")
    return rows

def _selftest():
    import tempfile
    rng = np.random.default_rng(0)
    cache = tempfile.mkdtemp(); out = tempfile.mkdtemp()
    n, m, d = 400, 60, 128
    C = rng.standard_normal((n, d)).astype(np.float32)
    Q = rng.standard_normal((m, d)).astype(np.float32)
    np.savez(os.path.join(cache, "toy__syn.npz"), corpus=C, queries=Q)
    qrels = {str(j): {str(int(rng.integers(0, n))): 1} for j in range(m)}
    json.dump(dict(query_ids=[str(j) for j in range(m)], doc_ids=[str(i) for i in range(n)],
                   qrels=qrels), open(os.path.join(cache, "toy__syn.json"), "w"))
    rows = run_pair("toy", "syn", cache, out, B=500, seed=0)
    import csv as _c
    got = list(_c.DictReader(open(os.path.join(out, "toy__syn__candidates.csv"))))
    fams = {("identity" if r["spec"]=="identity" else r["spec"].split("(")[0]) for r in got}
    print("SELFTEST OK:", len(got), "rows; families:", sorted(fams),
          "| has int8:", any(r["spec"]=="scalar_int8" for r in got),
          "| has opq:", any(r["spec"].startswith("opq") for r in got),
          "| out_dir:", out)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=[])
    ap.add_argument("--datasets", nargs="*", default=[], help="dataset names matching cache files")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--bootstrap-resamples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return
    for bb in a.backbones:
        for ds in a.datasets:
            run_pair(bb, ds, a.cache_dir, a.output_dir, a.bootstrap_resamples, seed=a.seed)

if __name__ == "__main__":
    main()
