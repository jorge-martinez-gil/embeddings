#!/usr/bin/env python3
"""
Two One-Sided Tests (TOST) equivalence analysis for near-lossless codecs.

Motivation
----------
A non-significant difference test (e.g. the paired randomization test already
reported by embedopt) does NOT establish equivalence: absence of evidence is
not evidence of absence. TOST [Schuirmann 1987; Lakens et al. 2018] reverses
the burden of proof. Given an equivalence margin eps on the per-query nDCG
delta  d = Q(codec) - Q(identity), the codec is declared *statistically
equivalent* to identity at level alpha iff the (1 - 2*alpha) confidence
interval of the mean delta lies entirely inside (-eps, +eps).

Two evaluation paths are provided:

1. exact_bootstrap_tost(...)  -- the artifact-grade path. Operates on per-query
   nDCG arrays, draws B paired resamples (sharing query IDs across the two
   layouts, exactly like embedopt's CI/randomization tests), and reports the
   90% bootstrap CI + a bootstrap TOST verdict. Use this when per-query nDCG
   is available.

2. ci_inclusion_tost(...)     -- the no-per-query fallback. Reuses embedopt's
   already-stored *95%* paired bootstrap CI on delta_vs_identity. Because the
   95% CI is wider than the 90% CI, if the stored 95% CI is inside (-eps, eps)
   then equivalence holds at alpha=0.05 AND at the stronger alpha=0.025. A
   normal-approximation TOST p-value is also returned (SE inferred from the CI
   width). This is what reproduces the numbers in the paper's TOST table from
   the committed results/*__candidates.csv files.

CLI
---
    python scripts/tost_equivalence.py --results results --spec scalar_int8 \
        --eps 0.01 --eps2 0.005 --out results/tost_scalar_int8.csv

Add --spec float16 once a float16 row exists in the candidate sweep (add
"float16" to the spec list and re-run the sweep; embeddings are cached so this
is cheap and deterministic).
"""
from __future__ import annotations
import argparse, csv, glob, math, os
from statistics import NormalDist

ND = NormalDist()
Z = lambda q: ND.inv_cdf(q)


def ci_inclusion_tost(delta: float, lo95: float, hi95: float, eps: float):
    """TOST from a stored 95% bootstrap CI (normal-approx for the p-value)."""
    se = (hi95 - lo95) / (2 * Z(0.975)) if hi95 > lo95 else 0.0
    lo90, hi90 = delta - Z(0.95) * se, delta + Z(0.95) * se
    if se > 0:
        p_lo = 1 - ND.cdf((delta + eps) / se)   # H0: mu <= -eps
        p_hi = 1 - ND.cdf((eps - delta) / se)   # H0: mu >= +eps
        p_tost = max(p_lo, p_hi)
    else:
        p_tost = 0.0 if abs(delta) < eps else 1.0
    return dict(se=se, ci90=(lo90, hi90), p_tost=p_tost,
                equiv_a05=(lo90 > -eps and hi90 < eps),     # alpha = 0.05  (90% CI)
                equiv_a025=(lo95 > -eps and hi95 < eps))    # alpha = 0.025 (95% CI)


def exact_bootstrap_tost(ndcg_identity, ndcg_codec, eps: float,
                         B: int = 5000, alpha: float = 0.05, seed: int = 0):
    """Artifact-grade paired bootstrap TOST from per-query nDCG arrays."""
    import numpy as np
    a = np.asarray(ndcg_identity, float)
    b = np.asarray(ndcg_codec, float)
    assert a.shape == b.shape, "per-query arrays must be aligned by query id"
    d = b - a
    n = d.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    means = d[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha))            # (1-2a) CI, lower
    hi = float(np.quantile(means, 1 - alpha))        # (1-2a) CI, upper
    return dict(delta=float(d.mean()), ci=(lo, hi),
                equiv=(lo > -eps and hi < eps), n=n, B=B, alpha=alpha)


def _read_candidates(path, spec):
    rid = r = None
    for row in csv.DictReader(open(path)):
        if row["spec"] == "identity":
            rid = row
        if row["spec"] == spec:
            r = row
    return rid, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--spec", default="scalar_int8")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--eps2", type=float, default=0.005)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(a.results, "*__candidates.csv"))):
        base = os.path.basename(f).replace("__candidates.csv", "")
        if "hashing" in base:
            continue
        rid, r = _read_candidates(f, a.spec)
        if r is None:
            continue
        bb, ds = base.split("__")
        delta = float(r["delta_vs_identity"])
        lo95, hi95 = float(r["ci_lower"]), float(r["ci_upper"])
        t1 = ci_inclusion_tost(delta, lo95, hi95, a.eps)
        t2 = ci_inclusion_tost(delta, lo95, hi95, a.eps2)
        rows.append((bb, ds, float(rid["ndcg_at_10"]), delta, lo95, hi95, t1, t2))

    hdr = (f"{'backbone':12s} {'dataset':11s} {'delta':>9s} {'95% CI':>20s} "
           f"{'90% CI':>20s} {'p_TOST':>9s}  eq{a.eps} eq{a.eps2}")
    print(hdr); print("-" * len(hdr))
    npass = npass2 = 0
    for bb, ds, q0, d, lo, hi, t1, t2 in rows:
        npass += t1["equiv_a05"]; npass2 += t2["equiv_a05"]
        print(f"{bb:12s} {ds:11s} {d:+9.5f} [{lo:+.5f},{hi:+.5f}] "
              f"[{t1['ci90'][0]:+.5f},{t1['ci90'][1]:+.5f}] {t1['p_tost']:9.2e}  "
              f"{'Y' if t1['equiv_a05'] else 'n'}    {'Y' if t2['equiv_a05'] else 'n'}")
    print(f"\nEquivalent (alpha=0.05): eps={a.eps}: {npass}/{len(rows)}   "
          f"eps={a.eps2}: {npass2}/{len(rows)}")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["backbone", "dataset", "ndcg_identity", "delta",
                        "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi",
                        f"p_tost_eps{a.eps}", f"equiv_eps{a.eps}",
                        f"equiv_eps{a.eps2}"])
            for bb, ds, q0, d, lo, hi, t1, t2 in rows:
                w.writerow([bb, ds, f"{q0:.4f}", f"{d:+.5f}", f"{lo:+.5f}",
                            f"{hi:+.5f}", f"{t1['ci90'][0]:+.5f}",
                            f"{t1['ci90'][1]:+.5f}", f"{t1['p_tost']:.2e}",
                            int(t1["equiv_a05"]), int(t2["equiv_a05"])])
        print("wrote", a.out)


if __name__ == "__main__":
    main()
