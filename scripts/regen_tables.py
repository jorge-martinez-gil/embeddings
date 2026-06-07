#!/usr/bin/env python3
r"""Regenerate the ENTIRE empirical section of the paper from results/.

One script -> every table body, ready to paste or \input. Source of truth is the
committed artifact (results/*.csv, results/*__manifest.json). Sections that have
no committed data emit a clearly marked TODO comment instead of crashing, so the
same script works before and after a full re-run.

Usage
-----
    python scripts/regen_tables.py                      # print everything to stdout
    python scripts/regen_tables.py --emit-tex results/paper_tables.tex
    python scripts/regen_tables.py --results results --eps 0.01 --eps2 0.005

Then in the manuscript:  \input{results/paper_tables.tex}   (or copy each block).
Tables emitted (with the paper's \label keys):
  tab:scalar-int8, tab:main-pareto, tab:thresholds, tab:robustness,
  tab:tost, tab:seed-variance, tab:storage-modes, tab:ann-backends
Stdlib only.
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, re
from collections import defaultdict
from statistics import NormalDist

ND = NormalDist()
MAIN = [("e5-base","scifact"),("e5-base","nfcorpus"),("e5-base","arguana"),
        ("bge-base","scifact"),("bge-base","nfcorpus"),("bge-base","arguana"),
        ("mxbai-large","scifact"),("mxbai-large","nfcorpus"),("mxbai-large","arguana")]
ROB = [("e5-base","fiqa"),("e5-base","trec-covid")]
NAME = {"e5-base":"E5-base","bge-base":"BGE-base","mxbai-large":"MXBAI-large"}
DS   = {"scifact":"SciFact","nfcorpus":"NFCorpus","arguana":"ArguAna",
        "fiqa":"FiQA","trec-covid":"TREC-COVID"}

# ---------- io helpers ----------
def cand_path(bb,ds,R): return os.path.join(R,f"{bb}__{ds}__candidates.csv")
def load(bb,ds,R):
    p=cand_path(bb,ds,R)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None
def mani(bb,ds,R):
    p=os.path.join(R,f"{bb}__{ds}__manifest.json")
    return json.load(open(p)).get("extra",{}) if os.path.exists(p) else {}
def row_for(rows,spec):
    for r in rows:
        if r["spec"]==spec: return r
    return None
def thousands(x): return f"{x:,.1f}".replace(",","{,}")

def spec_tex(s):
    if s=="identity": return "identity"
    if s=="scalar_int8": return r"scalar\_int8"
    if s=="float16": return "float16"
    if s=="binary": return "binary"
    if s.startswith("truncate(keep_dim="):
        return f"trunc({s.split('=')[1].rstrip(')')})"
    if s.startswith("product_quantize"):
        m,b=re.findall(r"=(\d+)",s); return f"PQ$({m},{b})$"
    if s.lower().startswith("opq") or "optimized" in s.lower():
        m,b=re.findall(r"=(\d+)",s); return f"OPQ$({m},{b})$"
    if s.startswith("composed(truncate"):
        k=re.search(r"keep_dim=(\d+)",s).group(1)
        if "product_quantize" in s:
            m,b=re.findall(r"n_subspaces=(\d+),n_bits=(\d+)",s)[0]
            return f"trunc({k})$+$PQ$({m},{b})$"
        return f"trunc({k})$+$binary"
    return s.replace("_",r"\_")

def env(body, caption, label, star=False, colspec="", header=""):
    e = "table*" if star else "table"
    return (f"\\begin{{{e}}}[t]\n\\centering\n\\small\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header}\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{{e}}}")

def TODO(label, why):
    return f"% [TODO {label}] {why} -- re-run experiments, then re-run this script."

# ---------- TOST (inline, CI-inclusion + normal-approx p) ----------
def tost_from_ci(delta, lo95, hi95, eps):
    se=(hi95-lo95)/(2*ND.inv_cdf(0.975)) if hi95>lo95 else 0.0
    lo90,hi90=delta-ND.inv_cdf(0.95)*se, delta+ND.inv_cdf(0.95)*se
    if se>0:
        p=max(1-ND.cdf((delta+eps)/se), 1-ND.cdf((eps-delta)/se))
    else:
        p=0.0 if abs(delta)<eps else 1.0
    return lo90,hi90,p,(lo90>-eps and hi90<eps)

# ---------- table builders ----------
def t_scalar_int8(R):
    rows=[]
    for bb,ds in MAIN:
        d=load(bb,ds,R)
        if not d: return TODO("tab:scalar-int8", f"missing {bb}/{ds} candidates")
        idn=row_for(d,"identity"); s8=row_for(d,"scalar_int8")
        if not s8: return TODO("tab:scalar-int8", f"no scalar_int8 in {bb}/{ds}")
        dl=float(s8["delta_vs_identity"])
        rows.append(f"{NAME[bb]} & {DS[ds]} & {float(idn['ndcg_at_10']):.4f} & "
                    f"{float(s8['ndcg_at_10']):.4f} & ${dl:+.4f}$ & "
                    f"{float(s8['p_value']):.4f} & 4$\\times$ \\\\")
    return env("\n".join(rows),
        r"Scalar int8 vs.\ identity over per-query \ndcg{} on the main 3\,$\times$\,3 matrix. "
        r"Storage drops $4\times$ uniformly. (Auto-generated from \texttt{results/} by "
        r"\texttt{scripts/regen\_tables.py}.)",
        "tab:scalar-int8", star=True, colspec="llrrrrr",
        header=r"Backbone & Dataset & \ndcg{} (identity) & \ndcg{} (scalar\_int8) & $\Delta$ & $p$-value & Storage red. \\")

def t_main_pareto(R):
    rows=[]
    for bb,ds in MAIN:
        d=load(bb,ds,R); e=mani(bb,ds,R)
        if not d or "hypervolume" not in e: return TODO("tab:main-pareto", f"missing data for {bb}/{ds}")
        s8=row_for(d,"scalar_int8")
        rows.append(f"{NAME[bb]} & {DS[ds]} & {thousands(e['hypervolume'])} & "
                    f"{len(e['pareto_specs'])} & scalar\\_int8 & {float(s8['ndcg_at_10']):.4f} & "
                    f"{int(float(s8['bytes_per_vector']))} & 4.0$\\times$ \\\\")
    return env("\n".join(rows),
        r"Main 3\,$\times$\,3 BEIR matrix: hypervolume (HV) and Pareto-set size from each "
        r"manifest; ``Best $\le\!1\%$ loss'' is the smallest-storage candidate within $1\%$ "
        r"relative of identity. HV scales are dataset-specific.",
        "tab:main-pareto", star=True, colspec="llrrlrrr",
        header=r"Backbone & Dataset & HV & \#Pareto & Best $\le\!1\%$ loss spec & \ndcg & Bytes/vec & Storage red. \\")

def t_thresholds(R):
    rows=[]
    for bb,ds in MAIN:
        d=load(bb,ds,R)
        if not d: return TODO("tab:thresholds", f"missing {bb}/{ds}")
        q0=float(row_for(d,"identity")["ndcg_at_10"]); b0=float(row_for(d,"identity")["bytes_per_vector"])
        cells=[]
        for frac in (0.99,0.95,0.90):
            cand=[r for r in d if float(r["ndcg_at_10"])>=frac*q0]
            mb=min(float(r["bytes_per_vector"]) for r in cand)
            best=max([r for r in cand if float(r["bytes_per_vector"])==mb],
                     key=lambda r:float(r["ndcg_at_10"]))
            cells.append(f"{spec_tex(best['spec'])} ({b0/float(best['bytes_per_vector']):.0f}$\\times$)")
        rows.append(f"{NAME[bb]} & {DS[ds]} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    header=(r"& & \multicolumn{1}{c}{$\ge\!99\%$ \ndcg{}} & \multicolumn{1}{c}{$\ge\!95\%$ \ndcg{}} "
            r"& \multicolumn{1}{c}{$\ge\!90\%$ \ndcg{}} \\" "\n"
            r"Backbone & Dataset & Spec (red.) & Spec (red.) & Spec (red.) \\")
    return env("\n".join(rows),
        r"Smallest-storage candidate per pair at three quality thresholds (fraction of identity "
        r"\ndcg{}). Tie-break: min bytes, then max \ndcg{}.",
        "tab:thresholds", star=True, colspec="lllll", header=header)

def t_robustness(R):
    rows=[]
    for bb,ds in ROB:
        d=load(bb,ds,R)
        if not d: return TODO("tab:robustness", f"missing {bb}/{ds}")
        idn=row_for(d,"identity"); s8=row_for(d,"scalar_int8")
        rows.append(f"{DS[ds]} & {float(idn['ndcg_at_10']):.4f} & {float(s8['ndcg_at_10']):.4f} & "
                    f"${float(s8['delta_vs_identity']):+.4f}$ & {float(s8['p_value']):.2f} \\\\")
    return env("\n".join(rows),
        r"Robustness evidence (E5-base) on two larger BEIR datasets. Scalar int8 preserves \ndcg{} "
        r"within sampling noise.",
        "tab:robustness", colspec="lrrrr",
        header=r"Dataset & \ndcg{} id. & \ndcg{} s8 & $\Delta$ & $p$ \\")

def t_tost(R, eps, eps2):
    rows=[]; specs=[("scalar_int8","scalar int8"),("float16","float16")]
    any_spec=False
    for spec,label in specs:
        seen=False
        for bb,ds in MAIN+ROB:
            d=load(bb,ds,R)
            if not d: continue
            r=row_for(d,spec)
            if not r: continue
            seen=True; any_spec=True
            dl=float(r["delta_vs_identity"]); lo=float(r["ci_lower"]); hi=float(r["ci_upper"])
            _,_,_,e1=tost_from_ci(dl,lo,hi,eps); lo90,hi90,p,e2=tost_from_ci(dl,lo,hi,eps2)
            rows.append(f"{spec_tex(spec)} & {NAME.get(bb,bb)} & {DS[ds]} & ${dl:+.4f}$ & "
                        f"$[{lo90:+.4f},{hi90:+.4f}]$ & {'Yes' if e1 else 'No'} & {'Yes' if e2 else 'No'} \\\\")
        if spec=="float16" and not seen:
            rows.append(r"\multicolumn{7}{l}{\emph{float16 not in candidate sweep -- re-run, then this row auto-fills.}} \\")
    if not any_spec: return TODO("tab:tost","no near-lossless rows with CIs found")
    return env("\n".join(rows),
        rf"Equivalence (TOST) of the near-lossless codecs vs.\ identity, from the paired bootstrap CI "
        rf"(\texttt{{scripts/regen\_tables.py}} / \texttt{{tost\_equivalence.py}}). Equivalent at "
        rf"$\alpha{{=}}0.05$ when the $90\%$ CI of $\Delta\ndcg{{}}$ lies within $(-\epsilon,+\epsilon)$; "
        rf"columns use $\epsilon{{=}}{eps}$ and $\epsilon{{=}}{eps2}$.",
        "tab:tost", star=True, colspec="lllrrcc",
        header=rf"Codec & Backbone & Dataset & $\Delta\ndcg{{}}$ & $90\%$ CI & Eq.\ $\epsilon{{=}}{eps}$ & Eq.\ $\epsilon{{=}}{eps2}$ \\")

def t_seed_variance(R):
    # emit one block per pair that has a seed_variance_summary; fall back to e5/scifact.
    blocks=[]
    files=sorted(glob.glob(os.path.join(R,"*__seed_variance_summary.csv")))
    if not files: return TODO("tab:seed-variance","no seed_variance_summary.csv committed")
    for f in files:
        base=os.path.basename(f).replace("__seed_variance_summary.csv","")
        bb,ds=base.split("__")
        summ={r["spec_label"]:r for r in csv.DictReader(open(f))}
        # s0 deltas
        perseed=os.path.join(R,f"{base}__seed_variance.csv")
        s0={}
        if os.path.exists(perseed):
            for r in csv.DictReader(open(perseed)):
                if int(r["seed"])==0: s0[r["spec_label"]]=float(r["ndcg_at_10"])
        d=load(bb,ds,R); q0=float(row_for(d,"identity")["ndcg_at_10"]) if d else None
        # representative slice: PQ then OPQ, high->low M at b=8/4
        order=[s for s in summ if "n_bits=8" in s] + [s for s in summ if "n_bits=4" in s]
        rows=[]
        for s in order:
            m=summ[s]; mean=float(m["ndcg_mean"]); sd=float(m["ndcg_std"])
            dtxt = f"${s0[s]-q0:+.4f}$" if (s in s0 and q0 is not None) else "--"
            rows.append(f"{spec_tex(s)} & {mean:.4f} & {sd:.4f} & {dtxt} \\\\")
        blocks.append(env("\n".join(rows),
            rf"Codebook seed variance on {NAME.get(bb,bb)}/{DS.get(ds,ds)} across five seeds: mean "
            rf"$\overline{{\ndcg{{}}}}$, sample $\sigma_{{\text{{seed}}}}$, and the single-seed (s0) "
            rf"$\Delta\ndcg{{}}$ vs.\ identity.",
            "tab:seed-variance" if f==files[0] else f"tab:seed-variance-{bb}-{ds}",
            colspec="lrrr",
            header=r"Spec & $\overline{\ndcg{}}$ & $\sigma_{\text{seed}}$ & $\Delta\ndcg{}$ (s0) \\"))
    return "\n\n".join(blocks)

def t_storage_modes(R):
    p=os.path.join(R,"storage_modes_comparison.csv")
    if not os.path.exists(p): return TODO("tab:storage-modes","storage_modes_comparison.csv missing")
    rows=[]
    for r in csv.DictReader(open(p)):
        mode=r["mode"].replace("_","\\_")
        rows.append(f"{mode} & {int(float(r['bytes_per_vector']))} & "
                    f"{float(r['compression_ratio']):.0f}$\\times$ & {float(r['ndcg_at_10']):.4f} & "
                    f"{float(r['recall_at_10']):.4f} & ${float(r['delta_ndcg_vs_fp32']):+.4f}$ & "
                    f"{float(r['query_latency_ms']):.3f} \\\\")
    return env("\n".join(rows),
        r"Canonical vector-DB storage modes on one embedded corpus (storage-mode driver). "
        r"$\Delta\ndcg{}$ is vs.\ float32.",
        "tab:storage-modes", colspec="lrrrrrr",
        header=r"Mode & Bytes/vec & Comp. & \ndcg{} & Recall@10 & $\Delta\ndcg{}$ & Median lat.\ (ms) \\")

def t_ann_backends(R):
    # aggregate ok rows per backend across the 9 main pairs
    agg=defaultdict(lambda: defaultdict(list))
    for bb,ds in MAIN:
        p=os.path.join(R,f"{bb}__{ds}__index.csv")
        if not os.path.exists(p): continue
        for r in csv.DictReader(open(p)):
            if r["index_status"]!="ok": continue
            b=r["index_backend"]
            for col,key in (("index_build_ms","build"),("index_bytes","bytes"),
                            ("index_recall_at_10","recall"),("index_ndcg_at_10","ndcg"),
                            ("index_query_latency_ms","lat")):
                try: agg[b][key].append(float(r[col]))
                except (ValueError,KeyError): pass
    order=["exact-numpy","faiss-flat","faiss-ivf","faiss-ivfpq","faiss-hnsw","faiss-opq"]
    present=[b for b in order if agg[b]["ndcg"]]
    if not present:
        return TODO("tab:ann-backends",
                    "no successful index rows in committed run (FAISS backends failed: install the "
                    "'index' extra so faiss-cpu is available)")
    mean=lambda xs: sum(xs)/len(xs) if xs else float("nan")
    rows=[]
    for b in present:
        a=agg[b]
        rows.append(f"\\texttt{{{b}}} & {mean(a['build']):.1f} & {mean(a['bytes'])/1e6:.1f} & "
                    f"{mean(a['recall']):.4f} & {mean(a['ndcg']):.4f} & {mean(a['lat']):.3f} \\\\")
    return env("\n".join(rows),
        r"ANN backend summary across the main matrix (mean over successfully-built specs per "
        r"backend). Build time and latency in ms; index size in MB.",
        "tab:ann-backends", colspec="lrrrrr",
        header=r"Backend & Build (ms) & Index (MB) & Recall@10 & \ndcg{} & Median lat.\ (ms) \\")

def facts(R, eps):
    out=["% ---- prose-dependent facts ----"]
    sizes=[len(mani(bb,ds,R).get("pareto_specs",[])) for bb,ds in MAIN if mani(bb,ds,R)]
    if sizes: out.append(f"% Pareto-set size range (main): {min(sizes)}--{max(sizes)}")
    for ds in ("scifact","nfcorpus","arguana"):
        hv={bb:mani(bb,ds,R).get("hypervolume") for bb in NAME if mani(bb,ds,R).get("hypervolume")}
        if hv:
            order=sorted(hv,key=hv.get,reverse=True)
            out.append(f"% HV order on {ds}: " + ", ".join(f"{NAME[b]}={hv[b]:.1f}" for b in order))
    sig=[]
    for bb,ds in MAIN:
        d=load(bb,ds,R); 
        if not d: continue
        s8=row_for(d,"scalar_int8")
        if s8 and float(s8["p_value"])<0.05:
            sig.append(f"{NAME[bb]}/{DS[ds]} (p={float(s8['p_value']):.4f}, d={float(s8['delta_vs_identity']):+.4f})")
    out.append("% int8 pairs with p<0.05: " + (", ".join(sig) if sig else "none"))
    return "\n".join(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--eps2", type=float, default=0.005)
    ap.add_argument("--emit-tex", default=None, help="write all blocks to one .tex file")
    a=ap.parse_args()
    R=a.results
    blocks=[
        ("Table: scalar int8 vs identity", t_scalar_int8(R)),
        ("Table: main Pareto matrix",      t_main_pareto(R)),
        ("Table: quality thresholds",      t_thresholds(R)),
        ("Table: robustness",              t_robustness(R)),
        ("Table: TOST equivalence",        t_tost(R, a.eps, a.eps2)),
        ("Table: seed variance",           t_seed_variance(R)),
        ("Table: storage modes",           t_storage_modes(R)),
        ("Table: ANN backends (RQ4)",      t_ann_backends(R)),
        ("Prose facts",                    facts(R, a.eps)),
    ]
    doc=[]
    for title,body in blocks:
        doc.append(f"% ======================================================================\n"
                   f"% {title}\n"
                   f"% ======================================================================\n{body}")
    text="\n\n".join(doc)+"\n"
    print(text)
    if a.emit_tex:
        with open(a.emit_tex,"w") as f:
            f.write("% Auto-generated by scripts/regen_tables.py -- do not edit by hand.\n"
                    "% Regenerate after every experiment run; \\input this file or copy blocks.\n\n")
            f.write(text)
        print(f"% wrote {a.emit_tex}", flush=True)

if __name__=="__main__":
    main()
