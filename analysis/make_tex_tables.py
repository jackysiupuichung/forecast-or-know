"""Table bodies (rows only) for Tables 1, 3 (A0) and 5 (yearly AUROC slopes) from the per-query results on the
full 1:10 test sets (2021-2026); the tabular/caption shells are in the paper source. Also prints the query
counts and chance levels quoted in the captions, and each model-minus-trial-history difference.

    python analysis/make_tex_tables.py   -> $OUTDIR/tables/tab_{headline,headline_full,slopes}_body.tex"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera_data import load  # noqa: E402
from src.paths import TABLES as OUT  # noqa: E402
OUT.mkdir(parents=True, exist_ok=True)
q = load(ids=True)
K = ["diseaseId", "year"]
MODELS = [("llama3.1-8b", "Llama 3.1-8B", "Dec 2023"), ("llama3.1-70b", "Llama 3.1-70B", "Dec 2023"),
          ("qwen3-8b", "Qwen 3-8B", "Jun 2024$^*$"), ("qwen3-32b", "Qwen 3-32B", "Jun 2024$^*$"),
          ("gemma-2-9b", "Gemma 2-9B", "Jun 2024$^*$"), ("gemma-2-27b", "Gemma 2-27B", "Jun 2024$^*$")]
POSTY = {"llama3.1-8b": 2024, "llama3.1-70b": 2024}
q["post"] = [y >= POSTY.get(s, 2025) for s, y in zip(q.series, q.year)]
rng = np.random.default_rng(0)

def sel(m, a, side=None):
    g = q[(q.series == m) & (q.arm == a)]
    if side is not None:
        g = g[g.post == (side == "post")]
    return g.set_index(K)

def boot(v, n=2000):
    v = np.asarray(v); b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return np.percentile(b, [2.5, 97.5])

def diff(x, y):
    d = (x - y).dropna(); lo, hi = boot(d.values); return d.mean(), lo, hi

def star(m, lo, hi, ci=False):
    s = f"${m:+.3f}$" + ("$^*$" if lo > 0 or hi < 0 else "")
    return s + (f" $[{lo:+.3f}, {hi:+.3f}]$".replace("+0.", ".").replace("-0.", "-.") if ci else "")

# ---- Table 1: levels, transposed: rows = model x cutoff side, columns = metric x condition
# (leading zeros dropped to fit 15 numeric columns; cutoffs and query counts go in the caption)
nolead = lambda v: f"{v:.3f}".replace("0.", ".", 1)
lines = []
for i, (m, M, cut) in enumerate(MODELS):
    for side in ["pre", "post"]:
        vals = [sel(m, a, side)[c].mean() for c in ["auroc", "ap", "mrr"] for a in ["A1", "A2", "A3"]]
        lines.append(f"{M if side == 'pre' else ''} & {side} & " + " & ".join(nolead(v) for v in vals) + r" \\")
    lines.append(r"\cmidrule{1-11}")   # thin rule under each model's pre/post pair
lines.pop()
(OUT / "tab_headline_body.tex").write_text("\n".join(lines) + "\n")
# ---- Appendix: levels for every condition (A0 is not in the main Table 1)
lines = []
for i, (m, M, cut) in enumerate(MODELS):
    for side in ["pre", "post"]:
        vals = [sel(m, a, side)[c].mean() for c in ["auroc", "ap", "mrr"] for a in ["A0", "A1", "A2", "A3"]]
        lines.append(f"{M if side == 'pre' else ''} & {side} & " + " & ".join(nolead(v) for v in vals) + r" \\")
    lines.append(r"\cmidrule{1-14}")
lines.pop()
(OUT / "tab_headline_full_body.tex").write_text("\n".join(lines) + "\n")
for m, M, cut in MODELS:
    print(f"{M}: cutoff {cut} | n_pre {len(sel(m, 'A1', 'pre'))} | n_post {len(sel(m, 'A1', 'post'))}")

# ---- Check: each model and condition minus trial history (per-query paired, all years; * = 95% CI excludes 0)
print('model minus trial history (AUROC, AP, MRR):')
for m, M, _ in MODELS:
    for j, a in enumerate(["A0", "A1", "A2", "A3"]):
        x = sel(m, a)
        cells = [star(*diff(x[c], q[q.series == "ta_breadth"].set_index(K)[c])) for c in ["auroc", "ap", "mrr"]]
        print(f"  {M:14s} {a}  " + "  ".join(c.replace('$', '').replace('^*', '*') for c in cells))

# ---- Appendix: year slopes, raw and query-size adjusted, pooled over models
rrf = q[q.series == "ta_breadth"].set_index(K).auroc   # trial-history baseline

def slope(X, y, adj):
    f = f"{y}~year" + ("+C(series)" if X.series.nunique() > 1 else "") + ("+np.log(n_cand)+np.log(n_pos)" if adj else "")
    r = smf.ols(f, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(X.diseaseId)[0]})
    lo, hi = r.conf_int().loc["year"]
    return f"${r.params.year:+.4f}$" + ("$^*$" if lo > 0 or hi < 0 else "") + f" ({r.pvalues.year:.3f})"

lines = []
for lab, X in [("Trial history", q[q.series == "ta_breadth"])] + \
              [(f"LLMs, {a}", q[(q.arm == a) & q.series.isin([m for m, _, _ in MODELS])]) for a in ["A0", "A1", "A2", "A3"]]:
    X = X.copy(); cells = [slope(X, "auroc", 0)]   # raw AUROC only: AUROC needs no query-size adjustment (2026-10-26)
    lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
(OUT / "tab_slopes_body.tex").write_text("\n".join(lines) + "\n")

# chance MRR / AP (exact, per query) for the 1:10 lists
from math import comb
def chance_mrr(n, N):
    return sum(comb(N - r, n - 1) / comb(N, n) / r for r in range(1, N - n + 2))
g = q[q.series == "ta_breadth"]
def chance_ap(n, N, reps=4000):
    r = np.random.default_rng(0); s = 0.0
    for _ in range(reps):
        h = np.sort(r.choice(N, n, replace=False)); s += np.mean(np.arange(1, n + 1) / (h + 1))
    return s / reps
print("chance MRR:", round(np.mean([chance_mrr(int(n), int(c)) for n, c in zip(g.n_pos, g.n_cand)]), 3),
      "| chance AP:", round(np.mean([chance_ap(int(n), int(c)) for n, c in zip(g.n_pos, g.n_cand)]), 3), "| chance AUROC 0.5")
print("queries:", len(g), "| per year:", g.groupby("year").size().to_dict())
print("wrote", sorted(p.name for p in OUT.iterdir()))
