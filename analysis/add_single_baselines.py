"""Single label-blind baselines (Table 4): the four trial counts stored in the candidate sets and the
genetic-association evidence (analysis/genetic_scores.py), scored per query on the full 1:10 test sets
with the metrics of analysis/per_query.py.

    python analysis/add_single_baselines.py   -> $AGG_DIR/per_query_single_baselines.csv,
                                                  $OUTDIR/tables/tab_single_baselines_body.tex
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import AGG_DIR, TABLES  # noqa: E402

COLS = ["auroc", "mrr", "hit1", "ndcg3", "p3", "p5", "ap", "rprec"]


def metrics(lab, s):  # as in analysis/per_query.py
    order = np.lexsort((lab, -s)); ls = lab[order]; n = len(ls); fp = np.flatnonzero(ls)[0]
    npos = lab.sum(); disc = 1 / np.log2(np.arange(2, 5))
    ndcg3 = (ls[:3] * disc[:min(3, n)]).sum() / disc[:min(npos, 3)].sum()
    hits = np.flatnonzero(ls); ap = np.mean(np.arange(1, len(hits) + 1) / (hits + 1))
    return [roc_auc_score(lab, s), 1 / (fp + 1), float(fp == 0), ndcg3, ls[:3].sum() / min(3, n),
            ls[:5].sum() / min(5, n), ap, ls[:npos].mean()]


gen = pd.read_csv(AGG_DIR / "genetic_scores.csv.gz")
frames = []
for Y in range(2021, 2027):
    df = pd.read_csv(f"data/evaluation/ratio10/test_yr{Y}/trial_entry_ratio10.csv",
                     usecols=["targetId", "diseaseId", "label", "breadth", "recent", "ta_breadth", "onto"])
    df["year"] = Y
    df = df.merge(gen, on=["year", "diseaseId", "targetId"], how="left", validate="one_to_one")
    assert df.genetic_direct.notna().all(), f"missing genetic scores for {Y}"
    frames.append(df)
bl = pd.concat(frames, ignore_index=True)
NAMES = ["breadth", "recent", "ta_breadth", "onto", "genetic_direct", "genetic_prop"]
rows = []
for name in NAMES:
    for (d, Y), g in bl.groupby(["diseaseId", "year"]):
        lab = g.label.to_numpy()
        rows.append(dict(series=name, arm="baseline", diseaseId=d, year=Y, n_pos=int(lab.sum()), n_cand=len(lab),
                         **dict(zip(COLS, metrics(lab, g[name].to_numpy(float))))))
out = pd.DataFrame(rows)
out.to_csv(AGG_DIR / "per_query_single_baselines.csv", index=False)
ref = pd.read_csv(AGG_DIR / "per_query.csv")
ref = ref[ref.series == "ta_breadth"].set_index(["diseaseId", "year"])
chk = out[out.series == "ta_breadth"].set_index(["diseaseId", "year"]).reindex(ref.index)
assert np.allclose(chk.auroc, ref.auroc) and np.allclose(chk.ap, ref.ap), "ta_breadth mismatch"
nz = bl.assign(**{n: bl[n] > 0 for n in NAMES}).groupby("label")[NAMES].mean()
print(f"{'baseline':16s}{'AUROC':>7s}{'AP':>7s}{'MRR':>7s}{'nonzero neg':>13s}{'nonzero pos':>13s}")
for n in NAMES:
    x = out[out.series == n]
    print(f"{n:16s}{x.auroc.mean():7.3f}{x.ap.mean():7.3f}{x.mrr.mean():7.3f}{nz.loc[0, n]:13.3f}{nz.loc[1, n]:13.3f}")

LAB = [("ta_breadth", "Trials for diseases in $d$'s area (trial history)"),
       ("breadth", "Trials for any disease"), ("recent", "Recent trial pairs ($C-2$ to $C$)"),
       ("onto", "Trials for ontology relatives of $d$"),
       ("genetic_direct", "Genetic association with $d$"),
       ("genetic_prop", "Genetic association with $d$ or a descendant")]
lines = []
for n, lab in LAB:
    x = out[out.series == n]
    lines.append(f"{lab} & ${x.auroc.mean():.3f}$ & ${x.ap.mean():.3f}$ & ${x.mrr.mean():.3f}$ & "
                 f"${100 * nz.loc[1, n]:.1f}\\%$ \\\\")
    if n == "onto":
        lines.append(r"\addlinespace[2pt]")
TABLES.mkdir(parents=True, exist_ok=True)
(TABLES / "tab_single_baselines_body.tex").write_text("\n".join(lines) + "\n")
print("wrote", TABLES / "tab_single_baselines_body.tex")
