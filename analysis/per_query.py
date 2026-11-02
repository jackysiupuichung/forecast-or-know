"""Per-query metrics on the full time-aware 1:10 test sets (2021-2026), the unit of every table and
figure: one row per (series, arm, disease, decision year).

series = the six models (arms A0-A3) and the trial-history baseline ta_breadth (arm "baseline").
LLM score of a candidate = logprob_yes - logprob_no; candidates whose answer did not parse are ranked
last in their query (min - 1). AP and MRR break ties against the positives.

    python analysis/per_query.py        -> $AGG_DIR/per_query.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_outputs import MODELS, YEARS, records  # noqa: E402
from src.paths import AGG_DIR  # noqa: E402

COLS = ["auroc", "mrr", "hit1", "ndcg3", "p3", "p5", "ap", "rprec"]
ARMS = ["A0", "A1", "A2", "A3"]


def metrics(lab, s):
    order = np.lexsort((lab, -s)); ls = lab[order]; n = len(ls); fp = np.flatnonzero(ls)[0]
    npos = lab.sum(); disc = 1 / np.log2(np.arange(2, 5))
    ndcg3 = (ls[:3] * disc[:min(3, n)]).sum() / disc[:min(npos, 3)].sum()
    hits = np.flatnonzero(ls); ap = np.mean(np.arange(1, len(hits) + 1) / (hits + 1))
    return [roc_auc_score(lab, s), 1 / (fp + 1), float(fp == 0), ndcg3, ls[:3].sum() / min(3, n),
            ls[:5].sum() / min(5, n), ap, ls[:npos].mean()]


bl = pd.concat([pd.read_csv(f"data/evaluation/ratio10/test_yr{Y}/trial_entry_ratio10.csv",
                            usecols=["targetId", "diseaseId", "label", "ta_breadth"]).assign(year=Y)
                for Y in YEARS])
Q = {}
for (d, Y), g in bl.groupby(["diseaseId", "year"]):
    lab = g.label.to_numpy()
    if 0 < lab.sum() < len(lab):
        Q[(d, Y)] = dict(idx={t: i for i, t in enumerate(g.targetId)}, lab=lab, g=g)

rows = []


def add(series, arm, d, Y, lab, s):
    rows.append(dict(series=series, arm=arm, diseaseId=d, year=Y, n_pos=int(lab.sum()), n_cand=len(lab),
                     **dict(zip(COLS, metrics(lab, s)))))


for (d, Y), qq in Q.items():                                   # trial-history baseline
    add("ta_breadth", "baseline", d, Y, qq["lab"], qq["g"].ta_breadth.to_numpy(float))

for m in MODELS:
    for arm in ARMS:
        P = {(r["disease_id"], Y): r["candidate_scores"] for Y in YEARS for r in records(m, arm, Y)}
        for (d, Y), qq in Q.items():
            cs = P.get((d, Y))
            if cs is None:
                continue
            s = np.full(len(qq["lab"]), np.nan)
            for c in cs:
                i = qq["idx"].get(c["target_id"])
                ly, ln = c.get("logprob_yes"), c.get("logprob_no")
                if i is not None and c.get("parse_ok") and ly is not None and ln is not None:
                    s[i] = ly - ln
            if np.isnan(s).all():
                continue
            s[np.isnan(s)] = np.nanmin(s) - 1
            add(m, arm, d, Y, qq["lab"], s)
        print(m, arm, flush=True)

AGG_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(AGG_DIR / "per_query.csv", index=False)
print("wrote", AGG_DIR / "per_query.csv", len(rows), "rows")
