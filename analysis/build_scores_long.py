"""One long table of per-candidate scores for every zero-shot run and the trial-history baseline on the
full time-aware 1:10 test sets: series, arm, year, diseaseId, targetId, label, score.
LLM score = logprob_yes - logprob_no (unparsed ranked last in its query).

    python analysis/build_scores_long.py   -> $AGG_DIR/scores_long.parquet"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_outputs import MODELS, YEARS, records  # noqa: E402
from src.paths import AGG_DIR  # noqa: E402
OUT = AGG_DIR / "scores_long.parquet"
frames = []
for m in MODELS:
    for arm in ["A0", "A1", "A2", "A3"]:
        for Y in YEARS:
            rows = []
            for r in records(m, arm, Y):
                cs = r["candidate_scores"]
                s = np.array([c["logprob_yes"] - c["logprob_no"] if c.get("parse_ok") and c.get("logprob_yes") is not None
                              else np.nan for c in cs], float)
                s[np.isnan(s)] = np.nanmin(s) - 1 if np.isfinite(s).any() else 0.0
                rows += [(r["disease_id"], c["target_id"], c["label"], v) for c, v in zip(cs, s)]
            df = pd.DataFrame(rows, columns=["diseaseId", "targetId", "label", "score"])
            frames.append(df.assign(series=m, arm=arm, year=Y))
for Y in YEARS:
    d = pd.read_csv(f"data/evaluation/ratio10/test_yr{Y}/trial_entry_ratio10.csv",
                    usecols=["targetId", "diseaseId", "label", "ta_breadth"])
    for h in ["ta_breadth"]:
        frames.append(d[["diseaseId", "targetId", "label"]].assign(score=d[h].astype(float), series=h, arm="baseline", year=Y))
long = pd.concat(frames, ignore_index=True)
for c in ["series", "arm", "diseaseId", "targetId"]:
    long[c] = long[c].astype("category")
long.to_parquet(OUT, index=False)
print(long.groupby(["arm"], observed=True).series.nunique().to_dict(), len(long))
print(long.groupby(["arm", "year"], observed=True).size().unstack().fillna(0).astype(int).to_string())
