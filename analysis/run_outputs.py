"""Zero-shot inference outputs, one record per (model, arm, decision year, disease).

Records have the shape of the lines of predictions.jsonl written by scripts/run_task1_experiment.py:
    {"disease_id", "candidate_scores": [{"target_id", "label", "p_yes", "logprob_yes",
                                          "logprob_no", "parse_ok"}, ...], "metrics": {"mrr"}}
They are read from the full run directory results/<model>/<run>/predictions.jsonl (scripts/run_inference.sh)
when it exists, otherwise from the compact copy shipped with the repository
(data/predictions/candidate_scores.parquet and run_metrics.parquet), which keeps the candidate order.
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PRED = REPO / "data" / "predictions"
MODELS = ["llama3.1-8b", "llama3.1-70b", "qwen3-8b", "qwen3-32b", "gemma-2-9b", "gemma-2-27b"]
YEARS = range(2021, 2027)
# arm -> run directory; A0 (identifiers only) is the A1 prompt on the ratio10_ids sets
RUN = {"A0": "ratio10_ids_test_yr{Y}_A1", "A1": "ratio10_test_yr{Y}_A1",
       "A2": "ratio10_test_yr{Y}_A2", "A3": "ratio10_test_yr{Y}_A3"}
FIELDS = ["target_id", "label", "p_yes", "logprob_yes", "logprob_no", "parse_ok"]


def run_file(model, arm, year):
    return REPO / "results" / model / RUN[arm].format(Y=year) / "predictions.jsonl"


@lru_cache(maxsize=1)
def _compact():
    c = pd.read_parquet(PRED / "candidate_scores.parquet")
    m = pd.read_parquet(PRED / "run_metrics.parquet")
    return c, m


def _none(v):
    return None if pd.isna(v) else v


def records(model, arm, year):
    f = run_file(model, arm, year)
    if f.exists():
        with open(f) as fh:
            for line in fh:
                yield json.loads(line)
        return
    c, m = _compact()
    sel = c[(c.model == model) & (c.arm == arm) & (c.year == year)]
    mrr = m[(m.model == model) & (m.arm == arm) & (m.year == year)].set_index("disease_id").mrr
    for d, g in sel.groupby("disease_id", sort=False, observed=True):
        cands = [{"target_id": t, "label": int(lab), "p_yes": _none(p), "logprob_yes": _none(ly),
                  "logprob_no": _none(ln), "parse_ok": bool(ok)}
                 for t, lab, p, ly, ln, ok in zip(g.target_id, g.label, g.p_yes, g.logprob_yes,
                                                  g.logprob_no, g.parse_ok)]
        yield {"disease_id": d, "candidate_scores": cands,
               "metrics": {"mrr": _none(mrr.get(d))}}
