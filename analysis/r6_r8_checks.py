"""Cutoff checks behind the camera-ready edits of 2026-10-27 (panel items R6, R8, R14).

Same regression as App. E "The floor across the cutoff": per-query values, pooled over the six
models with model fixed effects, standard errors clustered by disease.

R8: A1 floor change across the cutoff (validates App. E: -0.007 [-0.035, +0.021]; A1-TH +0.016),
    then two sensitivity checks: Llama only (documented cutoff) and each model's first
    post-cutoff decision year left out.
R6: dated-graph gain (A2-A1) before vs after the cutoff (no claim about the contrast is made in
    the text; recorded here).
R14: pre-cutoff A2-A1 per model.

Run from the repository root: python analysis/r6_r8_checks.py
"""
import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.paths import AGG_DIR  # noqa: E402
MODELS = ["llama3.1-8b", "llama3.1-70b", "qwen3-8b", "qwen3-32b", "gemma-2-9b", "gemma-2-27b"]
# First decision year after each model's training cutoff (Llama Dec 2023; Qwen/Gemma ~Jun 2024).
FIRST_POST = {"llama3.1-8b": 2024, "llama3.1-70b": 2024, "qwen3-8b": 2025,
              "qwen3-32b": 2025, "gemma-2-9b": 2025, "gemma-2-27b": 2025}
METRICS = ["auroc", "ap", "mrr"]


def load() -> pd.DataFrame:
    q = pd.read_parquet(AGG_DIR / "q_full.parquet")
    q["series"] = q.series.astype(str)
    q["arm"] = q.arm.astype(str)
    key = ["series", "year", "diseaseId", "post"]
    w = (q[q.series.isin(MODELS) & q.arm.isin(["A1", "A2"])]
         .pivot_table(index=key, columns="arm", values=METRICS).dropna().reset_index())
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    th = q[q.series == "ta_breadth"][["year", "diseaseId"] + METRICS]
    w = w.merge(th.rename(columns={m: f"th_{m}" for m in METRICS}), on=["year", "diseaseId"], how="left")
    for m in METRICS:
        w[f"dknow_{m}"] = w[f"{m}_A2"] - w[f"{m}_A1"]   # A2 - A1
        w[f"a1_th_{m}"] = w[f"{m}_A1"] - w[f"th_{m}"]   # A1 - trial history
    return w


def clustered(formula: str, df: pd.DataFrame, term: str):
    m = smf.ols(formula, data=df).fit(cov_type="cluster",
                                      cov_kwds={"groups": pd.factorize(df.diseaseId)[0]})
    lo, hi = m.conf_int().loc[term]
    return m.params[term], lo, hi, m.pvalues[term]


def post_effect(y: str, df: pd.DataFrame, label: str) -> None:
    df = df.assign(post=df.post.astype(int))
    f = f"{y} ~ post + C(series)" if df.series.nunique() > 1 else f"{y} ~ post"
    b, lo, hi, p = clustered(f, df, "post")
    print(f"{label:50s} {b:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.3f} n={len(df)}")


def mean_ci(y: str, df: pd.DataFrame, label: str) -> None:
    b, lo, hi, p = clustered(f"{y} ~ 1", df, "Intercept")
    print(f"{label:50s} {b:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.3f}")


def main() -> None:
    w = load()
    print("== R8: A1 floor, post minus pre (model FE, clustered by disease)")
    post_effect("auroc_A1", w, "all models, A1 AUROC (App. E)")
    post_effect("a1_th_auroc", w, "all models, A1-TH AUROC (App. E)")
    llama = w[w.series.str.startswith("llama")]
    post_effect("auroc_A1", llama, "Llama only, A1 AUROC")
    post_effect("a1_th_auroc", llama, "Llama only, A1-TH AUROC")
    nofirst = w[w.year != w.series.map(FIRST_POST)]
    post_effect("auroc_A1", nofirst, "first post-cutoff year left out, A1 AUROC")
    post_effect("a1_th_auroc", nofirst, "first post-cutoff year left out, A1-TH AUROC")

    print("== R6: dated-graph gain (A2-A1), pooled by period and post minus pre")
    for m in METRICS:
        mean_ci(f"dknow_{m}", w[~w.post], f"pre  {m}")
        mean_ci(f"dknow_{m}", w[w.post], f"post {m}")
        post_effect(f"dknow_{m}", w, f"post-pre {m}")

    print("== R14: pre-cutoff A2-A1 per model")
    pre = w[~w.post].groupby("series")[[f"dknow_{m}" for m in METRICS]].mean().reindex(MODELS)
    print(pre.round(3).to_string())


if __name__ == "__main__":
    main()
