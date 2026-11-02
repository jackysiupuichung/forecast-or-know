#!/usr/bin/env python3
"""Expanding-window slope, ONE arm, the 3 LARGER models (70B/32B/27B), 1x3.
Top row: raw+MA quarterly per-positive MRR (cutoff line). Bottom row: expanding-
window regression slope (start=8, exp-decay wt). Saved to paper figures dir.

  ARM=a1|a2|a3 python analysis/plot_expanding_arm_large.py   -> $OUTDIR/figures/expanding_large_<arm>.png
"""
import glob, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_outputs import records  # noqa: E402
from src.paths import FIGURES, OT_DIR  # noqa: E402
from src.evaluation.fig_style import clean_axes, paper_style, FS, LW  # noqa: E402
CP_DIR = str(OT_DIR / "sourceId=clinical_precedence")
ARM = os.environ.get("ARM", "a1")
ARM_CFG = {"a1": ("A1", "A1 (no retrieval)"), "a2": ("A2", "A2 (dated graph)"), "a3": ("A3", "A3 (newest graph)")}
ARM_TAG, ARM_LABEL = ARM_CFG[ARM]
FIGURES.mkdir(parents=True, exist_ok=True)
OUT = str(FIGURES / f"expanding_large_{ARM}.png")
START, HALFLIFE, MA = 8, 6.0, 2
# 3 larger models only
MODELS = [("Llama 3.1-70B","llama3.1-70b",2023.95),
          ("Qwen 3-32B","qwen3-32b",2024.45),
          ("Gemma 2-27B","gemma-2-27b",2024.45)]
LETTERS="abc"

def first_trial_dates():
    files=sorted(glob.glob(os.path.join(CP_DIR,"*.parquet")))
    df=pd.concat([pd.read_parquet(f,columns=["targetId","diseaseId","resolvedTrialDate"]) for f in files],ignore_index=True)
    df["resolvedTrialDate"]=pd.to_datetime(df["resolvedTrialDate"],errors="coerce")
    df=df.dropna(subset=["resolvedTrialDate"])
    return df.groupby(["diseaseId","targetId"])["resolvedTrialDate"].min().to_dict()

def series(slug, dates):
    rows=[]
    for d in (r for Y in range(2021, 2027) for r in records(slug, ARM_TAG, Y)):
        cs=d.get("candidate_scores")
        if not cs: continue
        order=sorted(cs,key=lambda c:-c["p_yes"]); lp=np.log(len(cs))
        for rank,c in enumerate(order,1):
            if c["label"]!=1: continue
            dt=dates.get((d["disease_id"],c["target_id"]))
            if dt is None: continue
            rows.append(dict(rr=1.0/rank,q=dt.year+(dt.quarter-1)/4.0,logpool=lp,disease=d["disease_id"]))
    df=pd.DataFrame(rows)
    if df.empty: return None
    r=smf.ols("rr ~ logpool",data=df).fit(cov_type="cluster",cov_kwds={"groups":df["disease"]})
    df=df.assign(adj=df["rr"]-r.params["logpool"]*(df["logpool"]-df["logpool"].mean()))
    df=df[(df["q"]>=2021.0)&(df["q"]<2026.0)]
    g=df.groupby("q")["adj"]; s=g.mean()[g.size()>=15]
    return s.sort_index()

def wls_slope(x,y,xe):
    w=0.5**((xe-x)/HALFLIFE); X=np.vstack([x,np.ones_like(x)]).T; W=np.diag(w)
    return np.linalg.lstsq(W@X,W@y,rcond=None)[0][0]

def main():
    dates=first_trial_dates()
    paper_style()
    fig=plt.figure(figsize=(4.95,2.0))
    gs=GridSpec(2,3,figure=fig,height_ratios=[2,1.2],hspace=0.15,wspace=0.55)
    for k,(mlabel,slug,cut) in enumerate(MODELS):
        s=series(slug,dates)
        aT=fig.add_subplot(gs[0,k]); aB=fig.add_subplot(gs[1,k],sharex=aT)
        if s is None or len(s)<START+2:
            aT.text(0.5,0.5,"insufficient data",ha="center"); continue
        x=s.index.to_numpy(); y=s.to_numpy()
        ma=pd.Series(y).rolling(MA,min_periods=1).mean().to_numpy()
        ends=[]; sl=[]
        for i in range(START-1,len(x)):
            ends.append(x[i]); sl.append(wls_slope(x[:i+1],y[:i+1],x[i]))
        aT.plot(x,y,color="#9aa8e0",lw=LW["data"]); aT.plot(x,ma,color="#1a237e",lw=LW["trend"])
        aT.axvline(cut,color="#1a237e",ls="--",lw=LW["ref"]); aT.set_ylabel("MRR",fontsize=FS["label"])
        aT.set_title(mlabel,fontsize=FS["title"],pad=2)
        aT.tick_params(labelbottom=False,labelsize=FS["tick"])
        aB.bar(ends,sl,width=0.2,color=["#1a237e" if v<0 else "#c62828" for v in sl])
        aB.axhline(0,color="k",lw=0.6); aB.axvline(cut,color="#1a237e",ls="--",lw=LW["ref"])
        aB.set_ylabel("slope",fontsize=FS["label"]); aB.set_xlabel("trial-entry quarter",fontsize=FS["label"])
        aB.tick_params(labelsize=FS["tick"])
        clean_axes(aT,grid_axis=None); clean_axes(aB,grid_axis=None)
    fig.suptitle(ARM_LABEL,fontsize=FS["title"],x=0.02,ha="left",y=1.04)
    fig.savefig(OUT,dpi=300,bbox_inches="tight",pad_inches=0.02); print("Wrote",OUT)

if __name__=="__main__":
    main()
