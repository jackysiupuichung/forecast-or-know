"""Data locations. Run every script from the repository root; override any of these with the
environment variable of the same name.

OT_DIR    Open Targets 26.03 evidence-dated layer. The repository ships a small extract
          (data/opentargets_min: disease therapeutic areas and first-trial dates of the evaluated
          pairs), enough for the figures. Rebuilding candidate sets or K-paths needs the full release.
SNAP_DIR  Yearly graph snapshots snapshot_<C>_datatype.pt (built by src/graph/build_snapshot.py).
AGG_DIR   Per-query and per-candidate aggregates (shipped; rebuilt by analysis/run_all.sh).
OUTDIR    Where tables (OUTDIR/tables) and figures (OUTDIR/figures) are written.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OT_DIR = Path(os.environ.get("OT_DIR", REPO / "data" / "opentargets_min"))
SNAP_DIR = Path(os.environ.get("SNAP_DIR", REPO / "data" / "snapshots"))
AGG_DIR = Path(os.environ.get("AGG_DIR", REPO / "data" / "aggregates"))
OUTDIR = Path(os.environ.get("OUTDIR", REPO / "outputs"))
TABLES = OUTDIR / "tables"
FIGURES = OUTDIR / "figures"


def snapshot(cutoff_year: int) -> Path:
    return SNAP_DIR / f"snapshot_{cutoff_year}_datatype.pt"
