#!/usr/bin/env python3
"""Identifiers-only (A0) copies of the candidate sets: both the target and the disease description are
blanked (the prompt renders "(no description available)"); every other column is unchanged.

    python scripts/make_ids_sets.py   -> data/evaluation/ratio10_ids/test_yr<Y>/trial_entry_ratio10.csv
"""
import argparse
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-root", default="data/evaluation/ratio10")
    p.add_argument("--years", nargs="+", type=int, default=list(range(2021, 2027)))
    args = p.parse_args()
    for y in args.years:
        src = Path(args.src_root) / f"test_yr{y}"
        dst = Path(f"{args.src_root}_ids") / f"test_yr{y}"
        dst.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(src / "trial_entry_ratio10.csv")
        df["target_description"] = ""
        df["disease_description"] = ""
        df.to_csv(dst / "trial_entry_ratio10.csv", index=False)
        print(f"test_yr{y}: {len(df)} rows -> {dst}")


if __name__ == "__main__":
    main()
