# Did the Model Forecast It, or Already Know It?

Code and data for **"Did the Model Forecast It, or Already Know It? Evaluating Large Language Models for
Novel Drug-Target Prioritisation with a Temporal Knowledge-Graph Oracle"** (Pui Chung Siu, Yuan Liang,
Claudia Cabrera, Mani Mudaliar, Arkaitz Zubiaga), NeurIPS 2026 Workshop on AI for Drug Discovery (AI4DD).

Each target–disease hypothesis is a dated event: the pair's first clinical trial. For a disease *d* and a
decision year *Y* (2021–2026), six open models score candidate targets, with the snapshot year *C = Y − 1*:

| Arm | What accompanies the pair |
|---|---|
| A0 | identifiers only |
| A1 | target and disease descriptions, no retrieval |
| A2 | paths from the dated knowledge graph G[C] |
| A3 | paths from the newest graph (2026) |

Positives are the targets whose first trial for *d* falls in *Y*. Negatives are drawn without future
information (time-aware): ten per positive, from targets already trialled in *d*'s therapeutic area by *C*.
The sets hold 1,373 queries, 6,394 positives and 70,334 scored pairs.

## Reproduce the tables and figures (no GPU)

```bash
pip install -r requirements.txt   # Python >= 3.11
bash analysis/run_all.sh    # writes outputs/tables/*.tex and outputs/figures/*.png
```

Run everything from the repository root. `run_all.sh` unpacks the candidate sets and recomputes the
per-query metrics from the shipped model outputs. It then writes Tables 1, 3, 4 and 5 and every figure of
the paper (Figure 1 is a diagram), and prints the numbers quoted in the text. With the versions in
`requirements.txt`, the table bodies match the paper's LaTeX byte for byte, and the figures match the
paper's PNG files byte for byte.

## What is shipped

| Path | Content |
|---|---|
| `data/evaluation/ratio10/test_yr<Y>/` | candidate sets (`trial_entry_ratio10.csv.gz`), `stats.json`, `dropped_queries.csv` |
| `data/predictions/` | every model output used in the paper, in compact form: per candidate the label, p(Yes) and the Yes/No log-probabilities; per disease the runner's MRR |
| `data/opentargets_min/` | therapeutic areas of the evaluated diseases and the clinical-precedence records of the positive pairs |
| `data/aggregates/genetic_scores.csv.gz` | genetic-association evidence of each pair at *C* (Table 4) |
| `data/aggregates/own_edge_disclosure.csv` | share of positives and negatives whose A3 (and A2) paths contain the pair's own trial edge |

The K-path caches (about 2.6 GB) and the graph snapshots are not in the repository.

## Rerun the pipeline

| Step | Command | Needs |
|---|---|---|
| 1. Graph snapshots | `SOURCE_GRAPH=... bash scripts/build_snapshots.sh` | the full THBKG graph |
| 2. Candidate sets | `python scripts/build_ratio_eval.py --split test --years 2021 2022 2023 2024 2025`, then the same command with `--years 2026` | Open Targets 26.03 evidence-dated layer (`OT_DIR`), `snapshot_2015` |
| 3. Identifiers-only sets (A0) | `python scripts/make_ids_sets.py` | step 2 |
| 4. K-path caches | `OT_DIR=... ARM=A2 bash scripts/build_kpaths.sh`, then the same with `ARM=A3` | steps 1–2; 13–26 CPU-hours per arm |
| 5. Inference | `MODEL=... SLUG=... ARM=A0\|A1\|A2\|A3 bash scripts/run_inference.sh` | one A100 80 GB; about 163 GPU-hours for all runs |
| 6. Tables and figures | `bash analysis/run_all.sh` | step 5 outputs in `results/` (read in place of the shipped copy) |

`scripts/run_inference.sh` lists the model revisions and the exact arguments of every run.

Step 2 must build 2026 in a separate call, as in the paper. A single seeded generator is shared across the
years of one call, so building all six years at once gives different 2026 negatives.

## Notes on the runs reported in the paper

- **A3 path annotations.** A2 strips the `[score=…]` annotation from each path edge, but A3 shows it
  (`src/reasoning/prompts.py`: `strip_scores = (ablation == "A2")`), so A3 − A2 combines the newest
  graph with the visible scores (paper, Appendices B and D).
- **Stated dates.** The A0 and A1 prompts state "December 2023" as the knowledge date for every model and
  year. A2 states December of *C*; A3 states December 2026.
- **Not reproduced here.**
  - The precision check (llama.cpp Q4_K_M against transformers, at most 0.019 AUROC) comes from earlier
    runs on the reviewed version's sets with a separate llama.cpp pipeline.
  - The package versions of the inference environment at run time were not recorded; `pyproject.toml`
    lists that environment's versions today.
- **Snapshots.** The 2020–2024 snapshots were sliced from an earlier build of the THBKG graph than
  2025–2026. Re-slicing today's graph may differ slightly.

## Data licences and attribution

- Open Targets Platform data are CC0.
- The candidate sets carry UniProt function annotations (CC BY 4.0) as target descriptions, and disease
  descriptions from the ontologies used by Open Targets (EFO and MONDO).
- Trial-entry labels derive from Open Targets clinical-precedence records, which trace to
  ClinicalTrials.gov (public domain).

## Licence

The code is MIT-licensed (`LICENSE`). `src/evaluation/figure_style_kernel.py` is vendored under
Apache-2.0 (`NOTICE`).

## Citation

```bibtex
@inproceedings{siu2026forecast,
  title     = {Did the Model Forecast It, or Already Know It? Evaluating Large Language Models for Novel
               Drug-Target Prioritisation with a Temporal Knowledge-Graph Oracle},
  author    = {Siu, Pui Chung and Liang, Yuan and Cabrera, Claudia and Mudaliar, Mani and Zubiaga, Arkaitz},
  booktitle = {NeurIPS 2026 Workshop on AI for Drug Discovery: Bridging the Translation Gap},
  year      = {2026}
}
```
