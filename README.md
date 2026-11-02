# Did the Model Forecast It, or Already Know It?

Code and data for **"Did the Model Forecast It, or Already Know It? Evaluating Large Language Models for
Novel Drug-Target Prioritisation with a Temporal Knowledge-Graph Oracle"** (Pui Chung Siu, Yuan Liang,
Claudia Cabrera, Mani Mudaliar, Arkaitz Zubiaga), NeurIPS 2026 Workshop on AI for Drug Discovery (AI4DD).

## Overview

When a language model ranks drug targets for a disease and is scored against past outcomes, a correct
ranking can mean two things. The model may have forecast the outcome, or the outcome may already have
been in its training data or in the evidence it was given. We separate the two by treating each
target–disease hypothesis as a dated event, the pair's first clinical trial, and by controlling the date
of the evidence the model sees.

**Task.** A query is a disease *d* and a decision year *Y* (2021–2026). The evidence cutoff is the end of
the previous year, *C = Y − 1*. The positives are the targets whose first trial for *d* falls in *Y*. Each
positive is matched with ten negatives, drawn from targets that had been trialled by *C* for a disease in
*d*'s therapeutic area. The negatives use no information from after *Y*, so a target that enters a trial
for *d* in a later year can still be a negative. Across the six years the test sets hold 1,373 queries,
6,394 positives and 70,334 scored pairs.

**Models and scoring.** Six open-weight instruction-tuned models (Llama 3.1 8B and 70B, Qwen3 8B and 32B,
Gemma 2 9B and 27B) are asked whether the target will enter its first trial for *d* in *Y*. Each query's
candidates are ranked by the model's probability of answering Yes rather than No.

**Conditions.** The conditions differ only in what accompanies the target–disease pair:

| Condition | Information given with the pair |
|---|---|
| A0 | gene symbol and disease name with their identifiers; no descriptions |
| A1 | target and disease descriptions; no retrieval |
| A2 | A1 plus up to eight paths (at most three hops) from the knowledge graph as dated at *C*, written G[*C*] |
| A3 | A1 plus the same kind of paths from the newest graph (2026) |

The paths come from THBKG, a temporal biomedical knowledge graph that can be queried as it stood at any
date. A2 − A1 (the knowledge gap) measures what evidence available at the decision adds. A3 − A2 (the
forecasting gap) measures what evidence from after the decision adds.

**Evaluation.** Each condition is compared with *trial history*, a model-free baseline that ranks a
target by the number of diseases in *d*'s therapeutic area it had been trialled for by *C*. AUROC,
average precision and MRR are computed per query and averaged over queries, with 95% confidence
intervals from 2,000 query resamples.

## Main findings

- Without retrieval (A1), the models rank about as well as trial history, both before and after their
  training cutoffs.
- Paths from the dated graph (A2) add at most a small gain over A1.
- Paths from the newest graph (A3) lift five of the six models clearly above trial history. These paths
  usually contain the answer. For 95–97% of positives they include the pair's own trial edge, which is
  dated after the decision, against under 1% of negatives
  ([`data/aggregates/own_edge_disclosure.csv`](data/aggregates/own_edge_disclosure.csv)).

## Reproducing the tables and figures

```bash
pip install -r requirements.txt   # Python >= 3.11
bash analysis/run_all.sh          # writes outputs/tables/*.tex and outputs/figures/*.png
```

Run both commands from the repository root. `run_all.sh` unpacks the candidate sets, recomputes the
per-query metrics from the model outputs in `data/predictions/`, and prints the numbers quoted in the
text. It writes Tables 1, 3, 4 and 5 and every data figure in the paper (Figure 1 is a diagram). With
the package versions pinned in `requirements.txt`, the table bodies and figure files are byte-identical
to those in the paper. No model is run in this step.

## Repository contents

| Path | Content |
|---|---|
| `src/` | graph snapshots, path retrieval and serialisation, prompts, LLM client, metrics |
| `scripts/` | the pipeline from graph snapshots to model inference (see below) |
| `analysis/` | per-query metrics, tables and figures (`run_all.sh` runs them in order) |
| `tests/` | unit tests for prompts, answer parsing, path extraction, ranking and metrics (`pytest tests`; needs `pytest` and `requirements-graph.txt`) |
| `data/evaluation/ratio10/test_yr<Y>/` | candidate set for year *Y* (`trial_entry_ratio10.csv.gz`), its summary (`stats.json`) and the queries dropped for having too few negatives (`dropped_queries.csv`) |
| `data/predictions/` | the model outputs behind every result in the paper. For each candidate: label, p(Yes), and the log-probabilities of Yes and No. For each query: the MRR recorded at inference time |
| `data/opentargets_min/` | therapeutic areas of the evaluated diseases, and the Open Targets clinical-precedence records of the positive pairs |
| `data/aggregates/genetic_scores.csv.gz` | genetic-association evidence for each pair at *C* (Table 4) |
| `data/aggregates/own_edge_disclosure.csv` | per year, the share of positives and negatives whose A2 or A3 paths contain the pair's own trial edge |

The graph snapshots and the cached paths (about 2.6 GB) are not included. Steps 1 and 4 below rebuild
them.

## Rerunning the full pipeline

| Step | Command | Requires |
|---|---|---|
| 1. Graph snapshots | `SOURCE_GRAPH=... bash scripts/build_snapshots.sh` | the full THBKG graph |
| 2. Candidate sets | `python scripts/build_ratio_eval.py --split test --years 2021 2022 2023 2024 2025`, then the same command with `--years 2026` | the evidence-dated layer of Open Targets 26.03 (set `OT_DIR`); `snapshot_2015`, used only for its list of target and disease nodes |
| 3. Identifiers-only sets (A0) | `python scripts/make_ids_sets.py` | step 2 |
| 4. Path caches | `OT_DIR=... ARM=A2 bash scripts/build_kpaths.sh`, then the same with `ARM=A3` | steps 1–2; 13–26 CPU-hours per condition |
| 5. Inference | `MODEL=... SLUG=... ARM=A0\|A1\|A2\|A3 bash scripts/run_inference.sh` | one 80 GB A100; about 163 GPU-hours for all runs |
| 6. Tables and figures | `bash analysis/run_all.sh` | step 5 |

`scripts/run_inference.sh` lists the pinned model revisions and the exact arguments of every run. Its
outputs go to `results/`. When `results/` exists, step 6 reads it instead of `data/predictions/`.

In step 2, build 2026 in its own call, as we did for the paper. All years in one call share a single
seeded random generator, so building the six years together draws different negatives for 2026.

## Known differences and limitations

- A3 prompts show an evidence-score annotation on each path edge (`--relation[score=0.84]-->`), and A2
  prompts strip it (`strip_scores = (ablation == "A2")` in `src/reasoning/prompts.py`). The forecasting
  gap A3 − A2 therefore combines the newer graph with the visible scores (paper, Appendices B and D).
- The prompts state a knowledge date. A0 and A1 state December 2023 for every model and year, A2 states
  December of *C*, and A3 states December 2026.
- The paper's precision check (llama.cpp Q4_K_M against transformers, at most 0.019 AUROC; Appendix I)
  cannot be rerun from this repository. It used earlier runs on the submitted version's candidate sets
  and a separate llama.cpp pipeline.
- The package versions of the inference environment were not recorded at run time.
  `requirements-inference.txt` lists the versions in that environment today.
- The 2020–2024 snapshots were sliced from an earlier build of THBKG than the 2025 and 2026 snapshots.
  Slicing today's graph may give slightly different snapshots.

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
