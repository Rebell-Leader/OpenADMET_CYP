# OpenADMET CYP Inhibition Blind Challenge — full reproducible package

Entry of team **`avaliev`** to the [OpenADMET CYP inhibition blind
challenge](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/):
predicting direct-inhibition pIC<sub>50</sub> for CYP1A2 / CYP2C9 / CYP2D6 / CYP3A4
and time-dependent-inhibition (TDI) class for CYP2D6 / CYP3A4, from structure alone.

This repository is the complete package — code, data, every experiment we ran
(including the failures), the submissions, the figures, and a paper draft. The
negative results are the more useful half: each one closes a direction with a number
attached.

## Results

| track | best submission | local CV | leaderboard | rank |
|---|---|---|---|---|
| Regression (MA-ST-RAE, lower better) | v2 (scored) | 0.7150 | **0.8284** | 20 / 45 |
| Regression | v4 (best, unsubmitted) | **0.6868** | ~0.807 projected | — |
| TDI classification (MA-MCC, higher better) | v1 | — | **0.3306** | **1 / 6** |

The classification rank-1 is real but not a demonstrated advantage: all six entries
on that board are statistically indistinguishable (largest z = 0.76).

Local scaffold-grouped CV runs **+0.113 to +0.127** optimistic relative to the
leaderboard, measured on two scored submissions. Add ≈ 0.12 to any local MA-ST-RAE
before comparing to a board score.

## What actually moved the needle

Ten approaches were benchmarked under identical scaffold-grouped folds. Each Δ below
is against an explicitly named comparator — they are *not* all against one baseline,
and they do not sum:

| change | Δ MA-ST-RAE | vs. what | note |
|---|---|---|---|
| Blending two model families (TabICL × multitask) | **−0.031** | best single parent (multitask, 0.7175) | beats *both* parents on *every* isoform |
| In-context tabular model instead of gradient boosting | −0.039 | LightGBM, same features | reproduced on two disjoint feature sets |
| Multitask shared trunk vs. four independent fits | −0.016 | single-task MLP, same width | isolates sharing, not "a net helped" |
| Auxiliary supervision from the single-concentration screen | −0.015 | same trunk without aux heads | as a *task*, not a feature |
| Cross-isoform stacking | −0.008 | TabICL n=32 (0.7226) | leak-free nested estimate |
| Screen pseudo-labels (floor-clipped, 2 of 4 isoforms) | −0.008 | TabICL n=32 | a-priori selection rule |
| Ensembling 8 → 32 estimators | −0.001 | TabICL n=8 | saturated |
| Frozen CheMeleon embeddings | **+0.010** | ECFP4 + descriptors, same heads | representation axis exhausted |
| Screen distillation (predicted log2fc as a feature) | **+0.036** | TabICL n=32 | see the leak warning below |
| Public ChEMBL/PubChem augmentation | **+0.057** | challenge data only | hurts 3 of 4 isoforms |

For reference, the full local progression across our four submissions was
0.7632 → 0.7150 → 0.7111 → **0.6868**.

## Three findings worth reusing elsewhere

**1. A dense auxiliary label on the *same* compounds adds nothing once it must be
predicted from structure.** The single-concentration screen correlates with pIC50 at
|ρ| = 0.83–0.94 and covers 89 % of the label matrix. Distilling it into a feature made
things *worse* (+0.036): the surrogate is a deterministic function of inputs the model
already has. The same data helps as extra *rows* (pseudo-labels, −0.008) or as an
auxiliary *task* (−0.015), because those add compounds or gradient rather than a lossy
re-encoding.

**2. That near-miss produced a 0.557 leak that looked like winning.** Fitting the
distillation donors on all rows — so each donor had seen the held-out compound's own
measured screen value — scored ST-RAE **0.2771** (R² 0.834) on CYP1A2, against
**0.8342** for the identical experiment run nested. Anyone building surrogate features
on a shared compound set should nest before believing the number.

**3. Gains from the same information compete; gains from different function classes
compose.** Pseudo-labels (−0.008) and cross-isoform stacking (−0.008) together gave
only −0.003 — both inject screen-derived estimates, so the second is redundant.
Blending TabICL with the multitask network gave −0.031 and beat both parents on every
isoform, despite error correlations of 0.89–0.94. That is variance reduction, not
decorrelation.

## Repository layout

Directory numbering is reading order; scripts resolve paths against the repo root, so
the names are load-bearing — don't rename them.

```
00_docs/           four memos (experiment ledger, data findings, improvement plan,
                   OpenBind assessment) + METHODS, strategy, go-live runbook, spec
01_env/            pip lock, hardware record, REBUILD.md, checkpoint fetch script
02_data/
  challenge_raw/   the five official challenge files
  *.parquet        curated public corpus (37k compounds), Octant DRC with credible
                   intervals, TDI auxiliary labels
03_code/           metric harness, splits, featurisation, models, experiment runners,
                   the organizers' own validators for cross-checking
04_experiments/    every benchmark CSV + findings JSON (cache/ is gitignored)
05_submissions/    all four submissions + SUBMISSION_LOG.md
06_figures/        figures, also used by the paper draft
07_leaderboards/   dated board snapshots
paper/             paper draft
```

## Reproducing

```bash
# 1. environment  (see 01_env/REBUILD.md for the two non-obvious dependencies)
pip install -r requirements.txt

# 2. rebuild the feature caches (~3 min; gitignored, 61 MB)
python 03_code/build_features.py

# 3. metric harness must pass before trusting any number
python -m pytest -q 03_code/test_harness.py        # 109 tests

# 4. any experiment, e.g.
python 03_code/run_multitask.py                    # multitask ablations
python 03_code/run_v3_experiments.py               # ensembling + full-width
```

`build_features.py` reproduces the cached arrays **bit-exactly**, which is how the
numbers above were verified after a workspace reset.

### Two environment traps that cost us time

* `socksio` must be installed *before* the process starts for the Hugging Face
  checkpoint fetch to work — the HTTP stack caches its capability check, so installing
  it mid-session is not enough.
* `zenodo.org` and `cas-server.xethub.hf.co` must both be reachable; the CheMeleon
  encoder and the TabICL checkpoint live there. `bash 01_env/fetch_checkpoints.sh`
  fetches and checksums the former.

### Methodological caveats, stated plainly

* **Transductive preprocessing — measured, and it gave us nothing.** The SVD basis and
  standardiser are fitted on train and blinded-test features stacked together
  (`build_features.py --fit-on all`, the default, because it is what every number here
  used). No test *label* exists to leak and the test SMILES are public, but it is not a
  strictly inductive protocol, so we measured it. Refitting strictly on train
  (`--fit-on train`) and re-running the identical folds makes the score **slightly
  better**, not worse: macro 0.7818 → 0.7740 (−0.0079; per isoform −0.020, −0.001,
  −0.011, 0.000; LightGBM instrument, same folds). **No result in this repository
  depends on test-set access** — if anything the transductive fit cost us a little.
  Full table: `04_experiments/transductive_check.csv`.
* **The organizers' validator is CSV-only.** `activity_validation.py` hardcodes
  `pd.read_csv` and rejects a parquet submission with a UTF-8 decode error, even
  though the Space itself scores parquet fine. Validate the `.csv` copy locally; both
  formats ship in `05_submissions/`.
* **v4 is unsubmitted.** Its blend weights are chosen by nested inner CV, but the
  decision to blend those two particular families was made after seeing both
  benchmarks. Treat 0.6868 as the best-supported local number, not as a board result.
* **TabPFN was never tested** — it requires registration and a `TABPFN_TOKEN`. So the
  architecture finding is "TabICL beats gradient boosting here", not "in-context
  learners beat gradient boosting".

## Data provenance and licensing

Challenge data in `02_data/challenge_raw/` is redistributed from the organizers'
[Hugging Face dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)
and remains under its original terms. The curated public corpus derives from ChEMBL
(CC BY-SA 3.0), PubChem (public domain), and the Octant CYP release. Code in this
repository is MIT-licensed (see `LICENSE`). No proprietary data was used in any
submission.
