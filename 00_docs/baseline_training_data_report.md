# OpenADMET CYP baselines: what the organizers trained on, and how beatable it is

Prepared 2026-08-06, eleven days before the challenge data drop (2026-08-17).
Everything below was read out of files actually downloaded from HuggingFace and
verified locally; nothing is inferred from the blog post.

## 1. The two "official" baselines are the same data twice

| | `…-chemeleon-v1` | `…-chemeleon-baseline` |
|---|---|---|
| HF revision | `ef24cf941ae21c7d7a64df378a846bd2066eceda` | `12265ee807889b4a3765b308c69d00022b99c176` |
| last modified | 2026-06-16 | 2026-05-06 |
| author (metadata.yaml) | Cynthia Xu | Devany West |
| `model.pth` | 51,669,359 B | 43,305,715 B |
| `X_train.csv` md5 | `69b425cb5e22fe51e57d7b52e15a8ae8` | `69b425cb5e22fe51e57d7b52e15a8ae8` |
| `y_train.csv` md5 | `e456868dc8f5f3b8d2dd6e61d189a0f2` | `0461c10639d6c9993521ba6eac7f53a7` |
| target column case | `…_CYP3A4` (upper) | `…_cyp3a4` (lower) |

**The training data is the same in both repositories.** Stated precisely, because
the md5s are not all equal and the difference is easy to misread:

* `X_train.csv` (the SMILES) is **byte-identical** — same md5 in both.
* `y_train.csv` (the labels) has a **different md5**, but the difference is
  confined entirely to the header line: the two headers are the same string up to
  ASCII case (`OPENADMET_LOGAC50_CYP3A4` vs `OPENADMET_LOGAC50_cyp3a4`), and
  **every byte after the header line is identical** (verified by splitting each
  file at its first newline and comparing the remainders; both files are 77,940 B).
  Parsed and case-normalized, the two frames have the same column order and
  compare equal element-for-element, NaNs included.

So the label *values* are identical and the differing md5 reflects column-name
casing only — not different labels.

This matters because the v1 README claims it "has been updated with **ChEMBL 37**".
Whatever was updated, it was not the shipped training data — v1 is a
re-hyperparameterized model fit to the same 8,068 rows as its predecessor. Treat
"ChEMBL 37" as a claim about provenance, not as evidence of a larger or fresher
training set.

## 2. Exactly what they trained on

`anvil_training/data/` — 8,068 rows, one column of SMILES
(`OPENADMET_CANONICAL_SMILES`), four sparse pIC50 target columns.

| target | n labelled | mean | median | std | min | max | at exactly 4.00 | frac < 5 |
|---|---|---|---|---|---|---|---|---|
| CYP3A4 | 4,800 | 5.456 | 5.30 | 0.862 | 4.00 | 10.30 | 18 (0.4%) | 31.6% |
| CYP2D6 | 2,583 | 5.388 | 5.24 | 0.816 | 4.00 | 10.00 | 13 (0.5%) | 35.4% |
| CYP2C9 | 2,523 | 5.312 | 5.21 | 0.727 | 4.00 |  9.30 |  8 (0.3%) | 34.7% |
| CYP1A2 | 1,361 | 5.576 | 5.43 | 0.953 | 4.00 |  9.90 | 17 (1.2%) | 30.6% |

* All 8,068 SMILES are unique; no duplicate rows.
* The label matrix is **sparse**, as the challenge train set will be: 5,980 rows
  carry exactly one isoform, 1,281 two, 503 three, and only **304 rows have all
  four**. No row is entirely empty.
* Every target is **floored at pIC50 = 4.00** — the same threshold the challenge
  calls unreliable (below the lowest tested dose). The organizers appear to have
  clipped rather than censored: values sit *at* 4.00 rather than below it.
* Labels are ChEMBL `pchembl_value`-like, not 2-decimal-rounded (604 distinct
  CYP3A4 values), consistent with a direct ChEMBL pull.

### Provenance cross-check
An independent ChEMBL pull I ran (targets CHEMBL3356 / CHEMBL3397 / CHEMBL289 /
CHEMBL340, `standard_type=IC50`, exact relations only, validity-flagged records
dropped, replicate-median aggregated) gives per-isoform counts very close to
theirs, which supports the ChEMBL-derived story:

| target | my curation | baseline |
|---|---|---|
| CYP1A2 | 1,571 | 1,361 |
| CYP2C9 | 2,820 | 2,523 |
| CYP2D6 | 2,942 | 2,583 |
| CYP3A4 | 5,446 | 4,800 |

I am consistently ~10–15% larger, most likely because I aggregate replicates by
median rather than dropping them and I do not apply an assay-level filter. Only
63 of my standardized SMILES match their `X_train` strings exactly, but that is a
lower bound only — their SMILES were standardized by a different pipeline than
mine, so string equality understates true compound overlap.

## 3. Architecture and hyperparameters

Both are **multitask ChemProp D-MPNN models initialized from the CheMeleon
foundation encoder** (`from_chemeleon: true`), 4 regression tasks, trained
through Anvil's `LightningTrainer`.

The CheMeleon foundation weights are **not** in either repo — they are fetched at
model-construction time from Zenodo record 15460715
(`chemeleon_mp.pt`, 34,859,448 B, md5 `6a80b54fdb7de37ef0374d302f01e8ce`).
Its hyperparameters, read from the checkpoint: `d_v=72, d_e=14, d_h=2048,
depth=6, bias=False, dropout=0.0, activation=relu, undirected=False`. Loading a
foundation encoder **overrides** the recipe's own `depth`, `message_hidden_dim`,
`messages` and `aggregation` settings — so those fields in the YAMLs are
decorative, and the real encoder is 2048-wide and 6 deep in both models.

| | v1 (`model.json`) | baseline (`model.json`) |
|---|---|---|
| n_tasks | 4 | 4 |
| ffn_hidden_dim | 1024 | 1024 |
| ffn_num_layers | **3** | **1** |
| batch_norm | false | true |
| dropout | 0.1 | 0.2 |
| scheduler | `plateau` (factor 0.5, patience 5) | Noam-like (warmup 2, init/max/final lr 1e-4/1e-3/1e-4) |
| mpnn_lr / ffn_lr | 1e-3 / 1e-3 | — |
| weight decay (mpnn/ffn) | 0 / 1e-4 | — |
| max_epochs (recipe) | 100 | 50 |
| early stopping | patience 20 on `val_loss` | patience 10 on `val_loss` |

Featurizer for both: `ChemPropFeaturizer`, `batch_size=128`, `n_jobs=4`,
`normalize_targets=true` (targets are z-scored; the scaler is folded into the
FFN's output transform on deserialization).

### The split is the headline problem
Both recipes use `ShuffleSplitter` with **`train_size: 1.0, val_size: 0,
test_size: 0`**. There is no validation set and no test set.

Consequences, all verifiable in the shipped files:
* `early_stopping: true` monitors `val_loss`, but no validation loader exists —
  so early stopping cannot have fired. The `-baseline` model ran all 50 epochs.
* `logs/model/version_0/metrics.csv` (only present in `-baseline`) contains
  **only `train_loss_epoch` / `train_loss_step`** across 114 rows / 50 epochs.
  Training loss falls monotonically from **0.763 (epoch 0) to 0.0156 (epoch 49)**
  in normalized-target units — i.e. ~0.016 MSE against a variance of 1.0, which
  is essentially memorization of the training set.
* **Neither baseline ships any held-out metric.** There is no published number
  for how well these models generalize. The v1 repo's `eval.yaml` is literally
  `eval: []`. Any claim that a submission "beats the baseline" has to be
  established on the challenge's own test set, because the organizers never
  measured one.

The `-baseline` README says so itself: "the model performance is quite poor."

## 4. Reproduced inference (see `baseline_predictions.csv`)

Both models were run through the documented Anvil path — `openadmet predict`'s
underlying `openadmet.models.inference.predict()` — on GPU, over the shipped
`compounds_for_inference.csv` (9,999 ZINC compounds, identical file in both
repos, md5 `5c88ee2e64b211891d9a805eb9771f83`).

| target | v1 mean | v1 std | v1 range | baseline mean | baseline std | baseline range |
|---|---|---|---|---|---|---|
| CYP3A4 | 5.180 | 0.416 | 4.14 – 7.81 | 4.980 | 0.648 | **−1.47** – 8.63 |
| CYP2D6 | 5.016 | 0.343 | 4.10 – 7.91 | 4.915 | 0.488 | −0.29 – **16.79** |
| CYP2C9 | 5.249 | 0.355 | 4.33 – 7.00 | 5.055 | 0.528 | 0.26 – 7.45 |
| CYP1A2 | 5.154 | 0.370 | 4.05 – 6.93 | 5.038 | 0.528 | −0.65 – **11.69** |

Two exploitable weaknesses:

**(a) v1 is severely regressed to the mean.** Its prediction std is roughly
*half* the training-label std on every isoform (CYP3A4 0.416 vs 0.862; CYP1A2
0.370 vs 0.953). A model that compresses its dynamic range this much will be
penalized by a *relative* absolute-error metric: MA-ST-RAE normalizes against a
trivial predictor, and a near-constant predictor scores close to that trivial
baseline by construction. Simply predicting with correctly calibrated spread is
worth points here.

**(b) The older baseline emits physically impossible values** — a pIC50 of 16.79
(≈16 femtomolar) and negative pIC50s, 8 predictions outside [0, 11] across the
four tasks. The challenge validator only requires "finite floats", so these
would be *accepted* and then scored. Clipping predictions to a physically sane
window is free insurance.

**(c) The two official baselines disagree with each other.** Pearson correlation
between v1 and `-baseline` predictions on the same 9,999 compounds is only
**0.43–0.52** (Spearman 0.45–0.54). This is despite the same 8,068 training rows
with the same label values (§1) and the same CheMeleon foundation encoder — the
two models differ only in FFN depth, batch-norm, dropout, LR schedule and epoch
budget (§3). Two fits that differ only in hyperparameters yet agree at only r≈0.5
indicate that model variance is large relative to the training signal, which is
direct evidence that an ensemble, and honest uncertainty estimates, will
outperform either single model.

## 5. Overlap risk against the challenge test set

The challenge test set is 750 compounds: 75 potent parents expanded into their
top-10 ECFP4-Tanimoto analogs **from Enamine US in-stock**. So the question is
not "does ChEMBL contain the test compounds" but "does ChEMBL-derived training
chemistry resemble Enamine screening space at all".

Proxy used: the 11,353 unique standardized SMILES in the Octant release's
`will_it_fly_in_mass_spec.tsv` — the actual library Octant screened for this
programme (Enamine DDS10 diversity + FDA-approved), so it is the closest
available stand-in for the test chemistry. ECFP4 (r=2, 2048-bit, binary)
Tanimoto, exactly the similarity the organizers used to pick analogs.

For each Octant/Enamine-like compound, similarity to its nearest neighbour in
the baseline's 8,068-compound training set:

| statistic | value |
|---|---|
| mean | 0.322 |
| median | **0.308** |
| 5th / 95th percentile | 0.235 / 0.429 |
| fraction with NN ≥ 0.40 | 8.3% (943 compounds) |
| fraction with NN ≥ 0.70 | 1.2% (138) |
| exact ECFP4 matches (T = 1.0) | **107 compounds (0.9%)** |

Reverse direction (baseline training compound → nearest Enamine-like compound)
is similar: mean 0.351, median 0.325.

**Reading.** A median nearest-neighbour Tanimoto of 0.31 is *below* the
conventional 0.4 "same series" floor: the baseline's ChEMBL chemistry and Enamine
screening chemistry are largely disjoint neighbourhoods. Two implications pull in
opposite directions and both matter:

1. **Leakage risk is low but not zero.** 107 compounds (0.9%) are exact ECFP4
   matches to a training compound. If any of those land in the test set, models
   trained on ChEMBL have effectively memorized them. Worth checking against the
   real test set on 2026-08-17, but too small to build a strategy on.
2. **Domain-shift risk is the dominant problem.** ~92% of screening-library
   chemistry has no close ChEMBL analog. A model whose only training signal is
   ChEMBL will be extrapolating on most of the test set. This is the strongest
   argument for training on the challenge's own ~1,500-DRC train set as the
   primary signal and using ChEMBL/CheMeleon only as pretraining or as an
   auxiliary task — not the reverse.

See `baseline_enamine_chemspace_overlap.png`.

## 6. Reference models: the bar to actually clear

Because the organizers published no held-out metric, I established one on public
data. Independent ChEMBL pull, replicate-median aggregated, **scaffold-grouped
5-fold CV** (Bemis-Murcko; GroupKFold) — grouped, because the test set is built
from analog series and the live/blind split keeps whole series together, so random
CV would be self-deception. Full table in `reference_model_metrics.csv`; best
configuration per isoform:

| isoform | n | scaffold groups | best model | MAE | RMSE | R² | RAE | Spearman |
|---|---|---|---|---|---|---|---|---|
| CYP1A2 | 1,571 | 855 | LightGBM, ECFP6+RDKit2D+MACCS | 0.525 | 0.698 | 0.453 | 0.675 | 0.664 |
| CYP2C9 | 2,820 | 1,795 | LightGBM, ECFP4 | 0.439 | 0.590 | 0.321 | 0.807 | 0.511 |
| CYP2D6 | 2,942 | 1,689 | LightGBM, ECFP4+RDKit2D | 0.493 | 0.651 | 0.372 | 0.753 | 0.601 |
| CYP3A4 | 5,446 | 3,056 | LightGBM, ECFP6+RDKit2D+MACCS | 0.458 | 0.625 | 0.451 | 0.706 | 0.612 |

Notes that will save time later:

* **RAE is the number to watch**, since the challenge's primary metric is a
  soft-threshold RAE. RAE < 1 beats the trivial mean predictor. The honest,
  scaffold-split RAE is **0.68–0.81** — a 19–32% improvement over predicting the
  mean, no more. CYP2C9 is the hardest (0.807) and CYP1A2 the most tractable
  (0.675), which tracks the narrow label spread of 2C9 (std 0.716).
* **Gradient boosting beats Ridge decisively, and Ridge is worse than useless
  here** — RAE 1.13–1.80, i.e. every Ridge fit is *worse than predicting the
  mean* out-of-scaffold. Linear models on fingerprints do not survive a grouped
  split.
* **RDKit 2D descriptors add real signal on CYP1A2** (RAE 0.711 → 0.678) and
  essentially none on CYP3A4/CYP2C9. Physicochemistry helps most where data is
  scarcest.
* **The experimental noise floor is ~0.16 pIC50** (median spread between
  replicate ChEMBL measurements of the same compound-isoform pair; 90th
  percentile 1.02, n=1,302 pairs with >1 measurement). Our best MAE of ~0.44–0.53
  is roughly 3× that floor, so there is genuine headroom — the models are not yet
  noise-limited.

## 7. Actionable conclusions

1. **Ensemble, and predict with honest spread.** The two official baselines
   correlate at only r≈0.5 with each other; model variance dominates. Ensembling
   also unlocks the `OADMET_STD` columns Anvil leaves empty for single models.
2. **Never submit an unclipped prediction.** Clip to a physically plausible pIC50
   window; the `-baseline` model demonstrates that the validator will happily
   accept a pIC50 of 16.8.
3. **Do not over-fit to ChEMBL.** 92% of Enamine-like screening chemistry has no
   close ChEMBL analog. Use ChEMBL/CheMeleon for pretraining; make the
   challenge's own DRC data the primary signal once it lands.
4. **Validate with scaffold- or series-grouped splits only.** Random CV on
   analog-rich data will over-report by a wide margin, and the leaderboard split
   is explicitly by chemisimilar series.
5. **Exploit the ST-RAE metric's structure.** Error is measured to the nearest
   bound of the ground-truth credible interval, and low-activity compounds are
   downweighted. Calibrated spread and well-placed predictions near pIC50 4
   matter more than raw R².
6. **Re-check for exact-match leakage on 2026-08-17.** 107 library compounds are
   exact ECFP4 matches to baseline training compounds; confirm whether any are in
   the released test set.

## 8. Files

| file | what it is |
|---|---|
| `baseline_predictions.csv` | both baselines' pIC50 predictions for all 4 isoforms on the 9,999 shipped ZINC compounds, reproduced via the documented Anvil path |
| `reference_model_metrics.csv` | 40 rows: 3 feature sets × 3 model families × 4 isoforms + mean-predictor controls, scaffold-grouped 5-fold CV |
| `chembl_cyp_pic50_curated.parquet` | my independent ChEMBL curation, long format (compound × isoform), replicate-median aggregated |
| `chembl_cyp_pic50_wide.parquet` | same, pivoted to a 9,048 × 4 sparse matrix |
| `baseline_enamine_chemspace_overlap.png` | nearest-neighbour Tanimoto distributions and coverage curve |
| `cyp_featurize.py` | the featurization module used for all of the above |

### Data dictionary — `chembl_cyp_pic50_curated.parquet`

| column | type | meaning |
|---|---|---|
| `smiles_std` | str | RDKit-standardized canonical SMILES (largest fragment, normalized, reionized, uncharged; stereo preserved) |
| `isoform` | str | one of CYP1A2 / CYP2C9 / CYP2D6 / CYP3A4 |
| `pIC50` | float | median of replicate measurements, = 9 − log10(IC50 in nM) |
| `n_meas` | int | number of ChEMBL records aggregated into this value |
| `pIC50_spread` | float | max − min across those records (0 when n_meas = 1); an experimental-noise proxy |
| `chembl_id` | str | one representative `molecule_chembl_id` |

Curation filters applied, in order: SMILES present → units `nM` → value present
→ relation exactly `=` (censored `>`/`<` dropped) → no `data_validity_comment` →
0 < value < 1e9 → pIC50 in [3, 11] → RDKit standardization succeeded. 14,861 of
36,015 raw records survive (41.3%), collapsing to 12,779 compound-isoform pairs.
