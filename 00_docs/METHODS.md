# Methods — OpenADMET CYP Inhibition Blind Challenge, submissions v1 and v2

Team/username **`avaliev`**. Written 2026-08-18. Every number here was read from a
saved artifact or printed by the run that produced it; nothing is recalled.

Purpose: make both submissions exactly reproducible and make the v1→v2 delta
attributable to a single change at a time.

---

## 0. Environment

| component | version |
|---|---|
| Python | 3.12.13 |
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| lightgbm | 4.7.0 |
| rdkit | 2026.3.4 |
| tabicl | 2.1.1 |
| torch | 2.13.0 |
| scipy | 1.18.0 |
| GPU | NVIDIA GeForce RTX 4080 Laptop (12 GB) |

Conda environment `cyp-models`. `tabicl` was installed via pip on 2026-08-18 and
downloads checkpoint `tabicl-regressor-v2-20260212.ckpt` (114,324,594 B) from the
Hugging Face repo `jingang/TabICL`. Two network dependencies are load-bearing and
easy to miss: `socksio` must be installed for `httpx`/`huggingface_hub` to work
through the sandbox proxy, and `cas-server.xethub.hf.co` must be allowlisted (HF's
Xet content-addressed storage serves the checkpoint bytes).

## 1. Data

Downloaded 2026-08-18 from `openadmet/cyp-challenge-train-test` (dataset last
modified 2026-08-06T20:25:31Z; challenge Space `CURRENT_PHASE = 1`, last modified
2026-08-17T15:42:26Z).

| file | rows × cols | used for |
|---|---|---|
| `cyp-challenge-TRAIN_inhibition.csv` | 4,905 × 18 | regression training (both versions) |
| `cyp-challenge-TEST-BLINDED.csv` | 750 × 2 | prediction target (`SMILES`, `Molecule_Name` only) |
| `cyp-challenge-TRAIN_TDI.csv` | 6,145 × 36 | classification training |
| `cyp-challenge-TRAIN_Emax.csv` | 6,146 × 30 | not used (available: `is_TDI` for all four isoforms) |
| `cyp-challenge-single-concentration-TRAIN.csv` | 17,504 × 12 | not used |

Column conventions, confirmed against the files rather than assumed:

* identifier is `Molecule_Name` (underscore), matching the Space's requirement;
* credible intervals are `{endpoint}_conf_low` / `{endpoint}_conf_high` — **not**
  `ci_lower`/`ci_upper`, and a standard deviation ships as `{endpoint}_std`;
* the direct arm is `{iso}_pIC50_direct_inhibition`; the pre-incubation arm is
  `{iso}_pIC50_TDI_condition`;
* the blinded test file contains **no labels and no intervals**, so ST-RAE cannot
  be computed locally on the test set at all.

Training label counts (direct arm): CYP1A2 1,412 · CYP2C9 1,285 · CYP2D6 1,493 ·
CYP3A4 2,335. The matrix is sparse — 3,596 compounds carry exactly one isoform's
label and only **41** carry all four — whereas the test set is dense across all
four. This asymmetry is what motivates the v2 change (§4).

Leakage checks against the blinded test set: zero exact `Molecule_Name` overlap,
zero exact SMILES overlap, and no compound at ECFP4 Tanimoto 1.0. Test→train
nearest-neighbour Tanimoto has median 0.587, and 95.7 % of test compounds have a
training neighbour at T ≥ 0.5 — consistent with the published design (test =
analogs of training-set potency hits).

## 2. Featurization (shared by both versions)

`featurize(smiles)` builds, per molecule:

1. **ECFP4** — RDKit `MorganGenerator(radius=2, fpSize=2048)`, as a dense binary
   `float32` block.
2. **RDKit 2D descriptors** — all 217 entries of `Descriptors._descList`, computed
   through `MoleculeDescriptors.MolecularDescriptorCalculator`, then
   `nan_to_num` and clipped to ±1e6 (several descriptors overflow `float32` on
   large molecules and would otherwise poison tree splits and standardization).

Concatenated: **2,265 features**. All 4,905 training and 750 test SMILES parsed
without failure.

Murcko scaffolds for grouping use `MurckoScaffoldSmiles`, with a fallback that
strips bond stereo and re-canonicalizes — a minority of molecules raise a
bond-stereo precondition error otherwise. Result: **4,520 scaffold groups** across
4,905 training compounds (5,367 across the 6,145 TDI rows).

## 3. Submission v1 (submitted 2026-08-18 19:01 UTC)

### 3.1 Regression — LightGBM, one model per isoform

Features: the full 2,265-dim block. Four independent `LGBMRegressor`s, each trained
on the rows where that isoform's direct-arm pIC50 is finite. No multitask coupling,
no external data.

```
n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=10,
subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=1.0,
n_jobs=16, verbosity=-1, random_state=0
```

Model selection used 5-fold `GroupKFold` on Murcko scaffolds. Note that
`parent_in_train_folds` from `cyp_splits.py` — designed for exactly this test-set
geometry — **could not be used**: only 16 % of training compounds fall in a
multi-member analog series at T = 0.55, so the series folds come back empty. Train
is a diversity library (internal NN Tanimoto median 0.450, 17 % ≥ 0.55) while test
is analog-dense (median 0.656, 85 % ≥ 0.55). Scaffold grouping is the closest
available surrogate and, because test compounds each have a training analog at
median T = 0.587, it is if anything pessimistic about the test task.

Final predictions come from refitting each isoform on all its labelled rows.

### 3.2 Classification — LightGBM with MCC-tuned thresholds

Trained on the released `CYP2D6_is_TDI` / `CYP3A4_is_TDI` labels (the two scored
endpoints), features as above, 5-fold scaffold-grouped `GroupKFold`.

```
n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20,
subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=1.0,
n_jobs=16, verbosity=-1, random_state=0
```

Decision thresholds were selected to maximize out-of-fold MCC over a 91-point grid
on [0.05, 0.95], then applied to full-data refits:

| isoform | n | positive rate | OOF MCC @ 0.5 | best OOF MCC | chosen threshold | test positive rate |
|---|---|---|---|---|---|---|
| CYP2D6 | 1,497 | 0.216 | 0.075 | **0.098** | 0.48 | 0.029 |
| CYP3A4 | 3,584 | 0.213 | 0.312 | **0.408** | 0.18 | 0.340 |

Threshold tuning is worth 0.096 MCC on CYP3A4 and is where most of the
classification result comes from; the tuned threshold (0.18) is far below 0.5
because MCC on an imbalanced problem rewards recall more than the default split
does.

**Caveat on the threshold grid.** Thresholds were chosen on the same out-of-fold
predictions used to report MCC, so the reported 0.408 / 0.098 are optimistically
biased. The leaderboard's 0.331 macro-MCC on held-out data is the unbiased number.

### 3.3 The TDI labeling rule, recovered from the released labels

Our pre-launch labeler reproduced the organizers' released `is_TDI` labels with
**100 % agreement on 5,080 of 5,081 rows** (1,497/1,497 CYP2D6, 3,583/3,584 CYP3A4;
the one exclusion has a NaN TDI arm) — but only after pinning down two cases the
published FAQ leaves undefined. Both resolve toward *negative*:

* the band `direct < 4` with `4 ≤ tdi ≤ 4.301` is **negative**, not excluded
  (18/18 CYP2D6 and 159/159 CYP3A4 such rows are `False`);
* a **missing direct arm** is **negative regardless of the TDI-arm value** —
  1,249/1,249 CYP3A4 and 4/4 CYP2D6 rows, including 1,055 CYP3A4 rows whose
  TDI-arm pIC50 exceeds 4.301, up to 7.54.

The second is consequential: 34.8 % of released CYP3A4 labels have no direct-arm
DRC and the organizers call them negative. Treating them as inferred positives
instead would move the CYP3A4 positive rate from 21.3 % to 50.8 %. We trained
against the organizers' convention because that is what scores, but a model
learning "measurable TDI with an absent direct arm ⇒ no TDI" is learning an
artifact of data availability, and down-weighting those rows is an open lever.

Both behaviours live in `cyp_tdi.py` under `organizer_rule=True` (default);
`organizer_rule=False` recovers the strict published reading, which can only score
60.7 % of CYP3A4 labels. 109/109 harness tests pass.

### 3.4 v1 results

Local, 5-fold scaffold-grouped CV on the training set:

| isoform | ST-RAE | MAE | R² | Spearman ρ |
|---|---|---|---|---|
| CYP1A2 | 0.8462 | 0.6616 | 0.2484 | 0.4997 |
| CYP2C9 | 0.6864 | 0.4745 | 0.3640 | 0.5895 |
| CYP2D6 | 0.9796 | 0.6194 | 0.1216 | 0.3431 |
| CYP3A4 | 0.5407 | 0.5552 | 0.5708 | 0.7528 |
| **macro** | **0.7632** | 0.5777 | 0.3262 | 0.5463 |

Leaderboard (blinded test, 2026-08-18): **regression MA-ST-RAE 0.8898 ± 0.0295,
rank 8 of 16**; **classification MA-MCC 0.3306 ± 0.0553, rank 1 of 6**.

## 4. Submission v2 (built 2026-08-18, regression only)

Two changes from v1, each measured separately.

### 4.1 Change 1 — TabICL replaces LightGBM

Motivation from the leaderboard rather than from taste: `TabICL-baseline` (0.6755)
is the **only** entry in its own statistical tier, with rank 2 sitting 2.69 σ
behind it, and it is the only significant gap anywhere in the top ten. That points
at model class, so we tested the architecture directly.

TabICL is an in-context-learning transformer for tabular data; it has a pretraining
feature-count regime far below 2,265, so the fingerprint block is compressed:

* `TruncatedSVD` on the 2,048-bit ECFP4 block, fit on train and test fingerprints
  jointly (unsupervised — no labels involved, so no label leakage);
* concatenated with the 217 raw descriptors;
* `StandardScaler` fit on train+test, then `nan_to_num`.

`TabICLRegressor(device="cuda", random_state=0, n_jobs=1)`, same 5-fold
scaffold-grouped folds, same metric. Feature-width sweep:

| representation | dims | ECFP4 variance retained | MA-ST-RAE |
|---|---|---|---|
| descriptors only | 217 | — | 0.7418 |
| SVD-128 + descriptors | 345 | 0.554 | 0.7243 |
| SVD-256 + descriptors | 473 | 0.687 | **0.7239** |
| *(v1 LightGBM, full 2,265)* | 2,265 | 1.000 | *0.7632* |

TabICL beat LightGBM on **all four isoforms** (deltas −0.042, −0.029, −0.051,
−0.033) in 82 s of GPU time. SVD-128 and SVD-256 differ by 0.0004, so fingerprint
compression is **not** the binding constraint — representation capacity is
saturated at this feature family.

### 4.2 Change 2 — cross-isoform stacking

Grounded in the sparse-train/dense-test asymmetry (§1): at test time every compound
has all four isoforms available, but in training only 41 compounds do. So each
isoform's model is given the *other three* isoforms' predictions as four extra
features (never its own — that would be its own label).

Cross-features are out-of-fold predictions where the donor label exists and
full-data-model predictions where it does not.

| isoform | plain TabICL | + cross-isoform | delta |
|---|---|---|---|
| CYP1A2 | 0.8038 | 0.8090 | +0.0052 |
| CYP2C9 | 0.6572 | 0.6391 | −0.0181 |
| CYP2D6 | 0.9284 | 0.9101 | −0.0183 |
| CYP3A4 | 0.5078 | 0.4987 | −0.0091 |
| **macro** | **0.7239** | **0.7142** | **−0.0097** |

**That 0.7142 was optimistic in principle, and we measured by how much.** Audit:
the donor full-data models were trained on every row carrying the donor label, and
those rows can be scaffold-mates of a row held out in the target isoform's fold —
so donor information crosses the CV boundary. The majority of cross-feature values
come from this path (71 %, 74 %, 70 %, 52 % of rows for CYP1A2/2C9/2D6/3A4).

`run_nested_stack.py` removes the leak by rebuilding the cross-features inside each
outer fold, refitting every donor isoform with the held-out scaffold groups deleted
first. Result (`nested_stack_benchmark.csv`):

| isoform | plain TabICL | stacked (optimistic) | stacked (nested, leak-free) | leak |
|---|---|---|---|---|
| CYP1A2 | 0.8038 | 0.8090 | 0.8088 | +0.0002 |
| CYP2C9 | 0.6572 | 0.6391 | 0.6394 | −0.0003 |
| CYP2D6 | 0.9284 | 0.9101 | 0.9112 | −0.0012 |
| CYP3A4 | 0.5078 | 0.4987 | 0.5007 | −0.0020 |
| **macro** | **0.7243** | **0.7142** | **0.7150** | **−0.0008** |

The leak is 0.0008 — negligible, because donor isoforms are labelled on largely
non-overlapping compound sets, so a donor refit rarely loses much when a target's
scaffold groups are removed. **The honest headline numbers are therefore: stacking
gain −0.0093, total v2 gain over v1 −0.0482 (0.7632 → 0.7150).** Quote 0.7150, not
0.7142.

The v2 submission itself was never affected — its cross-features for the 750 test
compounds come from models fit on training data only, and no test label exists to
leak. The concern was purely whether the local estimate of the gain was overstated;
it was, by 0.0008.

### 4.3 v2 prediction statistics

| isoform | v1 mean (sd) | v2 mean (sd) | corr(v1,v2) |
|---|---|---|---|
| CYP1A2 | 4.98 (0.70) | 5.02 (0.64) | 0.878 |
| CYP2C9 | 4.73 (0.71) | 4.85 (0.67) | 0.903 |
| CYP2D6 | 4.63 (0.47) | 4.65 (0.44) | 0.786 |
| CYP3A4 | 4.54 (0.88) | 4.71 (0.98) | 0.930 |

v2 keeps the compressed spread (0.44–0.98 against training label sd 0.78–1.09).
That is deliberate: see §5.2 on why expanding it is wrong under this metric.

## 5. Validation and rejected hypotheses

### 5.1 Submission validation

Both submissions pass our `validate_submission.py` **and** the organizers' own
`validation/activity_validation.py` / `validation/tdi_validation.py` from
`OpenADMET/CYP-Challenge-Tutorial`, checked against the real 750 test identifiers.
Both are 750 rows with exact column names, float64 pIC50s and genuine booleans.

One trap in that repo: `evaluation/config.py` is a PXR-era leftover
(`ENDPOINTS = ["pEC50"]`, plain RAE, TDI metrics that consume *probabilities*).
It is not the CYP scoring reference. The validators are the authoritative part.

SHA-256 prefixes: v1 regression `09a146090596c318`, v1 classification
`948a2dcc9416a78c`, v2 regression `c40b2833cf922588`.

### 5.2 Hypotheses tested and rejected

Recorded because each looked plausible and cost real time.

**Public-data augmentation — rejected.** Adding 33,008 leakage-free public
compounds (ChEMBL/PubChem/Octant corpus, InChIKey14-matched against both challenge
splits and removed; 5 test and 297 train collisions dropped) at sample weight 0.3
*degrades* MA-ST-RAE from 0.7632 to 0.8202, hurting three of four isoforms
(CYP2C9 worst at +0.156). Public CYP data needs domain adaptation, not
concatenation — consistent with our pre-launch finding that a public-only model
scores near no-skill on a real OpenADMET assay.

**Variance-expansion recalibration — rejected.** v1 predictions carry only
0.51–0.91 of the label spread, which looked like fixable miscalibration. Testing
α ∈ [1.0, 2.0] on `pred = mean + (pred − mean)·α`: the optimal α is **1.00 on all
four isoforms**, and the no-oracle variance-matching choice (α = 1.40–2.16)
degrades MA-ST-RAE from 0.7632 to 1.0041. Under an L1-type metric, shrinkage
toward the conditional median is correct behaviour, not a defect.

**"Ranks well but predicts magnitudes poorly" — not supported.** Tempting from
percentile ranks (56th on Spearman vs 50th on ST-RAE) but it does not survive a
test: against the field median we are z = −0.04 on ST-RAE and z = +0.64 on
Spearman. Both within noise.

**Denominator explanation of the local→leaderboard gap — wrong, and backwards.**
Local 0.7632 → leaderboard 0.8898 (+0.1265) while MAE went 0.5777 → 0.9987
(+73 %). So the *larger* test denominator cushioned ST-RAE rather than inflating
it: the test set is harder and wider-spread, as its potency-hit-plus-analog
construction implies.

*Correction to the arithmetic originally given here.* The first version compared a
test spread of 1.122 against a training spread of 0.718. Those two numbers are not
the same quantity. **0.718** is exact — the macro mean absolute deviation of the
training labels, computed directly (per isoform 0.7513 / 0.5926 / 0.6315 / 0.8959,
macro 0.7178), which is the basis of the ST-RAE denominator. **1.122** is a *proxy*,
`MA-MAE / MA-ST-RAE` on the test set, and it is biased upward: ST-RAE's numerator is
soft-threshold error, which is zero inside the credible interval and therefore ≤ raw
absolute error, so dividing raw MAE by it overstates the spread. Measured on
training, the inflation is 1.055× (proxy 0.757 against true 0.718).

De-biasing gives a test spread of ≈ **1.064**, so the honest comparison is training
0.718 versus test ≈ 1.064 — the test distribution is about **48 %** wider, not the
56 % the original figures implied. The direction and the conclusion are unchanged;
only the magnitude was overstated. Where an exact test spread is needed it cannot be
computed at all, because the blinded test file ships no labels.

## 6. Reproduction

```
# data
python -c "import urllib.request as u; [u.urlretrieve(
  'https://huggingface.co/datasets/openadmet/cyp-challenge-train-test/resolve/main/'+f,
  'challenge_data/'+f) for f in [...]]"

# v1: featurize -> LGBMRegressor per isoform -> refit on all rows -> predict test
# v1 TDI: LGBMClassifier per isoform, MCC-tuned threshold on scaffold-grouped OOF
# v2: SVD-256 + descriptors -> TabICLRegressor -> cross-isoform stack -> refit -> predict
python run_tabicl.py          # TabICL CV + test predictions
python run_nested_stack.py    # leak-free stacking estimate
python mods/validate_submission.py --track regression --file submission_regression_v2.parquet
```

Fixed seeds throughout (`random_state=0`; `GroupKFold` is deterministic). The one
irreducible source of variation is TabICL's GPU inference kernels.

## 7. Artifact index

| file | contents |
|---|---|
| `submission_regression.parquet` | v1 regression, submitted, scored 0.8898 |
| `submission_classification.parquet` | v1 classification, submitted, scored 0.3306 |
| `submission_regression_v2.parquet` | v2 regression, validated, not yet submitted |
| `method_provenance.json` | versions, hyperparameters, checksums, prediction stats |
| `real_data_benchmark.csv` | v1 local CV, random vs scaffold-grouped |
| `tabicl_benchmark.csv` | TabICL per-isoform CV |
| `tabicl_stacked_benchmark.csv` | cross-isoform stacking (optimistic) |
| `nested_stack_benchmark.csv` | cross-isoform stacking (leak-free) |
| `public_augmentation_benchmark.csv` | rejected public-data experiment |
| `subfloor_exploit_check.csv` | rejected sub-floor gaming experiment |
| `tdi_cv_report.csv` | TDI thresholds and OOF MCC |
| `leaderboard_tiering_regression.csv`, `leaderboard_tiering_classification.csv` | pairwise significance |
| `launch_day_findings.json` | schema checks, TDI rule recovery, leakage audit |
| `cyp_tdi.py`, `cyp_metrics.py`, `cyp_splits.py`, `validate_submission.py`, `test_harness.py` | harness |
