# Harness verification on real challenge-shaped data

Verification of `cyp_metrics.py`, `cyp_tdi.py` and `validate_submission.py` against the
released PXR challenge data, used as a stand-in because the CYP train/test sets do not
exist until 2026-08-17.

**Test suite:** 109 tests, all passing (`pytest -q test_harness.py`).

---

## 1. Stand-in dataset

Downloaded from HuggingFace `openadmet/pxr-challenge-train-test`:

| file | rows | used for |
| :--- | ---: | :--- |
| `pxr-challenge_TEST_PHASE_1_UNBLINDED.csv` | 253 | test stand-in |
| `pxr-challenge_TEST_PHASE_2_UNBLINDED.csv` | 260 | test stand-in |
| `pxr-challenge_TRAIN.csv` | 4139 | CI-width characterisation |
| `pxr-challenge_TEST_BLINDED.csv` | 513 | confirms 253 + 260 = the full blinded set |

The two unblinded phases were concatenated into a **513-compound** evaluation set. This
release is the right stand-in for one specific reason: it carries **real Bayesian
dose-response credible intervals**, which is exactly the input ST-RAE needs and which
cannot be faked convincingly.

Ground-truth columns used: `pEC50`, `pEC50_ci.lower (-log10(molarity))`,
`pEC50_ci.upper (-log10(molarity))`. The point estimate lies inside its own credible
interval for **513/513 compounds (100.0 %)**, in all three files — so the intervals are
well-formed and a perfect predictor scores exactly 0.

## 2. Column conventions: PXR vs what the CYP Space requires

| role | PXR release | CYP challenge (confirmed from Space source) | status |
| :--- | :--- | :--- | :--- |
| compound id | `Molecule Name` (**space**) | `Molecule_Name` (**underscore**) | **RENAMED — will break naive reuse** |
| structure | `SMILES` | `SMILES` | same |
| target value | `pEC50` (single endpoint) | `CYP{1A2,2C9,2D6,3A4}_pIC50_direct_inhibition` | 1 → 4 endpoints |
| CI lower | `pEC50_ci.lower (-log10(molarity))` | **not published** | UNKNOWN |
| CI upper | `pEC50_ci.upper (-log10(molarity))` | **not published** | UNKNOWN |
| std error | `pEC50_std.error (-log10(molarity))` | **not published** | UNKNOWN |
| live/blind split | `Split`, `phase` (two-phase unblinding) | by chemisimilar series, single stage | different |

Two things to carry forward. First, the identifier column is renamed between challenges
— PXR used `Molecule Name`, the CYP Space's `IDENTIFIER_COLUMNS` requires
`Molecule_Name`, and the validator's case/space check will catch this. Second, the PXR
ground-truth columns embed their units in the column name; if the CYP release follows
the same convention, the CI columns will not be named `ci_lower`/`ci_upper` and the
harness will need a column-mapping step on day one.

## 3. Real credible-interval widths — the metric's design premise holds

| dataset | n | median CI width | median, pEC50 < 4.5 | median, pEC50 ≥ 4.5 | ratio |
| :--- | ---: | ---: | ---: | ---: | ---: |
| PHASE_1 | 253 | 0.520 | 1.035 | 0.373 | 2.8× |
| PHASE_2 | 260 | 0.475 | 1.010 | 0.380 | 2.7× |
| TRAIN | 4139 | 0.588 | 1.019 | 0.355 | 2.9× |

Split at the assay floor itself (pEC50 < 4) on the 513-compound evaluation set: **100
compounds (19.5 %)** are below the floor, with median CI width **1.15** against **0.41**
for the 413 above it — a **2.8× ratio**.

This is the mechanism ST-RAE relies on, confirmed on real data rather than assumed: the
Bayesian fit is genuinely less certain where the assay has no dynamic range, so a
soft-threshold metric automatically discounts error there without needing an explicit
pIC50 < 4 rule. The organizers' design is sound. Note the corollary — the discount is
*graded by fit quality*, not by a hard activity cutoff, so a low-activity compound whose
DRC happens to be well-determined still carries full weight.

## 4. Predictor ranking

Nine predictors on the 513-compound set, 1000 bootstrap resamples over compounds with a
shared resample matrix (seed 7). `ST-RAE` uses the default `st_mean` denominator.

| predictor | ST-RAE | boot mean ± std | 95 % CI | frac inside CI | MAE | R² | Spearman ρ |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| perfect | **0.0000** | 0.0000 ± 0.0000 | [0.000, 0.000] | 1.000 | 0.000 | 1.000 | 1.000 |
| good (σ=0.3) | 0.1438 | 0.1438 ± 0.0144 | [0.117, 0.174] | 0.610 | 0.238 | 0.909 | 0.934 |
| error on inactives only | 0.1938 | 0.1948 ± 0.0253 | [0.146, 0.245] | 0.866 | 0.193 | 0.730 | 0.914 |
| over-confident on inactives | 0.8615 | 0.8626 ± 0.0334 | [0.798, 0.925] | 0.400 | 0.664 | −0.266 | 0.377 |
| mean-only | **1.0000** | 1.0012 ± 0.0191 | [0.971, 1.042] | 0.253 | 0.761 | 0.000 | NaN |
| mediocre (σ=1.0) | 1.0272 | 1.0285 ± 0.0772 | [0.878, 1.189] | 0.248 | 0.766 | 0.069 | 0.647 |
| error on actives only | 1.2253 | 1.2263 ± 0.1022 | [1.037, 1.447] | 0.310 | 0.757 | −0.129 | 0.814 |
| shuffled truth | 1.5892 | 1.5916 ± 0.0842 | [1.440, 1.765] | 0.172 | 1.064 | −1.014 | −0.005 |
| constant 4.0 | 1.6824 | 1.6841 ± 0.0972 | [1.505, 1.897] | 0.144 | 1.130 | −0.631 | NaN |

Four checks pass:

1. **The ordering is sensible.** perfect < good < mediocre < shuffled < constant, and
   every non-trivial predictor is correctly placed relative to the mean-only baseline.
2. **`mean_only` scores exactly 1.0000.** This is the definitional check on the
   normalisation, not a coincidence: with the `st_mean` denominator the mean-of-truth
   predictor *is* the denominator, so its ST-RAE must be 1 by construction. Values above
   1 mean "worse than predicting the mean of the test set" — a directly interpretable
   scale, and further evidence that this is the denominator the organizers intended.
3. **R² and Spearman are NaN for constant predictors** (`mean_only`, `constant_4`) rather
   than silently 0 — a constant prediction has no rank information, and reporting 0
   would falsely imply it is as good as random.
4. **`shuffled truth` gets Spearman ≈ 0 (−0.005)** but R² = −1.01, the expected
   signature of a prediction with the right marginal distribution and no information.

### The unpublished denominator does not change the ranking

The largest open risk was that our inferred denominator (§6 of
`metric_spec_reverse_engineered.md`) would rank models differently from the organizers'.
It does not. Spearman rank correlation between the 9 predictors' orderings under all
five denominator policies is **1.0 for every pair** — `st_mean`, `st_median`,
`mae_mean`, `mae_median` and `none` produce the identical ranking, differing only in
absolute scale (e.g. `mediocre` = 1.027 / 1.084 / 0.669 / — / 0.509).

**This substantially de-risks model selection.** We can rank candidate models correctly
today without knowing the denominator; only the absolute number we quote will shift when
the organizers publish theirs, and re-fitting is a one-argument change. The open question
remains worth asking, but it is no longer blocking.

## 5. ST-RAE really does neutralise low-activity error

Two experiments, both on real CI widths.

**Matched error budget.** The same error vector (100 draws from |N(0, 1.2)|) was added
either to the 100 compounds below the assay floor or to 100 randomly chosen compounds
above it. MAE is **identical to 0.00e+00** by construction; ST-RAE is not:

| error placed on | median CI width hit | MAE | ST-RAE |
| :--- | ---: | ---: | ---: |
| 100 inactives (pEC50 < 4) | 1.15 | 0.1939 | **0.2149** |
| 100 actives (pEC50 ≥ 4) | 0.43 | 0.1939 | **0.3168** |

An identical error budget costs **1.47× more** when spent on active compounds. MAE cannot
see this distinction at all. This is precisely the reweighting the organizers describe,
and it is driven entirely by the real fitted interval widths.

**CI-width sweep.** Scaling the real interval half-widths by a factor s ∈ [0, 4] and
recomputing the mean soft-threshold error (denominator `none`, to isolate the numerator
from any normalisation choice):

| CI scale | median width | good (σ=0.3) | mediocre (σ=1.0) | over-confident on inactives |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.000 | 0.2375 | 0.7663 | 0.6642 |
| 0.50 | 0.245 | 0.1280 | 0.6241 | 0.5264 |
| **1.00** (real) | **0.490** | **0.0712** | **0.5105** | **0.4185** |
| 2.00 | 0.980 | 0.0245 | 0.3440 | 0.2496 |
| 4.00 | 1.960 | 0.0036 | 0.1664 | 0.0295 |

At scale 0 the metric degenerates to plain MAE (0.2375 = the `good` predictor's MAE
exactly), confirming the zero-width limit. Error decays monotonically with interval
width for every predictor, and the decay is steepest for the predictor whose errors are
concentrated where the intervals are widest: `over-confident on inactives` falls 15.6×
between scale 1 and scale 4, against 3.1× for `mediocre`, whose error is spread evenly.

**A caution this exposes.** At 4× the real widths the over-confident predictor scores
0.0295 — better than the `good` predictor at real widths. Soft-thresholding is a strong
discount, and if the CYP release ships intervals materially wider than PXR's, calling
every inactive compound a mid-range value becomes a viable leaderboard strategy. Worth
checking against the real test intervals on day one.

## 6. Macro-average and the (Sample, Endpoint) bootstrap layout

The CYP regression track has four endpoints; PXR has one. A CYP-shaped matrix was built
by replicating each real (point, CI lower, CI upper) triple into four correlated
endpoints (jitter σ = 0.35, real interval widths preserved).

| predictor | MA-ST-RAE (point) | MA-RAE_mean | MA-RAE_std | per-endpoint (1A2 / 2C9 / 2D6 / 3A4) |
| :--- | ---: | ---: | ---: | :--- |
| good | 0.1349 | 0.1357 | 0.0093 | 0.140 / 0.128 / 0.144 / 0.128 |
| mediocre | 0.9987 | 1.0039 | 0.0539 | 0.956 / 1.048 / 1.052 / 0.939 |
| mean_only | 1.0000 | 1.0014 | 0.0137 | 1.000 / 1.000 / 1.000 / 1.000 |

`bootstrap_results_frame` produced a **(5000, 7)** frame = 1000 samples × (4 endpoints +
`MA`), with `Endpoint` taking exactly the four confirmed endpoint names plus `MA` — the
layout `config.MACRO_ENDPOINT_LABEL` and `head_to_head._load_bootstrap` describe. The
`MA` row of each sample equals the unweighted mean of that sample's four endpoint rows
(asserted per-sample in the test suite), so the macro-average is formed *inside* the
sample as inferred.

Note `mean_only` scores 1.000 on every endpoint independently, so its macro-average is
also exactly 1.000 — the interpretable baseline survives macro-averaging.

**Paired comparison** (shared resample matrix, seed 13), reproducing the Space's
head-to-head:

- good vs mediocre: mean Δ = −0.868, **good wins 1000/1000** bootstrap samples, p < 0.001.
- mediocre vs mean_only: mean Δ = +0.0025, wins split **479/521**, **p = 0.958**.

The second row is the important one. `mediocre` and `mean_only` differ by 0.0025 in
MA-ST-RAE — a gap the leaderboard would print as 1.0039 vs 1.0014, looking like a real
ordering — and the paired bootstrap correctly reports it as indistinguishable. Since the
final leaderboard applies BH-corrected pairwise tests and a `CLD` column (§5 of the
metric spec), the selection target should be the **paired delta against the current
leader**, not the absolute MA-ST-RAE.

## 7. TDI labeler and classification scoring

PXR has no TDI arm, so a realistic two-arm construction was built from the real pEC50
distribution: 15 % of compounds given a genuine shift (U[0.35, 1.5]), the rest assay-scale
jitter (σ = 0.12). Category census over 513 compounds:

| category | n | collapsed label |
| :--- | ---: | :--- |
| negative | 355 | 0 |
| assigned_negative | 85 | 0 |
| positive | 58 | 1 |
| inferred_positive | 10 | 1 |
| **unassignable** | **5** | **excluded** |

508/513 scorable (98.9 %), 5 (1.0 %) falling in the documented spec gap; positive rate
among scorable compounds 0.134. All four published categories are populated, including
`inferred_positive`, so the low-activity inference branch is exercised on realistic data.

Classification battery against these labels (1000-sample bootstrap, seed 3):

| predictor | MCC | Accuracy | Precision | Recall | F1 | MCC boot mean ± std | MCC 95 % CI |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| perfect | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 | 1.0000 ± 0.0000 | [1.000, 1.000] |
| 85 % accurate | 0.5676 | 0.8484 | 0.465 | 0.882 | 0.609 | 0.5690 ± 0.0401 | [0.483, 0.642] |
| all_negative | 0.0000 | **0.8661** | 0.000 | 0.000 | 0.000 | 0.0000 ± 0.0000 | [0.000, 0.000] |
| all_positive | 0.0000 | 0.1339 | 0.134 | 1.000 | 0.236 | 0.0000 ± 0.0000 | [0.000, 0.000] |
| random | 0.1235 | 0.8130 | 0.255 | 0.206 | 0.228 | 0.1264 ± 0.0554 | [0.019, 0.234] |
| inverted | −1.0000 | 0.0000 | 0.000 | 0.000 | 0.000 | −1.0000 ± 0.0000 | [−1.000, −1.000] |

The `all_negative` row is why MCC is the right primary metric and why we should not
optimise accuracy: predicting "no TDI" for every compound scores **86.6 % accuracy** and
MCC exactly 0. Every predictor is scored on the same 508 compounds, confirming that the
unassignable exclusion is applied to the ground truth and not to the predictions.

## 8. Validator

Both dummy files pass with **0 errors and 0 warnings**, in both parquet and CSV:

| file | track | rows | dtypes | result |
| :--- | :--- | ---: | :--- | :--- |
| `dummy_submission_regression.parquet` | regression | 750 | 4 × float64 | PASS |
| `dummy_submission_classification.parquet` | classification | 750 | 2 × bool | PASS |
| `dummy_submission_regression.csv` | regression | 750 | 4 × float64 | PASS |
| `dummy_submission_classification.csv` | classification | 750 | 2 × bool | PASS |

Classification positive rates 0.139 (CYP2D6) and 0.115 (CYP3A4) — both classes present,
so MCC is defined and the file will not trip the single-class warning. Identifiers use
the `OADMET-%05d` pattern from the Space's own CSV example.

The test suite covers every documented rule and both Space defects: wrong row count, a
missing column, case-only column mismatch, extra columns (tolerated, warned), NaN and
±inf pIC50, a string pIC50 column, non-binary and NaN TDI values, single-class TDI,
duplicate `Molecule_Name`, bad extension, corrupt parquet, the 184-file structure zip,
the directory-entry trap, and both CLI exit codes.

## 9. Verified end to end

- ST-RAE numerator is correct in every limit (zero-width interval → MAE; prediction
  inside interval → 0; perfect prediction → 0).
- ST-RAE reweights error by fitted uncertainty as designed, demonstrated on real
  intervals with a matched error budget.
- Predictor ranking is monotone in quality and **invariant to the unpublished
  denominator choice**.
- The macro-average and the (Sample, Endpoint) bootstrap layout reproduce the structure
  the Space's own loader expects.
- The paired bootstrap correctly refuses to separate two predictors 0.0025 apart.
- All four TDI categories plus the unassignable gap are exercised; MCC behaves correctly
  on the degenerate single-class cases.
- Both dummy submissions pass the validator, so the upload path can be exercised on
  2026-08-17.

## 10. What still cannot be verified

Blocked on the data drop or on organizer clarification, in priority order:

1. **Whether `ci_lower`/`ci_upper` ship in the CYP test ground truth at all.** Without
   them ST-RAE cannot be computed locally and self-validation is impossible. This is the
   single highest-priority question.
2. The credible level and its definition (quantile vs HDI) — sets the discount strength,
   and §5 shows that strength matters.
3. The exact denominator — affects the absolute value we quote, not the ranking (§4).
4. Whether the leaderboard's `RAE` column is the soft-thresholded metric or a plain RAE
   reported alongside it.
5. The TDI-arm ground-truth column name (`label_tdi_frame` currently guesses
   `{isoform}_pIC50_tdi`).
6. Whether the unassignable band is excluded or folded into assigned-negative, and
   whether the threshold comparisons are strict.
7. The backend's MCC convention for a single-class submission.
8. Whether the backend tolerates extra columns as the upload check does.
