# Reverse-engineered metric, leaderboard and validation specification
## OpenADMET CYP Inhibition Blind Challenge

**Source of truth for this document:** the HuggingFace Space source snapshot
(`cyp_challenge_space/`: `app.py`, `config.py`, `leaderboards.py`, `head_to_head.py`,
`submission.py`, `models.py`, `utils.py`) plus the announcement blog text. Read on
2026-08-06, pre-data-drop (`CURRENT_PHASE = 0`).

**Critical caveat.** The Space is the *display and upload* side only. The scoring
backend (referred to throughout the Space source as `backend.config`,
`backend.aws_leaderboards`, `backend/lambda_handler.py`) is **private and not in the
snapshot**. Every metric *value* is computed there. What we can recover from the Space
is: the metric **names**, the **column schema** they arrive in, the **file layout**,
the **ranking direction**, and the **complete set of upload-time validation rules**.
The mathematical definition of ST-RAE's normalisation denominator is *not* recoverable
and must be inferred — see §6.

Each section is split into **CONFIRMED IN CODE** (a literal quotable line in the
snapshot) and **INFERRED** (our reading, to be treated as a hypothesis).

---

## 1. Tracks and endpoints

### CONFIRMED IN CODE (`config.py`)

Three fully independent backend tracks, each with its own submissions/scores/leaderboard
S3 prefix, Lambda and per-track macro-average. The track slugs are:

| constant | value |
| :--- | :--- |
| `REGRESSION_TRACK` | `"regression"` |
| `CLASSIFICATION_TRACK` | `"classification"` |
| `STRUCTURE_TRACK` | `"structure"` |

Endpoint lists (these are the exact submission column names, and also the exact
`endpoint_slug` values used in leaderboard file paths):

```python
REGRESSION_ENDPOINTS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]
CLASSIFICATION_ENDPOINTS = ["CYP2D6_is_TDI", "CYP3A4_is_TDI"]
STRUCTURE_ENDPOINTS = ["structure"]
IDENTIFIER_COLUMNS = ["SMILES", "Molecule_Name"]
ACTIVITY_ENDPOINTS = REGRESSION_ENDPOINTS + CLASSIFICATION_ENDPOINTS   # 8 entries
ACTIVITY_DATASET_SIZE = 750
STRUCTURE_DATASET_SIZE = 184
MACRO_ENDPOINT_LABEL = "MA"
HOURS_BETWEEN_SUBMISSIONS = 4
CLASSIFICATION_ACTIVE = True
STRUCTURE_TRACK_LIVE = True
CURRENT_PHASE = 0
```

Note that `STRUCTURE_ENDPOINTS[0]` is lowercase `"structure"`, identical to the track
slug — so the structure leaderboard file is
`leaderboard/live/structure/structure_leaderboard_latest.csv`.

`ACTIVITY_ENDPOINTS` (the 8-entry union) is **not** a submission concept. `config.py`
states it is "the combined view of the shared underlying compound set — used by
`head_to_head.py`'s cross-track comparison, and as the shared row-count both tracks
validate against (same compounds, different columns)".

### INFERRED

- The structure track is live in the code (`STRUCTURE_TRACK_LIVE = True`) but absent
  from the blog post. Its metrics (§4) are PoseBusters/OpenStructure-style. It is
  gated on `CURRENT_PHASE`, and the blog says the structure track starts "halfway
  through". Treat 184 pose predictions as a real third deliverable.
- Because the three tracks have separate Lambdas and separate macro-averages, a
  regression submission can never affect the classification leaderboard, and vice
  versa. The only coupling is the 750-row count.

---

## 2. Metric names and leaderboard column conventions

### CONFIRMED IN CODE (`config.py`, `leaderboards.py`)

The regression track's metric keys, in display order, primary/sort metric first — the
`config.py` comment states these "Must mirror `backend.config.ACTIVITY_METRICS`":

```python
ACTIVITY_METRIC_DISPLAY_NAMES = {
    "RAE": "RAE",
    "MAE": "MAE",
    "R2": "R²",
    "Spearman_R": "Spearman ρ",
    "Kendall_Tau": "Kendall's τ",
}
CLASSIFICATION_METRIC_DISPLAY_NAMES = {
    "MCC": "MCC",
    "Accuracy": "Accuracy",
    "Precision": "Precision",
    "Recall": "Recall",
    "F1": "F1 Score",
}
```

**This is the single most important finding in this document: the primary regression
metric key is literally `RAE`, not `ST-RAE`.** There is no `ST_RAE`, `STRAE` or
`ST-RAE` identifier anywhere in the Space source. The blog post and the in-Space FAQ
both describe the primary metric as Soft-Threshold RAE / MA-ST-RAE, while every
machine-readable name in the code is plain `RAE`. See §6 and §9-Q1.

Raw leaderboard CSV → display column mapping (`_prepare_activity_df`):

| raw CSV column | display column (endpoint tab) | display column (Overall tab) |
| :--- | :--- | :--- |
| `RAE_mean` | `RAE` | `MA-RAE` |
| `RAE_std` | `RAE std` | `MA-RAE std` |
| `MAE_mean` | `MAE` | `MA-MAE` |
| `R2_mean` | `R²` | `MA-R²` |
| `Spearman_R_mean` | `Spearman ρ` | `MA-Spearman ρ` |
| `Kendall_Tau_mean` | `Kendall's τ` | `MA-Kendall's τ` |
| `MCC_mean` | `MCC` | `MA-MCC` |
| `Accuracy_mean` / `Precision_mean` / `Recall_mean` / `F1_mean` | `Accuracy` / `Precision` / `Recall` / `F1 Score` | same, `MA-` prefixed |
| `rank` | `Rank` | `Rank` |
| `username` | `Username` | `Username` |
| `submitted_at` | `Submitted` (`%Y-%m-%d %H:%M UTC`) | same |
| `model_report_link` | `Model Report Link` | same |
| `used_proprietary_data` | `Proprietary Data` (`Yes`/`No`) | same |
| `open_source_code` | `Open Code` (`Yes`/`No`) | same |
| `user_alias`, `anonymous` | consumed and dropped | consumed and dropped |
| `CLD` | `CLD` (kept, forced last) | same |

So **every metric is reported to participants as a bootstrap `mean ± std` pair**, not
as a percentile interval: `_prepare_activity_df` drops all `*_std` columns and rounds
means to 4 dp for the live table, and keeps both at full precision `for_download=True`.
`head_to_head.build_stats_table` computes `boot_a[metric].mean()` and
`boot_a[metric].std()` directly from the bootstrap file and prints
`f"{mean:.4f} ± {std:.4f}"`.

`sort_leaderboard_columns` fixes column order: `rank, username, Submitted,
model_report_link`, then all metric columns, then `Proprietary Data, Open Code, CLD`.
Any `Unnamed*` column is dropped.

Ranking direction: rows are sorted by the backend-supplied `rank` column ascending
(`df.sort_values("rank", ascending=True)`). `head_to_head.py` declares
`TRACK_PRIMARY_METRIC = {"activity": "RAE", "structure": "LDDT-PLI"}` and
`LOWER_IS_BETTER = ["RAE"]`. Structure is sorted by `LDDT-PLI_mean` descending.

### INFERRED

- `CLD` is a **compact letter display** — the standard way of rendering "which
  entries are not significantly different from each other" as shared letters. It is
  produced by the same pairwise-BH machinery as `pairwise_comparisons.csv` (§5).
  `sort_leaderboard_columns` reserves a slot for it, so expect it on final
  leaderboards even though it is not in the placeholder empty-DataFrame schema.
- `rank` is assigned on `MA-RAE` ascending for regression and on `MCC` descending for
  classification. The classification direction is not stated anywhere in the Space —
  `TRACK_PRIMARY_METRIC` has **no** `"classification"` key, so
  `build_stats_table` would silently fall back to its `"MAE"` default for a
  classification head-to-head. That is a stale-code artifact (§8), not evidence that
  MAE is used.
- Because only `mean` and `std` are surfaced, our percentile bootstrap CIs are for
  *our own* model selection; to reproduce the leaderboard number we must report the
  bootstrap **mean** of the metric, which for a non-linear statistic is not the
  point estimate on the full test set. We implement both.

---

## 3. Leaderboard file path conventions

### CONFIRMED IN CODE (`config.py`, `leaderboards.py::format_leaderboard_uri`)

```python
LEADERBOARD_URI_FORMAT = (
    "leaderboard/{leaderboard_type}/{track}/{endpoint_slug}_leaderboard_{version}.csv"
)
```
`config.py` states: "Every leaderboard file (all tracks, all stages) follows this one
convention — must mirror `backend.aws_leaderboards._leaderboard_file_path`."

- `leaderboard_type` ∈ `{"live", "interim", "final"}` (validated; anything else raises).
- `track` ∈ `{"regression", "classification", "structure"}` (validated; anything else raises).
- `endpoint_slug` = `"MA"` for a track's master macro-ranked leaderboard, or a literal
  endpoint name, or `"structure"`.
- `version` = `"latest"` (default) or an ISO timestamp for a dated snapshot.

Concrete expected keys inside the S3 bucket:

```
leaderboard/live/regression/MA_leaderboard_latest.csv
leaderboard/live/regression/CYP1A2_pIC50_direct_inhibition_leaderboard_latest.csv
leaderboard/live/regression/CYP2C9_pIC50_direct_inhibition_leaderboard_latest.csv
leaderboard/live/regression/CYP2D6_pIC50_direct_inhibition_leaderboard_latest.csv
leaderboard/live/regression/CYP3A4_pIC50_direct_inhibition_leaderboard_latest.csv
leaderboard/live/classification/MA_leaderboard_latest.csv
leaderboard/live/classification/CYP2D6_is_TDI_leaderboard_latest.csv
leaderboard/live/classification/CYP3A4_is_TDI_leaderboard_latest.csv
leaderboard/live/structure/structure_leaderboard_latest.csv
```
…and the same nine under `interim/` and `final/`. An `"MA"` master file exists only
when a track has more than one endpoint (`_add_activity_track_tabs`), so structure has
no master file.

Every activity-style leaderboard CSV — master or per-endpoint — has the **same bare
metric-column schema**, because "the backend narrows down to one endpoint's columns
before saving (see `aws_leaderboards._narrow_averaged_results_to_endpoint`)".

Live leaderboards auto-refresh on a `gr.Timer(value=30)` — one 30 s timer per track.

### Per-submission score artefacts

### CONFIRMED IN CODE (`head_to_head.py`, `submission.py`, `models.py`)

Submission storage layout (`models.Submission.s3_prefix`):
```
submissions/{regression|classification|structure}/{safe_username}/{submission_id}/
    metadata.json          # Submission model as JSON
    predictions.parquet     # or predictions.csv  (canonical name, original suffix kept)
    structures.zip          # structure track only
```
`safe_username` = `str(username).strip().lower().replace("/", "_").replace(" ", "_")`
(`utils._safeify_username`), lowercased specifically so that HF-casing variants cannot
bypass the 4-hour cooldown.

Score artefacts, from `_extract_username_and_submission_id`'s documented URI format
and the `.replace()` calls in `_load_bootstrap` / `_load_structure_per_compound`:
```
scores/all/{track}/{username}/{submission_id}/averaged-results.parquet
scores/all/{track}/{username}/{submission_id}/bootstrap-results.parquet
scores/all/{track}/{username}/{submission_id}/per-compound-results.parquet
```
Ground truth and manifests (`head_to_head.py` module constants):
```
ground_truth/activity-dataset.parquet
ground_truth/{track}-identifiers.parquet     # columns include Molecule_Name, phase
leaderboard/final/{track}/manifest.csv       # column all_scores_uri, username
leaderboard/final/{track}/pairwise_comparisons.csv
```

---

## 4. The macro-average pseudo-endpoint and the bootstrap protocol

### CONFIRMED IN CODE

`config.py`, verbatim:

> Activity `bootstrap-results.parquet` files hold one row per (Sample, Endpoint) pair,
> including this synthetic pseudo-endpoint holding the multi-endpoint macro-averaged
> scores for that bootstrap sample. Must mirror `backend.config.MACRO_ENDPOINT_LABEL`.
> ```python
> MACRO_ENDPOINT_LABEL = "MA"
> ```

`head_to_head._load_bootstrap`, verbatim docstring:

> Activity bootstrap files hold one row per (Sample, Endpoint) pair, including the
> "MA" pseudo-endpoint holding the multi-endpoint macro-averaged scores for that
> sample (see MACRO_ENDPOINT_LABEL). This comparison only ever uses the macro series —
> narrow down to it here so downstream code sees exactly one row per Sample.

and the code:
```python
bootstrap_df = bootstrap_df[bootstrap_df["Endpoint"] == MACRO_ENDPOINT_LABEL].drop(columns=["Endpoint"])
```

So `bootstrap-results.parquet` schema is:

| column | meaning |
| :--- | :--- |
| `Sample` | bootstrap resample index (join key across submissions) |
| `Endpoint` | one of the 4 (or 2) endpoint names, **or** `"MA"` |
| `RAE`, `MAE`, `R2`, `Spearman_R`, `Kendall_Tau` | regression metrics for that (Sample, Endpoint) |
| `MCC`, `Accuracy`, `Precision`, `Recall`, `F1` | classification metrics for that (Sample, Endpoint) |

For 4 regression endpoints + MA over 1000 samples that is 5000 rows.

`_prepare_activity_df` consumes `{metric}_mean` / `{metric}_std`, i.e. the leaderboard
is the mean and SD **over the `Sample` axis** of this file.

Number of resamples — `config.FAQ_MD`, shown to participants inside the Space:

> Secondary metrics (MAE, R², Spearman ρ, Kendall's τ) are also reported. All metrics
> are bootstrapped over 1,000 resamples.

`head_to_head.build_delta_plot` merges the two users' bootstrap frames **on `Sample`**:
```python
merged = boot_a[["Sample", metric]].merge(boot_b[["Sample", metric]], on="Sample", suffixes=("_a","_b"))
delta = merged[f"{metric}_a"] - merged[f"{metric}_b"]
```
and counts `(delta < 0).sum()` as A's wins when the metric is in `LOWER_IS_BETTER`.

### INFERRED

1. **The macro-average is computed inside each bootstrap sample, then averaged.** The
   `MA` row for `Sample = s` is `mean over the 4 endpoints of metric(s, endpoint)`.
   This is *not* the same as bootstrapping the macro-average of the point estimates,
   and it is the order our implementation uses.
2. **The resampling unit is compounds, and the resample indices are shared across
   submissions.** Merging on `Sample` is only meaningful if `Sample = s` means the
   same set of compounds for every participant — i.e. the backend draws 1000 fixed
   compound index vectors (almost certainly from a fixed seed) and reuses them for
   every submission. This is what makes the paired delta distribution and the
   BH-corrected pairwise p-values valid. Our `bootstrap_ci` therefore resamples
   **compounds** with replacement and exposes a fixed `seed`, and can be handed an
   explicit index matrix so we can pair predictors exactly.
3. **Macro-averaging is unweighted** across endpoints (a plain mean), so the sparsely
   measured endpoints count as much as the dense ones. Since the CYP test set is dense
   across all four isoforms this matters less than it did for PXR, but it does mean an
   endpoint with fewer non-NaN ground truths has the same weight.
4. **NaN ground truth is dropped per endpoint**, not per compound — otherwise a
   compound missing one isoform would silently shrink the other three endpoints. The
   dense test matrix should make this rare.
5. The `Sample` axis probably includes or excludes the point estimate; unknown. We
   report both the full-data point estimate and the bootstrap mean.

---

## 5. Significance testing (final-leaderboard only)

### CONFIRMED IN CODE (`head_to_head.py`)

```python
ALPHA_LEVEL = 0.05
PAIRWISE_URI = "leaderboard/final/{track}/pairwise_comparisons.csv"
```
`pairwise_comparisons.csv` columns read by `_get_pairwise_row` / `build_stats_table`:
`entry_a`, `entry_b` (order-independent lookup), `adjustment_rank`,
`adjusted_threshold`, `p_value`, `significant_difference`. The stats table labels these
"Target FDR (Q)", "BH Rank (k)", "BH Critical Value", "Observed p-value",
"Statistically Significant".

### INFERRED

Benjamini–Hochberg FDR control at Q = 0.05 over all pairs of entries within a track,
with the p-value coming from the paired bootstrap delta distribution on the primary
metric (`RAE` for regression). The `CLD` leaderboard column renders the resulting
significance groups. Practical consequence for us: a leaderboard gap smaller than the
paired-bootstrap noise will be reported as not significant, so the model-selection
target should be the **paired delta vs the current leader**, not the absolute MA-RAE.

---

## 6. ST-RAE: what is and is not pinned down

### CONFIRMED (blog post + in-Space FAQ text, `config.ABOUT_MD` / `config.FAQ_MD`)

> Error is measured as the distance between your prediction and the nearest credible
> interval bound of the fitted dose-response curve. Predictions falling anywhere inside
> the credible interval incur zero error penalty.

and

> pIC50 values below 4 are outside the reliable dynamic range of the physical assay.
> Standard MAE/RAE would unfairly penalize models for missing noisy point estimates in
> inactive regions.

So the **numerator** is unambiguous. For prediction `ŷ` and ground-truth credible
interval `[lo, hi]`:

```
st_ae(ŷ) = max(lo - ŷ, 0, ŷ - hi)          # 0 inside the interval
```

### NOT CONFIRMED ANYWHERE — must be inferred

1. **The RAE denominator.** "Relative absolute error" conventionally means
   `Σ|ŷ-y| / Σ|y-ȳ|`, i.e. normalisation by a trivial predictor. Neither the blog nor
   the Space says (a) whether the denominator is itself soft-thresholded, (b) whether
   the trivial predictor is the train mean, the test mean, or the test median, or
   (c) whether normalisation happens before or after macro-averaging. We therefore
   implement the denominator as a **swappable, documented policy** with five options
   and default to the one we consider most likely (`st_mean`: the *soft-thresholded*
   error of the test-set mean-of-`y_true` predictor), so the whole harness can be
   re-fitted with a one-line change the moment the organizers publish theirs.

   | policy | denominator |
   | :--- | :--- |
   | `st_mean` (default) | `Σ st_ae(mean(y_true))` — soft-thresholded, mean predictor |
   | `st_median` | `Σ st_ae(median(y_true))` — soft-thresholded, median predictor |
   | `mae_mean` | `Σ|y_true − mean(y_true)|` — plain (classical RAE) |
   | `mae_median` | `Σ|y_true − median(y_true)|` — plain, median (robust RAE) |
   | `none` | `1` — reduces ST-RAE to plain mean soft-threshold absolute error |

   Consequence to be aware of: with `st_mean` the denominator can be **small or even
   zero** if the credible intervals are wide enough that the constant mean predictor
   already lands inside most of them. Our implementation guards this and reports the
   guard.

2. **Whether the credible interval is published per compound per endpoint.** The
   metric is undefined without `ci_lower` / `ci_upper` in the ground truth. The Octant
   blog-release inhibition table carries `ci_lower` / `ci_upper` / `se`, so the machinery
   exists. `head_to_head.DATASET_URI` points at a single
   `ground_truth/activity-dataset.parquet` and the scatter plot only ever reads a bare
   `endpoint` column from it — no CI columns appear in the Space at all.
3. **The credible interval level** (50 %? 89 %? 95 %? HDI or quantile?).
4. **What happens when the DRC fit failed / the CI is missing or infinite.** We treat a
   missing bound as `±inf` on that side (i.e. no penalty in that direction) only if
   explicitly asked; the default is to exclude the compound, and we count exclusions.
5. **Whether the secondary metrics are soft-thresholded too.** `MAE`, `R2`,
   `Spearman_R`, `Kendall_Tau` are almost certainly computed against the point
   estimate, not the interval — an interval-aware R² is not a standard object. Our
   secondary battery uses point estimates.

---

## 7. Validation rules **actually enforced on upload**

This is the complete list, read off `submission.py::submit_predictions` and
`_read_tabular_submission`. Nothing else is checked client-side.

### CONFIRMED IN CODE — identity/metadata gates (fail before the file is read)

1. `username` non-empty after strip, **and** `utils.validate_hf_username` must find
   `https://huggingface.co/{username}` returning HTTP 200 (retries up to 10 times on 429).
2. If "submit anonymously" is checked, `user_alias` must be non-empty.
3. If `email` is non-empty it must match `^[^@\s]+@[^@\s]+\.[^@\s]+$`.
4. If "open-source code" is checked, `model_tag` must start with `https://` and be
   reachable (200), else the submission is rejected. If unchecked, the link is stored
   verbatim (or `"Not submitted"`) and never fetched.
5. A track must be selected and a file must be attached.

### CONFIRMED IN CODE — file gates, tabular tracks

6. Extension must be `.parquet` or `.csv` (lower-cased suffix). Parquet is read with
   `pd.read_parquet`, CSV with `pd.read_csv`; a read exception is surfaced as
   "Could not read …". **Any other extension is rejected** with "must be a .parquet or
   .csv file".
7. `len(df) != 750` → `"Error: Expected 750 rows, got N."` Exact, both tracks.
8. `missing = set(required_columns) - set(df.columns)`; non-empty → rejected.
   Required sets are exactly
   `["SMILES","Molecule_Name"] + REGRESSION_ENDPOINTS` (6 columns) and
   `["SMILES","Molecule_Name"] + CLASSIFICATION_ENDPOINTS` (4 columns).
9. **Extra columns are explicitly NOT an error** — the check is a one-sided set
   difference. Column *order* is not checked either. Column names are compared by
   exact string equality, so they are **case-sensitive**.
10. Regression only, for each of the 4 endpoint columns:
    `df[col].isnull().any()` → "contains NaN values";
    `not np.isfinite(df[col]).all()` → "contains infinite values".
11. Classification only, for each of the 2 endpoint columns:
    `df[col].dropna().isin([0, 1, True, False]).all()` must hold, else "contains
    non-binary values".
12. Rate limit: the most recent `metadata.json` under
    `submissions/{track}/{safe_username}/` must be more than
    `HOURS_BETWEEN_SUBMISSIONS = 4` hours old, **per track**.

### CONFIRMED IN CODE — file gates, structure track

13. Extension must be `.zip`; `zipfile.ZipFile(...).namelist()` length must equal
    exactly `184`. (Note: `namelist()` counts directory entries too, so a zip built
    from a folder will report 185 and be rejected.) Per-file `.pdb` naming and the
    `LIG` residue requirement are documented in the Submit-tab markdown but **not
    enforced** by the Space.

### Two real defects in the Space's checks (CONFIRMED by reading, worth flagging)

- **Classification NaNs pass validation.** `df[col].dropna().isin([...]).all()` drops
  the NaNs before testing, so an all-NaN classification column is accepted by the
  Space even though the FAQ says "must be fully populated for every row". Our validator
  rejects them (strict mode) and warns that the Space would not.
- **A non-numeric regression column raises rather than reporting.**
  `np.isfinite(df[col])` on an `object`/string dtype raises `TypeError`, which is not
  caught in `submit_predictions` — the participant would get a Gradio traceback rather
  than the intended message. Our validator reports a clean dtype failure.

### Not checked at all (INFERRED to be backend-side)

SMILES parseability; `Molecule_Name` matching the released test-set identifiers;
duplicate `Molecule_Name`s; row ordering; prediction range sanity; the 6-column /
4-column *count* stated in the Submit-tab markdown (only the required-subset is
enforced).

---

## 8. Stale code in the snapshot — do not treat as spec

`head_to_head.py` has not been migrated to the three-track split and contradicts
`config.py`. It is the PXR-era module. Specifically:

- `MANIFEST_URI = "leaderboard/final/{track}/manifest.csv"` and
  `LEADERBOARD_URI = "leaderboard/final/{track}/leaderboard_latest.csv"` do **not**
  match `LEADERBOARD_URI_FORMAT` (no `{endpoint_slug}` component).
- Its track vocabulary is `["activity", "structure"]`, not
  `["regression", "classification", "structure"]`.
- `_load_activity_predictions` hardcodes
  `submissions/activity/{username}/{submission_id}/predictions.csv`, while
  `submission.py` writes `submissions/regression/...` and prefers `predictions.parquet`.
- `TRACK_PRIMARY_METRIC` has no `"classification"` key.
- It iterates the 8-entry `ACTIVITY_ENDPOINTS` for scatter tabs, mixing regression and
  boolean endpoints.
- `ACTIVITY_TARGET_PHASE = 2` and the `phase` column reflect PXR's two-phase
  unblinding; the CYP challenge is a single continuous stage with a live/blind split
  **by chemisimilar series**.

Where `config.py` and `head_to_head.py` disagree, `config.py` wins — it carries the
"Must mirror `backend.config.*`" annotations.

---

## 9. Open questions to raise on Discord (`#cyp-challenge`)

1. **Is the leaderboard's `RAE` column the soft-thresholded metric?** The code's
   metric key is `RAE` everywhere; the FAQ calls the primary metric ST-RAE / MA-ST-RAE.
   Are they the same number under two names, or is a plain RAE also reported?
2. **What exactly is the ST-RAE denominator?** Which trivial predictor (mean/median,
   train or test), is the denominator itself soft-thresholded, and is the ratio taken
   per endpoint before macro-averaging or after?
3. **Will `ci_lower` / `ci_upper` ship in the test ground truth, and at what credible
   level and definition (quantile vs HDI)?** Without them ST-RAE cannot be reproduced
   locally, and self-validation is impossible. Will the *training* DRCs carry them too?
4. **Order of operations for MA:** macro-average within each bootstrap sample then
   average over samples (our reading of the `MA` pseudo-endpoint), or bootstrap the
   macro-average of point estimates?
5. **Are bootstrap resample indices fixed and shared across submissions?** The paired
   delta plot and BH pairwise tests imply yes; please confirm, and ideally publish the
   seed so participants can reproduce their own CIs exactly.
6. **Compounds with a failed or unbounded DRC fit** — excluded from scoring, or scored
   against the point estimate?
7. **TDI labelling gap:** the four documented cases do not cover
   `direct < 4 AND 4 ≤ tdi ≤ 4.301`. Is that band excluded ("only compounds whose label
   can be assigned with confidence contribute to the score"), or folded into
   assigned-negative? Also: are the comparisons strict or non-strict at exactly
   `direct = 4`, `tdi = 4.301`, `shift = 0.301`? (See `cyp_tdi.py`.)
8. **MCC when a leaderboard entry predicts a single class** (MCC's denominator is 0) —
   scored as 0, as NaN, or is the entry rejected?
9. **Is the TDI ground-truth label derived per isoform from that isoform's own two
   arms** (CYP3A4 direct vs CYP3A4 +NADPH), and is the *direct* arm used for labelling
   the same number that is the regression ground truth?
10. **Secondary metrics: point estimate or interval-aware?** We assume MAE/R²/ρ/τ use
    the DRC point estimate.
11. **Does the classification track's row-count really have to be 750**, including
    compounds whose TDI label is unassignable? (The code says yes — predictions are
    requested for all 750 so the scored subset leaks nothing.)
12. **`CLD`** — confirm it is a compact letter display and which α it uses.
13. **Structure track**: `STRUCTURE_TRACK_LIVE = True` and a 184-structure leaderboard
    with LDDT-PLI / BiSyRMSD / LDDT-LP / Coverage exist in the code but not in the
    announcement. Is it confirmed, and when does it open?
14. **Extra columns**: tolerated by the current check. Will the backend also tolerate
    them, or reject strictly?
