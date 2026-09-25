# Go-live runbook — OpenADMET CYP Challenge, 2026-08-17

Ordered actions for launch day and the weeks after. Written 2026-08-06.
Deadlines: **intermediate 2026-09-24 23:59 UTC**, **final 2026-11-03 23:59 UTC**.
Rate limit: **one submission per 4 hours**; only the latest valid submission
counts; **one submission per team/lab** across the whole challenge.

---

## Before launch day (do now, 2026-08-06 → 08-16)

1. **Cache the CheMeleon foundation encoder.** Anvil fetches
   `chemeleon_mp.pt` (34,859,448 B, md5 `6a80b54fdb7de37ef0374d302f01e8ce`) from
   Zenodo record 15460715 *at model-construction time*. Zenodo is not on our
   network allowlist by default — download and cache the file now, or model
   construction fails on launch day. Request `zenodo.org` access if needed.
2. **Pin the environment.** `cyp-models` is built and verified: Python 3.12.13,
   torch 2.13.0+cu130, CUDA available on the RTX 4080 Laptop (12 GB). Set
   `n_jobs=0` in any Anvil recipe — the shipped `n_jobs=4` DataLoader workers hang
   silently in this sandbox.
3. **Post the open questions to Discord `#cyp-challenge`** (§7 of
   `STRATEGY_AND_GAP_ANALYSIS.md`). Question 1 — whether `ci_lower`/`ci_upper` ship
   with the test ground truth — is blocking for local self-validation, so ask it
   early enough to get an answer before the drop.
4. **Register the Hugging Face account** that will submit, and decide now who the
   single submitting identity is (one per team/lab, no exceptions).

---

## Launch day, 2026-08-17

### Step 1 — acquire and inventory (target: 30 min)

The dataset link is not yet published (`DATASET_DOWNLOAD_LINK = False` in the
Space's `config.py`). Expect a HF dataset under the `openadmet` org, following the
`openadmet/pxr-challenge-train-test` pattern. Check the Space's Data tab and
Discord.

Download everything, then inventory before modelling:

```
n rows per file; column names verbatim; dtypes
per-isoform DRC counts (expect ~1,500 each)
which columns carry the DIRECT arm vs the +NADPH/TDI arm
presence and names of credible-interval columns
test set: expect exactly 750 rows, dense across 4 isoforms
primary screen: single-concentration, +NADPH condition, whole library
```

### Step 2 — the four schema checks that gate everything (target: 30 min)

Run these **before** writing any model code. Each has a known failure mode:

1. **Identifier column.** The CYP Space requires `Molecule_Name` (underscore).
   PXR used `Molecule Name` (space). Do not assume; read the header.
2. **Credible-interval columns.** PXR named them
   `pEC50_ci.lower (-log10(molarity))` — units embedded in the name. If CYP follows
   suit, they will **not** be called `ci_lower`/`ci_upper`, and
   `cyp_metrics.st_rae` needs a column-mapping step. If they are absent entirely,
   ST-RAE cannot be computed locally — fall back to `denominator="none"` on plain
   MAE for model selection and say so in the method report.
3. **TDI-arm column name.** `cyp_tdi.label_tdi_frame` currently guesses
   `{isoform}_pIC50_tdi`. Fix it to the real name.
4. **Arm identification.** Confirm which column is the −NADPH (direct) arm. Track 1
   scores **direct** inhibition; the primary screen and the Octant public data are
   both +NADPH. Getting this backwards invalidates everything downstream.

### Step 3 — check the sub-floor exploit (target: 20 min)

Compute the credible-interval width distribution, split at pIC50 4. Octant showed a
**23×** ratio (2.026 vs 0.088 log units); PXR showed 2.8×. Then score two
predictors under ST-RAE: a genuinely good model, and one that assigns a blanket
mid-range value to every sub-floor compound. If the blanket strategy wins, the
intervals are wide enough that sub-floor predictions should be set to the interval
midpoint rather than modelled. Decide this before training.

### Step 4 — build the challenge-matched validation protocol (target: 1 h)

```python
import cyp_splits
series = cyp_splits.leader_cluster(fps, threshold=0.55)
for train_idx, test_idx in cyp_splits.parent_in_train_folds(series, y):
    ...
```

Use `parent_in_train_folds` for model selection — it reproduces the real condition
(parent hit in training, its analogs held out). Use `series_held_out_folds` as the
pessimistic bound. Do **not** select models on random K-fold: it was optimistic by
0.04–0.05 RAE relative to scaffold-grouped in our measurements.

If the release identifies the parent/analog series explicitly (it should, since the
live/blind split is by chemisimilar series), use the organizers' series assignment
in place of our clustering.

### Step 5 — train (target: day 1–3)

Order of work, per `method_shortlist.csv`:

1. Fit the rank-3 GBM ensemble first — 2 days, no GPU, and it gives a defensible
   submission within 24 h of the drop.
2. Then the rank-1 multitask Chemprop/CheMeleon over all six endpoints jointly.
   **Fine-tune on the training pack**; do not ship a public-data-only model. Our
   external test showed a public-only model reaches only R² = 0.12 / ρ = 0.47 on a
   real OpenADMET CYP3A4 assay.
3. Then rank-2 shift regression for TDI. This is the only viable route for
   `CYP2D6_is_TDI` — public data supplies exactly **one** labelled CYP2D6 TDI
   compound, so that endpoint depends entirely on the training pack's ±NADPH pairs
   plus cross-isoform sharing from CYP3A4.

Pretraining material is ready: `cyp_public_corpus.parquet` (37,006 compounds,
per-isoform pIC50, arm/censoring/source flags) and `tdi_auxiliary_labels.parquet`
(625 tier-1 rows, 5 empirically enriched MBI alerts only).

### Step 6 — score, then stop tuning when inside the noise floor

```python
import cyp_metrics
cyp_metrics.ma_st_rae(...)              # primary, regression
cyp_metrics.bootstrap_results_frame(...) # 1000 resamples, (Sample, Endpoint) + 'MA'
cyp_metrics.paired_delta(...)            # the number that actually matters
```

In the PXR challenge, **28 of 100 activity entries were statistically
indistinguishable from rank 1** (MAE 0.4061–0.4360), with bootstrap sd ≈ 0.02
against top-10 gaps of ≈ 0.002. The final leaderboard applies BH-corrected pairwise
tests with a compact-letter-display column. Optimise the **paired delta against the
current leader**, and stop when the difference is inside the noise floor — further
tuning buys rank noise, not rank.

### Step 7 — validate and submit

```bash
python validate_submission.py --track regression      --file predictions_regression.parquet
python validate_submission.py --track classification  --file predictions_classification.parquet
```

Both tracks are **separate files**, each exactly 750 rows, parquet preferred.
Regression needs `SMILES`, `Molecule_Name` + the 4 `*_pIC50_direct_inhibition`
columns, all finite floats. Classification needs `SMILES`, `Molecule_Name` +
`CYP2D6_is_TDI`, `CYP3A4_is_TDI`, valid booleans. Known-good dummy files
(`dummy_submission_regression.parquet`, `dummy_submission_classification.parquet`)
pass the validator and can exercise the upload path immediately.

On the submission form: tick **open source code** with a real public repo link, and
**disclose proprietary data** honestly (we use none — public ChEMBL/PubChem/Octant
only, which is worth stating explicitly since it is a fair-comparison signal).
Provide the `model_report_link`.

Check `#cyp-challenge-submissions` on Discord for the automated receipt. If nothing
appears within 2 hours, ask in `#cyp-challenge`.

---

## Timeline after launch

| date | action |
|---|---|
| 2026-08-17 | Data drop. Steps 1–4 same day. Submit dummy files to prove the upload path. |
| 2026-08-18 → 08-20 | GBM baseline trained and submitted — a real, defensible entry inside 72 h. |
| 2026-08-21 → 09-10 | Multitask Chemprop/CheMeleon; TDI shift regression; iterate under `parent_in_train_folds`. |
| 2026-09-20 → 09-23 | Freeze the intermediate entry. Leave ≥ 24 h margin: the 4-hour rate limit means a failed validation close to the deadline can cost the slot. |
| **2026-09-24 23:59 UTC** | **Intermediate deadline.** |
| 2026-09-25 | Interim leaderboard — the one-time full-test-set reveal. Read the paired deltas and the CLD tiering, not the absolute ranks. |
| 2026-09-26 → 10-25 | Improve where the interim result says there is headroom. Consider ranks 6–7 for the Innovation award (explicitly decoupled from leaderboard rank). |
| 2026-10-30 → 11-02 | Freeze the final entry, re-validate, submit with margin. |
| **2026-11-03 23:59 UTC** | **Final deadline.** |

## If the Structure track opens

Rank it third. Boltz-2 accepts `ccd: HEM` alongside a SMILES ligand (verified), but
one complex took >36 GPU-min at minimal settings on our 12 GB GPU — >110 GPU-hours
for 184 structures. Use template docking into the holo CYP3A4 structures in
`cyp_pdb_inventory.csv`, or budget remote GPU. Submission is a `.zip` of exactly 184
`.pdb` files, ligand residue named `LIG`, filenames matching compound identifiers.
Space validation checks only the extension and the file count, so naming errors fail
silently at scoring. Primary metric is LDDT-PLI (higher better) — the contact
pattern is what scores, so a correct fold with a misplaced ligand scores near zero.

## Asset index

| file | use |
|---|---|
| `challenge_spec.json` | machine-readable spec: endpoints, metrics, rules, timeline |
| `cyp_metrics.py` | ST-RAE, MA-ST-RAE, secondary battery, bootstrap, paired delta |
| `cyp_tdi.py` | 4-way TDI labeler (incl. unassignable band), MCC battery |
| `validate_submission.py` | pre-upload gate, CLI, non-zero exit on failure |
| `test_harness.py` | 109 tests, all passing |
| `cyp_splits.py` | `parent_in_train_folds`, `series_held_out_folds`, series clustering |
| `cyp_featurize.py` | ECFP4/6, RDKit-2D, MACCS/Avalon, CheMeleon embeddings |
| `cyp_public_corpus.parquet` | 37,006-compound pretraining corpus (+ data dictionary) |
| `octant_cyp3a4_drc.parquet` | real pIC50 credible intervals — ST-RAE validation set |
| `tdi_auxiliary_labels.parquet` | 625 tier-1 TDI labels + enriched alerts |
| `STRATEGY_AND_GAP_ANALYSIS.md` | findings, risks, open questions |
| `literature_review.md`, `references.bib`, `method_shortlist.csv` | method basis |
| `metric_spec_reverse_engineered.md`, `harness_verification.md` | what is confirmed vs inferred |
| `structure_track_scoping.md`, `cyp_pdb_inventory.csv` | track 3 |
