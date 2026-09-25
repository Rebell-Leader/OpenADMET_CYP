# OpenADMET CYP Inhibition Blind Challenge — strategy and gap analysis

**Prepared 2026-08-06. Data drops 2026-08-17 (11 days).**
Everything in this document is traceable to a file we downloaded and read, a
command we ran, or a number we computed. Inferences and open questions are marked
as such rather than asserted.

---

## 1. What the challenge actually is

Recovered from the announcement post (DOI 10.5281/zenodo.21789716) and from the
live Hugging Face Space source, which we snapshotted on 2026-08-06 (Space last
modified 2026-08-05, `CURRENT_PHASE = 0`).

| | Track 1 | Track 2 | Track 3 |
|---|---|---|---|
| name | Direct Inhibition (Regression) | TDI (Classification) | **Structure Prediction (Pose)** |
| targets | CYP1A2, CYP2C9, CYP2D6, CYP3A4 | CYP3A4, CYP2D6 only | one pose leaderboard |
| columns | `CYP{1A2,2C9,2D6,3A4}_pIC50_direct_inhibition` | `CYP{2D6,3A4}_is_TDI` | 184-file `.zip` |
| primary metric | **MA-ST-RAE** (lower better) | **MCC** | **LDDT-PLI** (higher better) |
| rows | exactly 750 | exactly 750 | exactly 184 |
| announced in blog? | yes | yes | **no — code only** |

**Track 3 is not in the blog post.** `config.py` carries
`STRUCTURE_TRACK_LIVE = True`, `STRUCTURE_DATASET_SIZE = 184`,
`STRUCTURE_NAME = "Structure Prediction (Pose)"`, and `leaderboards.py` sorts it
by `LDDT-PLI_mean` descending with BiSyRMSD / LDDT-LP / Ligand RMSD / Coverage as
secondaries — the CASP15/OpenStructure ligand-pose metric set. Anyone reading only
the announcement will not know this track exists. Treat 184 as provisional: the
constant carries a `TODO: Update when final dataset is ready` comment and the PXR
release contains exactly 184 ground-truth PDBs, so the number may be copied.

Dataset design, which drives everything downstream:

* **Train**: ~1,500 12-point Bayesian-fitted DRCs per CYP, **both arms** (±NADPH
  pre-incubation), plus a single-concentration primary screen of the whole library
  run in the **+NADPH (TDI) condition**. The train label matrix is **sparse**.
* **Test**: 75 potent parents (top 25 hits each from CYP1A2/CYP2C9/CYP3A4 — *not*
  CYP2D6) × top 10 ECFP4-Tanimoto analogs from Enamine US in-stock = 750
  compounds, **dense** across all four isoforms.
* **Live/blind split**: by chemisimilar series, so a parent and all its analogs
  land on the same side.

Rules that constrain planning: one submission per team/lab, one submission per 4
hours, latest valid submission counts, external and pretrained data allowed but
proprietary data must be disclosed, and a discretionary Innovation in ML award
that is explicitly decoupled from leaderboard rank.

---

## 2. The five findings that should shape the entry

### 2.1 A model trained on public CYP data transfers poorly to an OpenADMET assay

This is the most consequential measurement we made. We trained LightGBM on
ECFP4+RDKit-2D over 24,340 leakage-free public CYP3A4 compounds and scored it on
the 1,084-compound Octant CYP3A4 DRC release — real OpenADMET data, real Bayesian
credible intervals, with every Octant compound removed from training.

| predictor | ST-RAE | MAE | R² | Spearman ρ |
|---|---|---|---|---|
| LightGBM (ChEMBL/PubChem-trained) | **0.870** | 0.640 | 0.117 | 0.472 |
| mean of training set | 1.012 | 0.726 | −0.002 | — |
| mean of test set | 1.000 | 0.720 | 0.000 | — |
| constant 5.0 | 1.095 | 0.776 | −0.051 | — |

**R² = 0.12 against a no-skill baseline of 0.** The model beats predicting the
mean, but only just, and it does so almost entirely through ranking (ρ = 0.47)
rather than calibrated values. The predictions span 4.43–6.49 (sd 0.31) against a
truth span of 1.80–7.47 (sd 1.01): the model compresses hard toward the training
mean and never predicts a potent or an inactive compound.

We checked whether recalibration rescues it. It does not: an oracle-fitted linear
rescaling changes ST-RAE from 0.870 to 0.871, and oracle rank-mapping to the true
distribution makes it *worse* (1.108). The deficiency is in the ranking itself,
not in the scale.

**Implication.** Public ChEMBL/PubChem data is useful for pretraining and for
ranking, but a model trained only on it will land near the no-skill line on the
real leaderboard. The organizers' own ~1,500 DRCs/CYP are worth more than all
37,006 public compounds, and the plan must be built around fine-tuning on them the
day they arrive — not around having a finished model beforehand. Note also that
this transfer gap partly reflects the arm mismatch in §2.4.

### 2.2 Random-split CV is optimistic; the challenge task is easier than
leave-series-out

Four split regimes, LightGBM, same features throughout (`split_regime_benchmark.csv`,
`challenge_matched_split_benchmark.csv`):

| isoform | random | scaffold-grouped | analog series held out | **parent in train, analogs out** |
|---|---|---|---|---|
| CYP1A2 | 0.675 | 0.726 | 0.599 | **0.602** |
| CYP2C9 | 0.857 | 0.896 | 0.783 | **0.803** |
| CYP2D6 | 0.796 | 0.847 | 0.763 | **0.794** |
| CYP3A4 | 0.760 | 0.806 | 0.717 | **0.716** |

(RAE, lower is better. The last two columns are restricted to multi-member analog
series, so they are not directly comparable to the first two in absolute terms —
the comparison that matters is random vs scaffold-grouped, where grouping costs
0.04–0.05 RAE.)

The important structural point: **the challenge is not a leave-novel-series-out
problem.** The 75 parent hits come from the training DRC set, so at test time the
model has already seen a potent member of every series and is asked to interpolate
within it. `cyp_splits.parent_in_train_folds()` reproduces exactly this condition
and is the protocol to select models with; `series_held_out_folds()` is the
pessimistic bound.

### 2.3 Analog series hide steep SAR — this is where the challenge is won or lost

Within multi-member analog series in the public corpus:

| isoform | series spanning >1 log | activity-cliff pairs (T ≥ 0.6, Δ > 1 log) |
|---|---|---|
| CYP1A2 | 47 % | 8.8 % |
| CYP2C9 | 43 % | 9.0 % |
| CYP2D6 | 45 % | 13.1 % |
| CYP3A4 | 43 % | 7.9 % |

Roughly 45 % of series span more than a log unit, and ~8–13 % of highly similar
pairs are outright activity cliffs. Since the test set is *entirely* composed of
such series, a model that regresses analogs toward their parent's activity will be
penalised precisely where the leaderboard is decided. The organizers say so
themselves — the test set was designed to contain "detailed SAR and activity
cliffs".

### 2.4 The best public CYP data is in the wrong arm

The Octant CYP3A4 release is the highest-quality public CYP data available and the
only public source with real pIC50 credible intervals. But the protocol file
(assay code **C3A4IAP**, "CYP3A4 Inhibition, **Active** Preincubation") specifies a
30-minute active-enzyme pre-incubation with NADP+ and a regenerating system, and
the dataset README states the consequence directly: the IC50 values reflect
reversible **plus** time-dependent inhibition.

**It therefore maps to the challenge's +NADPH / TDI arm, not to the
direct-inhibition arm that Track 1 scores.** The protocol names a sibling
inactive-preincubation assay (C3A4IIP) that would be the direct arm; that data is
not in the public release. Any model trained on Octant pIC50 as if it were direct
inhibition carries a systematic, TDI-shaped bias.

Used correctly, though, this is an asset: it is arguably the best available
training signal for the *TDI arm*, and it is what we used to validate ST-RAE.

### 2.5 The metric's discount is real and much stronger than expected

Measured on the Octant credible intervals: median CI width is **0.094** log units
overall, but **2.026** below pIC50 4 versus **0.088** at or above it — a **23×
ratio** (the PXR release showed 2.8×; the difference is that Octant's sub-floor
fits are far less determined). 7.7 % of compounds fall below the floor, and they
contribute 27.9 % of total soft-threshold error against 31.3 % of plain absolute
error.

So soft-thresholding does exactly what the organizers describe, and the discount
on inactives is steep. The harness track independently confirmed that the
unpublished ST-RAE denominator does **not** change model ranking (Spearman 1.0
across all five candidate normalisation policies), so model selection can proceed
now; only the absolute number we quote will shift.

The caution: if the real test intervals are wide, blanket mid-range guessing on
inactive compounds becomes a viable leaderboard tactic. Check this against the
real intervals on day one before deciding how to treat sub-floor compounds.

---

## 3. Data inventory

Curated corpus: **37,006 unique compounds**, of which 5,706 carry pIC50 for all
four challenge isoforms (`cyp_public_corpus.parquet`, dictionary in
`cyp_corpus_data_dictionary.md`).

| isoform | compounds with pIC50 | ChEMBL | PubChem | Octant (sole source) | qualitative-only | discordant >1 log |
|---|---|---|---|---|---|---|
| CYP1A2 | 14,228 | 4,533 | 9,773 | — | 6,896 | 59 |
| CYP2C9 | 18,397 | 5,935 | 12,693 | — | 8,214 | 189 |
| CYP2D6 | 16,033 | 6,652 | 9,652 | — | 11,245 | 209 |
| CYP3A4 | 25,424 | 9,713 | 15,002 | 1,036 | 8,355 | 372 |
| CYP2C19 (auxiliary) | 9,798 | — | 9,798 | — | 7,064 | 45 |

The Octant column counts compounds for which Octant is the **only** source; 1,084
Octant compounds credit it as a CYP3A4 source in total, of which 48 are also
present in ChEMBL or PubChem.

Curation decisions worth carrying: 51,872 ChEMBL rows are PubChem re-deposits
(`src_id = 7`, same NCGC AIDs) and were tagged and excluded from aggregation —
pooling them would have faked replicate agreement. AID 885 was excluded (CYP3A4
*activators*) and AID 1963596 rejected (CYP3A7, wrong isoform). ChEMBL targets
were resolved via UniProt accession and verified as SINGLE PROTEIN / *Homo
sapiens*: CHEMBL3356, CHEMBL3397, CHEMBL289, CHEMBL340. Censored records
(15,508 upper-bound, 698 lower-bound) are flagged, never dropped.

### The TDI gap is the single largest hole in our preparation

`tdi_auxiliary_labels.parquet` has 625 tier-1 rows with retrievable sources
(FDA DDI table footnote (a), Octant paired ±preincubation deltas, and the
623-compound non-proprietary TDI database from Faramarzi et al., *Front.
Pharmacol.* **15**, doi 10.3389/fphar.2024.1451164 — note the DOI carries 2024 but
CrossRef gives an issue date of 2025-02-12, so cite it as 2025 — each entry
carrying its own DOI/PMID). Broken down by isoform:

| isoform | tier-1 compounds | positives | negatives |
|---|---|---|---|
| CYP3A4 | 622 | 306 | 316 |
| **CYP2D6** | **1** | 1 | 0 |

**The challenge scores `CYP2D6_is_TDI`, and public data supplies exactly one
labelled CYP2D6 compound.** There is no way to pretrain a CYP2D6 TDI classifier
from public data. That endpoint is entirely dependent on the training pack's
±NADPH DRC pairs, which is the strongest argument for making the shift-regression
approach (§4, rank 2) work: it can share representation across isoforms and borrow
CYP3A4's far denser signal.

Structural alerts were **measured**, not asserted: of 22 candidate MBI alerts, only
5 are significantly enriched against the 621 labelled compounds — benzodioxole
(PPV 0.87, OR 7.3), cyclopropylamine (0.71), primary aliphatic amine (0.76),
catechol (0.65), furan (0.73). Thiophene (0.40) and nitroaromatic (0.25) sit at or
below the 0.493 base rate. Only the enriched five populate tier 2, and tier 2
carries `is_TDI = null` — an alert match is a feature, not a label.

---

## 4. Method shortlist

Full table with references in `method_shortlist.csv`; the literature basis is in
`literature_review.md`. Ranked recommendation:

1. **Multitask Chemprop v2 D-MPNN initialized from CheMeleon, all six endpoints
   jointly** (4 direct pIC50 + both TDI arms), descriptors concatenated. This is
   the configuration that dominated the ExpansionRx top 20 (Chemprop in ~15 of 20,
   CheMeleon in 8, multitask in 18). Treat it as the floor to match cheaply, not
   the frontier. ~6 days, one GPU.
2. **TDI as shift regression, thresholded for MCC.** Predict Δ = pIC50(+NADPH) −
   pIC50(direct) continuously, then apply the 0.301-log cutoff at an operating
   point tuned to maximise MCC. The training pack ships both arms, so the
   continuous quantity is learnable, and this is the only route we can see to a
   usable CYP2D6 classifier given §3. ~4 days, shares the rank-1 model.
3. **Gradient-boosted descriptor ensemble blended with the D-MPNN.** Two days, no
   GPU. Justified by the two careful CYP head-to-heads in which classical GBMs
   *beat* deep models (XGBoost 90.4 % vs 88.5 % external accuracy; GBM AUC 0.92 vs
   0.89), and three of the ExpansionRx top 10 used exactly this hybrid.
4. Probe-aware curation of the public data (CYP3A4 inhibition is probe-dependent).
5. Mechanism-class alert features for the TDI track, restricted to the five
   empirically enriched alerts.
6. **ST-RAE-matched loss with calibrated uncertainty** — interval-aware objective.
   Speculative; the honest innovation-award play.
7. **TabPFN v2 stacked on frozen CheMeleon embeddings** — the combination
   OpenADMET explicitly names as the untried next step. Cheap to try (2 days).
8. On-target pretraining/augmentation over the D-MPNN.

### The bar to beat is unmeasured

Both official baselines (`cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1` and
`…-chemeleon-baseline`) were trained with `train_size: 1.0, val_size: 0,
test_size: 0` — **no validation set, no test set**, so `early_stopping: true` on
`val_loss` could never have fired and the only logged metrics are training losses
(falling 0.763 → 0.0156 over 50 epochs, i.e. heavily fit). Their training data is
**byte-identical between the two repos** (same `X_train.csv` md5), 8,068 rows with
the same 4-isoform sparsity pattern the challenge train set will have, every target
floored at exactly pIC50 4.00. The v1 README's "updated with ChEMBL 37" is a
provenance claim, not a larger dataset.

There is therefore **no organizer-published generalization number for either
baseline.** Our scaffold-grouped reference (RAE 0.675–0.807 across isoforms) is the
realistic target to improve on. We reproduced official inference successfully via
the documented Anvil path on GPU after clearing two non-obvious blockers: the
shipped recipes' `n_jobs=4` DataLoader workers hang silently in this sandbox (fix
`n_jobs=0`), and the CheMeleon foundation encoder is fetched from Zenodo record
15460715 at model-construction time rather than shipped in either repo — so an
offline environment will fail to construct the model at all.

---

## 5. Structure track: capable but not affordable locally

Boltz-2 **does** accept `ccd: HEM` alongside a SMILES ligand — verified by
extracting the component itself (43 atoms, iron, 10 conformers), not inferred from
docs. But one CYP3A4+HEM+ligand complex took **>36 GPU-minutes at minimal
settings** on the 12 GB RTX 4080, which extrapolates to **>110 GPU-hours for 184
structures**. Template docking into one of the 122 holo CYP3A4 crystal structures
we inventoried (`cyp_pdb_inventory.csv`) is the practical route.

Scoring is LDDT-PLI-primary, meaning the pose is judged on its protein–ligand
contact pattern: a perfect fold with a misplaced ligand scores near zero. Space
validation is shallow (checks only `.zip` and exactly 184 entries), so filename and
`LIG` residue-naming errors will fail silently at scoring rather than at upload.

**Recommendation: rank this track third.** It is unannounced, starts mid-challenge,
and needs compute we do not have locally. Revisit if the organizers publish
details, and budget remote GPU if we commit.

---

## 6. Risk register

| risk | evidence | mitigation |
|---|---|---|
| Public-data model lands at no-skill on real assay | R² = 0.12, ρ = 0.47 on Octant external (§2.1) | Build the pipeline to **fine-tune on the training pack**, not to ship a pre-trained model |
| **CYP2D6 TDI has 1 public label** | §3 | Shift-regression with cross-isoform sharing (rank 2); accept this endpoint is weakest |
| Arm mismatch silently biases training | Octant = active preincubation, protocol C3A4IAP (§2.4) | `arm` column is in the corpus; never pool arms without a flag |
| Activity cliffs in every test series | 43–47 % of series span >1 log (§2.3) | Select models under `parent_in_train_folds`; consider cliff-aware losses |
| Leaderboard gaps are inside noise | PXR: 28 entries statistically tied with rank 1 (MAE 0.4061–0.4360); bootstrap sd 0.02 vs top-10 gaps of 0.002 | Optimise the **paired delta** against the leader, not absolute MA-ST-RAE; stop tuning once inside the noise floor |
| Sub-floor gaming | 23× CI-width ratio (§2.5) | Test mid-range-guess strategy against real intervals on day one |
| No `ci_lower`/`ci_upper` in test ground truth | not published | **Highest-priority Discord question** — without them ST-RAE cannot be computed locally at all |
| Column-name drift between challenges | PXR used `Molecule Name` (space); CYP requires `Molecule_Name` (underscore) | Validator catches it; expect a column-mapping step on day one |
| CheMeleon fetched from Zenodo at runtime | §4 | Cache `chemeleon_mp.pt` (md5 `6a80b54fdb7de37ef0374d302f01e8ce`) before launch day |

---

## 7. Open questions for the organizers (Discord `#cyp-challenge`)

In priority order — the first is blocking:

1. **Do `ci_lower`/`ci_upper` ship with the test ground truth?** Without them
   ST-RAE cannot be computed locally and self-validation is impossible.
2. What is the credible level, and is it a quantile interval or an HDI?
3. What is the ST-RAE denominator (which trivial predictor normalises it)?
4. Is the leaderboard's `RAE` column the soft-thresholded metric or a plain RAE
   reported alongside it?
5. What is the TDI-arm ground-truth column name in the training pack?
6. Is the `direct < 4` with `4 ≤ tdi ≤ 4.301` band excluded or folded into
   assigned-negative — and are the threshold comparisons strict or inclusive?
   (The published rules leave this band undefined; our labeler treats it as
   unassignable and excludes it.)
7. What is the backend's MCC convention for a single-class submission?
8. Is the Structure Prediction track real for CYP, and if so which isoform(s),
   how many structures, and when does it open?
