# Memo 3 — Improvement plan

Team `avaliev`. Written 2026-08-22. Intermediate deadline **2026-09-24** (33 days),
final **2026-11-03** (73 days). One submission per team; latest valid counts; one
submission per 4 hours.

Prerequisites: Memo 1 (what we tried) and Memo 2 (what the data says).

---

## 1. The problem, stated precisely

Current standing: **v2 = 0.8284 MA-ST-RAE, rank 20 of 45.** Rank 1 is 0.5209.

The gap is 0.3075, and we have decomposed it:

| component | size | evidence |
|---|---|---|
| raw predictive accuracy | ≥ 0.25 (≥ 82 %) | our MA-R² 0.116 vs rank 1's 0.507 — 4.4× less variance explained |
| interval-aware placement | ≤ 0.056 (≤ 18 %) | controlled experiment, MAE matched exactly (Memo 2 §3) |

**This is an accuracy problem, not a metric-gaming problem.** Every top-7 entry has
MA-R² ≥ 0.42; we have 0.116. Our best local R² across all experiments was 0.351
(TabICL) on the *easier* training distribution. No amount of interval-aware
post-processing bridges a 4.4× variance-explained gap.

A second fact sets the pace: the field advanced from best-0.6755 to best-0.5209 in
four days, and 38 of 45 entries postdate our v2. Standing still loses rank even when
our score improves — which is exactly what happened (v2's score would have ranked
4/16 on the 08-18 board).

---

## 2. What is closed — do not spend time here

Each of these was measured, not assumed:

| direction | why it is closed |
|---|---|
| Public-data augmentation | degrades 0.7632 → 0.8202; and the best public set (Octant) is in the wrong assay arm |
| Variance-expansion recalibration | optimal α = 1.00 on every isoform; variance-matching gives 1.0041 |
| Frozen CheMeleon embeddings | worse under both heads; concatenation also worse |
| Hand-built CYP2D6 pharmacophore features | RAE 1.088, below no-skill |
| Sub-floor metric exploitation | 0.74 vs 0.21 for honest prediction |
| Rank-then-calibrate two-stage designs | the rank/magnitude asymmetry does not exist (z = +0.64, −0.04) |
| Structure/pose track | `STRUCTURE_TRACK_LIVE = False` |

---

## 3. Priority 1 — buy accuracy (weeks 1–3)

Ordered by expected gain per unit effort. The unifying observation: **the one thing
that has reliably worked is changing the model class, twice measured** (TabICL beats
GBM by −0.039 on ECFP4 features and −0.034 on CheMeleon features). Feature
engineering has produced nothing. So push on architecture and on ensembling.

### 3.1 ~~TabICL capacity sweep~~ — DONE 2026-08-22, lever spent

Ensembling saturates at `n_estimators = 8`: macro 0.7388 (n=1) → 0.7239 (n=8) →
0.7226 (n=32), i.e. a further −0.0013 for 4× the runtime. Full-width 2,265 features
score **worse** (0.7611 vs 0.7239) and trigger CUDA OOM on the 12 GB GPU — so SVD
compression is helping, not costing us. Adopt `n_estimators=32` for the next
submission (free −0.0013), but treat tuning as finished. Details: Memo 1 §6.

### 3.2 ~~TabPFN as a second in-context learner~~ — BLOCKED

TabPFN 3.x aborts in `fit()` before touching data: it requires registration at
`ux.priorlabs.ai`, licence acceptance, and a `TABPFN_TOKEN` environment variable.
We have no such credential.

Worth unblocking if cheap, because it is the only cheap way to distinguish "in-context
learning beats GBM" from "TabICL beats GBM" — our two positive results are both
TabICL, on different features. **Action for the user:** if you want this tested,
register at `ux.priorlabs.ai` and add `TABPFN_TOKEN` as a credential; the experiment
itself takes minutes (`03_code/run_tabpfn.py` is written and ready).

### 3.3a STATUS 2026-08-22 — EXECUTED, and it worked; read Memo 1 §5c

Built (`03_code/multitask.py`, `03_code/run_multitask.py`) and benchmarked over five
variants × 3 seeds. Multitask beats single-task at equal architecture by **−0.016**;
auxiliary log2fc heads add a further **−0.015**; best single multitask config is
**0.7175**.

The larger result is the **blend**: multitask and TabICL are complementary per isoform
(multitask wins CYP2C9 by 0.051, TabICL wins CYP1A2 by 0.022), and a nested-weight
blend reaches **0.6868** — beating *both* parents on *every* isoform. That is
**−0.024 on v3** and the largest single gain of the project. Shipped as v4.

One recommendation below is now superseded: the CYP2D6 exclusion is correct only
*without* auxiliary supervision and reverses with it (Memo 2, updated).

### 3.3 Multitask neural network on all four isoforms jointly

Our four models are independent. The organizers' own baselines are multitask
(`n_tasks: 4`). With only 41 compounds carrying all four labels, a shared-trunk
network with per-isoform heads and masked loss uses the sparse label matrix far
better than four separate fits — and it is the natural generalisation of the
cross-isoform stacking that already gained −0.009. **Exception: exclude CYP2D6 from
the shared trunk, or give it its own** (Memo 2 §2 — its SAR is orthogonal, mean
cross-isoform ρ = −0.002, so sharing representation with it is actively harmful).

### 3.4a STATUS 2026-08-22 — this section has now been executed; read Memo 1 §5b

Both mechanisms were tried. **Distillation (predicted log2fc as a feature) failed**
(+0.036), because a surrogate predicted from structure adds no information the model
did not already have — and its leaky variant scored 0.2771 with R² 0.834, a 0.557 leak
that would have looked like winning the challenge. **Pseudo-labelling the ~2,900
compounds per isoform that have a screen reading but no fitted DRC worked** (−0.008),
but only after clipping pseudo-labels at the assay floor, and only on the two isoforms
an a-priori shift rule selects. Pseudo-labels and cross-isoform stacking **compete**
rather than compose (both together: −0.003, worse than either alone). The hybrid ships
as v3 at local 0.7111.

Net: the top-ranked lever in this plan yielded ~0.012, not the step change needed. The
projected board score (~0.83) is indistinguishable from v2's 0.8284. **The remaining
gap to the 0.521 frontier is structural, and §3.3 (a genuine multitask architecture)
is now the only untried item with plausible reach.** Everything below stands, but
recalibrate expectations: five successive levers have each returned ≤ 0.012.

### 3.4 Use the auxiliary data we have not touched

Two files ship unused:

* **`TRAIN_Emax.csv`** — 6,146 compounds carrying `is_TDI` for **all four** isoforms
  (CYP1A2 n=1,414 pos 1.6 %, CYP2C9 n=1,285 pos 3.0 %, CYP2D6 n=1,498 pos 21.6 %,
  CYP3A4 n=3,584 pos 21.3 %) plus `EmaxVsPosCtrl` values with credible intervals for
  the TDI condition. Emax (maximal effect) is mechanistically related to potency and
  is a free auxiliary regression target. It also contributes **1,241 compounds absent
  from the DRC training file**.
* **`single-concentration-TRAIN.csv`** — 17,504 measurements: 4,376 compounds × 4
  enzymes at 4.95e-5 M, as log2 fold-change with standard errors, p-values, FDR and
  Cohen's d.

  **Correction to an earlier draft of this memo:** I described this as "3× more
  compounds than the DRC training set". That is wrong — 4,375 of its 4,376 compounds
  are already in the DRC file, so it adds essentially no new molecules. Its actual
  value is different and still substantial: it is **dense across all four isoforms**,
  where the DRC file is sparse. Per-isoform, the number of compounds with a
  single-concentration reading but **no fitted DRC** is CYP1A2 2,964 · CYP2C9 3,091 ·
  CYP2D6 2,883 · CYP3A4 2,571. So it roughly **triples the per-isoform label count**
  (e.g. CYP2C9 goes from 1,285 DRC to 4,376 total), as a coarser but same-assay
  signal. Unlike the public corpus it is the organizers' own measurement, so the
  arm-mismatch objection does not apply — though note the screen was run in the
  **+NADPH (TDI) condition**, so it is not a direct-arm surrogate and should enter as
  an auxiliary task rather than as pseudo-labels for the direct endpoint.

  The natural use is multitask: predict log2FC alongside pIC50, letting the dense
  coarse labels regularise the sparse fine ones. This is also what makes §3.3 more
  attractive than it looks in isolation — the multitask architecture is the vehicle
  for this data.

I rank 3.4 above 3.3 on expected value, but they are the same piece of work: the
dense auxiliary labels need a multitask model to exploit them.

### 3.5 Fine-tuned CheMeleon for CYP2D6 only

Frozen embeddings failed globally but **CYP2D6 was the single winning cell** in the
whole representation × head matrix (0.9796 → 0.9472 under LightGBM). Combined with
CYP2D6's orthogonal, basicity-driven SAR, a learned graph encoder fine-tuned
end-to-end is the most promising route for the isoform that is currently near
no-skill. Scope it to CYP2D6; do not re-run it globally.

---

## 4. Priority 2 — capture the ≤ 0.056 placement gain (week 3–4)

Worth doing *after* accuracy work, because it is bounded and additive. The board
shows it is real: ranks 6–7 hold the two best MAEs on the entire board yet rank only
6th–7th on ST-RAE, while ranks 1–2 have worse MAE and better ST-RAE.

Mechanism: ST-RAE is zero inside the credible interval, so a prediction that lands
inside a wide interval is free. Concretely, predict the **interval midpoint** rather
than the conditional median where the model expects a wide interval — which means
also predicting interval width. The training data ships `_conf_low`, `_conf_high` and
`_std` per compound, so width is a learnable target. Then shift predictions toward
the predicted midpoint in proportion to predicted width.

Note this is *not* the rejected variance-recalibration idea (§2): that expanded
spread globally and failed. This is a width-conditional shift, and the controlled
experiment says it is worth ~0.056 at fixed accuracy.

---

## 5. On OpenBind-0 — assessed, and not applicable

The user flagged the OpenBind-0 release (2026-08-21). I read the announcement. My
assessment is that it does not help us, and I want to be explicit about why so the
reasoning can be checked.

**What OB0 is.** A fully open-source (Apache 2.0) **co-folding** model built on
OpenFold3, specialised for predicting the 3D structures of protein–small-molecule
complexes. Trained on PDB through June 2025, with inference-time "chemical steering"
to improve the physical validity of predicted ligand geometry. Released alongside 717
ligand-bound fragment-to-hit structures across three targets. Evaluated by pose
accuracy: success = lDDT-PLI > 0.8 and ligand RMSD < 2 Å.

**Why it does not address our bottleneck.** OB0 predicts **geometry, not affinity**.
The announcement contains no mention of affinity, IC50, potency, or any binding-strength
prediction — its metrics are pose-accuracy metrics throughout. Our problem is
regressing pIC50 (and classifying TDI). A predicted pose is not a potency, and the
step from pose to affinity is precisely the unsolved part.

Three further obstacles, each independently sufficient:

1. **The structure track is off.** `STRUCTURE_TRACK_LIVE = False`. The one place a
   co-folding model would score directly does not currently exist. (If the
   organizers re-enable it, OB0 becomes immediately relevant — that is the trigger
   to revisit, and worth periodically re-checking.)
2. **Cost.** Our earlier Boltz-2 feasibility test measured **> 36 GPU-min for a
   single CYP3A4+heme+ligand complex** at minimal settings on the local 12 GB GPU.
   For 750 test compounds × 4 isoforms that is ~1,800 GPU-hours. Docking-derived
   features for the training set as well would multiply that again.
3. **CYP-specific difficulty, on OpenBind's own evidence.** Their headline caveat is
   that co-folding performance varies enormously by target — from high accuracy on
   EV-A71 2A protease down to **success rates below 10 %** on both RdRp systems, and
   they report that target-specific fine-tuning does not reliably recover the gap.
   CYPs are a hard case of exactly this kind: a buried heme cofactor whose iron
   coordinates the ligand, and famously plastic active sites (CYP3A4 especially).
   Nothing in the release suggests CYP-like systems land on the favourable end of
   that spread.

**The honest counter-argument**, so this is not dismissed too fast: docking scores or
pose-derived descriptors *can* add signal to a ligand-based QSAR, and we do have a
CYP structure inventory (`cyp_pdb_inventory.csv`) prepared pre-launch. If we were
accuracy-saturated on ligand-based features, structure-derived features would be a
reasonable next axis. We are not: cheaper, in-domain levers (§3.4 alone triples the
compound count) remain untested. Revisit if §3 is exhausted, if the structure track
re-opens, or if remote GPU becomes available — the cost objection is the softest of
the three.

**Recommendation:** log it, do not pursue it now. It is a strong release aimed at a
different problem than the one blocking us.

---

## 6. Sequencing and submission discipline

| window | work | submission |
|---|---|---|
| now → 08-26 | §3.1 sweep, §3.2 TabPFN, both cheap | replace v2 if either gains |
| 08-26 → 09-08 | §3.4 auxiliary data (single-conc + Emax) — the biggest untapped lever | submit best |
| 09-08 → 09-18 | §3.3 multitask net (CYP2D6 separate), §3.5 CheMeleon fine-tune for CYP2D6 | submit best |
| 09-18 → 09-23 | §4 placement gain; freeze | **final intermediate entry** |
| **09-24** | — | **intermediate deadline** |
| 09-25 | interim full-test-set reveal — read paired deltas and tier structure, not ranks | — |
| 09-25 → 10-30 | act on what the reveal shows | iterate |
| 10-30 → 11-02 | freeze, re-validate with ≥ 24 h margin | **final entry** |

Rules discipline: validate every candidate against **both** our
`validate_submission.py` and the organizers' `validation/*.py` before upload; the
4-hour rate limit means a failed validation near a deadline can cost the slot.

**Forecasting rule.** Local CV runs +0.113 to +0.127 optimistic relative to the
board, calibrated on two submissions. Add ≈ 0.12 to any local MA-ST-RAE before
comparing to a leaderboard number. To reach the current rank-1 score of 0.521 we
need a **local** MA-ST-RAE near **0.40** — against our current best of 0.715. That
is the scale of the task, and it is why §3 is about model class and more data rather
than incremental tuning.

## 7. Classification track

One submission, currently rank 1 of 6 — but all six entries are statistically tied
(max z = 0.76), so that rank is a coin flip and will likely be displaced as the field
grows. The defensible improvements:

* **Shift regression instead of classification** — predict
  `pIC50(TDI) − pIC50(direct)` and threshold at 0.301, using the magnitude
  information the boolean discards.
* **Emax auxiliary labels** — `is_TDI` for all four isoforms, giving CYP1A2/CYP2C9 as
  auxiliary multitask signal for the two scored endpoints.
* **Reconsider the 1,249 no-direct-arm CYP3A4 negatives** — 34.8 % of the labels, of
  which 1,055 show measurable TDI. Training against the organizers' convention is
  correct for scoring, but down-weighting them may generalise better to test
  compounds that do have both arms.
