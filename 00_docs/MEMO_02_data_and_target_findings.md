# Memo 2 — What we learned about the data and the four targets

Team `avaliev`. Last updated 2026-08-22. These are properties of the challenge
itself, independent of which model we run. Several of them constrain what any model
can achieve, so they should be read before designing the next one.

---

## 1. The train/test asymmetry is the central design fact

| property | train (4,905) | test (750) |
|---|---|---|
| internal NN ECFP4 Tanimoto, median | 0.450 | **0.656** |
| fraction with a neighbour ≥ 0.55 | 17 % | **85 %** |
| compounds with all 4 isoform labels | **41** | all 750 |
| compounds with exactly 1 label | 3,596 | — |
| label spread (mean abs dev) | 0.718 (exact) | **≈ 1.064** (estimated¹) |

¹ The training figure is the macro mean absolute deviation of the labels, computed
directly (per isoform 0.751 / 0.593 / 0.632 / 0.896). The test figure cannot be
computed — the blinded file has no labels — so it is estimated from the
`MA-MAE / MA-ST-RAE` ratio (1.122 for v1) de-biased by 1.055×; that ratio runs high
because ST-RAE's numerator is zero inside the credible interval and therefore ≤ raw
absolute error. Treat it as "roughly 48 % wider than training", not as a measurement.

The training set is a **diversity library**; the test set is **analog-series dense**,
built from 75 potency hits × their top-10 ECFP4 analogs from Enamine. Three
consequences we have measured:

1. **The test set is wider-spread, not narrower.** This is why MAE degraded 73 %
   from local CV to the board while ST-RAE degraded only 0.127 — the larger RAE
   denominator absorbs part of the damage.
2. **Analog-series CV is impossible on the training data** (only 16 % of compounds
   sit in a multi-member series), so scaffold-grouped `GroupKFold` is the working
   surrogate. It is probably pessimistic: each test compound has a training analog
   at median T = 0.587, which is an easier interpolation than a held-out scaffold.
3. **Sparse train / dense test is exploitable** — this is the basis of the
   cross-isoform stacking that gained us −0.009 (Memo 1 §2.2).

**No leakage.** Zero exact `Molecule_Name` overlap, zero exact SMILES overlap, no
test compound at Tanimoto 1.0 to any training compound.

**Local→board conversion, now calibrated on two submissions:** local CV runs
**0.113–0.127 optimistic**. Use ≈ +0.12.

---

## 2. CYP2D6 is a different problem from the other three

It resists every method we have tried: ST-RAE 0.91–0.98 (near no-skill) across
LightGBM, TabICL and stacking, and TDI MCC 0.098 against CYP3A4's 0.408. It is also
the only isoform read out by a different assay technology — label-free Echo-MS
dextromethorphan depletion, versus fluorogenic probe displacement for the other
three. We tested the two candidate explanations against each other because they have
opposite implications.

### H1 — the Echo-MS readout is noisy: **REFUTED**

| isoform | median CI width | width above floor | median `_std` | noise/signal |
|---|---|---|---|---|
| CYP1A2 | 0.328 | 0.289 | 0.084 | 0.385 |
| CYP2C9 | 0.526 | 0.432 | 0.137 | 0.729 |
| **CYP2D6** | **0.272** | **0.261** | **0.069** | 0.414 |
| CYP3A4 | 0.379 | 0.223 | 0.095 | 0.249 |

CYP2D6 has the **narrowest** credible intervals of all four isoforms and the
smallest per-compound standard deviation. Its noise-to-signal ratio is mid-pack,
better than CYP2C9's. The Echo-MS arm is if anything the *cleanest* of the four.
None of the 0.98 is explained by measurement uncertainty.

### H2 — the SAR is genuinely orthogonal: **SUPPORTED**, three independent lines

**(a) Cross-isoform correlation.** Spearman ρ on co-measured compounds only:

|  | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---|---|---|---|
| CYP1A2 | 1.000 | 0.423 | 0.059 | 0.383 |
| CYP2C9 | 0.423 | 1.000 | −0.120 | **0.688** |
| **CYP2D6** | 0.059 | −0.120 | 1.000 | 0.056 |
| CYP3A4 | 0.383 | 0.688 | 0.056 | 1.000 |

Mean off-diagonal ρ: CYP1A2 +0.288, CYP2C9 +0.330, CYP3A4 +0.375, **CYP2D6 −0.002**.
CYP2D6 potency is essentially uncorrelated with everything else in the panel, while
CYP2C9↔CYP3A4 reach +0.688.

**(b) The physicochemical drivers are inverted.** Single-feature Spearman ρ:

| feature | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---|---|---|---|
| basic N count | −0.098 | −0.190 | **+0.302** | −0.119 |
| logP | +0.233 | +0.485 | **+0.034** | **+0.602** |
| aromatic atoms | +0.212 | +0.393 | **−0.001** | +0.393 |
| TPSA | −0.148 | −0.084 | −0.204 | −0.212 |

Basic-nitrogen count is the **only** simple feature positively correlated with
CYP2D6, and it is *negative* for all three others. Conversely lipophilicity — which
dominates CYP3A4 (+0.602) and CYP2C9 (+0.485) — is inert for CYP2D6, as is aromatic
content.

**(c) A clean dose-response on basicity.**

| basic N | n | median pIC50 | fraction potent (≥ 5) |
|---|---|---|---|
| 0 | 935 | 4.617 | 27.3 % |
| 1 | 522 | 5.006 | 51.0 % |
| 2+ | 36 | 5.514 | 69.4 % |

This is textbook CYP2D6 pharmacology — the active site engages a protonated basic
amine — recovered from the challenge data alone.

### Two findings that sharpen what to do about it

**The test set is depleted in basic nitrogen**: 11.6 % of test compounds carry at
least one, against 18.9 % of training compounds (mean count 0.123 vs 0.201). The one
strong CYP2D6 handle is scarcer at test time than in training.

**Hand-built pharmacophore features do not help** (Memo 1 §3.4) — RAE 1.088, worse
than the mean. The signal is real, weak, and already captured by the full feature
set.

**UPDATE 2026-08-22 — the architectural consequence of orthogonality does not
survive dense auxiliary supervision.** This memo previously implied CYP2D6 should be
kept out of any shared trunk, and Memo 3 §3.3 was written on that basis. The
multitask ablation (Memo 1 §5c.2) tested it both ways. Without an auxiliary task the
prediction holds — excluding CYP2D6 is worth 0.0056. **With** auxiliary log2fc heads
it reverses: sharing all four is worth 0.0023. The dense screen supervision appears
to give the trunk enough isoform-specific capacity that CYP2D6 no longer competes
with the other three for representation.

The orthogonality *finding* is unchanged and still correct (mean cross-isoform
ρ = −0.002, inverted physicochemical drivers). What changes is the recommendation
derived from it: **share all four isoforms when auxiliary supervision is present,
separate CYP2D6 when it is not.**

**Conclusion.** CYP2D6's ceiling is an orthogonal, basicity-driven SAR that the
shared ECFP4-plus-descriptor family represents poorly. It is neither assay noise nor
a missing descriptor we can hand-write. Notably, CheMeleon's *only* win in the entire
representation × head matrix was CYP2D6 — which makes a **fine-tuned** learned graph
encoder the most promising remaining route for this isoform specifically.

---

## 3. The metric rewards interval-aware prediction, but accuracy dominates

ST-RAE measures distance to the nearest bound of the ground-truth credible interval,
zero inside it. So two predictors with identical MAE can score differently.

**Controlled experiment** (identical |error| from truth, sign chosen to move either
toward or away from the interval centre — MAE matched to 0.00e+00):

| variant | MAE | ST-RAE | fraction inside CI |
|---|---|---|---|
| shifted away from CI centre | 0.3994 | 0.5275 | 0.349 |
| shifted toward CI centre | 0.3994 | **0.4710** | 0.414 |

Placement is worth **≈ 0.056 MA-ST-RAE at fixed accuracy**. Real, but an order
smaller than our 0.307 gap to rank 1 — so **at most ~18 % of that gap is metric
exploitation and ≥ 82 % is raw predictive accuracy.**

The board confirms both routes exist. Ranks 6–7 (`drp-to-admet`, `drp-to-admed`) hold
the two **best MAEs on the entire board** (0.692, 0.691) yet rank only 6th–7th on
ST-RAE — pure accuracy, no interval awareness. Ranks 1–2 have *worse* MAE (0.729,
0.760) but much better ST-RAE — accuracy plus placement.

Our v2 sits at MA-R² 0.116 against rank 1's **0.507**: they explain 4.4× more
variance. That is the wall, and it is not a metric trick.

Also relevant: **credible intervals are 23× wider below the pIC50 4 assay floor**
(median 2.03 vs 0.088 log units on the Octant data; the same pattern holds in the
challenge training data). There is no exploit here — blanket mid-range guessing on
sub-floor compounds scores 0.74 against 0.21 for honest prediction — but it does mean
errors on inactives are cheap and errors on potent compounds are expensive.

---

## 4. Schema and label conventions (confirmed against the files)

* Identifier is `Molecule_Name` (underscore). The prior PXR challenge used
  `Molecule Name` with a space — do not assume.
* Credible intervals are `{endpoint}_conf_low` / `{endpoint}_conf_high`, **not**
  `ci_lower`/`ci_upper`; a per-compound `{endpoint}_std` also ships.
* Direct arm `{iso}_pIC50_direct_inhibition`; pre-incubation arm
  `{iso}_pIC50_TDI_condition`.
* **The blinded test file contains no labels and no intervals**, so ST-RAE cannot be
  computed locally on the test set at all. All local scoring is CV on training data.
* The `TRAIN_Emax.csv` file ships `is_TDI` for **all four** isoforms, not just the two
  scored — unused auxiliary signal (Memo 3).
* The organizers' `evaluation/config.py` in the tutorial repo is a PXR-era leftover
  (`ENDPOINTS = ["pEC50"]`, plain RAE, TDI metrics consuming *probabilities*). It is
  **not** the CYP scoring reference. Their `validation/*.py` is authoritative and our
  submissions pass it.

Per-isoform training coverage: CYP1A2 1,412 · CYP2C9 1,285 · CYP2D6 1,493 ·
CYP3A4 2,335. Fraction below the pIC50 4 floor: 0.164 / 0.202 / 0.086 / **0.404**.

---

## 5. Track status

* **Regression** — live, 45 entries as of 2026-08-22.
* **TDI classification** — live, 6 entries as of 2026-08-18.
* **Structure/pose** — `STRUCTURE_TRACK_LIVE = False`. The organizers turned it off
  after listing it in the Space source (184 structures, LDDT-PLI). Worth re-checking
  periodically; see Memo 3 §5 for why this matters to the OpenBind question.
