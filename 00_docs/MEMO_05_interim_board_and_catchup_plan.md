# MEMO 05 — Interim leaderboard review, gap diagnosis, and catch-up plan

**Date:** 2026-09-28 · **Team:** `avaliev` · **Deadline:** 2026-11-03 23:59 UTC (36 days)
**Board snapshots:** `07_leaderboards/*_2026-09-28.csv` (18 tables scraped from the Space)

---

## 1. What changed on the organizers' side

| change | detail |
|---|---|
| Interim leaderboard | live (`CURRENT_PHASE = 2`). Scored on the **full** 750-compound test set; the live board still scores half of it. Regression 219 entries, TDI 114 entries. |
| Structure track | **live**, and it is **not** the 184-compound PXR-style task we scoped in August. It is **20 CYP3A4 cryo-EM complexes** (15 ligands from the training set, 2 from the blinded test set, 3 extra). Blinded SMILES released 2026-09-23 as `cyp-challenge-TEST-BLINDED_structures.csv`. |
| Structure scoring | LDDT-PLI primary, BiSyRMSD secondary, via OpenStructure. **New:** every pose is also run through PoseBusters — an OST failure scores LDDT-PLI 0 **and** a 20 Å BiSyRMSD penalty; a PoseBusters failure scores LDDT-PLI 0. |
| Structure submission | one `.zip` of 20 PDBs, `{id}.pdb`, ligand residue named exactly `LIG`, **at most 3 chains (protein + heme + ligand)** — the heme is expected. |
| Rate limit | **4 h → 12 h** between submissions. Budget the last week accordingly. |
| Anti-gaming | new `MIN_PREDICTION_STD = 0.01`; constant-ish regression columns are rejected. |
| Tutorial repo | `OpenADMET/CYP-Challenge-Tutorial` now ships the official scoring functions, the structure validator, PoseBusters helpers, and **Boltz-2 baseline poses for all 20 structure ligands** (they contain HEM). |

## 2. Where we stand

| track | our score | our rank | field best | organizers' TabICL baseline |
|---|---|---|---|---|
| Regression (interim, MA-ST-RAE ↓) | **0.7903** (Tier 23) | **139 / 219** | 0.3721 (`preheat-to-450`) | 0.6135 |
| Regression (live board) | 0.8486 | 149 / 234 | — | 0.6755 |
| TDI (interim, MA-MCC ↑) | **0.2857** (Tier 2) | **64 / 114** | 0.3980 (`nova`) | 0.3250 |
| Structure (LDDT-PLI ↑) | — | — | **board is empty: 0 entries** | — |

Per endpoint (interim board, `07_leaderboards/our_position_interim_2026-09-28.csv`):

| endpoint | ours | our rank | field best | field median | TabICL baseline | our R² |
|---|---|---|---|---|---|---|
| CYP1A2 | 0.7303 | 140 | 0.3752 | 0.6673 | 0.5193 | 0.245 |
| CYP2C9 | 0.5728 | 143 | 0.3130 | 0.5125 | 0.4476 | 0.516 |
| **CYP2D6** | **1.2774** | 148 | 0.4568 | 1.1794 | 1.0296 | **−0.682** |
| CYP3A4 | 0.5807 | 134 | 0.3331 | 0.5144 | 0.4576 | 0.605 |
| CYP2D6 TDI (MCC) | 0.1568 | 73 | 0.3450 | 0.1829 | 0.1107 | — |
| CYP3A4 TDI (MCC) | 0.4147 | 49 | 0.5207 | 0.4091 | 0.4225 | — |

We are **behind the organizers' own stock TabICL baseline on all four regression endpoints**. Our CYP2D6 prediction is worse than predicting the training mean (R² = −0.68).

### 2.1 Submission-log correction — the entry that counts is v3, not v2

The board records our latest submission at **2026-08-22 17:46 UTC**. `SUBMISSION_LOG.md` records v3 as "not uploaded". The timestamp matches v3's build time (17:19), not v4's (20:00), so **v3 was uploaded and v4 never was**. Consequences:

* v3 scored **0.8486** on the live board against v2's **0.8284** — the third submission was a **+0.020 regression**, and it is the one currently counting.
* Our local CV ranked v3 *above* v2 (0.7111 vs 0.7150). That ordering was wrong on the board. This is the second time local CV mis-ranked two submissions.
* v4, our best local model (0.6868), has never been scored.

## 3. Diagnosis: five hypotheses tested, four eliminated

**(a) A data leak or extra labels we missed — ruled out.** The 750 test compounds share **zero** `Molecule_Name` and **zero** exact SMILES with any of the four training files. The TDI file adds 1,240 compounds beyond the inhibition file but **zero** additional direct-inhibition labels, and its labels agree with the inhibition file to 0.0 on all 6,525 shared values. The usable label matrix is 1412 / 1285 / 1493 / 2335 compounds — that is all anyone has.

**(b) The metric — not the problem.** Verified against the organizers' own `rae_soft_threshold_absolute_error`, now used directly in our experiment scripts rather than our re-implementation.

**(c) Scaffold-grouped vs random folds — irrelevant, and a false comfort.** Re-run under identical features (`03_code/run_split_regime_diagnostic.py`, `04_experiments/split_regime_diagnostic.csv`):

| fold scheme | LightGBM macro ST-RAE | median validation→train NN Tanimoto |
|---|---|---|
| random 5-fold | 0.7577 | 0.396 |
| Murcko GroupKFold | 0.7665 | 0.398 |

The two regimes are the *same* regime: Murcko grouping does not separate analog series, so our "hard" split was never hard. Every conclusion in MEMO 01 that rests on "scaffold-grouped folds" was measured at NN ≈ 0.40.

**(d) Prediction shrinkage — real, but rescaling makes it worse.** Our submitted prediction/label std ratios are 0.62 / 0.68 / 0.43 / 0.75 (CYP2D6 worst). Variance-matched affine rescaling degrades macro ST-RAE from 0.758 to 0.997. **Do not rescale**; shrinkage is the metric-optimal response to a weak model, not the cause of the weakness.

**(e) The validation instrument was pointed at the wrong regime — confirmed, and now fixed.** The blind set is an *analog expansion of potent training hits*: its compounds sit at median NN Tanimoto **0.587** to the training library, i.e. **closer to training than training compounds are to each other** (LOO median 0.450) and far closer than anything our folds ever validated at (0.40). `03_code/run_analog_cv.py` builds folds that reproduce the deployment regime — a validation compound is admitted only when a close analogue (≥ 0.55) stays in training, with admission biased toward potent compounds:

| instrument | LightGBM macro ST-RAE | median val→train NN | our board score |
|---|---|---|---|
| random / Murcko folds | 0.757 / 0.767 | 0.396 / 0.398 | — |
| **analog-expansion folds (new)** | **0.798** | **0.600** | **0.790** |

The new instrument lands within 0.01 of the board for this model family; the old ones read 0.03–0.08 low, and for our submitted pipelines 0.08–0.14 low. **All future model selection runs on the analog folds.**

**What is left is a model-capability gap.** Our whole model set — LightGBM and TabICL over ECFP4(+SVD)+descriptors — tops out around 0.76–0.80 under any instrument. The stock TabICL baseline is 0.61 and the field leader is 0.37. No amount of recalibration closes 0.42 of macro ST-RAE; a different class of model does.

### 3.1 Rank sensitivity — where the macro gain is

Macro is a plain mean, so a 0.10 gain is worth the same on any endpoint, but the *headroom* is not:

| scenario | macro | rank |
|---|---|---|
| current (v3) | 0.7903 | 139 |
| CYP2D6 → field median | 0.7658 | 132 |
| CYP2D6 → TabICL-baseline level | 0.7284 | 120 |
| **all four → TabICL-baseline level** | **0.6135** | **72** |
| all four → current field best | 0.3695 | 1 |

CYP2D6 alone holds 0.82 of the 1.69 total per-endpoint headroom.

## 4. What the field is doing

Top-10 entries publish nothing. The most informative public write-up is rank 28 (`stir_bar`, 0.4647, [repo](https://github.com/lachrymator/openadmet-cyp-challenge-public)), which reports macro CV MAE 0.447 / R² 0.688 against our in-fold R² 0.25–0.33. Their recipe, and how it differs from ours:

| their ingredient | our version | why theirs works |
|---|---|---|
| Ensemble across *representation families*: SMILES transformer, D-MPNN (Chemprop), 3D-conformer model, tabular foundation models over frozen embeddings, fingerprint, fragment model | two families (LightGBM, TabICL) over one representation | they report representation diversity beating single-model tuning |
| Masked **multitask pretraining** on public bioactivity, then fine-tune | public data added as extra **labelled rows** (+0.057, i.e. it hurt) | same corpus, different mechanism: pretraining transfers, row-mixing imports assay noise |
| 88,683 near-neighbours of the blind compounds retrieved from a public catalogue; 4,999 admitted carrying **calculated properties only** | none | densifies the neighbourhood the board actually scores |
| Non-negative stacking per endpoint (unconstrained ridge produced cancelling ±1.9/−0.6 weights that transferred badly) | fixed 50/50-style blends | a documented failure mode we have not guarded against |
| Butina cluster-disjoint folds | Murcko GroupKFold (= random, §3c) | — |

They also report that **100 % of their regression endpoint-pairs scored worse on the blind set than in CV** (median MAE +0.208) — consistent with our own optimism gap, and a reminder that the analog folds are a *ranking* instrument, not a promise.

## 5. Plan to 2026-11-03

Ordered by expected macro gain per day of work.

**P0 — stop the bleeding (today).** v3 is our counting entry at 0.8486 live / 0.7903 interim, and it is worse than v2. Upload **v4** (best local, never scored) or re-upload v2; either is expected to beat v3. The 12 h limit means one shot per half-day — do this before starting anything long.

**P1 — rebuild the regression stack.** Base learners with genuinely different inductive biases: Chemprop D-MPNN; a pretrained SMILES/graph encoder fine-tuned per endpoint (ChemBERTa / MolFormer class); tabular foundation model over frozen embeddings (TabICL, and Mitra via AutoGluon, which the rank-28 write-up found beat TabPFN on 6 of 7 embedding sources); LightGBM on raw ECFP4 counts + descriptors as the floor. Per-endpoint **non-negative** stacking, selected on the analog folds.

**P2 — CYP2D6 specifically.** Largest single hole (1.28, R² −0.68) and the field median is also above 1.0, so this is where a real edge is available. CYP2D6 binding is dominated by a basic-nitrogen/Asp301 salt bridge plus aromatic stacking — a pharmacophore the current featurization does not encode. Actions: protonation-state-aware features (QUACPAC, §6), an explicit basic-centre/distance descriptor set, and a separate error analysis of which compounds drive the negative R².

**P3 — public data as pretraining, not as rows.** We already hold a curated 37k-compound CYP corpus (`02_data/cyp_public_corpus.parquet`). Re-enter it as a masked multitask pretraining stage for the trunk, with the calculated-property heads down-weighted, and keep the challenge data as the only fine-tuning signal.

**P4 — blind-neighbourhood densification.** Retrieve public analogues of the 750 test compounds, admit those above a similarity floor as unlabelled/property-only rows. This is the one ingredient of the rank-28 recipe we have never tried in any form.

**P5 — TDI.** We fell from 1/6 to 64/114 by standing still; the board is tightly packed (top 26 entries are one tier). CYP3A4 TDI is mechanism-driven — we hold `mbi_alert_definitions.csv` and the Emax file, neither of which the current classifier uses.

**P6 — structure track: the cheapest points on the board.** Zero entries so far. A validated 20-pose zip scores immediately and the track is worth an award slot. Details in §7.

## 6. OpenEye — what the challenge licence actually gives us

The licence file (expires **2026-11-13**, ten days after the deadline) carries 19 products. Installed and **verified licensed** in a fresh env (`cyp-oe`, toolkits 2026.1.0):

`oechem` · `oeomega` · `oedocking` (FRED/HYBRID/POSIT) · `oeshape` (ROCS) · `oespruce` · `oequacpac` · `oeszybki` · `oeszmap` · `oezap` · `oegraphsim` · `oemedchem` · `oemolprop` (Filter) · `oesitehopper` · `oespicoli` · `oebioisostere` · `oeiupac` · `oedepict` · `oegrapheme` — all return `True`.

**Install note.** The `openeye` conda channel is not on our network allowlist; the working path is the pip wheel:
`pip install https://pypi.anaconda.org/openeye/simple/openeye-toolkits/2026.1.0/openeye_toolkits-2026.1.0-py312-none-manylinux_2_28_x86_64.whl`
(the anaconda.org package-storage S3 host had to be allowlisted too). Set `OE_LICENSE` to the licence file. The desktop **applications** on eyesopen.com/downloads need a customer account we do not have — but the licence `FEATURES` lines are language bindings (`python;java;clr`), and every application capability we need is exposed in the licensed Python toolkits.

### Where each toolkit earns its place

**Structure track (highest value — this is what the licence is for):**

| toolkit | use |
|---|---|
| **Spruce** | Build CYP3A4 design units from the 122 heme-containing PDB entries: cap termini, build missing loops/side chains, keep `HEM` as cofactor, generate the receptor + binding site. This is the only clean way to get a scoring-ready holo receptor with the heme intact. |
| **OMEGA** | Conformer ensembles for the 20 ligands (they are flexible: sulfonamides, azoles, amides). |
| **POSIT** | Template-guided pose prediction against each co-crystal ligand, returning a **calibrated probability of being within 2 Å** — exactly the per-ligand selection signal this track needs, and it degrades gracefully to HYBRID/FRED when no similar reference ligand exists. |
| **ROCS / EON (shape, eontk)** | Choose *which* of the 122 structures to dock into, by 3D shape and electrostatic similarity of the query to each bound ligand — the CYP3A4 pocket is plastic, so template choice matters more than the docking function. |
| **SZYBKI** | In-pocket minimization of the final pose; the direct fix for PoseBusters geometry failures, which zero a score outright. |
| **SZMAP** | Water-thermodynamics map of the pocket — flags poses that displace a stable water or leave a hydrophobic hole. |
| **Filter / molprop, oemedchem, grapheme** | Sanity, MMP/MCS analysis against training analogues, and figures. |

**Regression track (secondary but real):**

* **QUACPAC** — one consistent tautomer/protomer per compound at pH 7.4 before featurization. Our current pipeline featurizes raw input SMILES; CYP2D6 in particular is driven by the protonated basic nitrogen, and this is a plausible part of the CYP2D6 hole (P2).
* **OMEGA + ROCS** — a 3D shape/colour-similarity model as an ensemble member with an inductive bias no fingerprint model has; the analog-expansion regime is exactly where 3D similarity should pay.
* **graphsim / oemedchem** — alternative fingerprint families and matched-molecular-pair features over the training analogue series.
* **zap** — Poisson–Boltzmann electrostatic descriptors.

Licensing note for the submission form: OpenEye is proprietary **software**, not proprietary **data**. Our "Proprietary Data: No" declaration stands.

## 7. Structure-track execution plan

Target: 20 CYP3A4 complexes. Board empty. Requirements are shallow but absolute — 20 files, `LIG` resname, ≤3 chains, PoseBusters-clean.

Ligand profile (computed, `02_data/challenge_raw/cyp-challenge-TEST-BLINDED_structures.csv`): 7 imidazole/triazole, 6 pyridine-type, 9 sulfonamide, 3 nitrile; **12 of 20 carry an sp²-N able to ligate the heme iron** — a strong pose prior, and the thing docking scores most often get wrong.

1. **Receptors.** Spruce design units from a diverse subset of the 122 CYP3A4 PDB entries (pocket volume and ligand-size spread), heme retained.
2. **Ligand prep.** QUACPAC protomer → OMEGA ensemble.
3. **Template choice.** ROCS/EON against each entry's bound ligand → top-k receptors per query.
4. **Posing.** POSIT per (ligand, receptor); HYBRID where the reference ligand is dissimilar; keep the full pose set with probabilities.
5. **Physics filter.** Fe–N distance/geometry check for the 12 coordinating ligands; SZYBKI minimization; SZMAP water check.
6. **Consensus.** Score each candidate against the organizers' Boltz-2 baseline pose (shipped in the tutorial repo) and against cross-receptor agreement; also run our own Boltz-2 with heme on the local GPU as an independent member.
7. **Assembly and validation.** Protein + HEM + LIG, three chains, resname `LIG`; run `validation/structure_validation.py` and PoseBusters locally before upload.

An honest floor: converting the organizers' 20 Boltz-2 poses into a valid zip is a same-day submission that is guaranteed to beat an empty board. Do that first, then improve.

## 8. Immediate next actions

1. Upload v4 (or v2) to retire v3 — one 12 h slot, today.
2. Land the Boltz-2-derived structure submission (valid zip, 20 poses) — first entry on that board.
3. Stand up the P1 stack under the analog folds; CYP2D6 error analysis in parallel.
4. Re-run every MEMO 01 ablation that mattered under the analog instrument before trusting any of it again.
