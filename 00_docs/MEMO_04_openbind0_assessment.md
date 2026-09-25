# Memo 4 — OpenBind-0: assessment for this challenge

Written 2026-08-22, from the release announcement of 2026-08-21
(<https://openbind.uk/news/blog-openbind-0-advancing-open-molecular-structure-prediction/>).

**Recommendation: log it, do not pursue it now.** It is a strong release aimed at a
different problem than the one blocking us. Revisit if the structure track re-opens,
if we exhaust the ligand-based levers in Memo 3 §3, or if substantial remote GPU
becomes available.

---

## 1. What OpenBind-0 is

A fully open-source (Apache 2.0) **co-folding** model — code, weights, training data
and training recipes all public — built on OpenFold3 and specialised for predicting
the 3D structure of a protein bound to a small-molecule ligand. Trained on PDB data
through June 2025, using a near-final OpenFold3 architecture, with checkpoints
selected for protein–ligand performance. It introduces inference-time "chemical
steering" during diffusion sampling to improve the physical validity of predicted
ligand geometry.

Released alongside a dataset of 717 ligand-bound structures capturing
fragment-to-hit progression across three targets (Dengue-2 and Zika NS5 RdRp, and
fatty acid thioesterase A), comprising 547 fragment and 170 hit binding events.

Evaluation follows the Runs N' Poses methodology on post-cutoff complexes binned by
similarity to the nearest training example. Success is defined as **lDDT-PLI > 0.8
and ligand RMSD < 2 Å** — pose-accuracy criteria throughout. On those benchmarks OB0
is reported as competitive with the strongest available models.

Where to get it: `github.com/aqlaboratory/openfold-3` (release tag v0.5.0).

## 2. Why it does not address our bottleneck

**OB0 predicts geometry, not affinity.** I searched the announcement for any
binding-strength capability — affinity, IC50, potency, binding free energy — and
found none; every reported metric is a pose-accuracy metric. Our task is regressing
pIC50 and classifying time-dependent inhibition. A predicted pose is not a potency,
and pose→affinity is precisely the step that remains unsolved.

Our measured bottleneck is raw predictive accuracy in a ligand-based regression:
MA-R² 0.116 against rank 1's 0.507, with ≥ 82 % of the 0.307 score gap attributable
to accuracy rather than metric handling (Memo 2 §3). Nothing in a pose predictor
speaks to that directly.

## 3. Three further obstacles, each independently sufficient

1. **The structure track is switched off.** `STRUCTURE_TRACK_LIVE = False` in the
   challenge Space config. The one place a co-folding model would score directly
   does not currently exist as a competition surface.

2. **Cost.** Our own earlier feasibility test with Boltz-2 measured **> 36 GPU-min
   for a single CYP3A4 + heme + ligand complex** at minimal settings on the local
   12 GB GPU. For 750 test compounds × 4 isoforms that is on the order of 1,800
   GPU-hours, before any training-set poses. OB0 is an OpenFold3-class diffusion
   model, so its per-complex cost is comparable or higher.

3. **CYPs are a hard case, on OpenBind's own evidence.** Their headline caveat is
   that co-folding accuracy varies enormously by target — from high accuracy on
   EV-A71 2A protease down to **success rates below 10 %** on both RdRp systems —
   and they report that target-specific fine-tuning does not reliably recover the
   gap. CYPs have the features that make this hard: a buried heme cofactor whose
   iron coordinates the ligand, and notoriously plastic active sites (CYP3A4
   especially, which can bind two substrates at once). Nothing in the release
   suggests CYP-like systems fall on the favourable end of that spread.

## 4. The counter-argument, stated fairly

Docking scores and pose-derived descriptors *can* add signal to a ligand-based QSAR,
and we are not starting from nothing: `00_docs/cyp_pdb_inventory.csv` catalogues
holo CYP structures, prepared pre-launch, and a Boltz-2 input template with heme
supplied as a CCD ligand was verified to work. So the structural route is
*available*, not merely hypothetical.

The reason to defer is opportunity cost, not impossibility. We have untested
in-domain levers that are far cheaper: the single-concentration screen roughly
triples the per-isoform label count using the organizers' own assay (Memo 3 §3.4),
TabPFN is installed and unrun, and a multitask architecture over the sparse label
matrix has not been built. Structure-derived features are the reasonable next axis
*once* ligand-based accuracy is saturated. It is not.

Of the three obstacles, cost is the softest — remote GPU would dissolve it. The
structure track being off and the pose→affinity gap are the substantive ones.

## 5. Trigger conditions to revisit

* The organizers re-enable the structure/pose track (worth re-checking the Space
  config periodically — it was live in the source before launch and then disabled).
* Memo 3 §3 is exhausted without closing the accuracy gap.
* A future OpenBind release adds affinity prediction — they state that later versions
  will "add capabilities tailored to small-molecule applications", so this is
  plausible rather than idle speculation.
