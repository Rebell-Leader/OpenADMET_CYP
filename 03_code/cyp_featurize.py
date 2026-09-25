"""
cyp_featurize.py — reusable molecular featurization for the OpenADMET CYP
Inhibition Blind Challenge (CYP1A2 / CYP2C9 / CYP2D6 / CYP3A4).

Design notes
------------
* Every featurizer returns ``(X, names, ok_mask)``. ``ok_mask`` is a boolean
  array over the *input* SMILES marking which parsed successfully; ``X`` has
  one row per input with NaN/0 rows for failures, so row alignment with an
  input DataFrame is never silently broken. This matters for the challenge:
  the submission file must have exactly 750 rows in input order.
* Standardization is explicit and separated from featurization, because the
  organizers' own pipeline uses an ``OPENADMET_CANONICAL_SMILES`` column and
  the reference models were trained on standardized structures.
* Nothing here imports torch at module level, so the fingerprint/descriptor
  path stays usable in a CPU-only environment.

Available blocks (``FEATURIZERS`` registry):
    ecfp4        Morgan r=2 count fingerprint, 2048 bits
    ecfp6        Morgan r=3 count fingerprint, 2048 bits
    fcfp4        Morgan r=2 FEATURE-invariant count fingerprint, 2048 bits
    rdkit2d      all RDKit 2D descriptors (~210), NaN/inf sanitized
    maccs        MACCS keys, 167 bits
    avalon       Avalon count fingerprint, 1024 bits
    atompair     Atom-pair count fingerprint, 2048 bits
    rdkitfp      RDKit path fingerprint, 2048 bits
    chemeleon    CheMeleon/ChemProp learned MPNN embedding (needs torch+chemprop)

Usage
-----
    from cyp_featurize import standardize_smiles, featurize, build_matrix
    smi_std, ok = standardize_smiles(df["SMILES"])
    X, names, ok2 = featurize(smi_std, "ecfp4")
    X, names, ok3 = build_matrix(smi_std, ["ecfp4", "rdkit2d"])

Grouped CV helper:
    from cyp_featurize import scaffold_groups
    groups = scaffold_groups(smi_std)   # Bemis-Murcko scaffold id per molecule

Author: prepared 2026-08 in advance of the 2026-08-17 challenge data drop.
"""

from __future__ import annotations

import warnings
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "standardize_smiles",
    "featurize",
    "build_matrix",
    "scaffold_groups",
    "FEATURIZERS",
    "ecfp_counts",
    "rdkit_2d",
    "maccs_keys",
    "avalon_counts",
    "atom_pair_counts",
    "rdkit_path_fp",
    "chemeleon_embedding",
]

# --------------------------------------------------------------------------- #
# standardization
# --------------------------------------------------------------------------- #


def standardize_smiles(
    smiles: Iterable[str],
    *,
    largest_fragment: bool = True,
    uncharge: bool = True,
    canonical_tautomer: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardize SMILES the way an ADMET model expects.

    Steps: parse -> (optional) keep largest organic fragment (strips counterions,
    which Enamine salt forms carry) -> normalize functional groups -> reionize ->
    (optional) neutralize charges -> (optional) canonical tautomer -> canonical
    SMILES without stereo removal (stereochemistry is kept: CYP2D6 and CYP2C9
    show stereo-dependent inhibition).

    Parameters
    ----------
    smiles : iterable of str
        Input SMILES.
    largest_fragment : bool
        Keep only the largest covalent fragment. Default True.
    uncharge : bool
        Neutralize formal charges where chemically possible. Default True.
    canonical_tautomer : bool
        Canonicalize tautomers. Slow (~10 ms/mol) and occasionally changes the
        chemotype; off by default.

    Returns
    -------
    std : np.ndarray of object
        Standardized canonical SMILES; None where standardization failed.
    ok : np.ndarray of bool
        True where standardization succeeded.

    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    smiles = list(smiles)
    normalizer = rdMolStandardize.Normalizer()
    reionizer = rdMolStandardize.Reionizer()
    chooser = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()
    tauto = rdMolStandardize.TautomerEnumerator() if canonical_tautomer else None

    std: list[str | None] = []
    for smi in smiles:
        if smi is None or (isinstance(smi, float) and np.isnan(smi)):
            std.append(None)
            continue
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                std.append(None)
                continue
            if largest_fragment:
                mol = chooser.choose(mol)
            mol = normalizer.normalize(mol)
            mol = reionizer.reionize(mol)
            if uncharge:
                mol = uncharger.uncharge(mol)
            if tauto is not None:
                mol = tauto.Canonicalize(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            std.append(Chem.MolToSmiles(mol))
        except Exception:
            std.append(None)

    arr = np.array(std, dtype=object)
    return arr, np.array([s is not None for s in std], dtype=bool)


def _mols(smiles: Sequence[str]):
    """Parse SMILES to RDKit mols, returning (mols, ok_mask). None where failed."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = []
    for smi in smiles:
        if smi is None or (isinstance(smi, float) and np.isnan(smi)):
            out.append(None)
            continue
        try:
            out.append(Chem.MolFromSmiles(str(smi)))
        except Exception:
            out.append(None)
    return out, np.array([m is not None for m in out], dtype=bool)


# --------------------------------------------------------------------------- #
# fingerprint blocks
# --------------------------------------------------------------------------- #


def ecfp_counts(
    smiles: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
    use_features: bool = False,
    binary: bool = False,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Morgan (ECFP/FCFP) fingerprint as folded counts.

    Counts rather than bits: CYP inhibition is driven partly by the *number* of
    lipophilic aromatic contacts, and count vectors consistently beat binary
    ones for gradient-boosted CYP models. Set ``binary=True`` for bit vectors.

    ``radius=2`` -> ECFP4, ``radius=3`` -> ECFP6 (diameter naming).
    ``use_features=True`` gives the FCFP (pharmacophoric-invariant) variant.
    """
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        atomInvariantsGenerator=(
            rdFingerprintGenerator.GetMorganFeatureAtomInvGen() if use_features else None
        ),
    )
    mols, ok = _mols(smiles)
    X = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is None:
            continue
        fp = gen.GetFingerprint(m) if binary else gen.GetCountFingerprint(m)
        for bit, val in fp.GetNonzeroElements().items() if not binary else (
            (b, 1) for b in fp.GetOnBits()
        ):
            X[i, bit] = val
    tag = ("FCFP" if use_features else "ECFP") + str(2 * radius)
    return X, [f"{tag}_{i}" for i in range(n_bits)], ok


def rdkit_2d(
    smiles: Sequence[str],
    sanitize_values: bool = True,
    clip_abs: float = 1e6,
    as_float64: bool = True,
):
    """
    All RDKit 2D descriptors (~210 columns: MW, cLogP, TPSA, ring counts, ...).

    Physicochemistry carries real signal for CYP inhibition — lipophilicity and
    aromatic ring count are the classic drivers — so this block is a useful
    complement to fingerprints even though it is low-dimensional.

    Two sanitation steps, both load-bearing:

    * NaN/inf values (from descriptors that fail on exotic valences) become 0.0
      when ``sanitize_values``.
    * Values are clipped to +/-``clip_abs``. This is not cosmetic: ``Ipc``
      (information content of the adjacency matrix) grows factorially and
      routinely reaches 1e30+ for drug-sized molecules, which overflows float32
      and makes any linear/kernel model raise or return garbage. Tree models
      tolerate it, linear ones do not, so the block is clipped at source.

    ``as_float64`` keeps the output in double precision, which matters for the
    same reason. Set ``clip_abs=None`` to disable clipping.
    """
    from rdkit.Chem import Descriptors

    names = [n for n, _ in Descriptors.descList]
    funcs = [f for _, f in Descriptors.descList]
    mols, ok = _mols(smiles)
    X = np.full((len(mols), len(funcs)), np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, m in enumerate(mols):
            if m is None:
                continue
            for j, f in enumerate(funcs):
                try:
                    X[i, j] = f(m)
                except Exception:
                    X[i, j] = np.nan
    if sanitize_values:
        X[~np.isfinite(X)] = 0.0
    if clip_abs is not None:
        np.clip(X, -clip_abs, clip_abs, out=X)
    X = X.astype(np.float64 if as_float64 else np.float32)
    return X, [f"RDKit2D_{n}" for n in names], ok


def maccs_keys(smiles: Sequence[str]):
    """MACCS 166 structural keys (167 bits incl. unused bit 0)."""
    from rdkit.Chem import MACCSkeys

    mols, ok = _mols(smiles)
    X = np.zeros((len(mols), 167), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is None:
            continue
        for b in MACCSkeys.GenMACCSKeys(m).GetOnBits():
            X[i, b] = 1.0
    return X, [f"MACCS_{i}" for i in range(167)], ok


def avalon_counts(smiles: Sequence[str], n_bits: int = 1024):
    """Avalon fingerprint as counts — a different substructure enumeration to Morgan."""
    from rdkit.Avalon import pyAvalonTools

    mols, ok = _mols(smiles)
    X = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is None:
            continue
        fp = pyAvalonTools.GetAvalonCountFP(m, nBits=n_bits)
        for bit, val in fp.GetNonzeroElements().items():
            X[i, bit] = val
    return X, [f"Avalon_{i}" for i in range(n_bits)], ok


def atom_pair_counts(smiles: Sequence[str], n_bits: int = 2048):
    """Atom-pair count fingerprint — encodes topological distance, unlike Morgan."""
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    mols, ok = _mols(smiles)
    X = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is None:
            continue
        for bit, val in gen.GetCountFingerprint(m).GetNonzeroElements().items():
            X[i, bit] = val
    return X, [f"AtomPair_{i}" for i in range(n_bits)], ok


def rdkit_path_fp(smiles: Sequence[str], n_bits: int = 2048, max_path: int = 7):
    """RDKit linear path fingerprint (counts)."""
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits, maxPath=max_path)
    mols, ok = _mols(smiles)
    X = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is None:
            continue
        for bit, val in gen.GetCountFingerprint(m).GetNonzeroElements().items():
            X[i, bit] = val
    return X, [f"RDKitFP_{i}" for i in range(n_bits)], ok


# --------------------------------------------------------------------------- #
# learned representation
# --------------------------------------------------------------------------- #


def chemeleon_embedding(
    smiles: Sequence[str],
    model_dir: str | None = None,
    batch_size: int = 128,
    device: str = "cpu",
):
    """
    Learned MPNN embedding from a trained ChemProp/CheMeleon Anvil model.

    Runs the message-passing encoder (and its aggregation) but *not* the
    prediction head, giving a fixed-length learned descriptor per molecule.
    With ``model_dir`` pointing at an OpenADMET baseline's ``anvil_training/``
    directory, the embedding comes from a network that was fine-tuned on CYP
    pIC50 — so it is a CYP-aware representation, not a generic one.

    Requires ``torch`` and ``chemprop``. Falls back to a clear RuntimeError if
    the checkpoint cannot be loaded, so callers can degrade to fingerprints.

    Parameters
    ----------
    model_dir : str, optional
        Path to a directory containing ``model.pth`` and ``model.json``
        (an OpenADMET ``anvil_training/`` folder). If None, a randomly
        initialized encoder is used, which is only useful for shape testing.

    Returns
    -------
    (X, names, ok) with X of shape (n_smiles, hidden_dim).

    """
    import torch
    from chemprop import nn as cpnn
    from chemprop.data import MoleculeDatapoint, MoleculeDataset
    from chemprop.data.collate import collate_batch
    from torch.utils.data import DataLoader

    mols_ok = []
    idx_ok = []
    for i, smi in enumerate(smiles):
        if smi is None or (isinstance(smi, float) and np.isnan(smi)):
            continue
        try:
            dp = MoleculeDatapoint.from_smi(str(smi))
            mols_ok.append(dp)
            idx_ok.append(i)
        except Exception:
            continue
    ok = np.zeros(len(list(smiles)), dtype=bool)
    ok[idx_ok] = True

    mp = cpnn.BondMessagePassing()
    agg = cpnn.NormAggregation()

    if model_dir is not None:
        from pathlib import Path

        ckpt_path = Path(model_dir) / "model.pth"
        if not ckpt_path.exists():
            raise RuntimeError(f"no model.pth under {model_dir}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = state.get("state_dict", state) if isinstance(state, dict) else state
        mp_sd = {
            k.split("message_passing.", 1)[1]: v
            for k, v in sd.items()
            if "message_passing." in k
        }
        if not mp_sd:
            raise RuntimeError(
                f"checkpoint at {ckpt_path} has no 'message_passing.*' weights; "
                f"keys look like {list(sd)[:5]}"
            )
        # infer hidden dim from the checkpoint rather than assuming the default
        w_h = mp_sd.get("W_h.weight")
        if w_h is not None:
            mp = cpnn.BondMessagePassing(d_h=w_h.shape[0])
            mp_sd_keys = set(mp.state_dict())
            mp.load_state_dict({k: v for k, v in mp_sd.items() if k in mp_sd_keys})

    mp.eval().to(device)
    agg.to(device)
    dset = MoleculeDataset(mols_ok)
    loader = DataLoader(
        dset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    embs = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg.to(device)
            h = mp(bmg)
            embs.append(agg(h, bmg.batch).cpu().numpy())
    E = np.vstack(embs) if embs else np.zeros((0, mp.output_dim), dtype=np.float32)
    dim = E.shape[1] if E.size else mp.output_dim
    X = np.zeros((len(ok), dim), dtype=np.float32)
    X[idx_ok] = E
    return X, [f"CheMeleon_{i}" for i in range(dim)], ok


# --------------------------------------------------------------------------- #
# registry + combination
# --------------------------------------------------------------------------- #

FEATURIZERS: dict[str, Callable] = {
    "ecfp4": lambda s: ecfp_counts(s, radius=2, n_bits=2048),
    "ecfp6": lambda s: ecfp_counts(s, radius=3, n_bits=2048),
    "fcfp4": lambda s: ecfp_counts(s, radius=2, n_bits=2048, use_features=True),
    "rdkit2d": rdkit_2d,
    "maccs": maccs_keys,
    "avalon": avalon_counts,
    "atompair": atom_pair_counts,
    "rdkitfp": rdkit_path_fp,
    "chemeleon": chemeleon_embedding,
}


def featurize(smiles: Sequence[str], kind: str, **kwargs):
    """Compute one named featurization block. See ``FEATURIZERS`` for names."""
    if kind not in FEATURIZERS:
        raise KeyError(f"unknown featurizer {kind!r}; have {sorted(FEATURIZERS)}")
    fn = FEATURIZERS[kind]
    return fn(smiles, **kwargs) if kwargs else fn(smiles)


def build_matrix(smiles: Sequence[str], kinds: Sequence[str], **per_kind):
    """
    Concatenate several featurization blocks column-wise.

    Returns ``(X, names, ok)`` where ``ok`` is the AND of each block's mask, so
    a molecule is only usable if every requested block succeeded for it.
    """
    Xs, names, ok = [], [], None
    for k in kinds:
        Xi, ni, oki = featurize(smiles, k, **per_kind.get(k, {}))
        Xs.append(Xi)
        names.extend(ni)
        ok = oki if ok is None else (ok & oki)
    return np.hstack(Xs), names, ok


def scaffold_groups(smiles: Sequence[str], generic: bool = False) -> np.ndarray:
    """
    Bemis-Murcko scaffold group id per molecule, for GroupKFold.

    This is the honest CV scheme for this challenge: the test set is built by
    taking 75 potent parents and expanding each into its 10 nearest Enamine
    analogs, and the live/blind split keeps whole analog series on one side.
    Random CV over an analog-rich training set therefore massively over-reports
    performance; grouping by scaffold approximates the real evaluation.

    ``generic=True`` erases atom identity and bond order (a "graph framework"),
    giving coarser, more conservative groups. Molecules that fail to parse and
    acyclic molecules (empty scaffold) each get their own singleton group.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups = np.empty(len(list(smiles)), dtype=object)
    for i, smi in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                groups[i] = f"__unparsed_{i}"
                continue
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            if generic:
                scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
            key = Chem.MolToSmiles(scaf)
            groups[i] = key if key else f"__acyclic_{i}"
        except Exception:
            groups[i] = f"__error_{i}"
    uniq = {k: j for j, k in enumerate(dict.fromkeys(groups))}
    return np.array([uniq[g] for g in groups], dtype=int)
