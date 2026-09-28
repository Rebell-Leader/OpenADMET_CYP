"""Assemble a valid structure-track submission zip from predicted complexes.

Submission contract (Space `submission.py` + tutorial `validation/structure_validation.py`):
  * one zip, exactly 20 members, named `{Molecule_Name}.pdb`
  * ligand residue named exactly `LIG`, exactly one such residue
  * at most three chains: protein, heme cofactor, ligand
  * unique atom serials, no duplicate (chain, resid) or duplicate atom names in a residue
  * ligand connectivity/bond orders must match the released SMILES

Input is a directory of Boltz-2 style CIFs (the organizers ship one per ligand in the
tutorial repo) or PDBs; the ligand residue may be named LIG1.

Usage:
    python 03_code/build_structure_submission.py <input_dir> <output_zip>
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile, get_structure
from biotite.structure.io.pdb import PDBFile

ROOT = Path(__file__).resolve().parents[1]
STRUCT_CSV = ROOT / "02_data" / "challenge_raw" / "cyp-challenge-TEST-BLINDED_structures.csv"
LIGAND_ALIASES = ("LIG", "LIG1", "UNL", "UNK")
HEME_NAMES = ("HEM", "HEC", "HEB")


def load_complex(path: Path) -> AtomArray:
    if path.suffix.lower() in (".cif", ".mmcif"):
        arr = get_structure(CIFFile.read(path), model=1)
    else:
        arr = PDBFile.read(path).get_structure(model=1)
    return arr


def normalise(arr: AtomArray) -> AtomArray:
    """Rename the ligand to LIG and place protein / heme / ligand in chains A / B / C."""
    res = np.char.upper(arr.res_name.astype(str))
    is_heme = np.isin(res, HEME_NAMES)
    is_lig = np.isin(res, LIGAND_ALIASES) & ~is_heme
    if not is_lig.any():
        # fall back: any HETATM residue that is neither heme nor water/ion
        het = arr.hetero & ~is_heme & ~np.isin(res, ("HOH", "WAT", "NA", "CL", "MG", "ZN"))
        is_lig = het
    is_protein = ~(is_heme | is_lig)

    arr.res_name[is_lig] = "LIG"
    arr.chain_id[is_protein] = "A"
    arr.chain_id[is_heme] = "B"
    arr.chain_id[is_lig] = "C"
    # single residue per cofactor/ligand chain, numbered after the protein
    last = int(arr.res_id[is_protein].max()) if is_protein.any() else 0
    arr.res_id[is_heme] = last + 1
    arr.res_id[is_lig] = last + 2
    arr.hetero[is_heme | is_lig] = True
    arr.hetero[is_protein] = False
    return arr[is_protein | is_heme | is_lig]


def dedupe_atom_names(arr: AtomArray) -> AtomArray:
    """PDB atom names must be unique inside a residue; Boltz ligand names usually are,
    but a fallback keeps the validator's duplicate-name check happy."""
    for chain in np.unique(arr.chain_id):
        for rid in np.unique(arr.res_id[arr.chain_id == chain]):
            sel = (arr.chain_id == chain) & (arr.res_id == rid)
            names = arr.atom_name[sel].astype(str).tolist()
            if len(set(names)) == len(names):
                continue
            seen: dict[str, int] = {}
            fixed = []
            for n, el in zip(names, arr.element[sel].astype(str)):
                seen[n] = seen.get(n, 0) + 1
                fixed.append(n if seen[n] == 1 else f"{el}{sum(seen.values())}"[:4])
            arr.atom_name[sel] = fixed
    return arr


def main(in_dir: Path, out_zip: Path) -> None:
    ids = pd.read_csv(STRUCT_CSV)["Molecule_Name"].tolist()
    files = {p.stem.lower(): p for p in in_dir.iterdir() if p.suffix.lower() in
             (".cif", ".mmcif", ".pdb")}
    out_dir = out_zip.parent / (out_zip.stem + "_pdbs")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for mol_id in ids:
        key = mol_id.lower()
        match = next((p for k, p in files.items() if key in k), None)
        if match is None:
            raise FileNotFoundError(f"no predicted structure found for {mol_id} in {in_dir}")
        arr = dedupe_atom_names(normalise(load_complex(match)))
        pdb = PDBFile()
        pdb.set_structure(arr)
        target = out_dir / f"{mol_id}.pdb"
        pdb.write(target)
        written.append(target)

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in written:
            zf.write(p, arcname=p.name)
    print(f"{len(written)} structures -> {out_zip}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
