"""Protonation-state features for the CYP2D6 hole (MEMO 05 §P2), via OpenEye QUACPAC.

CYP2D6 recognition is dominated by a protonated basic nitrogen pairing with Asp301;
MEMO 02 found basic-nitrogen count to be the only simple descriptor positively
correlated with CYP2D6 potency, and our featurisation has so far used the input SMILES
as written — i.e. the neutral form, with no protonation state at all.

This writes, for every training and blinded-test compound, the predominant protomer at
physiological pH plus its charge bookkeeping, so downstream featurisation can use the
species that actually binds.

Run in the `cyp-oe` environment with OE_LICENSE set:
    python 03_code/build_protomer_features.py
Output: 02_data/protomer_features.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openeye import oechem, oequacpac

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "02_data" / "challenge_raw"


def protomer_row(smiles: str) -> dict:
    mol = oechem.OEGraphMol()
    if not oechem.OESmilesToMol(mol, smiles):
        return dict(protomer_smiles=smiles, net_charge=0, n_pos_N=0, n_neg=0,
                    n_basic_N=0, quacpac_ok=False)
    ok = oequacpac.OEGetReasonableProtomer(mol)
    net = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    n_pos_N = sum(1 for a in mol.GetAtoms()
                  if a.GetAtomicNum() == 7 and a.GetFormalCharge() > 0)
    n_neg = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() < 0)
    # a basic nitrogen: neutral or protonated sp3 N that is not an amide/sulfonamide N
    n_basic = 0
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 7 or a.IsAromatic():
            continue
        neighbours = list(a.GetAtoms())
        withdrawing = any(
            nb.GetAtomicNum() in (6, 16) and any(
                b.GetOrder() == 2 and b.GetEnd().GetAtomicNum() in (8, 7, 16)
                for b in nb.GetBonds())
            for nb in neighbours)
        if not withdrawing:
            n_basic += 1
    return dict(protomer_smiles=oechem.OEMolToSmiles(mol), net_charge=net,
                n_pos_N=n_pos_N, n_neg=n_neg, n_basic_N=n_basic, quacpac_ok=bool(ok))


def main() -> None:
    frames = []
    for split, fname in [("train", "cyp-challenge-TRAIN_inhibition.csv"),
                         ("test", "cyp-challenge-TEST-BLINDED.csv")]:
        df = pd.read_csv(RAW / fname)[["Molecule_Name", "SMILES"]]
        rows = [protomer_row(s) for s in df.SMILES]
        frames.append(pd.concat([df.assign(split=split).reset_index(drop=True),
                                 pd.DataFrame(rows)], axis=1))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(ROOT / "02_data" / "protomer_features.parquet", index=False)
    print(out.groupby("split")[["net_charge", "n_pos_N", "n_basic_N", "quacpac_ok"]]
          .mean().round(3).to_string())
    print(f"\ncharge distribution:\n{out.net_charge.value_counts().to_string()}")
    print(f"protomer differs from input SMILES: "
          f"{(out.protomer_smiles != out.SMILES).mean():.1%}")


if __name__ == "__main__":
    main()
