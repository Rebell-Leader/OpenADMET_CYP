"""RDKit standardization pipeline for the OpenADMET CYP public corpus.

Order of operations (single pipeline used for every source):
  1. Cleanup           - sanitize, disconnect metals, normalize functional groups, reionize
  2. LargestFragment   - strip salts / solvents / counterions
  3. Uncharge          - neutralise charges where chemically sensible
  4. Canonical tautomer- RDKit TautomerEnumerator.Canonicalize
  5. Canonical SMILES + full InChIKey + 14-char skeleton InChIKey
"""
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

_lfc = rdMolStandardize.LargestFragmentChooser()
_unc = rdMolStandardize.Uncharger()
_te = rdMolStandardize.TautomerEnumerator()
_te.SetMaxTautomers(200)
_te.SetMaxTransforms(200)


def standardize(smi):
    out = dict(std_smiles=None, inchikey=None, inchikey14=None, std_status=None,
               mw=None, n_heavy=None)
    if not isinstance(smi, str) or not smi.strip():
        out["std_status"] = "empty"
        return out
    m = Chem.MolFromSmiles(smi)
    if m is None:
        out["std_status"] = "parse_fail"
        return out
    try:
        m = rdMolStandardize.Cleanup(m)
        m = _lfc.choose(m)
        m = _unc.uncharge(m)
        status = "ok"
        try:
            mt = _te.Canonicalize(m)
            if mt is not None:
                m = mt
        except Exception:
            status = "ok_no_tautomer"
        if m.GetNumHeavyAtoms() == 0:
            out["std_status"] = "no_heavy_atoms"
            return out
        Chem.SanitizeMol(m)
        out["std_smiles"] = Chem.MolToSmiles(m, canonical=True)
        ik = inchi.MolToInchiKey(m)
        out["inchikey"] = ik or None
        out["inchikey14"] = ik.split("-")[0] if ik else None
        out["mw"] = round(Descriptors.MolWt(m), 3)
        out["n_heavy"] = m.GetNumHeavyAtoms()
        out["std_status"] = status
    except Exception as e:
        out["std_status"] = "error:" + type(e).__name__
    return out
