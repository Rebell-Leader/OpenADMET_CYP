"""P1 base learners: out-of-fold predictions on the analog folds + blind-set predictions.

One learner per invocation so they can be run and re-run independently:

    python 03_code/run_p1.py lgbm_fp
    python 03_code/run_p1.py tabicl_svd
    python 03_code/run_p1.py chemberta
    python 03_code/run_p1.py chemprop_mt
    python 03_code/run_p1.py chemprop_pre     # public-corpus pretraining, then fine-tune

Each writes `04_experiments/p1/<learner>.npz` with `oof` (n_train, 4) and
`test_pred` (750, 4). `run_p1_stack.py` scores them and combines them.

The point of the set is *representation diversity*, not more capacity in one family:
count fingerprints (LightGBM), an in-context tabular learner over a compressed
fingerprint (TabICL), a pretrained SMILES transformer's frozen embedding, and a
learned graph representation (Chemprop D-MPNN), with and without transfer from the
37k-compound public CYP corpus.
"""

from __future__ import annotations

import copy
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from p1_common import (  # noqa: E402
    ISO, N_FOLDS, ROOT, SEED, TARGETS, build_folds, features, labels,
    load_challenge, save_learner,
)


# --------------------------------------------------------------------------- setup
def get_data() -> dict:
    inh, test = load_challenge()
    y, lo, hi = labels(inh)
    fp_tr, desc_tr, fp_te, desc_te, bits_tr, _ = features(
        inh.SMILES.tolist(), test.SMILES.tolist()
    )
    folds = build_folds(bits_tr, y)
    return dict(inh=inh, test=test, y=y, lo=lo, hi=hi, fp_tr=fp_tr, desc_tr=desc_tr,
                fp_te=fp_te, desc_te=desc_te, folds=folds,
                smiles_tr=inh.SMILES.tolist(), smiles_te=test.SMILES.tolist())


# ------------------------------------------------------------------ tabular models
def _lgbm():
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=700, learning_rate=0.04, num_leaves=31, colsample_bytree=0.6,
        subsample=0.8, subsample_freq=1, min_child_samples=10, reg_lambda=1.0,
        random_state=SEED, verbose=-1,
    )


def _per_endpoint_tabular(d: dict, make_X, make_model, name: str):
    """Fit one model per endpoint per fold on a standardised feature block."""
    from sklearn.preprocessing import StandardScaler

    y, folds = d["y"], d["folds"]
    fold_of = folds["fold_of"]
    n, n_te = len(y), len(d["smiles_te"])
    oof = np.full((n, 4), np.nan)
    test_pred = np.zeros((n_te, 4))

    for j, target in enumerate(TARGETS):
        labelled = np.isfinite(y[:, j])
        for f in range(N_FOLDS):
            tr = np.where(labelled & (fold_of != f))[0]
            va = np.where(labelled & (fold_of == f))[0]
            if len(va) == 0:
                continue
            Xtr, Xva, _ = make_X(d, tr, va, None)
            m = make_model()
            m.fit(Xtr, y[tr, j])
            oof[va, j] = m.predict(Xva)
        tr = np.where(labelled)[0]
        Xtr, _, Xte = make_X(d, tr, None, "test")
        m = make_model()
        m.fit(Xtr, y[tr, j])
        test_pred[:, j] = m.predict(Xte)
        print(f"  {name} {target}: {labelled.sum()} labelled rows done", flush=True)
    return oof, test_pred


def _X_raw(d, tr, va, want_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(d["desc_tr"][tr])
    Xtr = np.hstack([d["fp_tr"][tr], sc.transform(d["desc_tr"][tr])])
    Xva = (np.hstack([d["fp_tr"][va], sc.transform(d["desc_tr"][va])])
           if va is not None else None)
    Xte = (np.hstack([d["fp_te"], sc.transform(d["desc_te"])])
           if want_test else None)
    return Xtr, Xva, Xte


def _X_svd(d, tr, va, want_test, n_comp=256):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    svd = TruncatedSVD(n_components=n_comp, random_state=SEED).fit(d["fp_tr"][tr])
    sc = StandardScaler().fit(d["desc_tr"][tr])
    Xtr = np.hstack([svd.transform(d["fp_tr"][tr]), sc.transform(d["desc_tr"][tr])])
    Xva = (np.hstack([svd.transform(d["fp_tr"][va]), sc.transform(d["desc_tr"][va])])
           if va is not None else None)
    Xte = (np.hstack([svd.transform(d["fp_te"]), sc.transform(d["desc_te"])])
           if want_test else None)
    return Xtr, Xva, Xte


def learner_lgbm_fp(d):
    return _per_endpoint_tabular(d, _X_raw, _lgbm, "lgbm_fp")


def learner_lgbm_prot(d):
    """Same model class as lgbm_fp, but featurised on the predominant protomer at
    physiological pH (QUACPAC, `build_protomer_features.py`) with explicit charge
    bookkeeping appended. The CYP2D6 pharmacophore is a protonated basic nitrogen;
    the neutral input SMILES cannot express it."""
    from p1_common import _fps_and_desc
    prot = pd.read_parquet(ROOT / "02_data" / "protomer_features.parquet")
    chg_cols = ["net_charge", "n_pos_N", "n_neg", "n_basic_N"]
    pm = prot.set_index(["split", "Molecule_Name"])
    tr_rows = pm.loc["train"].reindex(d["inh"].Molecule_Name)
    te_rows = pm.loc["test"].reindex(d["test"].Molecule_Name)
    assert tr_rows.protomer_smiles.notna().all() and te_rows.protomer_smiles.notna().all()

    cache = ROOT / "04_experiments" / "cache" / "protomer_fp.npz"
    if cache.exists():
        z = np.load(cache)
        fp_tr, desc_tr, fp_te, desc_te = z["fp_tr"], z["desc_tr"], z["fp_te"], z["desc_te"]
    else:
        fp_tr, desc_tr, _ = _fps_and_desc(tr_rows.protomer_smiles.tolist())
        fp_te, desc_te, _ = _fps_and_desc(te_rows.protomer_smiles.tolist())
        np.savez_compressed(cache, fp_tr=fp_tr, desc_tr=desc_tr, fp_te=fp_te, desc_te=desc_te)
    desc_tr = np.hstack([desc_tr, tr_rows[chg_cols].to_numpy(float)])
    desc_te = np.hstack([desc_te, te_rows[chg_cols].to_numpy(float)])

    dd = dict(d, fp_tr=fp_tr, desc_tr=desc_tr, fp_te=fp_te, desc_te=desc_te)
    return _per_endpoint_tabular(dd, _X_raw, _lgbm, "lgbm_prot")


def learner_tabicl_svd(d):
    from tabicl import TabICLRegressor
    make = lambda: TabICLRegressor(n_estimators=32, random_state=SEED)  # noqa: E731
    return _per_endpoint_tabular(d, _X_svd, make, "tabicl_svd")


# ------------------------------------------------------- frozen transformer + LGBM
def chemberta_embeddings(smiles: list[str], tag: str,
                         model_id: str = "DeepChem/ChemBERTa-77M-MLM") -> np.ndarray:
    """Mean-pooled last-hidden-state embeddings, cached on disk."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    cache = ROOT / "04_experiments" / "cache" / f"chemberta_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModel.from_pretrained(model_id).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = mod.to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), 128):
            batch = tok(smiles[i:i + 128], padding=True, truncation=True,
                        max_length=256, return_tensors="pt").to(dev)
            h = mod(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    emb = np.vstack(out).astype(np.float32)
    np.save(cache, emb)
    return emb


def learner_chemberta(d):
    emb_tr = chemberta_embeddings(d["smiles_tr"], "train")
    emb_te = chemberta_embeddings(d["smiles_te"], "test")

    def make_X(dd, tr, va, want_test):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(np.hstack([emb_tr[tr], dd["desc_tr"][tr]]))
        f = lambda E, D: sc.transform(np.hstack([E, D]))  # noqa: E731
        return (f(emb_tr[tr], dd["desc_tr"][tr]),
                f(emb_tr[va], dd["desc_tr"][va]) if va is not None else None,
                f(emb_te, dd["desc_te"]) if want_test else None)

    return _per_endpoint_tabular(d, make_X, _lgbm, "chemberta")


# ----------------------------------------------------------------------- chemprop
def _chemprop_fit(smis_tr, Y_tr, smis_pred, seed, max_epochs=60, init_from=None):
    """Masked multitask D-MPNN. Returns predictions for `smis_pred` (n, n_tasks)."""
    import lightning.pytorch as pl
    import torch
    from chemprop import data, featurizers, models, nn
    from lightning.pytorch.callbacks import EarlyStopping

    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")
    fz = featurizers.SimpleMoleculeMolGraphFeaturizer()
    n_tasks = Y_tr.shape[1]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(smis_tr))
    n_val = max(40, int(0.1 * len(idx)))
    va_i, tr_i = idx[:n_val], idx[n_val:]
    mk = lambda ii: [data.MoleculeDatapoint.from_smi(smis_tr[i], Y_tr[i]) for i in ii]  # noqa: E731
    ds_tr = data.MoleculeDataset(mk(tr_i), fz)
    scaler = ds_tr.normalize_targets()
    ds_va = data.MoleculeDataset(mk(va_i), fz)
    ds_va.normalize_targets(scaler)
    dl_tr = data.build_dataloader(ds_tr, batch_size=64, num_workers=0)
    dl_va = data.build_dataloader(ds_va, batch_size=64, shuffle=False, num_workers=0)

    mp = nn.BondMessagePassing(depth=4, d_h=400)
    agg = nn.MeanAggregation()
    if init_from is not None:
        # deep-copy, or every fold would keep fine-tuning the SAME trunk object and
        # inherit weights already updated on its own validation compounds
        mp = copy.deepcopy(init_from.message_passing)
        agg = copy.deepcopy(init_from.agg)
    ffn = nn.RegressionFFN(n_tasks=n_tasks, input_dim=mp.output_dim, n_layers=2, dropout=0.1,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
    model = models.MPNN(mp, agg, ffn, batch_norm=True)
    trainer = pl.Trainer(
        max_epochs=max_epochs, accelerator="auto", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=12, mode="min")],
        gradient_clip_val=5.0,
    )
    trainer.fit(model, dl_tr, dl_va)

    ds_pred = data.MoleculeDataset(
        [data.MoleculeDatapoint.from_smi(s) for s in smis_pred], fz)
    dl_pred = data.build_dataloader(ds_pred, batch_size=128, shuffle=False, num_workers=0)
    pred = np.concatenate([p.numpy() for p in trainer.predict(model, dl_pred)])
    return pred, model


def _public_pretrain(seed: int):
    """Multitask pretraining on the curated public CYP corpus (5 isoforms)."""
    corpus = pd.read_parquet(ROOT / "02_data" / "cyp_public_corpus.parquet")
    cols = [f"{i}_pIC50" for i in ISO + ["CYP2C19"]]
    corpus = corpus.dropna(subset=cols, how="all")
    smis = corpus.std_smiles.tolist()
    Y = corpus[cols].to_numpy(float)
    print(f"  pretraining on {len(smis)} public compounds, {np.isfinite(Y).sum()} labels",
          flush=True)
    _, model = _chemprop_fit(smis, Y, smis[:2], seed, max_epochs=25)
    return model


def learner_chemprop(d, pretrain: bool, n_seeds: int = 2):
    y, folds = d["y"], d["folds"]
    fold_of = folds["fold_of"]
    n, n_te = len(y), len(d["smiles_te"])
    oof = np.zeros((n, 4)); oof_w = np.zeros((n, 4))
    test_pred = np.zeros((n_te, 4))

    for seed in range(n_seeds):
        base = _public_pretrain(seed) if pretrain else None
        for f in range(N_FOLDS):
            tr = np.where(fold_of != f)[0]
            va = np.where(fold_of == f)[0]
            t0 = time.time()
            pred, _ = _chemprop_fit([d["smiles_tr"][i] for i in tr], y[tr], 
                                    [d["smiles_tr"][i] for i in va], seed,
                                    init_from=base)
            oof[va] += pred; oof_w[va] += 1
            print(f"  seed {seed} fold {f}: {len(tr)} train / {len(va)} val "
                  f"({time.time()-t0:.0f}s)", flush=True)
        pred_te, _ = _chemprop_fit(d["smiles_tr"], y, d["smiles_te"], seed, init_from=base)
        test_pred += pred_te / n_seeds
    oof = np.where(oof_w > 0, oof / np.maximum(oof_w, 1), np.nan)
    oof[~np.isfinite(y)] = np.nan
    return oof, test_pred


def learner_public_lgbm(d):
    """A pure transfer member: LightGBM trained *only* on the 37k-compound public CYP
    corpus, applied to challenge compounds. It sees no challenge labels, so its
    predictions are automatically out-of-fold; the stack decides per endpoint how much
    the public assay consensus is worth. This is the third distinct way to use public
    data (prior/member), after extra rows (failed, MEMO 01) and pretraining
    (`chemprop_pre`)."""
    from p1_common import _fps_and_desc
    from sklearn.preprocessing import StandardScaler
    corpus = pd.read_parquet(ROOT / "02_data" / "cyp_public_corpus.parquet")
    cache = ROOT / "04_experiments" / "cache" / "public_corpus_fp.npz"
    if cache.exists():
        z = np.load(cache); fp_p, desc_p = z["fp"], z["desc"]
    else:
        fp_p, desc_p, _ = _fps_and_desc(corpus.std_smiles.tolist())
        np.savez_compressed(cache, fp=fp_p, desc=desc_p)
    oof = np.full((len(d["y"]), 4), np.nan)
    test_pred = np.zeros((len(d["smiles_te"]), 4))
    for j, iso in enumerate(ISO):
        yp = corpus[f"{iso}_pIC50"].to_numpy(float)
        m = np.isfinite(yp)
        sc = StandardScaler().fit(desc_p[m])
        Xp = np.hstack([fp_p[m], sc.transform(desc_p[m])])
        mdl = _lgbm(); mdl.fit(Xp, yp[m])
        oof[:, j] = mdl.predict(np.hstack([d["fp_tr"], sc.transform(d["desc_tr"])]))
        test_pred[:, j] = mdl.predict(np.hstack([d["fp_te"], sc.transform(d["desc_te"])]))
        print(f"  public_lgbm {iso}: trained on {int(m.sum())} public compounds", flush=True)
    oof[~np.isfinite(d["y"])] = np.nan
    return oof, test_pred


def learner_chemprop_prot(d, n_seeds: int = 2):
    """Chemprop on the QUACPAC protomer, i.e. the graph the enzyme actually sees.
    Protonation lifted the fingerprint model by 0.033 macro ST-RAE; a message-passing
    network reads formal charge as an atom feature, so it should transfer."""
    prot = pd.read_parquet(ROOT / "02_data" / "protomer_features.parquet")
    pm = prot.set_index(["split", "Molecule_Name"])
    dd = dict(d,
              smiles_tr=pm.loc["train"].reindex(d["inh"].Molecule_Name).protomer_smiles.tolist(),
              smiles_te=pm.loc["test"].reindex(d["test"].Molecule_Name).protomer_smiles.tolist())
    return learner_chemprop(dd, pretrain=False, n_seeds=n_seeds)


LEARNERS = {
    "public_lgbm": learner_public_lgbm,
    "chemprop_prot": learner_chemprop_prot,
    "lgbm_fp": learner_lgbm_fp,
    "lgbm_prot": learner_lgbm_prot,
    "tabicl_svd": learner_tabicl_svd,
    "chemberta": learner_chemberta,
    "chemprop_mt": lambda d: learner_chemprop(d, pretrain=False),
    "chemprop_pre": lambda d: learner_chemprop(d, pretrain=True),
}


if __name__ == "__main__":
    which = sys.argv[1]
    d = get_data()
    fo = d["folds"]
    print(f"folds: {np.bincount(fo['fold_of'])}, scored {fo['scored'].sum()}, "
          f"headline {fo['headline'].sum()}, median NN {np.median(fo['nn_sim']):.3f}",
          flush=True)
    t0 = time.time()
    oof, test_pred = LEARNERS[which](d)
    save_learner(which, oof, test_pred, meta=dict(seconds=round(time.time() - t0, 1)))
