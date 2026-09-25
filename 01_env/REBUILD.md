# Rebuilding the `cyp-models` environment

Conda-env export is unavailable inside the analysis sandbox (the env's metadata is
mounted read-only), so `pip_freeze.txt` is the authoritative lock. `hardware.json`
records the machine the results were produced on.

```bash
conda create -n cyp-models python=3.12
conda activate cyp-models
pip install -r 01_env/pip_freeze.txt
```

## Two dependencies that are easy to miss

1. **`socksio`** — without it `httpx`/`huggingface_hub` cannot fetch through a SOCKS
   proxy and the TabICL checkpoint download dies with an `ImportError` from
   `httpcore`. Note that installing it mid-session is not enough: `httpcore`
   caches its capability check, so the download must run in a fresh interpreter.
2. **Network allowlist** — `cas-server.xethub.hf.co` (Hugging Face Xet
   content-addressed storage, serves the TabICL checkpoint bytes) and `zenodo.org`
   (CheMeleon encoder) must both be reachable.

## Model checkpoints (not vendored — sizes/hashes for verification)

| file | source | bytes | hash |
|---|---|---|---|
| `tabicl-regressor-v2-20260212.ckpt` | HF `jingang/TabICL` | 114,324,594 | — |
| `chemeleon_mp.pt` | Zenodo record 15460715 | 34,859,448 | md5 `6a80b54fdb7de37ef0374d302f01e8ce` |

TabICL downloads its checkpoint automatically on first `fit()`. The CheMeleon
encoder must be fetched explicitly:

```bash
curl -L -o 01_env/checkpoints/chemeleon_mp.pt \
  "https://zenodo.org/records/15460715/files/chemeleon_mp.pt?download=1"
md5sum 01_env/checkpoints/chemeleon_mp.pt   # 6a80b54fdb7de37ef0374d302f01e8ce
```

## GPU note

`n_jobs=0` in any Anvil/chemprop DataLoader recipe — the shipped `n_jobs=4` workers
hang silently in a sandboxed environment.
