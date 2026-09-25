#!/bin/bash
# Fetch the CheMeleon foundation encoder used by the (negative) representation
# experiment in 03_code/run_chemeleon.py.
#
# chemprop downloads this at model-construction time, so the file must exist before
# that script runs. It is excluded from git (34.9 MB) but the checksum is pinned.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/01_env/checkpoints"
mkdir -p "$DEST"

URL="https://zenodo.org/records/15460715/files/chemeleon_mp.pt?download=1"
OUT="$DEST/chemeleon_mp.pt"
EXPECTED="6a80b54fdb7de37ef0374d302f01e8ce"

if [[ -f "$OUT" ]] && [[ "$(md5sum "$OUT" | cut -d' ' -f1)" == "$EXPECTED" ]]; then
  echo "already present and verified: $OUT"
  exit 0
fi

echo "downloading CheMeleon encoder (34,859,448 B) from Zenodo record 15460715 ..."
curl -L --fail -o "$OUT" "$URL"

GOT="$(md5sum "$OUT" | cut -d' ' -f1)"
if [[ "$GOT" != "$EXPECTED" ]]; then
  echo "CHECKSUM MISMATCH: got $GOT, expected $EXPECTED" >&2
  exit 1
fi
echo "verified $OUT (md5 $GOT)"
