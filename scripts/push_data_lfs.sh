#!/usr/bin/env bash
# Split heat data and prepare for Git LFS push.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC="${1:-${RECFNO_DATA_ROOT:-}/heat/temperature6000.h5}"
if [[ ! -f "$SRC" ]]; then
  # try sibling data dir
  SRC="$ROOT/../data/heat/temperature6000.h5"
fi
if [[ ! -f "$SRC" ]]; then
  echo "ERROR: temperature6000.h5 not found. Pass path as arg1 or set RECFNO_DATA_ROOT"
  exit 1
fi

echo "Source: $SRC"
python3 scripts/split_heat_dataset.py --src "$SRC" --out-dir data/heat

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "Install Git LFS: sudo apt install git-lfs && git lfs install"
  exit 1
fi

git lfs install
git lfs track "data/heat/*.h5"
git add .gitattributes data/heat/train.h5 data/heat/val.h5 data/heat/test.h5 data/heat/splits.json

echo ""
echo "Next:"
echo "  git commit -m 'Add train/val/test heat dataset (Git LFS)'"
echo "  git lfs push origin main --all"
echo "  git push origin main"
echo ""
ls -lh data/heat/*.h5
