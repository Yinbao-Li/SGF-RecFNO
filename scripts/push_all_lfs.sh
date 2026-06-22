#!/usr/bin/env bash
# Upload checkpoints + heat data via Git LFS.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "Install Git LFS: sudo apt install git-lfs && git lfs install"
  exit 1
fi
git lfs install

echo "=== 1/2 Heat data (train/val/test) ==="
bash scripts/push_data_lfs.sh "$@"

echo ""
echo "=== 2/2 Checkpoints ==="
bash scripts/push_checkpoints_lfs.sh

echo ""
echo "=== Commit & push (run in your terminal) ==="
echo "  git commit -m 'Add heat dataset and checkpoint LFS assets'"
echo "  git lfs push origin main --all"
echo "  git push origin main"
