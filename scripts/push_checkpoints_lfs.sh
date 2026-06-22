#!/usr/bin/env bash
# Upload checkpoint binaries to GitHub LFS (run from repo root).
#
# Problem: git push only uploads LFS *pointers* unless you also run git lfs push.
# Clone without LFS objects yields ~130-byte text files, not loadable .pth weights.
#
# Prerequisites:
#   sudo apt install git-lfs
#   Original .pth files (see SOURCE below)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "ERROR: install Git LFS first: sudo apt install git-lfs"
  exit 1
fi

git lfs install
git lfs track "checkpoints/**/*.pth"
git add .gitattributes

# Map: checkpoints dir -> source best checkpoint under heat2D/logs/ckpt
declare -A PAIRS=(
  [benchmark_sgf-recfno_300]="benchmark_sgf-recfno_300/best_epoch_296_loss_0.00027766.pth"
  [benchmark_isorecfno_300]="benchmark_isorecfno_300/best_epoch_294_loss_0.00062246.pth"
  [benchmark_recfno_300]="benchmark_recfno_300/best_epoch_299_loss_0.00014600.pth"
  [benchmark_pino_300]="benchmark_pino_300/best_epoch_291_loss_0.00007033.pth"
  [benchmark_geofno_300]="benchmark_geofno_300/best_epoch_299_loss_0.00074884.pth"
  [benchmark_gino_300]="benchmark_gino_300/best_epoch_295_loss_0.00738168.pth"
)

SRC_ROOT="${CHECKPOINT_SOURCE:-$ROOT/heat2D/logs/ckpt}"
missing=0

for exp in "${!PAIRS[@]}"; do
  fname="$(basename "${PAIRS[$exp]}")"
  src="$SRC_ROOT/${PAIRS[$exp]}"
  dst="$ROOT/checkpoints/$exp/$fname"

  if [[ ! -f "$src" ]]; then
    echo "MISSING source: $src"
    missing=1
    continue
  fi

  size=$(stat -c%s "$src" 2>/dev/null || stat -f%z "$src")
  if [[ "$size" -lt 1000000 ]]; then
    echo "SKIP (looks like LFS pointer, not binary): $src ($size bytes)"
    missing=1
    continue
  fi

  mkdir -p "$(dirname "$dst")"
  cp -f "$src" "$dst"
  echo "OK copied $(numfmt --to=iec "$size" 2>/dev/null || echo ${size}B) -> $dst"
done

if [[ "$missing" -eq 1 ]]; then
  echo ""
  echo "Some source checkpoints missing. Set CHECKPOINT_SOURCE to your heat2D/logs/ckpt path."
  echo "Example: CHECKPOINT_SOURCE=/path/to/heat2D/logs/ckpt bash scripts/push_checkpoints_lfs.sh"
  exit 1
fi

git add checkpoints/
git status --short checkpoints/

echo ""
echo "Commit if needed, then upload LFS objects AND git refs:"
echo "  git commit -m 'Fix checkpoint LFS binaries'   # if there are staged changes"
echo "  git lfs push origin main --all"
echo "  git push origin main"
echo ""
echo "Verify on GitHub: file size should be ~MB/GB, not ~130 bytes."
