#!/usr/bin/env bash
# Enable Git LFS for checkpoint files (>100 MB each).
set -euo pipefail

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "Git LFS is required to push checkpoints."
  echo "Install: sudo apt install git-lfs   (or: brew install git-lfs)"
  exit 1
fi

git lfs install
git lfs track "checkpoints/**/*.pth"
echo "Git LFS ready. Run: git add checkpoints/ && git commit && git push"
