#!/usr/bin/env bash
# Clone official baseline repositories into external/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="${RECFNO_EXTERNAL_ROOT:-$ROOT/external}"
mkdir -p "$EXT"

clone_if_missing() {
  local url="$1"
  local dest="$2"
  if [[ -d "$dest/.git" ]]; then
    echo "[skip] $dest"
  else
    echo "[clone] $url -> $dest"
    git clone --depth 1 "$url" "$dest"
  fi
}

clone_if_missing "https://github.com/neuraloperator/neuraloperator.git" "$EXT/neuraloperator"
clone_if_missing "https://github.com/neuraloperator/Geo-FNO.git" "$EXT/Geo-FNO"
clone_if_missing "https://github.com/neuraloperator/physics_informed.git" "$EXT/physics_informed"

echo ""
echo "External repos ready under: $EXT"
echo ""
echo "Optional — GINO dependencies:"
echo "  pip install tensorly opt_einsum"
echo "  pip install git+https://github.com/tensorly/torch"
echo "  pip install -e $EXT/neuraloperator"
