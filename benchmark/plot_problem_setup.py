#!/usr/bin/env python3
"""Draw annotated benchmark setup figure (BCs, sensors, sample field)."""
import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from utils.bootstrap import ensure_repo_context

ensure_repo_context(heat2d_cwd=True)

from benchmark.config import VIS_INDEX
from benchmark.visualize import plot_benchmark_problem_setup
from data.dataset import _load_fields

# Must match scripts/generate_temperature6000.py
N = 200
L = 0.1
DIRICHLET_J0 = N // 2 - 8
DIRICHLET_J1 = N // 2 + 8


def parse_args():
    p = argparse.ArgumentParser(description='Plot heat benchmark problem setup')
    p.add_argument('--sample-idx', type=int, default=VIS_INDEX)
    p.add_argument('--out-dir', default=os.path.join('logs', 'benchmark_comparison'))
    p.add_argument('--out-name', default='benchmark_problem_setup.png')
    p.add_argument('--copy-figures', action='store_true',
                   help='also save to ../figures/ for README')
    return p.parse_args()


def load_field_k(sample_idx: int) -> np.ndarray:
    arr = _load_fields([sample_idx])
    if arr.ndim == 4:
        field = arr[0, 0]
    elif arr.ndim == 3:
        field = arr[0]
    else:
        field = arr
    return np.asarray(field, dtype=np.float64)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    field_k = load_field_k(args.sample_idx)
    out_png = os.path.join(args.out_dir, args.out_name)
    plot_benchmark_problem_setup(
        field_k,
        out_png,
        sample_idx=args.sample_idx,
        domain_size=L,
        grid_size=N,
        dirichlet_j0=DIRICHLET_J0,
        dirichlet_j1=DIRICHLET_J1,
    )
    print(f'Saved: {out_png}', flush=True)

    if args.copy_figures:
        from benchmark.figures_paths import copy_to_figures
        dst = copy_to_figures(out_png, args.out_name, category='setup')
        print(f'Copied: {dst}', flush=True)


if __name__ == '__main__':
    main()
