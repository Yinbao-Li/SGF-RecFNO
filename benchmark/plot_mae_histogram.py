#!/usr/bin/env python3
"""Overlay histogram of per-sample MAE on the full test set (1000 cases × 6 models)."""
import argparse
import csv
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from utils.bootstrap import ensure_repo_context

ensure_repo_context(heat2d_cwd=True)

from benchmark.config import COMPARISON_MODELS, COMPARISON_OUT_DIR
from benchmark.registry import MODEL_SPECS, _find_ckpt
from benchmark.run_comparison import collect_per_sample_mae_k, ensure_checkpoints
from benchmark.visualize import plot_mae_distribution_overlay

OUT_DIR = COMPARISON_OUT_DIR
CACHE_NAME = 'mae_per_sample.npz'
SUMMARY_NAME = 'mae_per_sample_summary.csv'


def _cache_key(name):
    return name.replace('-', '_')


def parse_args():
    p = argparse.ArgumentParser(description='Plot per-sample MAE distribution (test set)')
    p.add_argument('--out-dir', default=OUT_DIR)
    p.add_argument('--out-name', default='test_mae_distribution.png')
    p.add_argument('--max-samples', type=int, default=0, help='0 = full test set (1000)')
    p.add_argument('--models', nargs='+', default=None)
    p.add_argument('--use-cache', action='store_true', help='load cached per-sample MAE if present')
    p.add_argument('--copy-figures', action='store_true')
    return p.parse_args()


def _load_cache(cache_path):
    data = np.load(cache_path)
    return {_cache_key(name): data[_cache_key(name)] for name in COMPARISON_MODELS if _cache_key(name) in data}


def _save_cache(cache_path, model_maes):
    np.savez_compressed(cache_path, **{_cache_key(n): v for n, v in model_maes})


def _save_summary_csv(path, model_maes):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(
            f,
            fieldnames=['model', 'n', 'mean_k', 'median_k', 'std_k', 'p95_k', 'max_k'],
        )
        w.writeheader()
        for name, maes in model_maes:
            maes = np.asarray(maes, dtype=np.float64)
            w.writerow({
                'model': name,
                'n': len(maes),
                'mean_k': float(maes.mean()),
                'median_k': float(np.median(maes)),
                'std_k': float(maes.std()),
                'p95_k': float(np.percentile(maes, 95)),
                'max_k': float(maes.max()),
            })


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    missing = ensure_checkpoints(train_missing=False)

    if args.models:
        model_list = args.models
    else:
        model_list = [
            name for name in COMPARISON_MODELS
            if name in MODEL_SPECS and name not in missing and _find_ckpt(MODEL_SPECS[name]['exp'])
        ]

    cache_path = os.path.join(args.out_dir, CACHE_NAME)
    cached = _load_cache(cache_path) if args.use_cache and os.path.exists(cache_path) else {}

    model_maes = []
    for name in model_list:
        key = _cache_key(name)
        if key in cached:
            maes = cached[key]
            if args.max_samples:
                maes = maes[:args.max_samples]
            print(f'Loaded cache: {name}  n={len(maes)}  mean={maes.mean():.4f} K', flush=True)
        else:
            print(f'\n>>> Collecting per-sample MAE: {name} ...', flush=True)
            maes, ckpt = collect_per_sample_mae_k(name, max_samples=args.max_samples)
            print(f'  n={len(maes)}  mean={maes.mean():.4f} K  ({ckpt})', flush=True)
        model_maes.append((name, maes))

    if not model_maes:
        raise SystemExit('No models evaluated.')

    summary_path = os.path.join(args.out_dir, SUMMARY_NAME)
    _save_summary_csv(summary_path, model_maes)

    _save_cache(cache_path, model_maes)

    out_png = os.path.join(args.out_dir, args.out_name)
    highlight = 'SGF-RecFNO (K=8)' if any(n == 'SGF-RecFNO (K=8)' for n, _ in model_maes) else 'SGF-RecFNO'
    plot_mae_distribution_overlay(model_maes, out_png, highlight_model=highlight)
    print(f'Saved: {out_png}', flush=True)
    print(f'Summary: {summary_path}', flush=True)

    if args.copy_figures:
        from benchmark.figures_paths import copy_to_figures
        dst = copy_to_figures(out_png, args.out_name, category='benchmark')
        print(f'Copied: {dst}', flush=True)


if __name__ == '__main__':
    main()
