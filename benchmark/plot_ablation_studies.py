#!/usr/bin/env python3
"""Evaluate ablation checkpoints and plot loss / quantile-K study figures."""
import argparse
import csv
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.bootstrap import ensure_repo_context

ensure_repo_context(heat2d_cwd=True)

from benchmark.ablation import collect_loss_ablation_results, collect_quantile_ablation_results
from benchmark.config import COMPARISON_OUT_DIR
from benchmark.visualize import plot_loss_ablation_bars, plot_quantile_k_curve

OUT_DIR = COMPARISON_OUT_DIR


def parse_args():
    p = argparse.ArgumentParser(description='Plot SGF ablation study results')
    p.add_argument('--out-dir', default=OUT_DIR)
    p.add_argument('--study', choices=['loss', 'quantile', 'all'], default='all')
    p.add_argument('--max-samples', type=int, default=0, help='0 = full test set')
    p.add_argument('--copy-figures', action='store_true')
    return p.parse_args()


def save_rows(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    loss_rows, quantile_rows = [], []

    if args.study in ('loss', 'all'):
        print('Evaluating loss-component ablations...', flush=True)
        loss_rows = collect_loss_ablation_results(max_samples=args.max_samples)
        for r in loss_rows:
            if r.get('missing'):
                print(f"  MISSING ckpt: {r['label']} ({r['exp']})", flush=True)
            else:
                print(f"  {r['label']:12s}  MAE={r['mae_k']:.4f} K", flush=True)

    if args.study in ('quantile', 'all'):
        print('\nEvaluating quantile-K ablations...', flush=True)
        quantile_rows = collect_quantile_ablation_results(max_samples=args.max_samples)
        for r in quantile_rows:
            if r.get('missing'):
                print(f"  MISSING ckpt: K={r['k']} ({r['exp']})", flush=True)
            else:
                print(f"  K={r['k']}  MAE={r['mae_k']:.4f} K  quantiles={r['quantiles']}", flush=True)

    results = {'loss': loss_rows, 'quantile': quantile_rows}
    json_path = os.path.join(args.out_dir, 'ablation_studies_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {json_path}', flush=True)

    if loss_rows:
        csv_path = os.path.join(args.out_dir, 'ablation_loss_results.csv')
        save_rows(csv_path, loss_rows)
        loss_png = os.path.join(args.out_dir, 'ablation_loss_mae.png')
        plot_loss_ablation_bars(loss_rows, loss_png, metric='mae_k')
        print(f'Saved: {loss_png}', flush=True)
        print(f'Saved: {csv_path}', flush=True)

    if quantile_rows:
        csv_path = os.path.join(args.out_dir, 'ablation_quantile_k_results.csv')
        save_rows(csv_path, quantile_rows)
        k_png = os.path.join(args.out_dir, 'ablation_quantile_k_mae.png')
        plot_quantile_k_curve(quantile_rows, k_png, metric='mae_k')
        print(f'Saved: {k_png}', flush=True)
        print(f'Saved: {csv_path}', flush=True)

    if args.copy_figures:
        from benchmark.figures_paths import copy_to_figures
        for name in ('ablation_loss_mae.png', 'ablation_quantile_k_mae.png'):
            src = os.path.join(args.out_dir, name)
            if os.path.exists(src):
                dst = copy_to_figures(src, name, category='ablation')
                print(f'Copied: {dst}', flush=True)

    missing = [r for r in loss_rows + quantile_rows if r.get('missing')]
    if missing:
        print(
            f'\n{len(missing)} experiment(s) missing checkpoints. '
            f'Train with: cd heat2D && python run_sgf_ablations.py --study all',
            flush=True,
        )


if __name__ == '__main__':
    main()
