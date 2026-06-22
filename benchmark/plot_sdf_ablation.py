#!/usr/bin/env python3
"""SDF ablation on test set: coarse only vs refine w/o SDF vs full SGF-RecFNO."""
import argparse
import csv
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from utils.bootstrap import ensure_repo_context

ensure_repo_context(heat2d_cwd=True)

from benchmark.config import COMPARISON_OUT_DIR, FIELD_STD, TEST_INDEX
from benchmark.metrics import compute_field_metrics
from benchmark.registry import inference_device, load_model
from benchmark.run_comparison import get_loaders
from benchmark.visualize import plot_sdf_ablation_bars

STD = FIELD_STD

VARIANTS = [
    ('Coarse only', dict(skip_refine=True)),
    ('Refine w/o SDF', dict(ablate_sdf=True)),
    ('Full SGF-RecFNO', dict()),
]


def parse_args():
    p = argparse.ArgumentParser(description='SDF ablation study (inference-time)')
    p.add_argument('--out-dir', default=COMPARISON_OUT_DIR)
    p.add_argument('--out-name', default='sdf_ablation.png')
    p.add_argument('--max-samples', type=int, default=0, help='0 = full test set (1000)')
    p.add_argument('--copy-figures', action='store_true')
    return p.parse_args()


@torch.no_grad()
def evaluate_variants(max_samples=0):
    device = inference_device()
    model, ckpt, _ = load_model('SGF-RecFNO')
    model.to(device).eval()
    loader, _ = get_loaders()

    totals = {
        label: {'mae_k': 0.0, 'mse': 0.0, 'max_ae_k': 0.0, 'n': 0}
        for label, _ in VARIANTS
    }

    n_seen = 0
    for inputs, targets in loader:
        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(np.asarray(inputs)).float()
        inputs, targets = inputs.to(device), targets.to(device)

        for label, kwargs in VARIANTS:
            out = model(inputs, return_aux=True, **kwargs)
            pred = out['field']
            bs = pred.size(0)
            t = totals[label]
            for b in range(bs):
                if max_samples and n_seen + b >= max_samples:
                    break
                m = compute_field_metrics(pred[b:b + 1], targets[b:b + 1], std=STD)
                t['mae_k'] += m['mae_k']
                t['mse'] += m['mse']
                t['max_ae_k'] = max(t['max_ae_k'], m['max_ae_k'])
                t['n'] += 1

        n_seen += inputs.size(0)
        if max_samples and n_seen >= max_samples:
            break

    rows = []
    for label, _ in VARIANTS:
        t = totals[label]
        n = t['n']
        rows.append({
            'variant': label,
            'mae_k': t['mae_k'] / n,
            'rmse_k': STD * (t['mse'] / n) ** 0.5,
            'max_ae_k': t['max_ae_k'],
            'num_samples': n,
            'checkpoint': ckpt,
        })
    return rows, ckpt


def save_table(rows, out_dir):
    csv_path = os.path.join(out_dir, 'sdf_ablation_metrics.csv')
    json_path = os.path.join(out_dir, 'sdf_ablation_metrics.json')
    fields = ['variant', 'mae_k', 'rmse_k', 'max_ae_k', 'num_samples', 'checkpoint']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)
    return csv_path, json_path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('Evaluating SDF ablation on test set...', flush=True)
    rows, ckpt = evaluate_variants(max_samples=args.max_samples)
    print(f'Checkpoint: {ckpt}', flush=True)
    for r in rows:
        print(
            f"  {r['variant']:18s}  MAE={r['mae_k']:.4f} K  "
            f"RMSE={r['rmse_k']:.4f} K  MaxAE={r['max_ae_k']:.4f} K",
            flush=True,
        )

    csv_path, json_path = save_table(rows, args.out_dir)
    out_png = os.path.join(args.out_dir, args.out_name)
    plot_sdf_ablation_bars(rows, out_png)
    print(f'\nSaved: {out_png}', flush=True)
    print(f'CSV : {csv_path}', flush=True)
    print(f'JSON: {json_path}', flush=True)

    if args.copy_figures:
        from benchmark.figures_paths import copy_to_figures
        dst = copy_to_figures(out_png, args.out_name, category='method')
        print(f'Copied: {dst}', flush=True)


if __name__ == '__main__':
    main()
