#!/usr/bin/env python3
"""Generate single-case error comparison across all trained models."""
import argparse
import csv
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from utils.bootstrap import ensure_repo_context

ensure_repo_context(heat2d_cwd=True)

from benchmark.config import ALL_MODELS, FIELD_STD, VIS_INDEX
from benchmark.registry import MODEL_SPECS, load_model
from benchmark.visualize import plot_case_error_comparison
from data.dataset import HeatDataset, HeatInterpolDataset

STD = FIELD_STD
MODELS = ALL_MODELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sample-idx', type=int, default=5500)
    p.add_argument('--out-dir', default=os.path.join('logs', 'benchmark_comparison'))
    return p.parse_args()


@torch.no_grad()
def predict_one(name, sample_idx):
    model, _, spec = load_model(name)
    if spec['type'] == 'sensor':
        inp, tgt = HeatDataset([sample_idx])[0]
        inp = torch.from_numpy(np.asarray(inp)).float().unsqueeze(0).cuda()
        pred = spec['forward'](model, inp)
    else:
        inp, tgt = HeatInterpolDataset([sample_idx])[0]
        inp = inp.unsqueeze(0).cuda()
        pred = spec['forward'](model, inp)
    truth = tgt[0].numpy() * STD
    pred_k = pred[0, 0].cpu().numpy() * STD
    mae = float(np.abs(pred_k - truth).mean())
    return name, pred_k, truth, mae


def main():
    args = parse_args()
    os.chdir(os.path.join(filename, 'heat2D'))
    os.makedirs(args.out_dir, exist_ok=True)

    preds = []
    truth = None
    rows = []
    for name in MODELS:
        try:
            n, pred_k, truth, mae = predict_one(name, args.sample_idx)
            preds.append((n, pred_k))
            rows.append({'model': n, 'case_mae_k': mae})
            print(f'{n:12s}  case MAE = {mae:.4f} K', flush=True)
        except FileNotFoundError as e:
            print(f'SKIP {name}: {e}', flush=True)

    preds.sort(key=lambda x: rows[[r['model'] for r in rows].index(x[0])]['case_mae_k']
               if x[0] in [r['model'] for r in rows] else 0)
    # sort by MAE ascending
    maes = {r['model']: r['case_mae_k'] for r in rows}
    preds.sort(key=lambda x: maes[x[0]])

    out_png = os.path.join(args.out_dir, f'case{args.sample_idx}_error_comparison.png')
    plot_case_error_comparison(truth, preds, out_png, sample_idx=args.sample_idx, std=STD)

    csv_path = os.path.join(args.out_dir, f'case{args.sample_idx}_mae.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'case_mae_k'])
        w.writeheader()
        for r in sorted(rows, key=lambda r: r['case_mae_k']):
            w.writerow(r)

    print(f'\nSaved: {out_png}', flush=True)
    print(f'Saved: {csv_path}', flush=True)


if __name__ == '__main__':
    main()
