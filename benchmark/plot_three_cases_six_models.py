#!/usr/bin/env python3
"""3 cases × 6 models: prediction + error comparison with GT on the left."""
import argparse
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
from benchmark.registry import inference_device, load_model
from benchmark.visualize import plot_three_cases_six_models
from data.dataset import HeatDataset, HeatInterpolDataset

STD = FIELD_STD
DEFAULT_SAMPLES = [5200, VIS_INDEX, 5800]


def parse_args():
    p = argparse.ArgumentParser(description='Plot 3 cases × 6 models comparison')
    p.add_argument(
        '--sample-idx', type=int, nargs='+', default=DEFAULT_SAMPLES,
        help='three global sample indices (test set: 5000–5999)',
    )
    p.add_argument('--out-dir', default=os.path.join('logs', 'benchmark_comparison'))
    p.add_argument('--out-name', default='three_cases_six_models.png')
    p.add_argument('--copy-figures', action='store_true')
    return p.parse_args()


@torch.no_grad()
def predict_case(model, spec, sample_idx, device):
    if spec['type'] == 'sensor':
        inp, tgt = HeatDataset([sample_idx])[0]
        inp = torch.from_numpy(np.asarray(inp)).float().unsqueeze(0).to(device)
        pred = spec['forward'](model, inp)
    else:
        inp, tgt = HeatInterpolDataset([sample_idx])[0]
        inp = inp.unsqueeze(0).to(device)
        pred = spec['forward'](model, inp)
    truth = tgt[0].numpy() * STD
    pred_k = pred[0, 0].cpu().numpy() * STD
    return pred_k, truth


def main():
    args = parse_args()
    if len(args.sample_idx) != 3:
        raise SystemExit('Please provide exactly 3 sample indices via --sample-idx')

    os.makedirs(args.out_dir, exist_ok=True)
    device = inference_device()

    # load each model once
    loaded = {}
    for name in ALL_MODELS:
        try:
            model, ckpt, spec = load_model(name)
            loaded[name] = (model, spec)
            print(f'Loaded {name}: {ckpt}', flush=True)
        except FileNotFoundError as e:
            print(f'SKIP {name}: {e}', flush=True)

    if len(loaded) < len(ALL_MODELS):
        missing = [m for m in ALL_MODELS if m not in loaded]
        raise SystemExit(f'Missing checkpoints for: {missing}')

    cases = []
    for sample_idx in args.sample_idx:
        preds = []
        truth = None
        print(f'\n--- sample {sample_idx} ---', flush=True)
        for name in ALL_MODELS:
            model, spec = loaded[name]
            pred_k, truth = predict_case(model, spec, sample_idx, device)
            mae = float(np.abs(pred_k - truth).mean())
            preds.append((name, pred_k))
            print(f'  {name:12s}  MAE = {mae:.4f} K', flush=True)
        cases.append({'sample_idx': sample_idx, 'truth': truth, 'preds': preds})

    out_png = os.path.join(args.out_dir, args.out_name)
    plot_three_cases_six_models(cases, out_png)
    print(f'\nSaved: {out_png}', flush=True)

    if args.copy_figures:
        from benchmark.figures_paths import copy_to_figures
        dst = copy_to_figures(out_png, args.out_name, category='benchmark')
        print(f'Copied: {dst}', flush=True)


if __name__ == '__main__':
    main()
