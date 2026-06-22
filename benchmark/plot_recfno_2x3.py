#!/usr/bin/env python3
"""2×3 comparison figure: RecFNO / IsoRecFNO / SGF-RecFNO pred + error, GT on the left."""
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

from benchmark.config import FIELD_STD, VIS_INDEX
from benchmark.registry import load_model
from benchmark.visualize import DEFAULT_ZOOM_BOX, plot_recfno_comparison_2x3, plot_recfno_comparison_2x3_zoom
from data.dataset import HeatDataset

MODELS = ['RecFNO', 'IsoRecFNO', 'SGF-RecFNO']
STD = FIELD_STD


def parse_args():
    p = argparse.ArgumentParser(description='Plot 2×3 RecFNO-family comparison')
    p.add_argument('--sample-idx', type=int, default=VIS_INDEX)
    p.add_argument('--out-dir', default=os.path.join('logs', 'benchmark_comparison'))
    p.add_argument('--out-name', default=None, help='output png filename')
    p.add_argument('--zoom', action='store_true',
                   help='add bottom-center zoom insets on each panel')
    p.add_argument('--zoom-box', type=int, nargs=4, default=list(DEFAULT_ZOOM_BOX),
                   metavar=('Y0', 'Y1', 'X0', 'X1'),
                   help='ROI for zoom (origin=upper, default: bottom-center)')
    return p.parse_args()


@torch.no_grad()
def predict_one(name, sample_idx):
    model, _, spec = load_model(name)
    inp, tgt = HeatDataset([sample_idx])[0]
    inp = torch.from_numpy(np.asarray(inp)).float().unsqueeze(0).cuda()
    pred = spec['forward'](model, inp)
    truth = tgt[0].numpy() * STD
    pred_k = pred[0, 0].cpu().numpy() * STD
    mae = float(np.abs(pred_k - truth).mean())
    return name, pred_k, truth, mae


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    preds = []
    truth = None
    for name in MODELS:
        n, pred_k, truth, mae = predict_one(name, args.sample_idx)
        preds.append((n, pred_k))
        print(f'{n:12s}  case MAE = {mae:.4f} K', flush=True)

    if args.zoom:
        out_name = args.out_name or f'case{args.sample_idx}_recfno_2x3_zoom.png'
        out_png = os.path.join(args.out_dir, out_name)
        plot_recfno_comparison_2x3_zoom(
            truth, preds, out_png,
            sample_idx=args.sample_idx,
            zoom_box=tuple(args.zoom_box),
        )
    else:
        out_name = args.out_name or f'case{args.sample_idx}_recfno_2x3.png'
        out_png = os.path.join(args.out_dir, out_name)
        plot_recfno_comparison_2x3(truth, preds, out_png, sample_idx=args.sample_idx)
    print(f'\nSaved: {out_png}', flush=True)


if __name__ == '__main__':
    main()
