#!/usr/bin/env python3
"""2×5 SGF-RecFNO pipeline figure: coarse → self-geometry SDF → refinement."""
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
from benchmark.registry import inference_device, load_model
from benchmark.visualize import plot_sgf_pipeline_figure
from data.dataset import HeatDataset
from model.sgf_recfno import extract_self_geometry
from utils.iso_geometry import DEFAULT_QUANTILES

STD = FIELD_STD


def parse_args():
    p = argparse.ArgumentParser(description='Plot SGF-RecFNO pipeline (2×5)')
    p.add_argument('--sample-idx', type=int, default=VIS_INDEX)
    p.add_argument('--out-dir', default=os.path.join('logs', 'benchmark_comparison'))
    p.add_argument('--out-name', default='sgf_pipeline_2x5.png')
    p.add_argument('--copy-figures', action='store_true')
    return p.parse_args()


@torch.no_grad()
def run_pipeline(sample_idx):
    recfno, _, rec_spec = load_model('RecFNO')
    sgf, _, sgf_spec = load_model('SGF-RecFNO')

    inp, tgt = HeatDataset([sample_idx])[0]
    dev = inference_device()
    inp_t = torch.from_numpy(np.asarray(inp)).float().unsqueeze(0).to(dev)

    coarse = rec_spec['forward'](recfno, inp_t)
    sgf_out = sgf(inp_t, return_aux=True)
    refined = sgf_out['field']

    truth_k = tgt[0].numpy() * STD
    coarse_k = coarse[0, 0].cpu().numpy() * STD
    refined_k = refined[0, 0].cpu().numpy() * STD
    delta_k = refined_k - coarse_k

    # SDF from coarse prediction only (same routine as SGF forward)
    sdf_self, levels = extract_self_geometry(coarse)
    sdf_np = sdf_self[0].cpu().numpy()  # (K, H, W)

    mae_coarse = float(np.abs(coarse_k - truth_k).mean())
    mae_refined = float(np.abs(refined_k - truth_k).mean())

    print(f'Sample {sample_idx}', flush=True)
    print(f'  RecFNO coarse  MAE = {mae_coarse:.4f} K', flush=True)
    print(f'  SGF refined    MAE = {mae_refined:.4f} K', flush=True)
    print(f'  Isotherm levels (from coarse, norm.): {levels[0].cpu().numpy()}', flush=True)

    return truth_k, coarse_k, refined_k, sdf_np, delta_k, mae_coarse, mae_refined


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    truth_k, coarse_k, refined_k, sdf_np, delta_k, mae_c, mae_r = run_pipeline(args.sample_idx)
    out_png = os.path.join(args.out_dir, args.out_name)
    plot_sgf_pipeline_figure(
        truth_k, coarse_k, refined_k, sdf_np, delta_k, out_png,
        sample_idx=args.sample_idx,
        quantiles=DEFAULT_QUANTILES,
        mae_coarse=mae_c,
        mae_refined=mae_r,
    )
    print(f'Saved: {out_png}', flush=True)

    if args.copy_figures:
        from benchmark.figures_paths import copy_to_figures
        dst = copy_to_figures(out_png, args.out_name, category='method')
        print(f'Copied: {dst}', flush=True)


if __name__ == '__main__':
    main()
