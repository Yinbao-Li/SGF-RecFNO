#!/usr/bin/env python3
"""Plot test-set Fourier spectrum error with SGF-RecFNO K=4 and K=8 variants."""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.bootstrap import ensure_repo_context

ensure_repo_context(heat2d_cwd=True)

from benchmark.config import COMPARISON_OUT_DIR, TEST_INDEX
from benchmark.figures_paths import copy_to_figures
from benchmark.metrics import batch_fourier_spectrum_curve
from benchmark.registry import MODEL_SPECS, _find_ckpt, _load_state, inference_device
from benchmark.visualize import plot_spectrum_error_figure
from data.dataset import HeatDataset, HeatInterpolDataset
from model.sgf_recfno import SGFRecFNO
from utils.ablation_config import quantile_ablation_exp_name, quantiles_for_k

OUT_DIR = COMPARISON_OUT_DIR

SPECTRUM_MODELS = [
    'SGF-RecFNO (K=4)',
    'SGF-RecFNO (K=8)',
    'IsoRecFNO',
    'RecFNO',
    'PINO',
    'Geo-FNO',
    'GINO',
]

LABEL_OFFSETS = {
    'SGF-RecFNO (K=4)': (8, 4),
    'SGF-RecFNO (K=8)': (8, -10),
    'IsoRecFNO': (8, 6),
    'RecFNO': (8, -8),
    'PINO': (8, 0),
    'Geo-FNO': (8, 10),
    'GINO': (8, -14),
}

LABEL_X_FRACS = {
    'SGF-RecFNO (K=4)': 0.62,
    'SGF-RecFNO (K=8)': 0.66,
    'IsoRecFNO': 0.58,
    'RecFNO': 0.72,
    'PINO': 0.55,
    'Geo-FNO': 0.50,
    'GINO': 0.45,
}


def parse_args():
    p = argparse.ArgumentParser(description='Plot spectrum error curves')
    p.add_argument('--out-dir', default=OUT_DIR)
    p.add_argument('--out-name', default='spectrum_error.png')
    p.add_argument('--max-samples', type=int, default=0, help='0 = full test set')
    p.add_argument('--use-cache', action='store_true',
                   help='reuse spectrum_err from comparison_results.json when available')
    p.add_argument('--copy-figures', action='store_true')
    return p.parse_args()


def _get_loader(spec):
    if spec['type'] == 'sensor':
        return DataLoader(HeatDataset(TEST_INDEX), batch_size=8, num_workers=4)
    return DataLoader(HeatInterpolDataset(TEST_INDEX), batch_size=8, num_workers=0)


def _load_sgf_k4():
    spec = MODEL_SPECS['SGF-RecFNO']
    ckpt = _find_ckpt(spec['exp'])
    if ckpt is None:
        raise FileNotFoundError('No checkpoint for SGF-RecFNO (K=4)')
    model = spec['build']()
    _load_state(model, ckpt)
    return model, spec


def _load_sgf_k8():
    exp = quantile_ablation_exp_name(8)
    ckpt = _find_ckpt(exp)
    if ckpt is None:
        raise FileNotFoundError('No checkpoint for SGF-RecFNO (K=8)')
    quantiles = quantiles_for_k(8)
    from benchmark.config import FC_SIZE, MODES, OUT_SIZE, SENSOR_NUM, WIDTH
    model = SGFRecFNO(
        SENSOR_NUM, FC_SIZE, OUT_SIZE,
        modes1=MODES, modes2=MODES, width=WIDTH,
        num_sdf=8, quantiles=quantiles,
    )
    _load_state(model, ckpt)
    spec = {
        'type': 'sensor',
        'forward': lambda m, x: m(x, return_aux=True)['field'],
    }
    return model, spec


def _resolve_model(name):
    if name == 'SGF-RecFNO (K=4)':
        return _load_sgf_k4()
    if name == 'SGF-RecFNO (K=8)':
        return _load_sgf_k8()
    from benchmark.registry import load_model
    return load_model(name)


@torch.no_grad()
def collect_spectrum_err(name, max_samples=0, device=None):
    if device is None:
        device = inference_device()
    model, spec = _resolve_model(name)
    loader = _get_loader(spec)
    fwd = spec['forward']

    spec_err_sum = None
    n = 0
    for inputs, targets in loader:
        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(np.asarray(inputs)).float()
        inputs, targets = inputs.to(device), targets.to(device)
        pred = fwd(model, inputs)
        bs = pred.size(0)
        _, _, se = batch_fourier_spectrum_curve(pred, targets)
        if spec_err_sum is None:
            spec_err_sum = se * bs
        else:
            mlen = min(len(spec_err_sum), len(se))
            spec_err_sum[:mlen] += se[:mlen] * bs
        n += bs
        if max_samples and n >= max_samples:
            break

    return np.asarray(spec_err_sum / n, dtype=np.float64)


def _load_cache(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        rows = json.load(f)
    cache = {}
    for row in rows:
        spec_err = row.get('spectrum_err')
        if spec_err is None:
            continue
        model = row['model']
        if model == 'SGF-RecFNO':
            cache['SGF-RecFNO (K=4)'] = np.asarray(spec_err, dtype=np.float64)
        elif model == 'SGF-RecFNO (K=8)':
            cache['SGF-RecFNO (K=8)'] = np.asarray(spec_err, dtype=np.float64)
        else:
            cache[model] = np.asarray(spec_err, dtype=np.float64)
    return cache


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cache = _load_cache(os.path.join(args.out_dir, 'comparison_results.json')) if args.use_cache else {}
    curves = {}

    for name in SPECTRUM_MODELS:
        if args.use_cache and name in cache:
            print(f'  {name}: cached', flush=True)
            curves[name] = cache[name]
            continue
        print(f'  {name}: evaluating...', flush=True)
        curves[name] = collect_spectrum_err(name, args.max_samples)

    out_png = os.path.join(args.out_dir, args.out_name)
    plot_spectrum_error_figure(
        curves,
        out_png,
        inline_labels=True,
        show_legend=False,
        label_offsets=LABEL_OFFSETS,
        label_x_fracs=LABEL_X_FRACS,
    )
    print(f'Saved: {out_png}', flush=True)

    if args.copy_figures:
        dst = copy_to_figures(out_png, args.out_name, category='benchmark')
        print(f'Copied to: {dst}', flush=True)


if __name__ == '__main__':
    main()
