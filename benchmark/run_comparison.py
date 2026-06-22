#!/usr/bin/env python3
"""Unified benchmark: RecFNO variants vs GINO, Geo-FNO, PINO (official GitHub)."""
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
from torch.utils.data import DataLoader

from utils.bootstrap import ensure_repo_context

ensure_repo_context(heat2d_cwd=True)

from benchmark.config import ALL_MODELS, COMPARISON_OUT_DIR, FIELD_STD, TEST_INDEX, VIS_INDEX
from benchmark.isotherm_metrics import isotherm_geometry_metrics
from benchmark.metrics import (
    batch_fourier_spectrum_curve,
    compute_field_metrics,
    count_parameters,
    measure_inference,
)
from benchmark.registry import MODEL_SPECS, inference_device, load_model, _find_ckpt as _find_ckpt_from_registry
from benchmark.visualize import plot_field_triplet, plot_spectrum_curves, plot_summary_bars
from data.dataset import HeatDataset, HeatInterpolDataset

OUT_DIR = COMPARISON_OUT_DIR
STD = FIELD_STD


def parse_args():
    p = argparse.ArgumentParser(description='Unified heat reconstruction benchmark')
    p.add_argument('--out-dir', default=OUT_DIR)
    p.add_argument('--train-missing', action='store_true', help='train GINO/Geo-FNO/PINO if ckpt missing')
    p.add_argument('--max-samples', type=int, default=0, help='0 = full test set')
    p.add_argument('--geometry-samples', type=int, default=50, help='samples for Chamfer/Hausdorff')
    p.add_argument('--models', nargs='+', default=None,
                   help='models to evaluate (default: all with checkpoints)')
    return p.parse_args()


_CACHED_LOADERS = {}


def get_loaders():
    if 'sensor' not in _CACHED_LOADERS:
        _CACHED_LOADERS['sensor'] = DataLoader(
            HeatDataset(TEST_INDEX), batch_size=8, num_workers=4,
        )
    if 'grid' not in _CACHED_LOADERS:
        print('Building grid dataset (nearest-neighbor interpolation)...', flush=True)
        _CACHED_LOADERS['grid'] = DataLoader(
            HeatInterpolDataset(TEST_INDEX), batch_size=8, num_workers=0,
        )
    return _CACHED_LOADERS['sensor'], _CACHED_LOADERS['grid']


_VIS_CACHE = None


def get_vis_batch():
    global _VIS_CACHE
    if _VIS_CACHE is None:
        s_ds = HeatDataset([VIS_INDEX])
        g_ds = HeatInterpolDataset([VIS_INDEX])
        s_in, s_tgt = s_ds[0]
        g_in, g_tgt = g_ds[0]
        _VIS_CACHE = {
            'sensor_in': torch.from_numpy(np.asarray(s_in)).float().unsqueeze(0).cuda(),
            'sensor_tgt': s_tgt.unsqueeze(0).cuda(),
            'grid_in': g_in.unsqueeze(0).cuda(),
            'grid_tgt': g_tgt.unsqueeze(0).cuda(),
        }
    return _VIS_CACHE


@torch.no_grad()
def evaluate_model(name, max_samples=0, geometry_samples=50):
    model, ckpt, spec = load_model(name)
    loader = get_loaders()[0 if spec['type'] == 'sensor' else 1]
    fwd = spec['forward']

    totals = {
        'relative_l2': 0, 'mse': 0, 'mae_k': 0, 'max_ae_k': 0,
        'psnr': 0, 'ssim': 0, 'grad_error': 0, 'spectrum_error': 0,
    }
    n = 0
    spec_pred_sum = spec_tgt_sum = spec_err_sum = None
    geo_totals = {}
    geo_n = 0

    for inputs, targets in loader:
        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(np.asarray(inputs)).float()
        inputs, targets = inputs.cuda(), targets.cuda()
        pred = fwd(model, inputs)

        bs = pred.size(0)
        sp, st, se = batch_fourier_spectrum_curve(pred, targets)
        if spec_pred_sum is None:
            spec_pred_sum, spec_tgt_sum, spec_err_sum = sp * bs, st * bs, se * bs
        else:
            mlen = min(len(spec_pred_sum), len(sp))
            spec_pred_sum[:mlen] += sp[:mlen] * bs
            spec_tgt_sum[:mlen] += st[:mlen] * bs
            spec_err_sum[:mlen] += se[:mlen] * bs

        for b in range(pred.size(0)):
            if max_samples and n >= max_samples:
                break
            pb, tb = pred[b:b + 1], targets[b:b + 1]
            m = compute_field_metrics(pb, tb, std=STD)
            for k in totals:
                totals[k] += m[k]

            if spec.get('geometry_model') and geo_n < geometry_samples:
                gm = isotherm_geometry_metrics(pb, tb, std=STD)
                for k, v in gm.items():
                    if not np.isnan(v):
                        geo_totals[k] = geo_totals.get(k, 0.0) + v
                geo_n += 1
            n += 1
        if max_samples and n >= max_samples:
            break

    result = {k: totals[k] / n for k in totals}
    result['rmse_k'] = STD * (result['mse'] ** 0.5)
    result['model'] = name
    result['checkpoint'] = ckpt
    result['params'] = count_parameters(model)
    result['num_samples'] = n

    sample_loader = get_loaders()[0 if spec['type'] == 'sensor' else 1]
    sample = next(iter(sample_loader))[0]
    if not torch.is_tensor(sample):
        sample = torch.from_numpy(np.asarray(sample)).float()
    sample = sample[:1].cuda()
    ms, mem = measure_inference(model, fwd, sample)
    result['inference_ms'] = ms
    result['gpu_mem_mb'] = mem

    if spec_pred_sum is not None:
        result['spectrum_pred'] = (spec_pred_sum / n).tolist()
        result['spectrum_tgt'] = (spec_tgt_sum / n).tolist()
        result['spectrum_err'] = (spec_err_sum / n).tolist()

    if geo_n:
        for k in geo_totals:
            result[k] = geo_totals[k] / geo_n
    return result, model, spec


@torch.no_grad()
def collect_per_sample_mae_k(name, max_samples=0, device=None):
    """Per-sample MAE (K) on the test loader for one model."""
    if device is None:
        device = inference_device()
    model, ckpt, spec = load_model(name)
    loader = get_loaders()[0 if spec['type'] == 'sensor' else 1]
    fwd = spec['forward']

    chunks = []
    n = 0
    for inputs, targets in loader:
        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(np.asarray(inputs)).float()
        inputs, targets = inputs.to(device), targets.to(device)
        pred = fwd(model, inputs)
        batch_mae = torch.mean(torch.abs((pred - targets) * STD), dim=(1, 2, 3))
        chunks.append(batch_mae.detach().cpu().numpy())
        n += pred.size(0)
        if max_samples and n >= max_samples:
            break

    maes = np.concatenate(chunks)
    if max_samples:
        maes = maes[:max_samples]
    return maes.astype(np.float64), ckpt


def ensure_checkpoints(train_missing):
    missing = []
    for name, spec in MODEL_SPECS.items():
        from benchmark.registry import _find_ckpt
        if _find_ckpt(spec['exp']) is None:
            missing.append(name)
    if not missing:
        return missing
    if train_missing:
        from benchmark.train_external import train_one
        for m in ['GINO', 'Geo-FNO', 'PINO']:
            if m in missing:
                try:
                    train_one(m)
                except ImportError as exc:
                    print(f'SKIP training {m}: {exc}', flush=True)
        return []
    print(f'Note: missing checkpoints for {missing}; will evaluate available models only.', flush=True)
    return missing


def save_table(rows, out_dir):
    csv_path = os.path.join(out_dir, 'comparison_table.csv')
    tex_path = os.path.join(out_dir, 'comparison_table.tex')
    json_path = os.path.join(out_dir, 'comparison_results.json')

    fields = [
        'model', 'relative_l2', 'mse', 'mae_k', 'psnr', 'ssim', 'grad_error',
        'spectrum_error', 'params', 'inference_ms', 'gpu_mem_mb',
        'chamfer_mean', 'hausdorff_mean',
    ]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Quantitative comparison on heat conduction test set (5000--5999).}',
        r'\begin{tabular}{lcccccc}',
        r'\toprule',
        r'Model & Rel.\,$L^2$ & MSE & MAE (K) & PSNR & SSIM & Grad Err \\',
        r'\midrule',
    ]
    for r in rows:
        lines.append(
            f"{r['model']} & {r['relative_l2']:.4e} & {r['mse']:.4e} & {r['mae_k']:.4f} & "
            f"{r['psnr']:.2f} & {r['ssim']:.4f} & {r['grad_error']:.4e} \\\\"
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))

    return csv_path, tex_path, json_path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    missing = ensure_checkpoints(args.train_missing)

    if args.models:
        model_list = args.models
    else:
        model_list = []
        for name in ALL_MODELS:
            if name not in MODEL_SPECS:
                continue
            if name in missing:
                continue
            ckpt = _find_ckpt_from_registry(name)
            if ckpt is None:
                continue
            model_list.append(name)

    rows = []
    spectra = {}
    vis = get_vis_batch()

    for name in model_list:
        if name in missing:
            print(f'SKIP {name}: no checkpoint', flush=True)
            continue
        print(f'\n>>> Evaluating {name}...', flush=True)
        try:
            result, model, spec = evaluate_model(name, args.max_samples, args.geometry_samples)
        except FileNotFoundError as e:
            print(f'SKIP {name}: {e}', flush=True)
            continue
        rows.append(result)
        if 'spectrum_pred' in result:
            spectra[name] = (np.array(result['spectrum_pred']), np.array(result['spectrum_err']))

        # visualization
        if spec['type'] == 'sensor':
            inp, tgt = vis['sensor_in'], vis['sensor_tgt']
        else:
            inp, tgt = vis['grid_in'], vis['grid_tgt']
        with torch.no_grad():
            pred = spec['forward'](model, inp)
        truth = tgt[0, 0].cpu().numpy() * STD
        pred_k = pred[0, 0].cpu().numpy() * STD
        plot_field_triplet(
            truth, pred_k,
            os.path.join(args.out_dir, f'{name.replace("-", "").lower()}_sample{VIS_INDEX}.png'),
            title=f'{name} @ sample {VIS_INDEX}',
        )
        print(f'  RelL2={result["relative_l2"]:.4e} MAE={result["mae_k"]:.4f}K PSNR={result["psnr"]:.2f} SSIM={result["ssim"]:.4f}', flush=True)

    csv_path, tex_path, json_path = save_table(rows, args.out_dir)
    plot_spectrum_curves(spectra, os.path.join(args.out_dir, 'spectrum_error.png'))
    plot_summary_bars(rows, ['relative_l2', 'mae_k', 'ssim', 'spectrum_error'],
                      os.path.join(args.out_dir, 'metrics_bar.png'))

    print(f'\n=== Benchmark complete ===', flush=True)
    print(f'CSV : {csv_path}', flush=True)
    print(f'TeX : {tex_path}', flush=True)
    print(f'JSON: {json_path}', flush=True)
    print(f'Figures in: {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
