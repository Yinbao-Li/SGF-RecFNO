#!/usr/bin/env python3
"""Regenerate all fluid benchmark README figures."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_HEAT2D = os.path.join(_ROOT, 'heat2D')
_SCRIPT = os.path.join(_ROOT, 'benchmark')

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.bootstrap import ensure_repo_context

ensure_repo_context(heat2d_cwd=True)

import numpy as np
import torch

from benchmark.fluid_config import FLUID_MODELS, TASKS
from benchmark.fluid_eval import (
    FLUID_CASE_SAMPLES,
    FLUID_OUT_DIR,
    collect_per_sample_mae,
    evaluate_fluid_models,
    load_fluid_model,
    predict_fluid_sample,
)
from benchmark.fluid_visualize import (
    plot_fluid_geometry_overlay,
    plot_fluid_mae_distribution,
    plot_fluid_metrics_bars,
    plot_fluid_three_cases,
)
from data.fluid_dataset import CYLINDER_SENSOR_4, DARCY_SENSOR_POS

FIG_DIR = os.path.join('figures', 'fluid')
COMPARISON_MODELS = [
    'SGF-RecFNO', 'SGF-RecFNO (K=8)', 'IsoRecFNO', 'RecFNO', 'PINO',
]


def parse_args():
    p = argparse.ArgumentParser(description='Regenerate fluid benchmark figures')
    p.add_argument('--copy-figures', action='store_true')
    p.add_argument('--skip-eval', action='store_true', help='reuse comparison_results.json')
    p.add_argument('--skip-mae-cache', action='store_true')
    return p.parse_args()


def _export_tables(rows):
    os.makedirs(FLUID_OUT_DIR, exist_ok=True)
    fields = ['task', 'model', 'relative_l2', 'mse', 'mae_k', 'psnr', 'ssim', 'checkpoint']
    csv_path = os.path.join(FLUID_OUT_DIR, 'comparison_table.csv')
    json_path = os.path.join(FLUID_OUT_DIR, 'comparison_results.json')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'Saved {csv_path}', flush=True)


def _build_cases(task: str, models: list[str], sample_indices: list[int]):
    loaded = {}
    for name in models:
        try:
            loaded[name] = load_fluid_model(task, name)
            print(f'  Loaded {name}', flush=True)
        except FileNotFoundError as e:
            print(f'  SKIP {name}: {e}', flush=True)

    cases = []
    for idx in sample_indices:
        preds = []
        truth = None
        for name in models:
            if name not in loaded:
                continue
            net, _, _ = loaded[name]
            pred, truth = predict_fluid_sample(task, name, idx, net=net)
            preds.append((name, pred))
        if preds:
            cases.append({'sample_idx': idx, 'truth': truth, 'preds': preds})
    for net, _, _ in loaded.values():
        del net
    torch.cuda.empty_cache()
    return cases


def _plot_setup(task: str, sample_idx: int, out_path: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    from benchmark.fluid_eval import load_ground_truth
    field = load_ground_truth(task, sample_idx)
    if task == 'cylinder':
        sensors = CYLINDER_SENSOR_4
        vmax = np.quantile(np.abs(field), 0.99)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = 'RdBu_r'
        label = r'$\omega$'
        title = f'Cylinder wake setup (#{sample_idx})'
    else:
        sensors = DARCY_SENSOR_POS
        norm = None
        cmap = 'viridis'
        label = 'pressure'
        title = f'Darcy flow setup (#{sample_idx})'

    fig, ax = plt.subplots(figsize=(5.5, 4))
    if norm:
        im = ax.imshow(field, cmap=cmap, norm=norm, origin='upper')
    else:
        im = ax.imshow(field, cmap=cmap, origin='upper')
    ax.scatter(sensors[:, 1], sensors[:, 0], c='lime', s=40, edgecolors='k', linewidths=0.6, zorder=3)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, label=label)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_sgf_vs_recfno(task: str, sample_idx: int, out_path: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    from benchmark.fluid_eval import load_ground_truth
    truth = load_ground_truth(task, sample_idx)
    pred_r, _ = predict_fluid_sample(task, 'RecFNO', sample_idx)
    pred_s, _ = predict_fluid_sample(task, 'SGF-RecFNO', sample_idx)

    if task == 'cylinder':
        vmax = np.quantile(np.abs(truth), 0.99)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = 'RdBu_r'
        suptitle = 'Cylinder: SGF vs RecFNO'
    else:
        norm = None
        cmap = 'viridis'
        vmin, vmax = truth.min(), truth.max()
        suptitle = 'Darcy: SGF vs RecFNO'

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), constrained_layout=True)
    kw = dict(cmap=cmap, origin='upper')
    if norm:
        kw['norm'] = norm
    else:
        kw.update(vmin=vmin, vmax=vmax)
    axes[0, 0].imshow(truth, **kw)
    axes[0, 0].set_title(f'GT (#{sample_idx})')
    for j, (name, pred) in enumerate([('RecFNO', pred_r), ('SGF-RecFNO', pred_s)], 1):
        mae = float(np.abs(pred - truth).mean())
        axes[0, j].imshow(pred, **kw)
        axes[0, j].set_title(f'{name}  MAE={mae:.2e}')
        err = np.abs(pred - truth)
        axes[1, j].imshow(err, cmap='hot', origin='upper')
        axes[1, j].set_title(f'{name} |error|')
    axes[1, 0].axis('off')
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(FLUID_OUT_DIR, exist_ok=True)

    rows = []
    if args.skip_eval:
        json_path = os.path.join(FLUID_OUT_DIR, 'comparison_results.json')
        with open(json_path) as f:
            rows = json.load(f)
    else:
        for task in TASKS:
            print(f'\n=== Evaluate {task} ===', flush=True)
            rows.extend(evaluate_fluid_models(task))
        _export_tables(rows)

    for task in TASKS:
        plot_fluid_metrics_bars(
            rows, os.path.join(FIG_DIR, f'{task}_test_metrics.png'), task,
        )
        print(f'Saved {task}_test_metrics.png', flush=True)

    for task, samples in FLUID_CASE_SAMPLES.items():
        print(f'\n--- {task} cases ---', flush=True)
        cases = _build_cases(task, COMPARISON_MODELS, samples)
        if not cases:
            continue
        label = r'$\omega$' if task == 'cylinder' else 'pressure'
        plot_fluid_three_cases(
            cases, os.path.join(FIG_DIR, f'{task}_three_cases.png'), task, label,
        )
        plot_fluid_geometry_overlay(
            cases, os.path.join(FIG_DIR, f'{task}_geometry_overlay.png'), task,
        )
        print(f'Saved {task}_three_cases.png, {task}_geometry_overlay.png', flush=True)

        cache = os.path.join(FLUID_OUT_DIR, f'{task}_mae_per_sample.npz')
        model_maes = []
        if args.skip_mae_cache and os.path.exists(cache):
            data = np.load(cache)
            for name in COMPARISON_MODELS:
                key = name.replace('-', '_')
                if key in data:
                    model_maes.append((name, data[key]))
        else:
            for name in COMPARISON_MODELS:
                try:
                    maes = collect_per_sample_mae(task, name)
                    model_maes.append((name, maes))
                    print(f'  {name} MAE mean={maes.mean():.2e}', flush=True)
                except FileNotFoundError as e:
                    print(f'  SKIP {name}: {e}', flush=True)
            if model_maes:
                np.savez_compressed(cache, **{n.replace('-', '_'): v for n, v in model_maes})
        if model_maes:
            plot_fluid_mae_distribution(
                model_maes, os.path.join(FIG_DIR, f'{task}_mae_distribution.png'), task,
            )

        _plot_setup(task, samples[1], os.path.join(FIG_DIR, f'{task}_problem_setup.png'))
        _plot_sgf_vs_recfno(task, samples[1], os.path.join(FIG_DIR, f'{task}_sgf_vs_recfno.png'))

    if args.copy_figures:
        from benchmark.figures_paths import copy_to_figures
        for fn in os.listdir(FIG_DIR):
            if fn.endswith('.png'):
                copy_to_figures(os.path.join(FIG_DIR, fn), fn, category='fluid')
                print(f'Copied {fn}', flush=True)

    print('\nDone. See figures/fluid/', flush=True)


if __name__ == '__main__':
    main()
