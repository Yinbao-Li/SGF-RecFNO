# -*- coding: utf-8 -*-
"""Visualization helpers for fluid benchmark figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from utils.ablation_config import quantiles_for_k

MODEL_COLORS = {
    'SGF-RecFNO': '#B2182B',
    'SGF-RecFNO (K=8)': '#D6604D',
    'IsoRecFNO': '#2166AC',
    'RecFNO': '#4DAF4A',
    'PINO': '#984EA3',
    'Geo-FNO': '#FF7F00',
    'GINO': '#A65628',
}

LEVEL_COLORS = ['#084594', '#2171b5', '#ef8a62', '#b30000']


def _task_cmap(task: str, signed: bool = True):
    if task == 'cylinder' and signed:
        return 'RdBu_r'
    return 'viridis'


def _sym_norm(field: np.ndarray, q: float = 0.99):
    vmax = np.quantile(np.abs(field), q)
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _draw_levels(ax, field, levels, colors, linestyle='-', lw=1.2):
    for lv, c in zip(levels, colors):
        ax.contour(field, levels=[float(lv)], colors=[c], linewidths=lw,
                   linestyles=linestyle, origin='upper')


def plot_fluid_three_cases(cases, out_path, task: str, field_label: str):
    """Three cases × N models: GT + pred row + error row per case."""
    nmodels = len(cases[0]['preds'])
    ncases = len(cases)
    signed = task == 'cylinder'
    cmap = _task_cmap(task, signed)

    nrows = ncases * 2
    ncols = nmodels + 1
    fig = plt.figure(figsize=(2.35 * ncols + 0.8, 2.05 * nrows + 0.6))
    gs = fig.add_gridspec(nrows, ncols, wspace=0.22, hspace=0.28)

    for i, case in enumerate(cases):
        truth = case['truth']
        preds = case['preds']
        sample_idx = case.get('sample_idx')
        row_p, row_e = 2 * i, 2 * i + 1

        if signed:
            norm = _sym_norm(truth)
            vmin = vmax = None
        else:
            norm = None
            vmin, vmax = float(truth.min()), float(truth.max())

        err_maps = [np.abs(p - truth) for _, p in preds]
        err_vmax = max(max(float(e.max()) for e in err_maps), 1e-8)

        ax_gt = fig.add_subplot(gs[row_p:row_e + 1, 0])
        kw = dict(cmap=cmap, origin='upper', norm=norm)
        if norm is None:
            kw.update(vmin=vmin, vmax=vmax)
        im_gt = ax_gt.imshow(truth, **kw)
        title = 'Ground Truth'
        if sample_idx is not None:
            title += f'\n(#{sample_idx})'
        ax_gt.set_title(title, fontsize=9)
        ax_gt.set_aspect('equal')
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.02).set_label(field_label, fontsize=7)

        for j, ((name, pred), err) in enumerate(zip(preds, err_maps)):
            mae = float(err.mean())
            col = j + 1
            ax_p = fig.add_subplot(gs[row_p, col])
            kw_p = dict(cmap=cmap, origin='upper', norm=norm)
            if norm is None:
                kw_p.update(vmin=vmin, vmax=vmax)
            ax_p.imshow(pred, **kw_p)
            if i == 0:
                ax_p.set_title(name, fontsize=9, fontweight='semibold', pad=4)
            ax_p.set_aspect('equal')
            ax_p.set_xticks([])
            ax_p.set_yticks([])

            ax_e = fig.add_subplot(gs[row_e, col])
            ax_e.imshow(err, cmap='hot', origin='upper', vmin=0, vmax=err_vmax)
            ax_e.set_title(f'MAE={mae:.2e}', fontsize=8)
            ax_e.set_aspect('equal')
            ax_e.set_xticks([])
            ax_e.set_yticks([])

    task_title = 'Cylinder wake (vorticity)' if task == 'cylinder' else 'Darcy flow (pressure)'
    fig.suptitle(f'{task_title}: 3 test cases × models', fontsize=12, y=1.01)
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _grad_mag_np(field: np.ndarray, log1p: bool = False) -> np.ndarray:
    gy, gx = np.gradient(field)
    gm = np.sqrt(gx ** 2 + gy ** 2)
    return np.log1p(gm) if log1p else gm


def plot_fluid_geometry_overlay(cases, out_path, task: str):
    """Level-set geometry overlays (task-specific contours on GT field)."""
    quantiles = quantiles_for_k(4)
    q_half = quantiles[:2]
    colors = LEVEL_COLORS[:2]
    ncases = len(cases)
    ncols = len(cases[0]['preds']) + 1

    fig, axes = plt.subplots(ncases, ncols, figsize=(2.65 * ncols + 0.6, 2.75 * ncases + 1.0),
                             squeeze=False)
    signed = task == 'cylinder'
    geom_title = r'$|\omega|$ + $|\nabla\omega|$' if signed else r'$p$ + $\log(1+|\nabla p|)$'

    for i, case in enumerate(cases):
        truth = case['truth']
        preds = case['preds']
        sample_idx = case.get('sample_idx')

        if signed:
            norm = _sym_norm(truth)
            field_levels = np.quantile(np.abs(truth).ravel(), q_half)
            grad_levels = np.quantile(_grad_mag_np(truth).ravel(), q_half)
            gm = _grad_mag_np(truth)
            vmin = vmax = None
        else:
            norm = None
            vmin, vmax = truth.min(), truth.max()
            gm = _grad_mag_np(truth, log1p=True)
            field_levels = np.quantile(truth.ravel(), q_half)
            grad_levels = np.quantile(gm.ravel(), q_half)

        ax_gt = axes[i, 0]
        if signed:
            ax_gt.imshow(truth, cmap='RdBu_r', norm=norm, origin='upper')
        else:
            ax_gt.imshow(truth, cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
        _draw_levels(ax_gt, np.abs(truth) if signed else truth, field_levels, colors, lw=1.6)
        _draw_levels(ax_gt, gm, grad_levels, ['#2ca02c', '#98df8a'], lw=1.2, linestyle='--')
        ax_gt.set_title('GT geometry', fontsize=10, fontweight='semibold')
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        if sample_idx is not None:
            ax_gt.text(0.02, 0.98, f'#{sample_idx}', transform=ax_gt.transAxes,
                       fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.85))

        for j, (name, pred) in enumerate(preds):
            ax = axes[i, j + 1]
            if signed:
                ax.imshow(pred, cmap='RdBu_r', norm=_sym_norm(truth), origin='upper', alpha=0.55)
            else:
                ax.imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax, origin='upper', alpha=0.55)
            _draw_levels(ax, np.abs(pred) if signed else pred, field_levels, colors, lw=1.0)
            _draw_levels(ax, np.abs(truth) if signed else truth, field_levels, colors,
                         lw=1.0, linestyle='--')
            ax.set_title(name, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f'Level-set geometry ({geom_title}): dashed=GT, solid=pred', fontsize=11, y=1.02)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_fluid_metrics_bars(rows, out_path, task: str, highlight='SGF-RecFNO'):
    """Bar charts for MAE / PSNR / SSIM on one fluid task."""
    task_rows = [r for r in rows if r.get('task') == task]
    if not task_rows:
        return
    models = [r['model'] for r in task_rows]
    metrics = [
        ('mae_k', 'Test MAE', True),
        ('psnr', 'PSNR', False),
        ('ssim', 'SSIM', False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (key, label, lower_better) in zip(axes, metrics):
        vals = [r[key] for r in task_rows]
        colors = [MODEL_COLORS.get(m, '0.5') for m in models]
        bars = ax.bar(range(len(models)), vals, color=colors, edgecolor='0.2', linewidth=0.5)
        if highlight in models:
            bars[models.index(highlight)].set_edgecolor('#B2182B')
            bars[models.index(highlight)].set_linewidth(2.0)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha='right', fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    title = 'Cylinder wake' if task == 'cylinder' else 'Darcy flow'
    fig.suptitle(f'{title}: test-set metrics (500 epochs)', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_fluid_mae_distribution(model_maes: list[tuple[str, np.ndarray]], out_path, task: str):
    """ECDF overlay of per-sample MAE (mirrors heat benchmark style)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, maes in model_maes:
        xs = np.sort(maes)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        lw = 2.2 if name == 'SGF-RecFNO' else 1.2
        ax.plot(xs, ys, label=name, color=MODEL_COLORS.get(name, None), linewidth=lw)
    ax.set_xlabel('Per-sample MAE')
    ax.set_ylabel('ECDF')
    ax.set_title(f'{"Cylinder" if task == "cylinder" else "Darcy"}: MAE distribution (test set)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
