# -*- coding: utf-8 -*-
"""Visualization helpers for benchmark outputs."""
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrum_curves(curves, out_path, title='Fourier Spectrum Error'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for name, (sp, _) in curves.items():
        ax.plot(sp, label=name, linewidth=1.2)
    ax.set_xlabel('Radial frequency bin')
    ax.set_ylabel('Mean |FFT| magnitude')
    ax.set_title('Radial Spectrum (prediction)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, (_, err) in curves.items():
        ax.plot(err, label=name, linewidth=1.2)
    ax.set_xlabel('Radial frequency bin')
    ax.set_ylabel('Squared spectrum error')
    ax.set_title('Spectrum Error vs Ground Truth')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_field_triplet(truth, pred, out_path, title=''):
    err = np.abs(pred - truth)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, t in zip(axes, [truth, pred, err], ['Ground Truth', 'Prediction', '|Error|']):
        im = ax.imshow(data, cmap='coolwarm' if t != '|Error|' else 'viridis', origin='upper')
        ax.set_title(t)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_bars(summary_rows, metrics, out_path):
    names = [r['model'] for r in summary_rows]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for ax, metric in zip(axes, metrics):
        vals = [r[metric] for r in summary_rows]
        bars = ax.bar(names, vals, color=colors, edgecolor='white')
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=25)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{v:.4g}',
                    ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_case_error_comparison(truth, preds, out_path, sample_idx=None, std=50.0):
    """
    Compare |error| maps for multiple models on one case.
    preds: list of (name, prediction_array_in_K)
    """
    n = len(preds)
    fig = plt.figure(figsize=(14, 2.2 * (n + 1)))
    gs = fig.add_gridspec(n + 1, 3, width_ratios=[1, 1, 1], hspace=0.35, wspace=0.25)

    ax_gt = fig.add_subplot(gs[0, :])
    im = ax_gt.imshow(truth, cmap='coolwarm', origin='upper')
    ax_gt.set_title(f'Ground Truth (K)' + (f'  sample {sample_idx}' if sample_idx is not None else ''))
    ax_gt.set_aspect('equal')
    plt.colorbar(im, ax=ax_gt, fraction=0.02, pad=0.01)

    err_maps = []
    maes = []
    for name, pred in preds:
        err = np.abs(pred - truth)
        err_maps.append(err)
        maes.append(float(err.mean()))

    vmax = max(e.max() for e in err_maps)
    vmax = max(vmax, 1e-6)

    for i, ((name, pred), err, mae) in enumerate(zip(preds, err_maps, maes)):
        ax_p = fig.add_subplot(gs[i + 1, 0])
        ax_e = fig.add_subplot(gs[i + 1, 1])
        ax_b = fig.add_subplot(gs[i + 1, 2])

        im_p = ax_p.imshow(pred, cmap='coolwarm', origin='upper')
        ax_p.set_title(f'{name}  Pred')
        ax_p.set_aspect('equal')
        plt.colorbar(im_p, ax=ax_p, fraction=0.046)

        im_e = ax_e.imshow(err, cmap='hot', origin='upper', vmin=0, vmax=vmax)
        ax_e.set_title(f'|Error|  MAE={mae:.4f} K')
        ax_e.set_aspect('equal')
        plt.colorbar(im_e, ax=ax_e, fraction=0.046)

        ax_b.barh([0], [mae], color='steelblue', height=0.4)
        ax_b.set_xlim(0, max(maes) * 1.15)
        ax_b.set_yticks([])
        ax_b.set_xlabel('MAE (K)')
        ax_b.set_title('Case MAE')
        ax_b.text(mae, 0, f' {mae:.4f}', va='center', fontsize=9)

    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
    return maes
