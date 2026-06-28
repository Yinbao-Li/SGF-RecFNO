# -*- coding: utf-8 -*-
"""Visualization helpers for benchmark outputs."""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils.iso_geometry import DEFAULT_QUANTILES

# Bottom-center ROI on 200×200 fields (origin='upper', y grows downward).
DEFAULT_ZOOM_BOX = (125, 200, 60, 140)  # y0, y1, x0, x1

ISOTHERM_LEVEL_COLORS = ['#084594', '#2171b5', '#ef8a62', '#b30000']
ISOTHERM_LEVEL_LABELS = [f'$q={q:.1f}$' for q in DEFAULT_QUANTILES]


def _draw_isotherm_contours(ax, field_k, levels_k, colors, linewidths, linestyles='-'):
    """Draw isotherm contours with origin='upper' to match imshow."""
    styles = linestyles
    if not isinstance(styles, (list, tuple)):
        styles = [styles] * len(levels_k)
    lws = linewidths
    if not isinstance(lws, (list, tuple, np.ndarray)):
        lws = [lws] * len(levels_k)
    for lv, c, lw, ls in zip(levels_k, colors, lws, styles):
        ax.contour(
            field_k, levels=[float(lv)], colors=[c],
            linewidths=lw, linestyles=ls, origin='upper',
        )


def plot_three_cases_isotherm_overlay(
    cases,
    out_path,
    quantiles=DEFAULT_QUANTILES,
    highlight_model='SGF-RecFNO',
):
    """
    Three cases × (GT + six models): isotherm contour overlays vs GT.

    cases: list of dicts with ``sample_idx``, ``truth`` (K), ``preds`` [(name, pred_K), ...].
    Isotherm levels are GT quantiles; each model panel overlays GT (dashed) and prediction (solid).
    """
    from benchmark.isotherm_metrics import gt_isotherm_levels_k, mean_chamfer_isotherms

    if not cases:
        raise ValueError('cases must be non-empty')
    nmodels = len(cases[0]['preds'])
    if any(len(c['preds']) != nmodels for c in cases):
        raise ValueError('all cases must have the same number of models')

    nlevels = len(quantiles)
    colors = ISOTHERM_LEVEL_COLORS[:nlevels]
    level_labels = ISOTHERM_LEVEL_LABELS[:nlevels]
    ncases = len(cases)
    ncols = nmodels + 1

    with plt.rc_context({
        'font.size': 10,
        'font.family': 'serif',
        'axes.linewidth': 0.8,
    }):
        fig, axes = plt.subplots(
            ncases, ncols,
            figsize=(2.65 * ncols + 0.6, 2.75 * ncases + 1.2),
            squeeze=False,
        )

        model_names = [name for name, _ in cases[0]['preds']]

        for i, case in enumerate(cases):
            truth = case['truth']
            preds = case['preds']
            sample_idx = case.get('sample_idx')
            levels_k = case.get('levels_k')
            if levels_k is None:
                levels_k = gt_isotherm_levels_k(truth, quantiles)

            chamfers = [
                mean_chamfer_isotherms(pred, truth, levels_k=levels_k)
                for _, pred in preds
            ]
            best_j = int(np.nanargmin(chamfers)) if chamfers else -1

            row_title = f'Case {i + 1}'
            if sample_idx is not None:
                row_title += f'  (sample #{sample_idx})'

            # --- GT reference ---
            ax_gt = axes[i, 0]
            ax_gt.imshow(
                truth, cmap='coolwarm', origin='upper',
                vmin=float(truth.min()), vmax=float(truth.max()),
                interpolation='nearest',
            )
            _draw_isotherm_contours(
                ax_gt, truth, levels_k, colors, linewidths=2.0, linestyles='-',
            )
            ax_gt.set_title('Ground truth', fontsize=10, fontweight='semibold', pad=6)
            ax_gt.set_aspect('equal')
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            ax_gt.text(
                0.02, 0.98, row_title, transform=ax_gt.transAxes,
                fontsize=9, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2),
            )

            # --- model overlays ---
            for j, ((name, pred), chamfer) in enumerate(zip(preds, chamfers)):
                ax = axes[i, j + 1]
                ax.imshow(
                    truth, cmap='coolwarm', origin='upper', alpha=0.35,
                    vmin=float(truth.min()), vmax=float(truth.max()),
                    interpolation='nearest',
                )
                _draw_isotherm_contours(
                    ax, truth, levels_k, colors, linewidths=1.6, linestyles='--',
                )
                _draw_isotherm_contours(
                    ax, pred, levels_k, colors, linewidths=1.1, linestyles='-',
                )
                title = name
                if not np.isnan(chamfer):
                    title += f'\nChamfer = {chamfer:.2f} px'
                ax.set_title(title, fontsize=9, fontweight='semibold', pad=4)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])

                if j == best_j:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#1a9850')
                        spine.set_linewidth(2.8)
                elif name == highlight_model:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#2166ac')
                        spine.set_linewidth(1.8)

        # column headers for models (row 0 only handled via titles)
        for j, name in enumerate(model_names):
            axes[0, j + 1].set_title(
                axes[0, j + 1].get_title(),
                fontsize=9, fontweight='semibold',
            )

        legend_handles = [
            Line2D([0], [0], color=colors[k], lw=2.2, linestyle='-',
                   label=f'{level_labels[k]} isotherm')
            for k in range(nlevels)
        ]
        legend_handles += [
            Line2D([0], [0], color='0.25', lw=1.6, linestyle='--', label='GT contour'),
            Line2D([0], [0], color='0.25', lw=1.1, linestyle='-', label='Model contour'),
            Line2D([0], [0], color='#1a9850', lw=2.8, label='Best Chamfer (per case)'),
        ]
        fig.legend(
            handles=legend_handles, loc='lower center', ncol=nlevels + 3,
            fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01),
        )

        fig.suptitle(
            'Isotherm contour comparison: six models vs. ground truth (GT quantile levels)',
            fontsize=13, y=0.98,
        )
        fig.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.10, wspace=0.12, hspace=0.32)
        plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
        plt.close()


MAE_HIST_COLORS = {
    'SGF-RecFNO': '#B2182B',
    'SGF-RecFNO (K=4)': '#B2182B',
    'SGF-RecFNO (K=8)': '#D6604D',
    'IsoRecFNO': '#2166AC',
    'RecFNO': '#4393C3',
    'PINO': '#4DAF4A',
    'Geo-FNO': '#FF7F00',
    'GINO': '#984EA3',
}

# Inline labels for panel (a) ECDF — target CDF level and text offset per model.
ECDF_LABEL_CONFIG = {
    'SGF-RecFNO': {'target_y': 0.35, 'xytext': (8, 2)},
    'SGF-RecFNO (K=4)': {'target_y': 0.35, 'xytext': (8, 2)},
    'SGF-RecFNO (K=8)': {'target_y': 0.45, 'xytext': (8, -6)},
    'PINO': {'target_y': 0.55, 'xytext': (8, -8)},
    'IsoRecFNO': {'target_y': 0.65, 'xytext': (8, 2)},
    'RecFNO': {'target_y': 0.75, 'xytext': (8, -6)},
    'Geo-FNO': {'target_y': 0.85, 'xytext': (8, 4)},
    'GINO': {'target_y': 0.94, 'xytext': (8, 0)},
}


def _ecdf_display_name(name):
    if name == 'SGF-RecFNO':
        return 'SGF-RecFNO (K=4)'
    return name


def _label_ecdf_curve(ax, xs, ys, label, color, *, target_y=0.5, xytext=(8, 0)):
    """Annotate an ECDF curve with the model name (no legend)."""
    idx = int(np.argmin(np.abs(ys - target_y)))
    ax.annotate(
        label,
        (xs[idx], ys[idx]),
        color=color,
        fontsize=9,
        fontweight='semibold',
        ha='left',
        va='center',
        xytext=xytext,
        textcoords='offset points',
        clip_on=False,
    )


def plot_mae_distribution_overlay(
    model_maes,
    out_path,
    bins=45,
    highlight_model='SGF-RecFNO',
    zoom_percentile=99.5,
):
    """
    Per-sample MAE distribution: ECDF (log-x, all models) + ridge KDE (zoomed core group).

    model_maes: ordered list of (model_name, mae_array_K).
    """
    if not model_maes:
        raise ValueError('model_maes must be non-empty')

    from matplotlib.gridspec import GridSpec
    from scipy.stats import gaussian_kde

    n_samples = len(model_maes[0][1])
    arrays = [(name, np.asarray(maes, dtype=np.float64)) for name, maes in model_maes]

    # Models within the zoom panel: exclude extreme outliers (e.g. GINO) by p99 threshold
    all_core = np.concatenate([m for _, m in arrays])
    zoom_hi = float(np.percentile(all_core, zoom_percentile))
    core = [(n, m) for n, m in arrays if np.percentile(m, 99) <= zoom_hi * 1.5]
    outliers = [(n, m) for n, m in arrays if (n, m) not in core]
    if not core:
        core, outliers = arrays, []

    # Ridge order: best median MAE on top
    core = sorted(core, key=lambda x: np.median(x[1]))

    with plt.rc_context({
        'font.size': 10,
        'font.family': 'serif',
        'axes.linewidth': 0.9,
    }):
        fig = plt.figure(figsize=(11.5, 7.2))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.35], hspace=0.38)

        # --- (a) ECDF, log-x: all models on one comparable curve plot ---
        ax_cdf = fig.add_subplot(gs[0, 0])
        for name, maes in arrays:
            color = MAE_HIST_COLORS.get(name, '#666666')
            lw = 2.4 if name == highlight_model else 1.5
            z = 5 if name == highlight_model else 3
            xs = np.sort(maes)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax_cdf.plot(xs, ys, color=color, linewidth=lw, zorder=z)
            cfg = ECDF_LABEL_CONFIG.get(name, {'target_y': 0.5, 'xytext': (8, 0)})
            _label_ecdf_curve(
                ax_cdf, xs, ys, _ecdf_display_name(name), color,
                target_y=cfg['target_y'],
                xytext=cfg['xytext'],
            )
        ax_cdf.set_xscale('log')
        ax_cdf.set_xlim(max(1e-4, float(all_core.min()) * 0.7), float(all_core.max()) * 1.15)
        ax_cdf.set_ylim(0, 1.02)
        ax_cdf.set_xlabel('Per-sample MAE (K)')
        ax_cdf.set_ylabel('Fraction of test cases')
        ax_cdf.set_title(
            f'(a) Empirical CDF  ($n={n_samples}$ per model)',
            fontsize=11, loc='left', pad=8,
        )
        ax_cdf.axhline(0.5, color='0.75', linewidth=0.8, linestyle=':', zorder=0)
        ax_cdf.axhline(0.9, color='0.85', linewidth=0.6, linestyle=':', zorder=0)
        ax_cdf.text(0.99, 0.51, '50%', transform=ax_cdf.transAxes, ha='right', va='bottom',
                    fontsize=8, color='0.45')
        ax_cdf.text(0.99, 0.91, '90%', transform=ax_cdf.transAxes, ha='right', va='bottom',
                    fontsize=8, color='0.45')
        ax_cdf.spines['top'].set_visible(False)
        ax_cdf.spines['right'].set_visible(False)
        ax_cdf.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.6)

        # --- (b) Ridge KDE: zoomed view for the accurate-model cluster ---
        ax_ridge = fig.add_subplot(gs[1, 0])
        ridge_hi = float(np.percentile(np.concatenate([m for _, m in core]), zoom_percentile)) * 1.08
        xs_grid = np.linspace(0, ridge_hi, 300)
        ridge_scale = 0.85

        for i, (name, maes) in enumerate(core):
            color = MAE_HIST_COLORS.get(name, '#666666')
            lw = 2.0 if name == highlight_model else 1.2
            maes_clip = maes[maes <= ridge_hi * 1.2]
            if len(maes_clip) < 2:
                continue
            kde = gaussian_kde(maes_clip)
            dens = kde(xs_grid)
            dens = dens / dens.max() * ridge_scale
            y0 = len(core) - 1 - i
            ax_ridge.fill_between(
                xs_grid, y0, y0 + dens, color=color, alpha=0.55, zorder=2,
            )
            ax_ridge.plot(xs_grid, y0 + dens, color=color, linewidth=lw, zorder=3)
            ax_ridge.text(
                -0.02, y0 + ridge_scale * 0.35, name,
                transform=ax_ridge.get_yaxis_transform(),
                ha='right', va='center', fontsize=9,
                fontweight='bold' if name == highlight_model else 'normal',
                color=color,
            )

        ax_ridge.set_xlim(0, ridge_hi)
        ax_ridge.set_ylim(-0.15, len(core) - 0.15 + ridge_scale)
        ax_ridge.set_yticks([])
        ax_ridge.set_xlabel('Per-sample MAE (K)')
        ax_ridge.set_title(
            f'(b) Ridge density (zoomed to MAE $\\leq$ {ridge_hi:.3f} K, top-{len(core)} models)',
            fontsize=11, loc='left', pad=8,
        )
        ax_ridge.spines['top'].set_visible(False)
        ax_ridge.spines['right'].set_visible(False)
        ax_ridge.spines['left'].set_visible(False)
        ax_ridge.grid(axis='x', linestyle='--', alpha=0.35, linewidth=0.6)

        if outliers:
            note = 'Not shown in (b): ' + ', '.join(
                f'{n} (med={np.median(m):.3f} K)' for n, m in outliers
            )
            fig.text(0.5, 0.01, note, ha='center', fontsize=8.5, color='0.35', style='italic')

        fig.suptitle(
            'Per-sample MAE distribution on test set (samples 5000–5999)',
            fontsize=12, y=0.98,
        )
        fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.08)
        plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.18)
        plt.close()


def _annotate_curve_label(ax, x, y, label, color, *, log_y=False, x_frac=0.70, xytext=(6, 0)):
    """Place a model name on its curve (no legend)."""
    n = len(x)
    if n == 0:
        return
    idx = int(np.clip(x_frac, 0.05, 0.95) * (n - 1))
    if log_y:
        for j in range(idx, n):
            if y[j] > 0:
                idx = j
                break
        for j in range(idx, -1, -1):
            if y[j] > 0:
                idx = j
                break
    ax.annotate(
        label,
        (x[idx], y[idx]),
        color=color,
        fontsize=9,
        fontweight='semibold',
        ha='left',
        va='center',
        xytext=xytext,
        textcoords='offset points',
        clip_on=False,
    )


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


def plot_spectrum_error_figure(
    curves,
    out_path,
    *,
    title='',
    inline_labels=True,
    show_legend=False,
    label_x_frac=0.68,
    label_x_fracs=None,
    label_offsets=None,
    colors=None,
    figsize=(6.8, 4.6),
    dpi=300,
):
    """
    Plot radial squared spectrum error vs. ground truth.

    ``curves``: {model_name: err_array} or {model_name: (pred_spec, err_array)}.
    When ``inline_labels`` is True, model names are drawn on the curves (no legend).
    """
    if label_offsets is None:
        label_offsets = {}
    if label_x_fracs is None:
        label_x_fracs = {}

    default_colors = [
        '#B2182B', '#D6604D', '#2166AC', '#4393C3', '#4DAF4A', '#984EA3', '#FF7F00',
    ]

    with plt.rc_context({
        'font.size': 10,
        'font.family': 'serif',
        'axes.linewidth': 0.9,
    }):
        fig, ax = plt.subplots(figsize=figsize)
        names = list(curves.keys())

        for i, name in enumerate(names):
            data = curves[name]
            err = data[1] if isinstance(data, (tuple, list)) else data
            err = np.asarray(err, dtype=np.float64)
            x = np.arange(len(err))
            color = (colors or {}).get(name, default_colors[i % len(default_colors)])
            ax.plot(x, err, color=color, linewidth=1.6, zorder=3)
            if inline_labels:
                _annotate_curve_label(
                    ax, x, err, name, color,
                    log_y=True,
                    x_frac=label_x_fracs.get(name, label_x_frac),
                    xytext=label_offsets.get(name, (6, 0)),
                )

        ax.set_xlabel('Radial frequency bin')
        ax.set_ylabel('Squared spectrum error')
        ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, which='both', linestyle='--', alpha=0.35, linewidth=0.6)
        if show_legend:
            ax.legend(fontsize=8, frameon=False)
        if title:
            ax.set_title(title)

        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.08)
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


def plot_recfno_comparison_2x3(truth, preds, out_path, sample_idx=None):
    """
    2×3 panel: row0 = predictions, row1 = |error| for three models.
    Ground truth is drawn in a dedicated left column spanning both rows.

    preds: list of (name, prediction_in_K) in column order.
    """
    ncols = len(preds)
    fig = plt.figure(figsize=(3.4 * ncols + 3.2, 6.8))
    gs = fig.add_gridspec(
        2, ncols + 1,
        width_ratios=[1.05] + [1] * ncols,
        height_ratios=[1, 1],
        wspace=0.28,
        hspace=0.12,
    )

    vmin = float(min(truth.min(), min(p.min() for _, p in preds)))
    vmax = float(max(truth.max(), max(p.max() for _, p in preds)))
    err_maps = [np.abs(p - truth) for _, p in preds]
    err_vmax = max(float(e.max()) for e in err_maps)
    err_vmax = max(err_vmax, 1e-6)

    ax_gt = fig.add_subplot(gs[:, 0])
    im_gt = ax_gt.imshow(truth, cmap='coolwarm', origin='upper', vmin=vmin, vmax=vmax)
    title_suffix = f' (sample {sample_idx})' if sample_idx is not None else ''
    ax_gt.set_title(f'Ground Truth{title_suffix}', fontsize=11)
    ax_gt.set_aspect('equal')
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])
    cbar_gt = fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.02)
    cbar_gt.set_label('T (K)', fontsize=9)

    for j, ((name, pred), err) in enumerate(zip(preds, err_maps)):
        mae = float(err.mean())

        ax_p = fig.add_subplot(gs[0, j + 1])
        im_p = ax_p.imshow(pred, cmap='coolwarm', origin='upper', vmin=vmin, vmax=vmax)
        ax_p.set_title(name, fontsize=11, fontweight='semibold')
        ax_p.set_aspect('equal')
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        if j == 0:
            ax_p.set_ylabel('Prediction', fontsize=10)
        cbar_p = fig.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.02)
        cbar_p.set_label('T (K)', fontsize=8)

        ax_e = fig.add_subplot(gs[1, j + 1])
        im_e = ax_e.imshow(err, cmap='hot', origin='upper', vmin=0, vmax=err_vmax)
        ax_e.set_title(f'|Error|  MAE={mae:.4f} K', fontsize=10)
        ax_e.set_aspect('equal')
        ax_e.set_xticks([])
        ax_e.set_yticks([])
        if j == 0:
            ax_e.set_ylabel('Error', fontsize=10)
        cbar_e = fig.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.02)
        cbar_e.set_label('|ΔT| (K)', fontsize=8)

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def _draw_field(ax, data, cmap, vmin, vmax):
    im = ax.imshow(data, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _add_error_zoom_inset(ax, err, zoom_box, err_vmax, err_patch_vmax,
                          inset_loc='upper left', inset_frac=0.50):
    """Draw |error| field; ROI box + magnified inset (no connector lines)."""
    y0, y1, x0, x1 = zoom_box
    im = _draw_field(ax, err, 'hot', 0, err_vmax)

    rect = Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, edgecolor='lime', linewidth=1.6, linestyle='-',
    )
    ax.add_patch(rect)

    axins = inset_axes(
        ax, width=f'{int(inset_frac * 100)}%', height=f'{int(inset_frac * 100)}%',
        loc=inset_loc, borderpad=0.8,
    )
    patch = err[y0:y1, x0:x1]
    axins.imshow(
        patch, cmap='hot', origin='upper', vmin=0, vmax=err_patch_vmax,
        interpolation='nearest',
    )
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title('Local zoom', fontsize=8, pad=2)
    for spine in axins.spines.values():
        spine.set_edgecolor('0.35')
        spine.set_linewidth(1.2)
    return im


def _overlay_local_zoom(ax, data, zoom_box, cmap, vmin, vmax, *,
                        inset_vmin=None, inset_vmax=None, inset_frac=0.72,
                        inset_loc='lower right', label='Local zoom', inset_crop=0.5):
    """Add ROI box + magnified inset on an existing image axis (no connector lines).

    inset_crop: fraction of ROI center used in inset (<1 => stronger magnification).
    """
    y0, y1, x0, x1 = zoom_box
    rect = Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, edgecolor='lime', linewidth=1.4, linestyle='-',
        zorder=4,
    )
    ax.add_patch(rect)

    h, w = y1 - y0, x1 - x0
    if inset_crop < 1.0:
        ch = max(int(h * inset_crop), 4)
        cw = max(int(w * inset_crop), 4)
        iy0 = y0 + (h - ch) // 2
        ix0 = x0 + (w - cw) // 2
        iy1, ix1 = iy0 + ch, ix0 + cw
    else:
        iy0, iy1, ix0, ix1 = y0, y1, x0, x1

    ivmin = vmin if inset_vmin is None else inset_vmin
    ivmax = vmax if inset_vmax is None else inset_vmax
    axins = inset_axes(
        ax, width=f'{int(inset_frac * 100)}%', height=f'{int(inset_frac * 100)}%',
        loc=inset_loc, borderpad=0.35,
    )
    patch = data[iy0:iy1, ix0:ix1]
    axins.imshow(
        patch, cmap=cmap, origin='upper', vmin=ivmin, vmax=ivmax,
        interpolation='nearest',
    )
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title(label, fontsize=8, pad=2)
    for spine in axins.spines.values():
        spine.set_edgecolor('0.35')
        spine.set_linewidth(1.2)


def plot_recfno_comparison_2x3_zoom(truth, preds, out_path, sample_idx=None,
                                    zoom_box=DEFAULT_ZOOM_BOX):
    """
    Single-case 2×3 RecFNO-family comparison with bottom-center zoom insets.

    preds: list of (name, prediction_in_K) for RecFNO / IsoRecFNO / SGF-RecFNO.
    """
    ncols = len(preds)
    fig = plt.figure(figsize=(3.5 * ncols + 3.4, 7.2))
    gs = fig.add_gridspec(
        2, ncols + 1,
        width_ratios=[1.05] + [1] * ncols,
        height_ratios=[1, 1],
        wspace=0.30,
        hspace=0.14,
    )

    vmin = float(min(truth.min(), min(p.min() for _, p in preds)))
    vmax = float(max(truth.max(), max(p.max() for _, p in preds)))
    err_maps = [np.abs(p - truth) for _, p in preds]
    err_vmax = max(float(e.max()) for e in err_maps)
    err_vmax = max(err_vmax, 1e-6)

    y0, y1, x0, x1 = zoom_box
    err_patch_vmax = max(float(e[y0:y1, x0:x1].max()) for e in err_maps)
    err_patch_vmax = max(err_patch_vmax, 1e-6)

    ax_gt = fig.add_subplot(gs[:, 0])
    im_gt = _draw_field(ax_gt, truth, 'coolwarm', vmin, vmax)
    title_suffix = f' (sample {sample_idx})' if sample_idx is not None else ''
    ax_gt.set_title(f'Ground Truth{title_suffix}', fontsize=11)
    fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.02).set_label('T (K)', fontsize=9)

    for j, ((name, pred), err) in enumerate(zip(preds, err_maps)):
        mae = float(err.mean())
        mae_patch = float(err[y0:y1, x0:x1].mean())

        ax_p = fig.add_subplot(gs[0, j + 1])
        im_p = _draw_field(ax_p, pred, 'coolwarm', vmin, vmax)
        ax_p.set_title(name, fontsize=11, fontweight='semibold')
        if j == 0:
            ax_p.set_ylabel('Prediction', fontsize=10)
        fig.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.02).set_label('T (K)', fontsize=8)

        ax_e = fig.add_subplot(gs[1, j + 1])
        im_e = _add_error_zoom_inset(
            ax_e, err, zoom_box, err_vmax, err_patch_vmax, inset_loc='upper left',
        )
        ax_e.set_title(f'|Error|  MAE={mae:.4f} K  (patch {mae_patch:.4f} K)', fontsize=9)
        if j == 0:
            ax_e.set_ylabel('Error', fontsize=10)
        fig.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.02).set_label('|ΔT| (K)', fontsize=8)

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_three_cases_six_models(cases, out_path):
    """
    Three test cases × six models: per case, GT on the left and a 2×6 grid
    (prediction row + |error| row) on the right.

    cases: list of dicts with keys ``sample_idx``, ``truth``, ``preds``
           where ``preds`` is [(model_name, prediction_K), ...] (length 6).
    """
    if not cases:
        raise ValueError('cases must be non-empty')
    nmodels = len(cases[0]['preds'])
    if any(len(c['preds']) != nmodels for c in cases):
        raise ValueError('all cases must have the same number of models')

    ncases = len(cases)
    nrows = ncases * 2
    ncols = nmodels + 1

    fig = plt.figure(figsize=(2.35 * ncols + 0.8, 2.05 * nrows + 0.6))
    gs = fig.add_gridspec(
        nrows, ncols,
        width_ratios=[1.0] + [1.0] * nmodels,
        height_ratios=[1.0] * nrows,
        wspace=0.22,
        hspace=0.28,
    )

    model_names = [name for name, _ in cases[0]['preds']]

    for i, case in enumerate(cases):
        truth = case['truth']
        preds = case['preds']
        sample_idx = case.get('sample_idx')
        row_p = 2 * i
        row_e = 2 * i + 1

        vmin = float(min(truth.min(), min(p.min() for _, p in preds)))
        vmax = float(max(truth.max(), max(p.max() for _, p in preds)))
        err_maps = [np.abs(p - truth) for _, p in preds]
        err_vmax = max(max(float(e.max()) for e in err_maps), 1e-6)

        ax_gt = fig.add_subplot(gs[row_p:row_e + 1, 0])
        im_gt = ax_gt.imshow(truth, cmap='coolwarm', origin='upper', vmin=vmin, vmax=vmax)
        gt_title = 'Ground Truth'
        if sample_idx is not None:
            gt_title += f'\n(sample {sample_idx})'
        ax_gt.set_title(gt_title, fontsize=9)
        ax_gt.set_aspect('equal')
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        if i == 0:
            ax_gt.text(-0.22, 0.5, 'GT', transform=ax_gt.transAxes,
                       rotation=90, va='center', ha='center', fontsize=10)
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.02).set_label('T (K)', fontsize=7)

        for j, ((name, pred), err) in enumerate(zip(preds, err_maps)):
            mae = float(err.mean())
            col = j + 1

            ax_p = fig.add_subplot(gs[row_p, col])
            im_p = ax_p.imshow(pred, cmap='coolwarm', origin='upper', vmin=vmin, vmax=vmax)
            if i == 0:
                ax_p.set_title(name, fontsize=9, fontweight='semibold', pad=4)
            ax_p.set_aspect('equal')
            ax_p.set_xticks([])
            ax_p.set_yticks([])
            if j == nmodels - 1:
                fig.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.02).set_label('T (K)', fontsize=7)

            ax_e = fig.add_subplot(gs[row_e, col])
            im_e = ax_e.imshow(err, cmap='hot', origin='upper', vmin=0, vmax=err_vmax)
            ax_e.set_title(f'MAE={mae:.4f} K', fontsize=8)
            ax_e.set_aspect('equal')
            ax_e.set_xticks([])
            ax_e.set_yticks([])
            if j == nmodels - 1:
                fig.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.02).set_label('|ΔT| (K)', fontsize=7)

        # case label on the left margin
        fig.text(
            0.01, 1 - (row_p + 1) / nrows - 0.5 / nrows,
            f'Case {i + 1}\nPred',
            va='center', ha='left', fontsize=9, rotation=90,
        )
        fig.text(
            0.01, 1 - (row_e + 1) / nrows - 0.5 / nrows,
            'Err',
            va='center', ha='left', fontsize=9, rotation=90,
        )

    fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.02)
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()


def _format_bar_value(key, v):
    if key in ('psnr',):
        return f'{v:.2f}'
    if key in ('ssim',):
        return f'{v:.4f}'
    if key in ('relative_l2', 'spectrum_error', 'mse'):
        return f'{v:.2e}'
    if v < 0.1:
        return f'{v:.4f}'
    return f'{v:.2f}'


def plot_paper_error_comparison(rows, out_path, highlight_model='SGF-RecFNO'):
    """
    Paper-style bar charts for all test-set metrics.

    rows: list of dicts with model + metric keys.
    """
    metrics = [
        ('mae_k', 'MAE (K)'),
        ('rmse_k', 'RMSE (K)'),
        ('max_ae_k', 'MaxAE (K)'),
        ('spectrum_error', 'Spectrum error'),
        ('psnr', 'PSNR (dB)'),
        ('ssim', 'SSIM'),
    ]
    log_keys = {'max_ae_k', 'spectrum_error'}
    names = [r['model'] for r in rows]
    n = len(names)
    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols
    panel_labels = [f'({chr(ord("a") + i)})' for i in range(len(metrics))]

    with plt.rc_context({
        'font.size': 10,
        'axes.linewidth': 0.9,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'font.family': 'serif',
    }):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows))
        axes = np.atleast_1d(axes).flatten()
        palette = ['#2166AC' if name != highlight_model else '#B2182B' for name in names]

        for idx, ((key, ylabel), panel) in enumerate(zip(metrics, panel_labels)):
            ax = axes[idx]
            vals = [r[key] for r in rows]
            bars = ax.bar(
                range(n), vals, color=palette, edgecolor='white',
                linewidth=0.9, width=0.72, zorder=3,
            )
            ax.set_xticks(range(n))
            ax.set_xticklabels(names, rotation=22, ha='right')
            ax.set_ylabel(ylabel)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', linestyle='--', alpha=0.35, linewidth=0.6, zorder=0)

            pos_vals = [v for v in vals if v > 0]
            if key in log_keys and pos_vals and max(pos_vals) > 5 * sorted(pos_vals)[0]:
                ax.set_yscale('log')

            ymax = ax.get_ylim()[1]
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02 * ymax,
                    _format_bar_value(key, v),
                    ha='center', va='bottom', fontsize=7,
                )
            ax.text(
                -0.12, 1.05, panel, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left',
            )

        for ax in axes[len(metrics):]:
            ax.set_visible(False)

        fig.suptitle('Test-set metrics (samples 5000–5999)', fontsize=12, y=1.01)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_sdf_ablation_bars(rows, out_path):
    """
    Bar charts for SDF ablation: coarse / refine w/o SDF / full SGF-RecFNO.

    rows: list of dicts with ``variant`` plus ``mae_k``, ``rmse_k``, ``max_ae_k``.
    """
    metrics = [
        ('mae_k', 'MAE (K)'),
        ('rmse_k', 'RMSE (K)'),
        ('max_ae_k', 'MaxAE (K)'),
    ]
    names = [r['variant'] for r in rows]
    colors = ['#969696', '#F4A582', '#B2182B']
    panel_labels = ['(a)', '(b)', '(c)']

    with plt.rc_context({
        'font.size': 10,
        'font.family': 'serif',
        'axes.linewidth': 0.9,
    }):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
        for idx, (ax, (key, ylabel), panel) in enumerate(zip(axes, metrics, panel_labels)):
            vals = [r[key] for r in rows]
            bars = ax.bar(
                range(len(vals)), vals, color=colors[:len(vals)],
                edgecolor='white', linewidth=0.9, width=0.62, zorder=3,
            )
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(names, rotation=14, ha='right', fontsize=9)
            ax.set_ylabel(ylabel)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', linestyle='--', alpha=0.35, linewidth=0.6, zorder=0)

            ymax = ax.get_ylim()[1]
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.03 * ymax,
                    f'{v:.4f}',
                    ha='center', va='bottom', fontsize=8,
                )
            ax.text(
                -0.14, 1.06, panel, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left',
            )

        fig.suptitle(
            'Ablation: contribution of self-geometry SDF vs. refinement block alone',
            fontsize=12, y=1.02,
        )
        fig.text(
            0.5, -0.02,
            'Same SGF-RecFNO checkpoint; w/o SDF zeros SDF channels at inference (coarse only to refine block).',
            ha='center', fontsize=9, color='0.35', style='italic',
        )
        fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.22, wspace=0.32)
        plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
        plt.close()


def plot_loss_ablation_bars(rows, out_path, metric='mae_k'):
    """Bar chart for loss-component ablations (test MAE by default)."""
    labels = {
        'mae_k': 'Test MAE (K)',
        'rmse_k': 'Test RMSE (K)',
        'max_ae_k': 'Test MaxAE (K)',
    }
    ylabel = labels.get(metric, metric)
    names = [r['label'] for r in rows]
    vals = [r[metric] for r in rows]
    colors = ['#B2182B' if r['key'] == 'full' else '#4393C3' for r in rows]

    with plt.rc_context({'font.size': 10, 'font.family': 'serif'}):
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor='white', width=0.62, zorder=3)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, rotation=12, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title('Loss-component ablation (300 epochs, test set)', fontsize=12, pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.35)

        ymax = ax.get_ylim()[1]
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.01 * ymax, 'N/A',
                        ha='center', va='bottom', fontsize=8, color='0.5')
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * ymax,
                        f'{v:.4f}', ha='center', va='bottom', fontsize=8)

        fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.20)
        plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.15)
        plt.close()


def plot_quantile_k_curve(rows, out_path, metric='mae_k'):
    """Line plot: test error vs. number of SDF channels K."""
    labels = {
        'mae_k': 'Test MAE (K)',
        'rmse_k': 'Test RMSE (K)',
        'max_ae_k': 'Test MaxAE (K)',
    }
    ylabel = labels.get(metric, metric)
    ks = [r['k'] for r in rows]
    vals = [r[metric] for r in rows]

    with plt.rc_context({'font.size': 10, 'font.family': 'serif'}):
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.plot(ks, vals, 'o-', color='#B2182B', linewidth=2.0, markersize=9, zorder=3)
        for k, v in zip(ks, vals):
            if not np.isnan(v):
                ax.annotate(f'{v:.4f}', (k, v), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=8)

        ax.set_xticks(ks)
        ax.set_xlabel('Number of SDF channels $K$')
        ax.set_ylabel(ylabel)
        ax.set_title('SDF depth ablation: test error vs. $K$ (300 epochs)', fontsize=12, pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.35)

        fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.14)
        plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.15)
        plt.close()


def plot_benchmark_problem_setup(
    field_k,
    out_path,
    sample_idx=5500,
    sensor_positions=None,
    domain_size=0.1,
    grid_size=200,
    dirichlet_j0=96,
    dirichlet_j1=108,
):
    """
    Annotated temperature field showing benchmark geometry, sensors, and BCs.

    field_k: (H, W) temperature in Kelvin.
    sensor_positions: (25, 2) array of [row, col] indices.
    """
    from data.dataset import HEAT_SENSOR_POSITIONS

    if sensor_positions is None:
        sensor_positions = HEAT_SENSOR_POSITIONS

    n = grid_size
    L = domain_size
    dx = L / (n - 1)
    sensors = np.asarray(sensor_positions)

    # Physical coords: x = col * dx, y = row * dx (origin at top-left of domain)
    x_dir0 = dirichlet_j0 * dx
    x_dir1 = dirichlet_j1 * dx

    with plt.rc_context({
        'font.size': 9,
        'font.family': 'serif',
        'axes.linewidth': 1.0,
    }):
        fig, ax = plt.subplots(figsize=(8.2, 8.6))
        extent = [0, L, L, 0]  # x left-right, y: top=0 bottom=L (matches i=0 at top)
        im = ax.imshow(field_k, cmap='coolwarm', origin='upper', extent=extent, aspect='equal')
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.82)
        cbar.set_label('Temperature  $T$  (K)', fontsize=10)

        # --- sensors: white box + black dot ---
        box_hw = 1.8 * dx
        for r, c in sensors:
            xc = c * dx
            yc = r * dx
            ax.add_patch(Rectangle(
                (xc - box_hw, yc - box_hw), 2 * box_hw, 2 * box_hw,
                fill=True, facecolor='white', edgecolor='black',
                linewidth=0.9, zorder=5,
            ))
            ax.plot(xc, yc, 'o', color='black', markersize=3.2, zorder=6)

        # --- boundary overlays (outside domain edges) ---
        pad = 0.006
        lw_bc = 3.0

        # Top: adiabatic | Dirichlet | adiabatic
        ax.plot([0, x_dir0], [-pad, -pad], color='#1B9E77', lw=lw_bc, solid_capstyle='butt', clip_on=False)
        ax.plot([x_dir0, x_dir1], [-pad, -pad], color='#D95F02', lw=lw_bc, solid_capstyle='butt', clip_on=False)
        ax.plot([x_dir1, L], [-pad, -pad], color='#1B9E77', lw=lw_bc, solid_capstyle='butt', clip_on=False)

        # Bottom, left, right: adiabatic
        ax.plot([0, L], [L + pad, L + pad], color='#1B9E77', lw=lw_bc, solid_capstyle='butt', clip_on=False)
        ax.plot([-pad, -pad], [0, L], color='#1B9E77', lw=lw_bc, solid_capstyle='butt', clip_on=False)
        ax.plot([L + pad, L + pad], [0, L], color='#1B9E77', lw=lw_bc, solid_capstyle='butt', clip_on=False)

        # BC labels
        ax.text(x_dir0 / 2, -0.012, 'Adiabatic', ha='center', va='bottom', fontsize=8, color='#1B9E77')
        ax.text((x_dir0 + x_dir1) / 2, -0.012, r'Dirichlet  $T=T_b$', ha='center', va='bottom',
                fontsize=8.5, color='#D95F02', fontweight='bold')
        ax.text((x_dir1 + L) / 2, -0.012, 'Adiabatic', ha='center', va='bottom', fontsize=8, color='#1B9E77')
        ax.text(L / 2, L + 0.012, r'Adiabatic  ($\partial T/\partial n=0$)', ha='center', va='top',
                fontsize=8, color='#1B9E77')
        ax.text(-0.012, L / 2, r'Adiabatic', ha='right', va='center', fontsize=8, color='#1B9E77',
                rotation=90)
        ax.text(L + 0.012, L / 2, r'Adiabatic', ha='left', va='center', fontsize=8, color='#1B9E77',
                rotation=90)

        # Dirichlet segment highlight on domain edge
        ax.add_patch(Rectangle(
            (x_dir0, -0.001), x_dir1 - x_dir0, 0.002,
            transform=ax.transData, facecolor='#D95F02', edgecolor='none', alpha=0.85, zorder=4,
        ))

        ax.set_xlim(-0.018, L + 0.018)
        ax.set_ylim(L + 0.022, -0.022)
        ax.set_xlabel(r'$x$  (m)', fontsize=10)
        ax.set_ylabel(r'$y$  (m)', fontsize=10)
        ax.set_title(
            f'Heat conduction benchmark setup  (test sample #{sample_idx})',
            fontsize=11, pad=10,
        )

        # Legend for sensors / BC
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='w', markeredgecolor='k',
                   markersize=8, label='Sensor (25, $5\\times5$ grid)'),
            Line2D([0], [0], color='#D95F02', lw=3, label=r'Dirichlet BC ($T=T_b$)'),
            Line2D([0], [0], color='#1B9E77', lw=3, label=r'Adiabatic BC ($\partial T/\partial n=0$)'),
        ]
        ax.legend(handles=legend_handles, loc='lower right', framealpha=0.92, fontsize=8)

        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_sgf_pipeline_figure(
    truth_k,
    coarse_k,
    refined_k,
    sdf_maps,
    delta_k,
    out_path,
    sample_idx=5500,
    quantiles=(0.2, 0.4, 0.6, 0.8),
    mae_coarse=None,
    mae_refined=None,
):
    """
    2×5 figure: row1 field reconstruction; row2 self-geometry SDF + delta.

    sdf_maps: (K, H, W) from coarse prediction only, values in (-1, 1).
    """
    from matplotlib.gridspec import GridSpec

    err_coarse = np.abs(coarse_k - truth_k)
    err_refined = np.abs(refined_k - truth_k)
    if mae_coarse is None:
        mae_coarse = float(err_coarse.mean())
    if mae_refined is None:
        mae_refined = float(err_refined.mean())

    t_vmin = float(min(truth_k.min(), coarse_k.min(), refined_k.min()))
    t_vmax = float(max(truth_k.max(), coarse_k.max(), refined_k.max()))
    err_vmax = max(float(err_coarse.max()), float(err_refined.max()), 1e-6)
    d_vmax = max(float(np.abs(delta_k).max()), 1e-6)

    with plt.rc_context({
        'font.size': 10,
        'font.family': 'serif',
        'axes.linewidth': 0.8,
    }):
        # 5 data columns + 1 right strip with stacked colorbar pairs
        fig = plt.figure(figsize=(24, 10))
        gs = GridSpec(
            2, 6, figure=fig,
            width_ratios=[1, 1, 1, 1, 1, 0.11],
            height_ratios=[1, 1],
            wspace=0.10, hspace=0.28,
            left=0.05, right=0.94, top=0.90, bottom=0.11,
        )
        gs_cb_row0 = gs[0, 5].subgridspec(2, 1, hspace=0.55)
        gs_cb_row1 = gs[1, 5].subgridspec(2, 1, hspace=0.55)
        panel_ids = [f'({chr(ord("a") + i)})' for i in range(10)]

        def _add_cbar(cax, mappable, label, orientation='vertical'):
            cb = fig.colorbar(mappable, cax=cax, orientation=orientation)
            cb.set_label(label, fontsize=10, labelpad=8)
            cb.ax.tick_params(labelsize=9, pad=2)
            return cb

        def _style_ax(ax, panel_id, aspect=1.0):
            ax.set_aspect(aspect)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.02, 0.98, panel_id, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2),
            )

        # --- Row 1 ---
        row1 = [
            (truth_k, 'coolwarm', t_vmin, t_vmax, 'Ground truth'),
            (coarse_k, 'coolwarm', t_vmin, t_vmax, f'Coarse (RecFNO), MAE={mae_coarse:.4f} K'),
            (refined_k, 'coolwarm', t_vmin, t_vmax, f'Refined (SGF-RecFNO), MAE={mae_refined:.4f} K'),
            (err_coarse, 'hot', 0, err_vmax, f'|Error| coarse, MAE={mae_coarse:.4f} K'),
            (err_refined, 'hot', 0, err_vmax, f'|Error| refined, MAE={mae_refined:.4f} K'),
        ]
        ims_t, ims_e = None, None
        for j, (data, cmap, vmin, vmax, title) in enumerate(row1):
            ax = fig.add_subplot(gs[0, j])
            im = ax.imshow(data, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_title(title, fontsize=10, pad=6)
            _style_ax(ax, panel_ids[j])
            if j == 0:
                ax.set_ylabel('Field reconstruction', fontsize=11, labelpad=10)
            if j <= 2:
                ims_t = im
            else:
                ims_e = im

        cax_t = fig.add_subplot(gs_cb_row0[0, 0])
        _add_cbar(cax_t, ims_t, '$T$ (K)')
        cax_e = fig.add_subplot(gs_cb_row0[1, 0])
        _add_cbar(cax_e, ims_e, r'$|\Delta T|$ (K)')

        # --- Row 2 ---
        ims_sdf = None
        for j, q in enumerate(quantiles):
            ax = fig.add_subplot(gs[1, j])
            im = ax.imshow(sdf_maps[j], cmap='RdBu', origin='upper', vmin=-1, vmax=1,
                           interpolation='nearest')
            ax.set_title(f'SDF, $q={q:.1f}$', fontsize=10, pad=6)
            _style_ax(ax, panel_ids[5 + j])
            if j == 0:
                ax.set_ylabel('Self-geometry & correction', fontsize=11, labelpad=10)
            ims_sdf = im

        ax_d = fig.add_subplot(gs[1, 4])
        im_d = ax_d.imshow(delta_k, cmap='RdBu_r', origin='upper',
                           vmin=-d_vmax, vmax=d_vmax, interpolation='nearest')
        ax_d.set_title(r'$\Delta T$ = refined $-$ coarse', fontsize=10, pad=6)
        _style_ax(ax_d, panel_ids[9])

        cax_s = fig.add_subplot(gs_cb_row1[0, 0])
        _add_cbar(cax_s, ims_sdf, 'SDF')
        cax_dcb = fig.add_subplot(gs_cb_row1[1, 0])
        _add_cbar(cax_dcb, im_d, r'$\Delta T$ (K)')

        fig.suptitle(
            f'SGF-RecFNO pipeline: coarse prediction $\\rightarrow$ self-geometry $\\rightarrow$ refinement'
            f'  (sample #{sample_idx})',
            fontsize=13, y=0.97,
        )
        fig.text(
            0.5, 0.03,
            'Row 2: all SDF channels are computed from the coarse prediction only (not from ground truth). '
            'Blue / red in SDF: below / above the corresponding isotherm level.',
            ha='center', fontsize=9.5, color='0.25', style='italic',
        )
        plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.25)
        plt.close()
