#!/usr/bin/env python3
"""Evaluate 6 models on the test set; export full metrics table and paper-style plots."""
import argparse
import csv
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.bootstrap import ensure_repo_context

ensure_repo_context(heat2d_cwd=True)

from benchmark.config import ALL_MODELS, COMPARISON_OUT_DIR, FIELD_STD
from benchmark.registry import MODEL_SPECS, _find_ckpt
from benchmark.run_comparison import ensure_checkpoints, evaluate_model
from benchmark.visualize import plot_paper_error_comparison

OUT_DIR = COMPARISON_OUT_DIR
STD = FIELD_STD

# All exported metrics (raw data columns)
METRIC_FIELDS = [
    'model',
    'mse',
    'mae_k',
    'rmse_k',
    'max_ae_k',
    'psnr',
    'ssim',
    'spectrum_error',
    'num_samples',
    'checkpoint',
]


def parse_args():
    p = argparse.ArgumentParser(description='Test-set full metrics comparison')
    p.add_argument('--out-dir', default=OUT_DIR)
    p.add_argument('--max-samples', type=int, default=0, help='0 = full test set (1000)')
    p.add_argument('--models', nargs='+', default=None)
    p.add_argument('--copy-figures', action='store_true')
    return p.parse_args()


def _row_from_result(result):
    return {
        'model': result['model'],
        'mse': result['mse'],
        'mae_k': result['mae_k'],
        'rmse_k': result['rmse_k'],
        'max_ae_k': result['max_ae_k'],
        'psnr': result['psnr'],
        'ssim': result['ssim'],
        'spectrum_error': result['spectrum_error'],
        'num_samples': result['num_samples'],
        'checkpoint': result['checkpoint'],
    }


def save_metrics_table(rows, out_dir):
    csv_path = os.path.join(out_dir, 'test_metrics_full.csv')
    json_path = os.path.join(out_dir, 'test_metrics_full.json')
    tex_path = os.path.join(out_dir, 'test_metrics_full.tex')
    md_path = os.path.join(out_dir, 'test_metrics_full.md')

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)

    # Markdown table (easy to read / copy)
    md_lines = [
        '# Test-set metrics (samples 5000–5999, n=1000 per model)',
        '',
        '| Model | MSE | MAE (K) | RMSE (K) | MaxAE (K) | PSNR | SSIM | Spectrum err |',
        '|-------|-----|---------|----------|-----------|------|------|--------------|',
    ]
    for r in rows:
        md_lines.append(
            f"| {r['model']} "
            f"| {r['mse']:.4e} "
            f"| {r['mae_k']:.4f} "
            f"| {r['rmse_k']:.4f} "
            f"| {r['max_ae_k']:.4f} "
            f"| {r['psnr']:.2f} "
            f"| {r['ssim']:.6f} "
            f"| {r['spectrum_error']:.4e} |"
        )
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')

    lines = [
        r'\begin{table*}[ht]',
        r'\centering',
        r'\caption{Full test-set metrics. Samples 5000--5999 ($n=1000$).}',
        r'\label{tab:test-metrics-full}',
        r'\small',
        r'\begin{tabular}{lccccccc}',
        r'\toprule',
        r'Model & MSE & MAE & RMSE & MaxAE & PSNR & SSIM & Spec.\,Err \\',
        r'\midrule',
    ]
    for r in rows:
        lines.append(
            f"{r['model']} & {r['mse']:.4e} & "
            f"{r['mae_k']:.4f} & {r['rmse_k']:.4f} & {r['max_ae_k']:.4f} & "
            f"{r['psnr']:.2f} & {r['ssim']:.4f} & {r['spectrum_error']:.4e} \\\\"
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))

    # Keep legacy filenames as symlinks/copies for backward compatibility
    legacy_csv = os.path.join(out_dir, 'test_error_metrics.csv')
    with open(legacy_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return csv_path, json_path, tex_path, md_path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    missing = ensure_checkpoints(train_missing=False)

    if args.models:
        model_list = args.models
    else:
        model_list = [
            name for name in ALL_MODELS
            if name in MODEL_SPECS and name not in missing and _find_ckpt(MODEL_SPECS[name]['exp'])
        ]

    rows = []
    for name in model_list:
        print(f'\n>>> Evaluating {name} on test set...', flush=True)
        try:
            result, _, _ = evaluate_model(name, max_samples=args.max_samples)
        except FileNotFoundError as exc:
            print(f'SKIP {name}: {exc}', flush=True)
            continue

        row = _row_from_result(result)
        rows.append(row)
        print(
            f"  MAE={row['mae_k']:.4f} K  RMSE={row['rmse_k']:.4f} K  "
            f"MaxAE={row['max_ae_k']:.4f} K",
            flush=True,
        )
        print(
            f"  PSNR={row['psnr']:.2f} dB  SSIM={row['ssim']:.6f}  "
            f"SpecErr={row['spectrum_error']:.4e}  (n={row['num_samples']})",
            flush=True,
        )

    if not rows:
        raise SystemExit('No models evaluated.')

    rows.sort(key=lambda r: r['mae_k'])

    csv_path, json_path, tex_path, md_path = save_metrics_table(rows, args.out_dir)
    fig_path = os.path.join(args.out_dir, 'test_metrics_comparison.png')
    plot_paper_error_comparison(rows, fig_path)

    print('\n=== Test-set full metrics ===', flush=True)
    print(f'CSV (raw): {csv_path}', flush=True)
    print(f'Markdown : {md_path}', flush=True)
    print(f'JSON     : {json_path}', flush=True)
    print(f'TeX      : {tex_path}', flush=True)
    print(f'Figure   : {fig_path}', flush=True)

    if args.copy_figures:
        from benchmark.figures_paths import copy_to_figures
        dst = copy_to_figures(fig_path, 'test_metrics_comparison.png', category='benchmark')
        print(f'Copied: {dst}', flush=True)


if __name__ == '__main__':
    main()
