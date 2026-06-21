#!/usr/bin/env python3
"""Evaluate and compare RecFNO, IsoRecFNO, SGF-RecFNO on heat test set."""
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(filename)

from data.dataset import HeatDataset
from model.fno import FNORecon
from model.iso_recfno import IsoRecFNO
from model.sgf_recfno import SGFRecFNO

MEAN, STD = 308.0, 50.0
BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'benchmark_300epoch')
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'ckpt')
TEST_INDEX = list(range(5000, 6000))
SAMPLE_IDX = 5500


def denorm(t):
    return t * STD + MEAN


def find_best_ckpt(exp_name):
    paths = sorted(glob.glob(os.path.join(CKPT_DIR, exp_name, 'best_epoch_*.pth')))
    if not paths:
        raise FileNotFoundError(f'no checkpoint for {exp_name}')
    return paths[-1]


def load_recfno(ckpt):
    net = FNORecon(25, (12, 12), (200, 200), modes1=50, modes2=50, width=32).cuda()
    net.load_state_dict(torch.load(ckpt, map_location='cuda', weights_only=False)['state_dict'])
    net.eval()
    return net


def load_iso(ckpt):
    net = IsoRecFNO(
        sensor_num=25, fc_size=(12, 12), out_size=(200, 200),
        modes1=50, modes2=50, width=32, num_iso_levels=4,
    ).cuda()
    net.load_state_dict(torch.load(ckpt, map_location='cuda', weights_only=False)['state_dict'])
    net.eval()
    return net


def load_sgf(ckpt):
    net = SGFRecFNO(
        sensor_num=25, fc_size=(12, 12), out_size=(200, 200),
        modes1=50, modes2=50, width=32, num_sdf=4,
    ).cuda()
    net.load_state_dict(torch.load(ckpt, map_location='cuda', weights_only=False)['state_dict'])
    net.eval()
    return net


@torch.no_grad()
def evaluate_model(name, net, loader, forward_fn):
    mae, rmse, max_ae, l1_norm, n = 0.0, 0.0, 0.0, 0.0, 0
    last_pred, last_target = None, None
    for inputs, targets in loader:
        inputs = torch.from_numpy(inputs.numpy()).float().cuda() if hasattr(inputs, 'numpy') else inputs.float().cuda()
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(np.asarray(inputs)).float().cuda()
        targets = targets.cuda()
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        if inputs.shape[-1] == 25 and inputs.dim() == 2:
            pass
        else:
            inputs = inputs.cuda()

        # HeatDataset returns numpy for inputs
        if isinstance(inputs, torch.Tensor) and inputs.dtype != torch.float32:
            inputs = inputs.float()

        pred = forward_fn(net, inputs)
        l1_norm += F.l1_loss(pred, targets).item() * targets.size(0)
        diff = denorm((pred - targets)[..., 0, :, :].cpu().numpy())
        mae += np.mean(np.abs(diff), axis=(1, 2)).sum()
        rmse += np.sqrt(np.mean(diff ** 2, axis=(1, 2))).sum()
        max_ae += np.max(np.abs(diff), axis=(1, 2)).sum()
        n += targets.size(0)
        last_pred, last_target = pred, targets

    return {
        'name': name,
        'l1_norm': l1_norm / n,
        'mae_k': mae / n,
        'rmse_k': rmse / n,
        'max_ae_k': max_ae / n,
        'last_pred': last_pred,
        'last_target': last_target,
    }


def main():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    loader = DataLoader(
        HeatDataset(index=TEST_INDEX),
        batch_size=8,
        num_workers=4,
    )

    models = [
        ('RecFNO', find_best_ckpt('benchmark_recfno_300'), load_recfno,
         lambda net, x: net(x)),
        ('IsoRecFNO', find_best_ckpt('benchmark_isorecfno_300'), load_iso,
         lambda net, x: net(x, return_aux=True)['field']),
        ('SGF-RecFNO', find_best_ckpt('benchmark_sgf-recfno_300'), load_sgf,
         lambda net, x: net(x, return_aux=True)['field']),
    ]

    results = []
    ckpt_info = []
    print('=== 三模型测试集评估 (samples 5000–5999) ===\n')
    for name, ckpt, loader_fn, fwd in models:
        net = loader_fn(ckpt)
        # fix input handling
        @torch.no_grad()
        def eval_loop(net, loader, fwd):
            mae, rmse, max_ae, l1_norm, n = 0.0, 0.0, 0.0, 0.0, 0
            last_pred, last_target = None, None
            for inputs, targets in loader:
                if not torch.is_tensor(inputs):
                    inputs = torch.from_numpy(np.asarray(inputs)).float()
                inputs, targets = inputs.cuda(), targets.cuda()
                pred = fwd(net, inputs)
                l1_norm += F.l1_loss(pred, targets).item() * targets.size(0)
                diff_k = ((pred - targets)[:, 0].cpu().numpy()) * STD
                mae += np.mean(np.abs(diff_k), axis=(1, 2)).sum()
                rmse += np.sqrt(np.mean(diff_k ** 2, axis=(1, 2))).sum()
                max_ae += np.max(np.abs(diff_k), axis=(1, 2)).sum()
                n += targets.size(0)
                last_pred, last_target = pred, targets
            return {
                'name': name, 'l1_norm': l1_norm / n,
                'mae_k': mae / n, 'rmse_k': rmse / n, 'max_ae_k': max_ae / n,
                'last_pred': last_pred, 'last_target': last_target,
            }

        r = eval_loop(net, loader, fwd)
        results.append(r)
        ckpt_info.append((name, ckpt, r))
        print(f'{name}:')
        print(f'  checkpoint: {ckpt}')
        print(f'  L1(norm)={r["l1_norm"]:.6f}  MAE={r["mae_k"]:.4f} K  RMSE={r["rmse_k"]:.4f} K  MaxAE={r["max_ae_k"]:.4f} K\n')

    # save CSV
    cmp_csv = os.path.join(BENCHMARK_DIR, 'test_comparison.csv')
    with open(cmp_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'l1_norm', 'mae_k', 'rmse_k', 'max_ae_k', 'checkpoint'])
        w.writeheader()
        for name, ckpt, r in ckpt_info:
            w.writerow({
                'model': name,
                'l1_norm': r['l1_norm'],
                'mae_k': r['mae_k'],
                'rmse_k': r['rmse_k'],
                'max_ae_k': r['max_ae_k'],
                'checkpoint': ckpt,
            })
    print(f'结果已保存: {cmp_csv}')

    # --- plots ---
    names = [r['name'] for r in results]
    metrics = ['mae_k', 'rmse_k', 'max_ae_k', 'l1_norm']
    labels = ['MAE (K)', 'RMSE (K)', 'Max AE (K)', 'L1 (norm)']
    colors = ['#2563eb', '#dc2626', '#059669']

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, metric, label in zip(axes, metrics, labels):
        vals = [r[metric] for r in results]
        bars = ax.bar(names, vals, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.tick_params(axis='x', rotation=15)
        for bar, v in zip(bars, vals):
            fmt = f'{v:.4f}' if metric != 'l1_norm' else f'{v:.5f}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt,
                    ha='center', va='bottom', fontsize=8)
    plt.suptitle('Test Set Performance (5000–5999, 1000 samples)', fontsize=12)
    plt.tight_layout()
    bar_path = os.path.join(BENCHMARK_DIR, 'test_comparison_bar.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()

    # convergence comparison: val field for iso/sgf, val_loss for recfno
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for fname, label, color in [
        ('recfno_history.csv', 'RecFNO val', '#2563eb'),
        ('isorecfno_history.csv', 'IsoRecFNO val field', '#dc2626'),
        ('sgf-recfno_history.csv', 'SGF-RecFNO val field', '#059669'),
    ]:
        path = os.path.join(BENCHMARK_DIR, fname)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        ep = [int(r['epoch']) for r in rows]
        if 'val_field' in rows[0]:
            y = [float(r['val_field']) for r in rows]
        else:
            y = [float(r['val_loss']) for r in rows]
        ax.plot(ep, y, '-', label=label, color=color, linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation field L1 (normalized)')
    ax.set_title('Convergence Comparison (field metric)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = axes[1]
    for fname, label, color in [
        ('recfno_history.csv', 'RecFNO', '#2563eb'),
        ('isorecfno_history.csv', 'IsoRecFNO', '#dc2626'),
        ('sgf-recfno_history.csv', 'SGF-RecFNO', '#059669'),
    ]:
        path = os.path.join(BENCHMARK_DIR, fname)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        ep = [int(r['epoch']) for r in rows]
        y = [float(r['val_loss']) for r in rows]
        ax.plot(ep, y, '-', label=label, color=color, linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val total loss')
    ax.set_title('Convergence Comparison (total val loss)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    conv_path = os.path.join(BENCHMARK_DIR, 'three_model_convergence.png')
    plt.savefig(conv_path, dpi=150, bbox_inches='tight')
    plt.close()

    # single sample visualization
    ds = HeatDataset(index=[SAMPLE_IDX])
    inputs, target = ds[0]
    inputs_t = torch.from_numpy(np.asarray(inputs)).float().unsqueeze(0).cuda()
    target_np = denorm(target[0].numpy())

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(target_np, cmap='coolwarm', origin='upper')
    axes[0].set_title(f'Truth (#{SAMPLE_IDX})')
    for ax, (name, ckpt, load_fn, fwd) in zip(axes[1:], models):
        net = load_fn(ckpt)
        with torch.no_grad():
            pred = fwd(net, inputs_t)
        pred_np = denorm(pred[0, 0].cpu().numpy())
        err = np.abs(pred_np - target_np)
        ax.imshow(pred_np, cmap='coolwarm', origin='upper')
        mae = err.mean()
        ax.set_title(f'{name}\nMAE={mae:.3f} K')
    for ax in axes:
        ax.set_aspect('equal')
    plt.suptitle(f'Sample #{SAMPLE_IDX} Reconstruction', fontsize=12)
    plt.tight_layout()
    sample_path = os.path.join(BENCHMARK_DIR, 'test_sample_5500.png')
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'图表: {bar_path}')
    print(f'      {conv_path}')
    print(f'      {sample_path}')

    # training best summary from CSV
    print('\n=== 训练最优 val field / val loss ===')
    for fname, name in [
        ('recfno_history.csv', 'RecFNO'),
        ('isorecfno_history.csv', 'IsoRecFNO'),
        ('sgf-recfno_history.csv', 'SGF-RecFNO'),
    ]:
        with open(os.path.join(BENCHMARK_DIR, fname)) as f:
            rows = list(csv.DictReader(f))
        val_loss = min(float(r['val_loss']) for r in rows)
        ep_loss = next(int(r['epoch']) for r in rows if float(r['val_loss']) == val_loss)
        if 'val_field' in rows[0]:
            val_field = min(float(r['val_field']) for r in rows)
            ep_field = next(int(r['epoch']) for r in rows if float(r['val_field']) == val_field)
            print(f'{name}: best val_field={val_field:.6f} @ ep{ep_field}, best val_total={val_loss:.6f} @ ep{ep_loss}')
        else:
            print(f'{name}: best val_loss={val_loss:.6f} @ ep{ep_loss}')


if __name__ == '__main__':
    main()
