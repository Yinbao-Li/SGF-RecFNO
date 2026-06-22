# -*- coding: utf-8 -*-
"""Load and evaluate SGF-RecFNO ablation checkpoints."""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmark.config import FIELD_STD, SENSOR_NUM, FC_SIZE, OUT_SIZE, MODES, WIDTH, TEST_INDEX
from benchmark.metrics import compute_field_metrics
from benchmark.registry import inference_device
from data.dataset import HeatDataset
from model.sgf_recfno import SGFRecFNO
from utils.ablation_config import (
    LOSS_ABLATIONS,
    MAIN_SGF_EXP,
    QUANTILE_K_VALUES,
    loss_ablation_exp_name,
    quantile_ablation_exp_name,
    quantiles_for_k,
)

STD = FIELD_STD
CKPT_ROOTS = (
    os.path.join('logs', 'ckpt'),
    os.path.join('..', 'checkpoints'),
)


def _find_ablation_ckpt(exp_name):
    paths = []
    for root in CKPT_ROOTS:
        paths.extend(glob.glob(os.path.join(root, exp_name, 'best_epoch_*.pth')))
    if not paths:
        return None
    def _loss(p):
        try:
            return float(p.split('_loss_')[-1].replace('.pth', ''))
        except ValueError:
            return float('inf')
    return min(paths, key=_loss)


def build_sgf_for_quantiles(quantiles, device):
    model = SGFRecFNO(
        SENSOR_NUM, FC_SIZE, OUT_SIZE,
        modes1=MODES, modes2=MODES, width=WIDTH,
        num_sdf=len(quantiles),
        quantiles=tuple(quantiles),
    )
    return model.to(device).eval()


@torch.no_grad()
def evaluate_ablation_ckpt(ckpt, quantiles, max_samples=0, device=None):
    if device is None:
        device = inference_device()
    model = build_sgf_for_quantiles(quantiles, device)
    state = torch.load(ckpt, map_location=device, weights_only=False)['state_dict']
    state = {k: v for k, v in state.items() if not k.endswith('_metadata')}
    model.load_state_dict(state, strict=False)
    model.eval()

    loader = DataLoader(HeatDataset(TEST_INDEX), batch_size=8, num_workers=4)
    totals = {'mae_k': 0.0, 'mse': 0.0, 'max_ae_k': 0.0, 'n': 0}

    for inputs, targets in loader:
        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(np.asarray(inputs)).float()
        inputs, targets = inputs.to(device), targets.to(device)
        pred = model(inputs)
        for b in range(pred.size(0)):
            if max_samples and totals['n'] >= max_samples:
                break
            m = compute_field_metrics(pred[b:b + 1], targets[b:b + 1], std=STD)
            totals['mae_k'] += m['mae_k']
            totals['mse'] += m['mse']
            totals['max_ae_k'] = max(totals['max_ae_k'], m['max_ae_k'])
            totals['n'] += 1
        if max_samples and totals['n'] >= max_samples:
            break

    n = totals['n']
    return {
        'mae_k': totals['mae_k'] / n,
        'rmse_k': STD * (totals['mse'] / n) ** 0.5,
        'max_ae_k': totals['max_ae_k'],
        'num_samples': n,
        'checkpoint': ckpt,
    }


def _resolve_loss_ckpt(key):
    if key == 'full':
        return _find_ablation_ckpt(MAIN_SGF_EXP)
    return _find_ablation_ckpt(loss_ablation_exp_name(key))


def _resolve_quantile_ckpt(k):
    if k == 4:
        return _find_ablation_ckpt(MAIN_SGF_EXP)
    return _find_ablation_ckpt(quantile_ablation_exp_name(k))


def collect_loss_ablation_results(max_samples=0):
    rows = []
    for key, cfg in LOSS_ABLATIONS.items():
        exp = MAIN_SGF_EXP if key == 'full' else loss_ablation_exp_name(key)
        ckpt = _resolve_loss_ckpt(key)
        row = {
            'study': 'loss',
            'key': key,
            'label': cfg['label'],
            'exp': exp,
            'k': 4,
            'quantiles': list(quantiles_for_k(4)),
        }
        if ckpt is None:
            row.update({'mae_k': float('nan'), 'rmse_k': float('nan'), 'max_ae_k': float('nan'),
                        'checkpoint': None, 'missing': True})
        else:
            metrics = evaluate_ablation_ckpt(ckpt, quantiles_for_k(4), max_samples=max_samples)
            row.update(metrics)
            row['missing'] = False
        rows.append(row)
    return rows


def collect_quantile_ablation_results(max_samples=0):
    rows = []
    for k in QUANTILE_K_VALUES:
        qs = quantiles_for_k(k)
        exp = MAIN_SGF_EXP if k == 4 else quantile_ablation_exp_name(k)
        ckpt = _resolve_quantile_ckpt(k)
        row = {
            'study': 'quantile',
            'key': f'k{k}',
            'label': f'K={k}',
            'exp': exp,
            'k': k,
            'quantiles': list(qs),
        }
        if ckpt is None:
            row.update({'mae_k': float('nan'), 'rmse_k': float('nan'), 'max_ae_k': float('nan'),
                        'checkpoint': None, 'missing': True})
        else:
            metrics = evaluate_ablation_ckpt(ckpt, qs, max_samples=max_samples)
            row.update(metrics)
            row['missing'] = False
        rows.append(row)
    return rows
