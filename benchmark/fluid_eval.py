# -*- coding: utf-8 -*-
"""Shared fluid benchmark evaluation and prediction helpers."""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmark.fluid_ckpt import find_fluid_ckpt
from benchmark.fluid_config import FLUID_MODELS, TASKS
from benchmark.metrics import compute_field_metrics
from data.fluid_dataset import (
    CylinderDataset,
    CylinderInterpolDataset,
    DarcyDataset,
    DarcyInterpolDataset,
)
from fluid2D.run_fluid_benchmark import _build_model, _forward

# Default test-case indices per task (global dataset index).
FLUID_CASE_SAMPLES = {
    'cylinder': [4300, 4500, 4800],
    'darcy': [5100, 5123, 5800],
}

FLUID_OUT_DIR = os.path.join('logs', 'fluid_benchmark')


def _dataset_cls(task: str, model: str, interpol: bool = False):
    if task == 'cylinder':
        return CylinderInterpolDataset if interpol or model == 'PINO' else CylinderDataset
    return DarcyInterpolDataset if interpol or model == 'PINO' else DarcyDataset


@torch.no_grad()
def load_fluid_model(task: str, model: str):
    task_cfg = TASKS[task]
    ckpt, exp = find_fluid_ckpt(task, model)
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint for {model} on {task} ({exp})')
    net = _build_model(task_cfg, model).cuda()
    state = torch.load(ckpt, map_location='cuda', weights_only=False)['state_dict']
    state = {k: v for k, v in state.items() if not k.endswith('_metadata')}
    net.load_state_dict(state, strict=False)
    net.eval()
    return net, ckpt, task_cfg


def load_ground_truth(task: str, sample_idx: int) -> np.ndarray:
    task_cfg = TASKS[task]
    ds_cls = CylinderDataset if task == 'cylinder' else DarcyDataset
    _, tgt = ds_cls([sample_idx], task_cfg['mean'], task_cfg['std'])[0]
    return tgt[0].numpy()


@torch.no_grad()
def predict_fluid_sample(task: str, model: str, sample_idx: int, net=None):
    if net is None:
        net, _, task_cfg = load_fluid_model(task, model)
    else:
        task_cfg = TASKS[task]
    ds_cls = _dataset_cls(task, model)
    inp, tgt = ds_cls([sample_idx], task_cfg['mean'], task_cfg['std'])[0]
    pred = _forward(net, model, inp.unsqueeze(0).cuda())
    field = pred[0, 0].cpu().numpy()
    truth = tgt[0].numpy()
    return field, truth


@torch.no_grad()
def evaluate_fluid_models(task: str, models: list[str] | None = None) -> list[dict]:
    from fluid2D.run_fluid_benchmark import evaluate_task
    return evaluate_task(TASKS[task], models or FLUID_MODELS)


@torch.no_grad()
def collect_per_sample_mae(task: str, model: str, max_samples: int = 0) -> np.ndarray:
    task_cfg = TASKS[task]
    net, _, _ = load_fluid_model(task, model)
    test_i = task_cfg['test_index']
    if max_samples:
        test_i = test_i[:max_samples]
    ds_cls = _dataset_cls(task, model)
    ds = ds_cls(test_i, task_cfg['mean'], task_cfg['std'])
    loader = DataLoader(ds, batch_size=16, num_workers=4)
    std = task_cfg['field_std']
    maes = []
    for inputs, targets in loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        pred = _forward(net, model, inputs)
        for b in range(pred.size(0)):
            m = compute_field_metrics(pred[b:b + 1], targets[b:b + 1], std=std)
            maes.append(m['mae_k'])
    del net
    torch.cuda.empty_cache()
    return np.asarray(maes, dtype=np.float64)


def load_comparison_rows(json_path: str | None = None) -> list[dict]:
    path = json_path or os.path.join(FLUID_OUT_DIR, 'comparison_results.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)
