#!/usr/bin/env python3
"""Train & evaluate RecFNO-family models on cylinder / Darcy fluid tasks."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

from utils.bootstrap import ensure_repo_context

ensure_repo_context(heat2d_cwd=True)

from benchmark.fluid_ckpt import find_fluid_ckpt, fluid_exp_name
from benchmark.fluid_adapters import GINOFluidRecon, GeoFNOFluidRecon, PINOFluidRecon
from benchmark.fluid_config import FLUID_MODELS, TASKS
from benchmark.metrics import compute_field_metrics
from data.fluid_dataset import (
    CYLINDER_SENSOR_4,
    CylinderDataset,
    CylinderInterpolDataset,
    DARCY_SENSOR_POS,
    DarcyDataset,
    DarcyInterpolDataset,
)
from model.fno import FNORecon
from model.iso_recfno import IsoRecFNO
from model.sgf_recfno import SGFRecFNO
from utils.ablation_config import quantiles_for_k
from utils.sgf_loss import iso_recfno_geometry_loss, sgf_recfno_loss

import re
import glob

LR = 0.001
SCHED_GAMMA = 0.98
_CKPT_RE = re.compile(r'best_epoch_(\d+)_loss_')
_SGF_NAMES = frozenset({'SGF-RecFNO', 'SGF-RecFNO (K=8)'})


def parse_args():
    p = argparse.ArgumentParser(description='Fluid benchmark (cylinder / darcy)')
    p.add_argument('--task', choices=['cylinder', 'darcy', 'all'], default='all')
    p.add_argument('--models', nargs='+', default=FLUID_MODELS)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--compare-only', action='store_true', help='skip training, evaluate checkpoints')
    p.add_argument('--skip-existing', action='store_true', help='skip models that already have a best checkpoint')
    return p.parse_args()


def _exp_name(task: str, model: str) -> str:
    return fluid_exp_name(task, model)


def _resolve_ckpt(task: str, model: str):
    return find_fluid_ckpt(task, model)


def _ckpt_epoch(ckpt_path: str) -> int:
    m = _CKPT_RE.search(os.path.basename(ckpt_path))
    return int(m.group(1)) if m else -1


def _training_complete(ckpt_path: str, exp: str, epochs: int) -> bool:
    ep = _ckpt_epoch(ckpt_path)
    if ep < 10:
        return False
    if ep >= epochs - 10:
        return True
    for log_path in glob.glob(os.path.join('logs', 'ckpt', exp, 'log_*.log')):
        with open(log_path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            tail = f.read().decode('utf-8', errors='ignore')
        if f'ep {epochs - 1:03d} ' in tail or f'ep {epochs - 1} ' in tail:
            return True
    return False


def _get_loaders(task_cfg: dict, model: str):
    mean, std = task_cfg['mean'], task_cfg['std']
    bs = task_cfg['batch_size']
    train_i, val_i = task_cfg['train_index'], task_cfg['val_index']
    if task_cfg['name'] == 'cylinder':
        if model == 'PINO':
            train_ds = CylinderInterpolDataset(train_i, mean, std)
            val_ds = CylinderInterpolDataset(val_i, mean, std)
        else:
            train_ds = CylinderDataset(train_i, mean, std)
            val_ds = CylinderDataset(val_i, mean, std)
    else:
        if model == 'PINO':
            train_ds = DarcyInterpolDataset(train_i, mean, std)
            val_ds = DarcyInterpolDataset(val_i, mean, std)
        else:
            train_ds = DarcyDataset(train_i, mean, std)
            val_ds = DarcyDataset(val_i, mean, std)
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val_ds, batch_size=bs, num_workers=4, pin_memory=True),
    )


def _build_model(task_cfg: dict, model: str):
    s = task_cfg['sensor_num']
    fc = task_cfg['fc_size']
    out = task_cfg['out_size']
    m = task_cfg['modes']
    w = task_cfg['width']
    geom = task_cfg['geometry']
    if model == 'RecFNO':
        return FNORecon(s, fc, out, modes1=m, modes2=m, width=w)
    if model == 'IsoRecFNO':
        q = quantiles_for_k(4)
        return IsoRecFNO(
            s, fc, out, modes1=m, modes2=m, width=w,
            num_iso_levels=geom.num_sdf_channels(4), quantiles=q,
        )
    if model == 'SGF-RecFNO':
        q = quantiles_for_k(4)
        return SGFRecFNO(
            s, fc, out, modes1=m, modes2=m, width=w,
            num_sdf=geom.num_sdf_channels(4), quantiles=q, geometry=geom,
        )
    if model == 'SGF-RecFNO (K=8)':
        q = quantiles_for_k(8)
        return SGFRecFNO(
            s, fc, out, modes1=m, modes2=m, width=w,
            num_sdf=geom.num_sdf_channels(8), quantiles=q, geometry=geom,
        )
    if model == 'PINO':
        return PINOFluidRecon()
    if model == 'Geo-FNO':
        pos = CYLINDER_SENSOR_4 if task_cfg['name'] == 'cylinder' else DARCY_SENSOR_POS
        return GeoFNOFluidRecon(pos, out, modes=min(m, 12))
    if model == 'GINO':
        pos = CYLINDER_SENSOR_4 if task_cfg['name'] == 'cylinder' else DARCY_SENSOR_POS
        return GINOFluidRecon(pos, out)
    raise KeyError(model)


def _forward(model, name: str, inputs):
    if name in _SGF_NAMES or name == 'IsoRecFNO':
        return model(inputs, return_aux=True)['field']
    return model(inputs)


def _train_loss(model, name: str, inputs, targets, task_cfg: dict):
    if name in _SGF_NAMES:
        aux = model(inputs, return_aux=True)
        k = 8 if name == 'SGF-RecFNO (K=8)' else 4
        geom = task_cfg['geometry']
        loss, _, _ = sgf_recfno_loss(
            aux['field'], targets, aux['sdf_self'],
            quantiles=quantiles_for_k(k),
            geometry=geom,
            sdf_scale=geom.sdf_scale,
            k=k,
            lambda_grad=geom.lambda_grad,
        )
        return loss
    if name == 'IsoRecFNO':
        aux = model(inputs, return_aux=True)
        geom = task_cfg['geometry']
        loss, _ = iso_recfno_geometry_loss(
            aux['field'], targets, aux['sdf_pred'],
            quantiles=quantiles_for_k(4),
            geometry=geom,
            sdf_scale=geom.sdf_scale,
            k=4,
            lambda_grad=geom.lambda_grad,
        )
        return loss
    pred = _forward(model, name, inputs)
    return F.l1_loss(pred, targets)


def train_one(task_cfg: dict, model: str, epochs: int):
    geom = task_cfg['geometry']
    exp = _exp_name(task_cfg['name'], model)
    print(
        f'\n>>> Train {model} on {task_cfg["name"]} ({epochs} ep, {geom.label}) → {exp}',
        flush=True,
    )

    class Args:
        pass

    args = Args()
    args.exp = exp
    args.ckpt = os.path.join('logs', 'ckpt')
    args.tb_path = os.path.join('logs', 'tb')
    args.gpu_id = 0
    args.lr = LR
    args.batch_size = task_cfg['batch_size']
    args.val_interval = 1

    torch.cuda.set_device(args.gpu_id)
    tb_writer = prep_experiment(args)
    args.best_record = {'epoch': -1, 'loss': 1e10}

    net = _build_model(task_cfg, model).cuda()
    train_loader, val_loader = _get_loaders(task_cfg, model)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHED_GAMMA)

    log_dir = task_cfg['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f'{model.lower().replace("-", "")}_history.csv')
    header = ['epoch', 'train_loss', 'val_loss', 'elapsed_s']

    t0 = time.time()
    for epoch in range(epochs):
        net.train()
        train_loss, n = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            loss = _train_loss(net, model, inputs, targets, task_cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            n += inputs.size(0)
        train_loss /= n
        scheduler.step()

        net.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                pred = _forward(net, model, inputs)
                val_loss += F.l1_loss(pred, targets).item() * inputs.size(0)
                vn += inputs.size(0)
        val_loss /= vn

        elapsed = time.time() - t0
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'elapsed_s': round(elapsed, 1)}
        new_file = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            if new_file:
                w.writeheader()
            w.writerow(row)
        print(f'[{task_cfg["name"]}/{model}] ep {epoch:03d} train={train_loss:.6f} val={val_loss:.6f}', flush=True)
        if val_loss < args.best_record['loss']:
            save_model(args, epoch, val_loss, net)
        net.train()

    tb_writer.close()
    del net
    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_task(task_cfg: dict, models: list[str]):
    test_i = task_cfg['test_index']
    std = task_cfg['field_std']
    rows = []
    for model in models:
        ckpt, exp = _resolve_ckpt(task_cfg['name'], model)
        if ckpt is None:
            print(f'SKIP {model}: no checkpoint ({exp})', flush=True)
            continue
        net = _build_model(task_cfg, model).cuda()
        state = torch.load(ckpt, map_location='cuda', weights_only=False)['state_dict']
        state = {k: v for k, v in state.items() if not k.endswith('_metadata')}
        net.load_state_dict(state, strict=False)
        net.eval()

        if model == 'PINO':
            ds = (CylinderInterpolDataset if task_cfg['name'] == 'cylinder' else DarcyInterpolDataset)(
                test_i, task_cfg['mean'], task_cfg['std']
            )
        else:
            ds = (CylinderDataset if task_cfg['name'] == 'cylinder' else DarcyDataset)(
                test_i, task_cfg['mean'], task_cfg['std']
            )
        loader = DataLoader(ds, batch_size=16, num_workers=4)

        totals = {k: 0.0 for k in ('relative_l2', 'mse', 'mae_k', 'psnr', 'ssim')}
        n = 0
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            pred = _forward(net, model, inputs)
            for b in range(pred.size(0)):
                m = compute_field_metrics(pred[b:b + 1], targets[b:b + 1], std=std)
                for k in totals:
                    totals[k] += m[k]
                n += 1

        row = {k: totals[k] / n for k in totals}
        row['model'] = model
        row['task'] = task_cfg['name']
        row['checkpoint'] = ckpt
        rows.append(row)
        print(f"  {model}: MAE={row['mae_k']:.6f} RelL2={row['relative_l2']:.4e} PSNR={row['psnr']:.2f}", flush=True)
        del net
        torch.cuda.empty_cache()

    out_dir = task_cfg['log_dir']
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'comparison_results.json')
    with open(out_json, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'Saved {out_json}', flush=True)
    return rows


def main():
    args = parse_args()
    tasks = list(TASKS.values()) if args.task == 'all' else [TASKS[args.task]]

    if not args.compare_only:
        for task_cfg in tasks:
            epochs = args.epochs or task_cfg['epochs']
            for model in args.models:
                exp = _exp_name(task_cfg['name'], model)
                ckpt, _ = _resolve_ckpt(task_cfg['name'], model)
                if args.skip_existing and ckpt is not None and _training_complete(ckpt, exp, epochs):
                    print(f'SKIP train {model} on {task_cfg["name"]} ({ckpt})', flush=True)
                    continue
                train_one(task_cfg, model, epochs)

    for task_cfg in tasks:
        print(f'\n=== Evaluate {task_cfg["name"]} ===', flush=True)
        evaluate_task(task_cfg, args.models)


if __name__ == '__main__':
    main()
