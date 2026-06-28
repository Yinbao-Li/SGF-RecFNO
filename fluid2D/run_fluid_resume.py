#!/usr/bin/env python3
"""Resume fluid benchmark training to 500 epochs (+ fresh train for updated models)."""
from __future__ import annotations

import argparse
import csv
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

from benchmark.fluid_config import FLUID_MODELS, TASKS
from benchmark.training_resume import align_scheduler, find_best_checkpoint, load_model_weights, parse_checkpoint_meta
from fluid2D.run_fluid_benchmark import (
    LR,
    SCHED_GAMMA,
    _build_model,
    _exp_name,
    _forward,
    _get_loaders,
    _resolve_ckpt,
    _train_loss,
)
from utils.misc import prep_experiment, save_model

BASE_EPOCHS = 300
TOTAL_EPOCHS = 500
# Task-specific geometry changed — must train from scratch.
FRESH_MODELS = frozenset({'IsoRecFNO', 'SGF-RecFNO (K=8)'})
# Resume 300→500 for models already trained with final geometry / stable setup.
RESUME_MODELS = frozenset(m for m in FLUID_MODELS if m not in FRESH_MODELS)


def parse_args():
    p = argparse.ArgumentParser(description='Fluid benchmark: resume to 500 ep (+ fresh Iso/K8)')
    p.add_argument('--task', choices=['cylinder', 'darcy', 'all'], default='all')
    p.add_argument('--models', nargs='+', default=None, help='default: all FLUID_MODELS')
    p.add_argument('--total-epochs', type=int, default=TOTAL_EPOCHS)
    p.add_argument('--base-epochs', type=int, default=BASE_EPOCHS)
    p.add_argument('--compare-only', action='store_true')
    return p.parse_args()


def _load_resume_weights(task: str, model: str, net) -> tuple[int, float]:
    ckpt, _ = _resolve_ckpt(task, model)
    if ckpt is None:
        ckpt = find_best_checkpoint(_exp_name(task, model))
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint to resume: {task}/{model}')
    epoch, loss = load_model_weights(net, ckpt)
    meta_ep, meta_loss = parse_checkpoint_meta(ckpt)
    return max(epoch, meta_ep), meta_loss


def train_one(
    task_cfg: dict,
    model: str,
    total_epochs: int,
    start_epoch: int = 0,
    resume: bool = False,
):
    geom = task_cfg['geometry']
    exp = _exp_name(task_cfg['name'], model)
    mode = 'resume' if resume else 'fresh'
    print(
        f'\n>>> {mode.upper()} {model} on {task_cfg["name"]} '
        f'(ep {start_epoch}→{total_epochs - 1}, {geom.label}) → {exp}',
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

    net = _build_model(task_cfg, model).cuda()
    if resume:
        best_ep, best_loss = _load_resume_weights(task_cfg['name'], model, net)
        args.best_record = {'epoch': best_ep, 'loss': best_loss}
        # Promoted / copied ckpts may not match exp_path filename — avoid save_model assert.
        snap = os.path.join(
            args.ckpt, exp,
            f'best_epoch_{best_ep}_loss_{best_loss:.8f}.pth',
        )
        if not os.path.exists(snap):
            args.best_record = {'epoch': -1, 'loss': 1e10}
    else:
        args.best_record = {'epoch': -1, 'loss': 1e10}

    train_loader, val_loader = _get_loaders(task_cfg, model)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHED_GAMMA)
    if resume and start_epoch > 0:
        align_scheduler(optimizer, scheduler, start_epoch)

    log_dir = task_cfg['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    slug = model.lower().replace('-', '').replace(' ', '')
    csv_path = os.path.join(log_dir, f'{slug}_history_500.csv')
    header = ['epoch', 'train_loss', 'val_loss', 'elapsed_s']

    t0 = time.time()
    for epoch in range(start_epoch, total_epochs):
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
        train_loss /= max(n, 1)
        scheduler.step()

        net.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                pred = _forward(net, model, inputs)
                val_loss += F.l1_loss(pred, targets).item() * inputs.size(0)
                vn += inputs.size(0)
        val_loss /= max(vn, 1)

        elapsed = time.time() - t0
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'elapsed_s': round(elapsed, 1),
        }
        new_file = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            if new_file:
                w.writeheader()
            w.writerow(row)
        print(
            f'[{task_cfg["name"]}/{model}] ep {epoch:03d} train={train_loss:.6f} val={val_loss:.6f}',
            flush=True,
        )
        if val_loss < args.best_record['loss']:
            save_model(args, epoch, val_loss, net)
        net.train()

    tb_writer.close()
    del net
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    models = args.models or list(FLUID_MODELS)
    tasks = list(TASKS.values()) if args.task == 'all' else [TASKS[args.task]]

    if not args.compare_only:
        for task_cfg in tasks:
            for model in models:
                if model in FRESH_MODELS:
                    train_one(task_cfg, model, args.total_epochs, start_epoch=0, resume=False)
                elif model in RESUME_MODELS:
                    train_one(
                        task_cfg, model, args.total_epochs,
                        start_epoch=args.base_epochs, resume=True,
                    )
                else:
                    print(f'SKIP unknown model {model}', flush=True)

    from fluid2D.run_fluid_benchmark import evaluate_task
    for task_cfg in tasks:
        print(f'\n=== Evaluate {task_cfg["name"]} (500 ep) ===', flush=True)
        evaluate_task(task_cfg, models)


if __name__ == '__main__':
    main()
