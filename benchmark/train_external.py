#!/usr/bin/env python3
"""Train official external baselines (GINO, Geo-FNO, PINO) on heat data."""
import argparse
import csv
import gc
import glob
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.bootstrap import ensure_repo_context

ensure_repo_context(heat2d_cwd=True)

from benchmark.config import DEFAULT_BATCH, EPOCHS, EXTERNAL_MODELS
from benchmark.registry import MODEL_SPECS
from data.dataset import HeatDataset, HeatInterpolDataset
from utils.misc import prep_experiment, save_model

_DATA_CACHE = {}


def parse_args():
    p = argparse.ArgumentParser(description='Train external baselines on heat data')
    p.add_argument('--models', nargs='+', default=EXTERNAL_MODELS,
                   help='models to train (default: all external)')
    p.add_argument('--skip-existing', action='store_true',
                   help='skip models that already have a best checkpoint')
    return p.parse_args()


def _has_ckpt(exp_name):
    return bool(glob.glob(os.path.join('logs', 'ckpt', exp_name, 'best_epoch_*.pth')))


def _get_loaders(spec, batch_size):
    key = (spec['type'], batch_size)
    if key not in _DATA_CACHE:
        if spec['type'] == 'sensor':
            _DATA_CACHE[key] = (
                DataLoader(HeatDataset(list(range(4000))), batch_size=batch_size, num_workers=4, shuffle=True),
                DataLoader(HeatDataset(list(range(4000, 5000))), batch_size=batch_size, num_workers=4),
            )
        else:
            print('Building HeatInterpolDataset (one-time grid interpolation)...', flush=True)
            _DATA_CACHE[key] = (
                DataLoader(HeatInterpolDataset(list(range(4000))), batch_size=batch_size, num_workers=0, shuffle=True),
                DataLoader(HeatInterpolDataset(list(range(4000, 5000))), batch_size=batch_size, num_workers=0),
            )
    return _DATA_CACHE[key]


def _release_gpu(net=None):
    if net is not None:
        del net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def train_one(name):
    spec = MODEL_SPECS[name]
    if name not in EXTERNAL_MODELS:
        return

    bs = DEFAULT_BATCH

    class Args:
        exp = spec['exp']
        ckpt = 'logs/ckpt'
        tb_path = 'logs/tb'
        gpu_id = 0
        lr = 0.001
        epochs = EPOCHS
        val_interval = 1

    Args.batch_size = bs

    print(f'\n=== Training {name} ({spec["source"]}) batch={bs} ===', flush=True)
    torch.cuda.set_device(Args.gpu_id)
    args = Args()
    tb_writer = prep_experiment(args)
    args.best_record = {'epoch': -1, 'loss': 1e10}

    try:
        net = spec['build']().cuda()
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        print(f'SKIP {name}: {exc}', flush=True)
        return

    train_loader, val_loader = _get_loaders(spec, bs)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    fwd = spec['forward']

    csv_path = os.path.join('logs', 'benchmark_300epoch', f'{name.lower().replace("-", "")}_history.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = ['epoch', 'train_loss', 'val_loss', 'elapsed_s']
    t0 = time.time()

    for epoch in range(EPOCHS):
        net.train()
        train_loss, n = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            pred = fwd(net, inputs)
            loss = F.l1_loss(pred, targets)
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
                pred = fwd(net, inputs)
                val_loss += F.l1_loss(pred, targets).item() * inputs.size(0)
                vn += inputs.size(0)
        val_loss /= vn

        elapsed = time.time() - t0
        print(f'[{name}] Epoch {epoch} | train={train_loss:.6f} | val={val_loss:.6f}', flush=True)
        row = [epoch, train_loss, val_loss, elapsed]
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)

        if val_loss < args.best_record['loss']:
            save_model(args, epoch, val_loss, net)

    print(f'[{name}] done. best epoch={args.best_record["epoch"]} val={args.best_record["loss"]:.6f}', flush=True)
    _release_gpu(net)


def main():
    args = parse_args()
    os.chdir(os.path.join(filename, 'heat2D'))
    _release_gpu(None)

    for name in args.models:
        if name not in EXTERNAL_MODELS:
            print(f'SKIP unknown model: {name}', flush=True)
            continue
        if args.skip_existing and _has_ckpt(MODEL_SPECS[name]['exp']):
            print(f'SKIP {name}: checkpoint already exists', flush=True)
            continue
        train_one(name)
        _release_gpu(None)


if __name__ == '__main__':
    main()
