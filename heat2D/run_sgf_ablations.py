#!/usr/bin/env python3
"""Train SGF-RecFNO ablation studies: loss components & SDF depth K (300 epochs each)."""
import argparse
import csv
import logging
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

from data.dataset import HeatDataset
from model.sgf_recfno import SGFRecFNO
from utils.ablation_config import (
    LOSS_ABLATIONS,
    LOSS_TRAIN_VARIANTS,
    QUANTILE_K_VALUES,
    QUANTILE_TRAIN_K,
    loss_ablation_exp_name,
    quantile_ablation_exp_name,
    quantiles_for_k,
)
from utils.misc import prep_experiment, save_model
from utils.sgf_loss import sgf_recfno_loss

ABLATION_LOG_DIR = os.path.join('logs', 'ablation_studies')
EPOCHS = 300
BATCH_SIZE = 8
PLOT_FREQ = 50

CSV_HEADER = [
    'epoch', 'train_loss', 'val_loss',
    'val_field', 'val_grad', 'val_sdf', 'val_ssim', 'elapsed_s',
]


def parse_args():
    p = argparse.ArgumentParser(description='SGF-RecFNO ablation training')
    p.add_argument(
        '--study', choices=['loss', 'quantile', 'all'], default='all',
        help='which ablation group to run',
    )
    p.add_argument(
        '--variant', choices=list(LOSS_ABLATIONS.keys()), default=None,
        help='loss ablation variant (loss study only)',
    )
    p.add_argument(
        '--k', type=int, choices=QUANTILE_K_VALUES, default=None,
        help='SDF depth K (quantile study only)',
    )
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--gpu-id', type=int, default=0)
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_csv_row(csv_path, row, header=None):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header or list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def build_sgf(args):
    return SGFRecFNO(
        sensor_num=args.sensor_num,
        fc_size=(args.fc_h, args.fc_w),
        out_size=(args.out_h, args.out_w),
        modes1=args.modes1,
        modes2=args.modes2,
        width=args.width,
        num_sdf=len(args.quantiles),
        quantiles=args.quantiles,
        sdf_scale=args.sdf_scale,
        refine_modes1=args.refine_modes1,
        refine_modes2=args.refine_modes2,
        refine_width=args.refine_width,
    ).cuda()


def compute_loss(net, inputs, outputs, args):
    aux = net(inputs, return_aux=True)
    loss, parts, _ = sgf_recfno_loss(
        aux['field'],
        outputs,
        aux['sdf_self'],
        quantiles=args.quantiles,
        field_loss=args.field_loss,
        lambda_field=args.lambda_field,
        lambda_grad=args.lambda_grad,
        lambda_sdf=args.lambda_sdf,
        lambda_ssim=args.lambda_ssim,
        sdf_scale=args.sdf_scale,
    )
    return loss, parts, aux


def make_base_args(exp_name, quantiles, lambdas):
    class Args:
        pass

    args = Args()
    args.exp = exp_name
    args.ckpt = os.path.join('logs', 'ckpt')
    args.tb_path = os.path.join('logs', 'tb')
    args.gpu_id = 0
    args.lr = 0.001
    args.epochs = EPOCHS
    args.batch_size = BATCH_SIZE
    args.plot_freq = PLOT_FREQ
    args.val_interval = 1
    args.sensor_num = 25
    args.fc_h = 12
    args.fc_w = 12
    args.out_h = 200
    args.out_w = 200
    args.modes1 = 50
    args.modes2 = 50
    args.width = 32
    args.train_end = 4000
    args.val_end = 5000
    args.quantiles = list(quantiles)
    args.field_loss = 'l1'
    args.lambda_field = lambdas['lambda_field']
    args.lambda_grad = lambdas['lambda_grad']
    args.lambda_sdf = lambdas['lambda_sdf']
    args.lambda_ssim = lambdas['lambda_ssim']
    args.sdf_scale = 5.0
    args.refine_modes1 = 24
    args.refine_modes2 = 24
    args.refine_width = 32
    return args


def train_one(scheme_label, args, csv_name):
    print(f'\n{"=" * 60}\n[{scheme_label}] {args.epochs} epoch training\n{"=" * 60}', flush=True)
    ensure_dir(ABLATION_LOG_DIR)
    csv_path = os.path.join(ABLATION_LOG_DIR, csv_name)

    torch.cuda.set_device(args.gpu_id)
    tb_writer = prep_experiment(args)
    args.fig_path = args.exp_path + '/figure'
    ensure_dir(args.fig_path)
    args.best_record = {'epoch': -1, 'loss': 1e10}

    net = build_sgf(args)
    train_loader = DataLoader(
        HeatDataset(index=list(range(args.train_end))),
        batch_size=args.batch_size, num_workers=4, shuffle=True,
    )
    val_loader = DataLoader(
        HeatDataset(index=list(range(args.train_end, args.val_end))),
        batch_size=args.batch_size, num_workers=4,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        train_loss, train_num = 0.0, 0
        train_parts = {'field': 0.0, 'grad': 0.0, 'sdf': 0.0, 'ssim': 0.0}

        for inputs, outputs in train_loader:
            inputs, outputs = inputs.cuda(), outputs.cuda()
            loss, parts, _ = compute_loss(net, inputs, outputs, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += parts['total'].item() * inputs.size(0)
            for key in train_parts:
                train_parts[key] += parts[key].item() * inputs.size(0)
            train_num += inputs.size(0)

        train_loss /= train_num
        for key in train_parts:
            train_parts[key] /= train_num
        scheduler.step()

        net.eval()
        val_loss, val_num = 0.0, 0
        val_parts = {'field': 0.0, 'grad': 0.0, 'sdf': 0.0, 'ssim': 0.0}
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs, outputs = inputs.cuda(), outputs.cuda()
                _, parts, _ = compute_loss(net, inputs, outputs, args)
                val_loss += parts['total'].item() * inputs.size(0)
                for key in val_parts:
                    val_parts[key] += parts[key].item() * inputs.size(0)
                val_num += inputs.size(0)
        val_loss /= val_num
        for key in val_parts:
            val_parts[key] /= val_num
            tb_writer.add_scalar(f'val/{key}', val_parts[key], epoch)
        tb_writer.add_scalar('val/total', val_loss, epoch)

        elapsed = time.time() - t0
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_field': val_parts['field'],
            'val_grad': val_parts['grad'],
            'val_sdf': val_parts['sdf'],
            'val_ssim': val_parts['ssim'],
            'elapsed_s': round(elapsed, 1),
        }
        write_csv_row(csv_path, row, CSV_HEADER)
        msg = (
            f'[{scheme_label}] Epoch {epoch:03d}/{args.epochs - 1} | '
            f'train={train_loss:.6f} | val={val_loss:.6f} | {elapsed / 60:.1f}min'
        )
        print(msg, flush=True)
        logging.info(msg)

        if val_loss < args.best_record['loss']:
            save_model(args, epoch, val_loss, net)
        net.train()

    tb_writer.close()
    print(f'[{scheme_label}] done → {csv_path}', flush=True)
    return csv_path


def run_loss_ablations(variant=None, epochs=EPOCHS, gpu_id=0):
    if variant:
        if variant == 'full':
            print('Skip: full loss uses existing benchmark_sgf-recfno_300 checkpoint.', flush=True)
            return []
        variants = [variant]
    else:
        variants = list(LOSS_TRAIN_VARIANTS)
        print('Note: full loss baseline = benchmark_sgf-recfno_300 (not retrained).', flush=True)
    paths = []
    for key in variants:
        cfg = LOSS_ABLATIONS[key]
        exp = loss_ablation_exp_name(key)
        args = make_base_args(exp, quantiles_for_k(4), cfg)
        args.epochs = epochs
        args.gpu_id = gpu_id
        paths.append(train_one(f'Loss ablation: {cfg["label"]}', args, f'loss_{key}_history.csv'))
    return paths


def run_quantile_ablations(k=None, epochs=EPOCHS, gpu_id=0):
    if k is not None:
        if k == 4:
            print('Skip: K=4 uses existing benchmark_sgf-recfno_300 checkpoint.', flush=True)
            return []
        ks = [k]
    else:
        ks = list(QUANTILE_TRAIN_K)
        print('Note: K=4 baseline = benchmark_sgf-recfno_300 (not retrained).', flush=True)
    paths = []
    for ki in ks:
        qs = quantiles_for_k(ki)
        exp = quantile_ablation_exp_name(ki)
        args = make_base_args(exp, qs, LOSS_ABLATIONS['full'])
        args.epochs = epochs
        args.gpu_id = gpu_id
        paths.append(train_one(f'Quantile K={ki}', args, f'quantile_k{ki}_history.csv'))
    return paths


def main():
    args = parse_args()
    if args.study in ('loss', 'all'):
        run_loss_ablations(variant=args.variant, epochs=args.epochs, gpu_id=args.gpu_id)
    if args.study in ('quantile', 'all'):
        run_quantile_ablations(k=args.k, epochs=args.epochs, gpu_id=args.gpu_id)


if __name__ == '__main__':
    main()
