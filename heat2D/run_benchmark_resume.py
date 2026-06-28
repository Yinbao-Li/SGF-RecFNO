#!/usr/bin/env python3
"""Resume benchmark models for extra epochs (default: +200 on top of 300)."""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, filename)

from benchmark.config import ALL_MODELS, DEFAULT_BATCH, EXTERNAL_MODELS, TRAIN_INDEX, VAL_INDEX
from benchmark.registry import MODEL_SPECS
from benchmark.training_resume import align_scheduler, find_best_checkpoint, load_model_weights
from data.dataset import HeatDataset, HeatInterpolDataset
from model.fno import FNORecon
from model.iso_recfno import IsoRecFNO
from model.sgf_recfno import SGFRecFNO
from utils.ablation_config import quantile_ablation_exp_name, quantiles_for_k
from utils.iso_geometry import iso_recfno_loss
from utils.misc import prep_experiment, save_model
from utils.sgf_loss import sgf_recfno_loss

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'benchmark_300epoch')
BASE_EPOCHS = 300
EXTRA_EPOCHS = 200
BATCH_SIZE = DEFAULT_BATCH
LR = 0.001
SCHED_GAMMA = 0.98

ISO_HEADER = ['epoch', 'train_loss', 'val_loss', 'val_field', 'val_grad', 'val_sdf', 'val_ssim', 'elapsed_s']
SIMPLE_HEADER = ['epoch', 'train_loss', 'val_loss', 'elapsed_s']


def parse_args():
    p = argparse.ArgumentParser(description='Resume benchmark training (+200 epochs)')
    p.add_argument('--base-epochs', type=int, default=BASE_EPOCHS)
    p.add_argument('--extra-epochs', type=int, default=EXTRA_EPOCHS)
    p.add_argument('--models', nargs='+', default=None,
                   help='subset of benchmark models (default: all + SGF K=8)')
    p.add_argument('--include-sgf-k8', action='store_true', default=True,
                   help='also resume ablation_k8 for SGF-RecFNO (K=8)')
    p.add_argument('--no-sgf-k8', action='store_true', help='skip SGF K=8 resume')
    p.add_argument('--start-epoch', type=int, default=None,
                   help='first epoch to run (default: --base-epochs)')
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


def _make_args(exp: str):
    class Args:
        pass

    args = Args()
    args.exp = exp
    args.ckpt = os.path.join('logs', 'ckpt')
    args.tb_path = os.path.join('logs', 'tb')
    args.gpu_id = 0
    args.lr = LR
    args.batch_size = BATCH_SIZE
    args.val_interval = 1
    args.sensor_num = 25
    args.fc_h = args.fc_w = 12
    args.out_h = args.out_w = 200
    args.modes1 = args.modes2 = 50
    args.width = 32
    args.train_end = len(TRAIN_INDEX)
    args.val_end = len(TRAIN_INDEX) + len(VAL_INDEX)
    args.quantiles = [0.2, 0.4, 0.6, 0.8]
    args.field_loss = 'l1'
    args.lambda_field = 1.0
    args.lambda_grad = 0.1
    args.lambda_sdf = 0.5
    args.lambda_ssim = 0.1
    args.sdf_scale = 5.0
    args.refine_modes1 = args.refine_modes2 = 24
    args.refine_width = 32
    return args


def _init_resume(exp: str, net, best_epoch: int, best_loss: float):
    args = _make_args(exp)
    torch.cuda.set_device(args.gpu_id)
    tb_writer = prep_experiment(args)
    args.best_record = {'epoch': best_epoch, 'loss': best_loss}
    ckpt = find_best_checkpoint(exp)
    load_model_weights(net, ckpt)
    return args, tb_writer


def resume_recfno(start_epoch: int, total_epochs: int):
    scheme = 'RecFNO'
    exp = 'benchmark_recfno_300'
    ckpt = find_best_checkpoint(exp)
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint for {exp}')
    from benchmark.training_resume import parse_checkpoint_meta
    best_epoch, best_loss = parse_checkpoint_meta(ckpt)

    print(f'\n{"=" * 60}\n[{scheme}] resume {start_epoch}–{total_epochs - 1} from {ckpt}\n{"=" * 60}', flush=True)
    csv_path = os.path.join(BENCHMARK_DIR, f'{scheme.lower()}_history.csv')

    net = FNORecon(25, (12, 12), (200, 200), modes1=50, modes2=50, width=32).cuda()
    args, tb_writer = _init_resume(exp, net, best_epoch, best_loss)
    train_loader = DataLoader(HeatDataset(index=TRAIN_INDEX), batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    val_loader = DataLoader(HeatDataset(index=VAL_INDEX), batch_size=BATCH_SIZE, num_workers=4)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHED_GAMMA)
    align_scheduler(optimizer, scheduler, start_epoch)

    t0 = time.time()
    for epoch in range(start_epoch, total_epochs):
        net.train()
        train_loss, train_num = 0.0, 0
        for inputs, outputs in train_loader:
            inputs, outputs = inputs.cuda(), outputs.cuda()
            pre = net(inputs)
            loss = F.l1_loss(outputs, pre)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_num += inputs.size(0)
        train_loss /= train_num
        scheduler.step()

        net.eval()
        val_loss, val_num = 0.0, 0
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs, outputs = inputs.cuda(), outputs.cuda()
                val_loss += F.l1_loss(outputs, net(inputs)).item() * inputs.size(0)
                val_num += inputs.size(0)
        val_loss /= val_num

        elapsed = time.time() - t0
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'elapsed_s': round(elapsed, 1)}
        write_csv_row(csv_path, row, SIMPLE_HEADER)
        msg = f'[{scheme}] Epoch {epoch:03d}/{total_epochs - 1} | train={train_loss:.6f} | val={val_loss:.6f} | {elapsed / 60:.1f}min'
        print(msg, flush=True)
        logging.info(msg)
        if val_loss < args.best_record['loss']:
            save_model(args, epoch, val_loss, net)
        net.train()

    tb_writer.close()
    print(f'[{scheme}] done → {csv_path}', flush=True)


def _resume_composite(scheme, exp, build_fn, loss_fn, csv_header, start_epoch, total_epochs, quantiles=None):
    ckpt = find_best_checkpoint(exp)
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint for {exp}')
    from benchmark.training_resume import parse_checkpoint_meta
    best_epoch, best_loss = parse_checkpoint_meta(ckpt)

    print(f'\n{"=" * 60}\n[{scheme}] resume {start_epoch}–{total_epochs - 1} from {ckpt}\n{"=" * 60}', flush=True)
    csv_key = scheme.lower().replace('-', '').replace(' ', '')
    if scheme == 'SGF-RecFNO (K=8)':
        csv_key = 'sgfrecfnok8'
    csv_path = os.path.join(BENCHMARK_DIR, f'{csv_key}_history.csv')

    args = _make_args(exp)
    if quantiles is not None:
        args.quantiles = list(quantiles)
    torch.cuda.set_device(args.gpu_id)
    tb_writer = prep_experiment(args)
    args.best_record = {'epoch': best_epoch, 'loss': best_loss}

    net = build_fn(args)
    load_model_weights(net, ckpt)
    train_loader = DataLoader(HeatDataset(index=TRAIN_INDEX), batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    val_loader = DataLoader(HeatDataset(index=VAL_INDEX), batch_size=BATCH_SIZE, num_workers=4)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHED_GAMMA)
    align_scheduler(optimizer, scheduler, start_epoch)

    t0 = time.time()
    for epoch in range(start_epoch, total_epochs):
        net.train()
        train_loss, train_num = 0.0, 0
        for inputs, outputs in train_loader:
            inputs, outputs = inputs.cuda(), outputs.cuda()
            loss, parts, _ = loss_fn(net, inputs, outputs, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += parts['total'].item() * inputs.size(0)
            train_num += inputs.size(0)
        train_loss /= train_num
        scheduler.step()

        net.eval()
        val_loss, val_num = 0.0, 0
        val_parts = {'field': 0.0, 'grad': 0.0, 'sdf': 0.0, 'ssim': 0.0}
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs, outputs = inputs.cuda(), outputs.cuda()
                _, parts, _ = loss_fn(net, inputs, outputs, args)
                val_loss += parts['total'].item() * inputs.size(0)
                for key in val_parts:
                    val_parts[key] += parts[key].item() * inputs.size(0)
                val_num += inputs.size(0)
        val_loss /= val_num
        for key in val_parts:
            val_parts[key] /= val_num

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
        write_csv_row(csv_path, row, csv_header)
        msg = (
            f'[{scheme}] Epoch {epoch:03d}/{total_epochs - 1} | train={train_loss:.6f} | val={val_loss:.6f} '
            f'| field={val_parts["field"]:.6f} sdf={val_parts["sdf"]:.6f} | {elapsed / 60:.1f}min'
        )
        print(msg, flush=True)
        logging.info(msg)
        if val_loss < args.best_record['loss']:
            save_model(args, epoch, val_loss, net)
        net.train()

    tb_writer.close()
    print(f'[{scheme}] done → {csv_path}', flush=True)


def build_iso(args):
    return IsoRecFNO(
        sensor_num=args.sensor_num, fc_size=(args.fc_h, args.fc_w),
        out_size=(args.out_h, args.out_w), modes1=args.modes1, modes2=args.modes2,
        width=args.width, num_iso_levels=len(args.quantiles), quantiles=args.quantiles,
    ).cuda()


def build_sgf(args):
    return SGFRecFNO(
        sensor_num=args.sensor_num, fc_size=(args.fc_h, args.fc_w),
        out_size=(args.out_h, args.out_w), modes1=args.modes1, modes2=args.modes2,
        width=args.width, num_sdf=len(args.quantiles), quantiles=args.quantiles,
        sdf_scale=args.sdf_scale, refine_modes1=args.refine_modes1,
        refine_modes2=args.refine_modes2, refine_width=args.refine_width,
    ).cuda()


def compute_iso_loss(net, inputs, outputs, args):
    aux = net(inputs, return_aux=True)
    loss, parts = iso_recfno_loss(
        aux['field'], outputs, aux['sdf_pred'],
        quantiles=args.quantiles, field_loss=args.field_loss,
        lambda_grad=args.lambda_grad, lambda_sdf=args.lambda_sdf,
        lambda_ssim=args.lambda_ssim, sdf_scale=args.sdf_scale,
    )
    return loss, parts, aux


def compute_sgf_loss(net, inputs, outputs, args):
    aux = net(inputs, return_aux=True)
    loss, parts, _ = sgf_recfno_loss(
        aux['field'], outputs, aux['sdf_self'],
        quantiles=args.quantiles, field_loss=args.field_loss,
        lambda_field=args.lambda_field,
        lambda_grad=args.lambda_grad, lambda_sdf=args.lambda_sdf,
        lambda_ssim=args.lambda_ssim, sdf_scale=args.sdf_scale,
    )
    return loss, parts, aux


def resume_external(name: str, start_epoch: int, total_epochs: int):
    spec = MODEL_SPECS[name]
    exp = spec['exp']
    ckpt = find_best_checkpoint(exp)
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint for {name} ({exp})')
    from benchmark.training_resume import parse_checkpoint_meta
    best_epoch, best_loss = parse_checkpoint_meta(ckpt)

    print(f'\n{"=" * 60}\n[{name}] resume {start_epoch}–{total_epochs - 1} from {ckpt}\n{"=" * 60}', flush=True)
    csv_path = os.path.join(BENCHMARK_DIR, f'{name.lower().replace("-", "")}_history.csv')

    args = _make_args(exp)
    torch.cuda.set_device(args.gpu_id)
    tb_writer = prep_experiment(args)
    args.best_record = {'epoch': best_epoch, 'loss': best_loss}

    net = spec['build']().cuda()
    load_model_weights(net, ckpt)
    if spec['type'] == 'sensor':
        train_loader = DataLoader(HeatDataset(TRAIN_INDEX), batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
        val_loader = DataLoader(HeatDataset(VAL_INDEX), batch_size=BATCH_SIZE, num_workers=4)
    else:
        train_loader = DataLoader(
            HeatInterpolDataset(TRAIN_INDEX),
            batch_size=BATCH_SIZE, num_workers=4, shuffle=True, pin_memory=True,
        )
        val_loader = DataLoader(
            HeatInterpolDataset(VAL_INDEX),
            batch_size=BATCH_SIZE, num_workers=4, pin_memory=True,
        )
    fwd = spec['forward']
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHED_GAMMA)
    align_scheduler(optimizer, scheduler, start_epoch)

    t0 = time.time()
    for epoch in range(start_epoch, total_epochs):
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
                val_loss += F.l1_loss(fwd(net, inputs), targets).item() * inputs.size(0)
                vn += inputs.size(0)
        val_loss /= vn

        elapsed = time.time() - t0
        write_csv_row(csv_path, {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'elapsed_s': round(elapsed, 1)}, SIMPLE_HEADER)
        msg = f'[{name}] Epoch {epoch:03d}/{total_epochs - 1} | train={train_loss:.6f} | val={val_loss:.6f} | {elapsed / 60:.1f}min'
        print(msg, flush=True)
        logging.info(msg)
        if val_loss < args.best_record['loss']:
            save_model(args, epoch, val_loss, net)
        net.train()

    tb_writer.close()
    del net
    torch.cuda.empty_cache()
    print(f'[{name}] done → {csv_path}', flush=True)


def main():
    args = parse_args()
    start_epoch = args.start_epoch if args.start_epoch is not None else args.base_epochs
    total_epochs = args.base_epochs + args.extra_epochs
    ensure_dir(BENCHMARK_DIR)
    master_log = os.path.join(BENCHMARK_DIR, f'resume_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(master_log), logging.StreamHandler(sys.stdout)],
    )

    include_k8 = args.include_sgf_k8 and not args.no_sgf_k8
    if args.models:
        selected = args.models
    else:
        selected = list(ALL_MODELS)
        if include_k8:
            selected.append('SGF-RecFNO (K=8)')

    print(f'Resuming epochs {start_epoch}–{total_epochs - 1} (+{args.extra_epochs}) for: {selected}', flush=True)
    total_t0 = time.time()

    if 'RecFNO' in selected:
        resume_recfno(start_epoch, total_epochs)
    if 'IsoRecFNO' in selected:
        _resume_composite('IsoRecFNO', 'benchmark_isorecfno_300', build_iso, compute_iso_loss, ISO_HEADER, start_epoch, total_epochs)
    if 'SGF-RecFNO' in selected:
        _resume_composite('SGF-RecFNO', 'benchmark_sgf-recfno_300', build_sgf, compute_sgf_loss, ISO_HEADER, start_epoch, total_epochs)
    if 'SGF-RecFNO (K=8)' in selected:
        k8_q = quantiles_for_k(8)
        _resume_composite(
            'SGF-RecFNO (K=8)', quantile_ablation_exp_name(8), build_sgf, compute_sgf_loss,
            ISO_HEADER, start_epoch, total_epochs, quantiles=k8_q,
        )
    for name in EXTERNAL_MODELS:
        if name in selected:
            resume_external(name, start_epoch, total_epochs)

    print(f'\nResume training complete in {(time.time() - total_t0) / 3600:.2f} h', flush=True)
    print(f'Log: {master_log}', flush=True)


if __name__ == '__main__':
    main()
