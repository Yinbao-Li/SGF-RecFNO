#!/usr/bin/env python3
"""Train all three heat-reconstruction schemes for 300 epochs and log metrics to CSV."""
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
sys.path.append(filename)

from data.dataset import HeatDataset
from model.fno import FNORecon
from model.iso_recfno import IsoRecFNO
from model.sgf_recfno import SGFRecFNO
from utils.iso_geometry import iso_recfno_loss
from utils.misc import prep_experiment, save_model
from utils.sgf_loss import sgf_recfno_loss

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'benchmark_300epoch')
EPOCHS = 300
BATCH_SIZE = 8
PLOT_FREQ = 50


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


def train_recfno():
    scheme = 'RecFNO'
    print(f'\n{"=" * 60}\n[{scheme}] 开始 300 epoch 训练\n{"=" * 60}', flush=True)
    csv_path = os.path.join(BENCHMARK_DIR, f'{scheme.lower()}_history.csv')
    header = ['epoch', 'train_loss', 'val_loss', 'elapsed_s']

    class Args:
        exp = 'benchmark_recfno_300'
        ckpt = os.path.join('logs', 'ckpt')
        tb_path = os.path.join('logs', 'tb')
        gpu_id = 0
        lr = 0.001
        epochs = EPOCHS
        batch_size = BATCH_SIZE
        plot_freq = PLOT_FREQ
        val_interval = 1

    args = Args()
    torch.cuda.set_device(args.gpu_id)
    tb_writer = prep_experiment(args)
    args.fig_path = args.exp_path + '/figure'
    ensure_dir(args.fig_path)
    args.best_record = {'epoch': -1, 'loss': 1e10}

    net = FNORecon(25, (12, 12), (200, 200), modes1=50, modes2=50, width=32).cuda()
    train_loader = DataLoader(HeatDataset(index=list(range(4000))), batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    val_loader = DataLoader(HeatDataset(index=list(range(4000, 5000))), batch_size=BATCH_SIZE, num_workers=4)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    t0 = time.time()
    for epoch in range(EPOCHS):
        net.train()
        train_loss, train_num = 0.0, 0
        for inputs, outputs in train_loader:
            inputs, outputs = inputs.cuda(), outputs.cuda()
            pre = net(inputs)
            loss = F.l1_loss(outputs, pre)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tb_writer.add_scalar('train_loss', loss.item(), epoch)
            train_loss += loss.item() * inputs.size(0)
            train_num += inputs.size(0)
        train_loss /= train_num
        scheduler.step()

        net.eval()
        val_loss, val_num = 0.0, 0
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs, outputs = inputs.cuda(), outputs.cuda()
                pre = net(inputs)
                val_loss += F.l1_loss(outputs, pre).item() * inputs.size(0)
                val_num += inputs.size(0)
        val_loss /= val_num
        tb_writer.add_scalar('val_loss', val_loss, epoch)

        elapsed = time.time() - t0
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'elapsed_s': round(elapsed, 1)}
        write_csv_row(csv_path, row, header)
        msg = f'[{scheme}] Epoch {epoch:03d}/{EPOCHS - 1} | train={train_loss:.6f} | val={val_loss:.6f} | {elapsed / 60:.1f}min'
        print(msg, flush=True)
        logging.info(msg)

        if val_loss < args.best_record['loss']:
            save_model(args, epoch, val_loss, net)
        net.train()

    tb_writer.close()
    print(f'[{scheme}] 完成，历史记录: {csv_path}', flush=True)
    return csv_path


def _train_composite(scheme, build_model_fn, compute_loss_fn, csv_header):
    print(f'\n{"=" * 60}\n[{scheme}] 开始 300 epoch 训练\n{"=" * 60}', flush=True)
    csv_path = os.path.join(BENCHMARK_DIR, f'{scheme.lower()}_history.csv')

    class Args:
        exp = f'benchmark_{scheme.lower()}_300'
        ckpt = os.path.join('logs', 'ckpt')
        tb_path = os.path.join('logs', 'tb')
        gpu_id = 0
        lr = 0.001
        epochs = EPOCHS
        batch_size = BATCH_SIZE
        plot_freq = PLOT_FREQ
        val_interval = 1
        sensor_num = 25
        fc_h = fc_w = 12
        out_h = out_w = 200
        modes1 = modes2 = 50
        width = 32
        train_end = 4000
        val_end = 5000
        quantiles = [0.2, 0.4, 0.6, 0.8]
        field_loss = 'l1'
        lambda_grad = 0.1
        lambda_sdf = 0.5
        lambda_ssim = 0.1
        sdf_scale = 5.0
        refine_modes1 = refine_modes2 = 24
        refine_width = 32

    args = Args()
    torch.cuda.set_device(args.gpu_id)
    tb_writer = prep_experiment(args)
    args.fig_path = args.exp_path + '/figure'
    ensure_dir(args.fig_path)
    args.best_record = {'epoch': -1, 'loss': 1e10}

    net = build_model_fn(args)
    train_loader = DataLoader(HeatDataset(index=list(range(args.train_end))), batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    val_loader = DataLoader(HeatDataset(index=list(range(args.train_end, args.val_end))), batch_size=BATCH_SIZE, num_workers=4)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    t0 = time.time()
    for epoch in range(EPOCHS):
        net.train()
        train_loss, train_num = 0.0, 0
        train_parts = {k: 0.0 for k in csv_header if k.startswith('train_') and k != 'train_loss'}
        if 'train_total' not in train_parts:
            train_parts = {'field': 0.0, 'grad': 0.0, 'sdf': 0.0, 'ssim': 0.0}

        for inputs, outputs in train_loader:
            inputs, outputs = inputs.cuda(), outputs.cuda()
            loss, parts, _ = compute_loss_fn(net, inputs, outputs, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tb_writer.add_scalar('train/total', parts['total'].item(), epoch)
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
        val_parts = {k: 0.0 for k in train_parts}
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs, outputs = inputs.cuda(), outputs.cuda()
                _, parts, _ = compute_loss_fn(net, inputs, outputs, args)
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
        write_csv_row(csv_path, row, csv_header)
        msg = (
            f'[{scheme}] Epoch {epoch:03d}/{EPOCHS - 1} | train={train_loss:.6f} | val={val_loss:.6f} '
            f'| field={val_parts["field"]:.6f} sdf={val_parts["sdf"]:.6f} | {elapsed / 60:.1f}min'
        )
        print(msg, flush=True)
        logging.info(msg)

        if val_loss < args.best_record['loss']:
            save_model(args, epoch, val_loss, net)
        net.train()

    tb_writer.close()
    print(f'[{scheme}] 完成，历史记录: {csv_path}', flush=True)
    return csv_path


def build_iso(args):
    return IsoRecFNO(
        sensor_num=args.sensor_num, fc_size=(args.fc_h, args.fc_w),
        out_size=(args.out_h, args.out_w), modes1=args.modes1, modes2=args.modes2,
        width=args.width, num_iso_levels=len(args.quantiles), quantiles=args.quantiles,
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


def build_sgf(args):
    return SGFRecFNO(
        sensor_num=args.sensor_num, fc_size=(args.fc_h, args.fc_w),
        out_size=(args.out_h, args.out_w), modes1=args.modes1, modes2=args.modes2,
        width=args.width, num_sdf=len(args.quantiles), quantiles=args.quantiles,
        sdf_scale=args.sdf_scale, refine_modes1=args.refine_modes1,
        refine_modes2=args.refine_modes2, refine_width=args.refine_width,
    ).cuda()


def compute_sgf_loss(net, inputs, outputs, args):
    aux = net(inputs, return_aux=True)
    loss, parts, _ = sgf_recfno_loss(
        aux['field'], outputs, aux['sdf_self'],
        quantiles=args.quantiles, field_loss=args.field_loss,
        lambda_grad=args.lambda_grad, lambda_sdf=args.lambda_sdf,
        lambda_ssim=args.lambda_ssim, sdf_scale=args.sdf_scale,
    )
    return loss, parts, aux


ISO_HEADER = ['epoch', 'train_loss', 'val_loss', 'val_field', 'val_grad', 'val_sdf', 'val_ssim', 'elapsed_s']
SGF_HEADER = ISO_HEADER


def main():
    ensure_dir(BENCHMARK_DIR)
    master_log = os.path.join(BENCHMARK_DIR, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(master_log),
            logging.StreamHandler(sys.stdout),
        ],
    )

    results = {}
    total_t0 = time.time()

    results['RecFNO'] = train_recfno()
    results['IsoRecFNO'] = _train_composite('IsoRecFNO', build_iso, compute_iso_loss, ISO_HEADER)
    results['SGF-RecFNO'] = _train_composite('SGF-RecFNO', build_sgf, compute_sgf_loss, SGF_HEADER)

    summary_path = os.path.join(BENCHMARK_DIR, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'Benchmark completed in {(time.time() - total_t0) / 3600:.2f} hours\n')
        for name, path in results.items():
            f.write(f'{name}: {path}\n')

    print(f'\n全部完成！总耗时 {(time.time() - total_t0) / 3600:.2f} 小时', flush=True)
    print(f'CSV 记录目录: {BENCHMARK_DIR}', flush=True)


if __name__ == '__main__':
    main()
