#!/usr/bin/env python3
"""Train RecFNO on heat conduction for 5 epochs and visualize results."""
import glob
import logging
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
from utils.misc import prep_experiment, save_model
from utils.options import parses
from utils.visualization import plot3x1

MEAN, STD = 308.0, 50.0


def denorm(t):
    return t * STD + MEAN


def train_5_epochs():
    args = parses()
    args.exp = 'recon_fno_heat_5epoch'
    args.epochs = 5
    args.batch_size = 8
    args.plot_freq = 1
    args.val_interval = 1

    print('=== 热传导 RecFNO 训练 (5 epochs) ===', flush=True)
    print(args, flush=True)
    torch.cuda.set_device(args.gpu_id)

    tb_writer = prep_experiment(args)
    args.fig_path = args.exp_path + '/figure'
    os.makedirs(args.fig_path, exist_ok=True)
    args.best_record = {'epoch': -1, 'loss': 1e10}

    net = FNORecon(
        sensor_num=25, fc_size=(12, 12), out_size=(200, 200),
        modes1=50, modes2=50, width=32,
    ).cuda()

    train_loader = DataLoader(
        HeatDataset(index=list(range(4000))),
        batch_size=args.batch_size, num_workers=4, shuffle=True,
    )
    val_loader = DataLoader(
        HeatDataset(index=list(range(4000, 5000))),
        batch_size=args.batch_size, num_workers=4,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    history = []
    for epoch in range(args.epochs):
        net.train()
        train_loss, train_num = 0.0, 0
        for i, (inputs, outputs) in enumerate(train_loader):
            inputs, outputs = inputs.cuda(), outputs.cuda()
            pre = net(inputs)
            loss = F.l1_loss(outputs, pre)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tb_writer.add_scalar('train_loss', loss.item(), i + epoch * len(train_loader))
            train_loss += loss.item() * inputs.size(0)
            train_num += inputs.size(0)

        train_loss /= train_num
        scheduler.step()

        net.eval()
        val_loss, val_num = 0.0, 0
        last_outputs, last_pre = None, None
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs, outputs = inputs.cuda(), outputs.cuda()
                pre = net(inputs)
                val_loss += F.l1_loss(outputs, pre).item() * inputs.size(0)
                val_num += inputs.size(0)
                last_outputs, last_pre = outputs, pre

        val_loss /= val_num
        tb_writer.add_scalar('val_loss', val_loss, epoch)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

        msg = f'Epoch {epoch:02d}/{args.epochs - 1} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}'
        print(msg, flush=True)
        logging.info(msg)

        if val_loss < args.best_record['loss']:
            save_model(args, epoch, val_loss, net)

        if last_outputs is not None:
            plot3x1(
                last_outputs[-1, 0].cpu().numpy(),
                last_pre[-1, 0].cpu().numpy(),
                os.path.join(args.fig_path, f'epoch{epoch}_norm.png'),
            )

    tb_writer.close()
    return args, history, net


def visualize_test(args, history):
    ckpts = sorted(glob.glob(os.path.join(args.exp_path, 'best_epoch_*.pth')))
    if not ckpts:
        raise FileNotFoundError('no checkpoint saved')
    ckpt_path = ckpts[-1]
    print(f'加载最优模型: {ckpt_path}', flush=True)

    net = FNORecon(
        sensor_num=25, fc_size=(12, 12), out_size=(200, 200),
        modes1=50, modes2=50, width=32,
    ).cuda()
    net.load_state_dict(torch.load(ckpt_path, map_location='cuda')['state_dict'])
    net.eval()

    test_idx = 5500
    ds = HeatDataset(index=[test_idx])
    inputs, target_norm = ds[0]
    inputs = torch.from_numpy(np.asarray(inputs)).float().unsqueeze(0).cuda()

    with torch.no_grad():
        pred_norm = net(inputs)[0, 0].cpu().numpy()
    target_norm = target_norm[0].numpy()

    target = denorm(target_norm)
    pred = denorm(pred_norm)
    err = pred - target

    out_dir = args.fig_path
    os.makedirs(out_dir, exist_ok=True)

    # 三联图（物理温度 K）
    plot3x1(target, pred, os.path.join(out_dir, 'test_truth_pred_error.png'))

    # 等温线对比图
    h, w = target.shape
    x = np.linspace(0, w / 100.0, w)
    y = np.linspace(h / 100.0, 0, h)
    X, Y = np.meshgrid(x, y)
    levels = np.linspace(target.min(), target.max(), 12)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, data, title in zip(
        axes,
        [target, pred, np.abs(err)],
        ['真值温度 (K)', '重建温度 (K)', '绝对误差 (K)'],
    ):
        cf = ax.contourf(X, Y, data, levels=50, cmap='coolwarm')
        if title.startswith('真值') or title.startswith('重建'):
            ax.contour(X, Y, data, levels=levels, colors='k', linewidths=0.4, alpha=0.6)
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, fraction=0.046)
    plt.tight_layout()
    iso_path = os.path.join(out_dir, 'test_isotherm.png')
    plt.savefig(iso_path, dpi=150, bbox_inches='tight')
    plt.close()

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_ae = float(np.max(np.abs(err)))
    summary = f'test sample {test_idx} | MAE={mae:.4f} K | RMSE={rmse:.4f} K | MaxAE={max_ae:.4f} K'
    print(summary, flush=True)

    # loss 曲线
    fig, ax = plt.subplots(figsize=(6, 4))
    ep = [h['epoch'] for h in history]
    ax.plot(ep, [h['train_loss'] for h in history], 'o-', label='train')
    ax.plot(ep, [h['val_loss'] for h in history], 's-', label='val')
    ax.set_xlabel('epoch')
    ax.set_ylabel('L1 loss (normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    hist_path = os.path.join(out_dir, 'loss_curve.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()

    return ckpt_path, iso_path, hist_path, summary


if __name__ == '__main__':
    args, history, _ = train_5_epochs()
    print('\n=== 训练历史 ===', flush=True)
    for h in history:
        print(f"  epoch {h['epoch']}: train={h['train_loss']:.6f}, val={h['val_loss']:.6f}", flush=True)

    ckpt, iso_path, hist_path, summary = visualize_test(args, history)
    print(f'\n可视化已保存至: {args.fig_path}', flush=True)
    print(f'  - loss 曲线: {hist_path}', flush=True)
    print(f'  - 等温线对比: {iso_path}', flush=True)
    print(f'  - {summary}', flush=True)
