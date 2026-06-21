#!/usr/bin/env python3
"""Quick IsoRecFNO smoke run (5 epochs) with visualization."""
import glob
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(filename)
heat2d_dir = os.path.dirname(__file__)
if heat2d_dir not in sys.path:
    sys.path.insert(0, heat2d_dir)

from data.dataset import HeatDataset
from heat2D_iso_recfno import build_model, compute_loss
from utils.misc import prep_experiment, save_model
from utils.options import parse_iso_args
from utils.visualization import plot3x1

MEAN, STD = 308.0, 50.0


def denorm(t):
    return t * STD + MEAN


def train_5_epochs():
    args = parse_iso_args()
    args.exp = 'iso_recfno_heat_5epoch'
    args.epochs = 5
    args.batch_size = 8
    args.plot_freq = 1
    args.val_interval = 1

    print('=== IsoRecFNO 热传导训练 (5 epochs) ===', flush=True)
    print(args, flush=True)
    torch.cuda.set_device(args.gpu_id)

    tb_writer = prep_experiment(args)
    args.fig_path = args.exp_path + '/figure'
    os.makedirs(args.fig_path, exist_ok=True)
    args.best_record = {'epoch': -1, 'loss': 1e10}

    net = build_model(args)
    train_loader = DataLoader(
        HeatDataset(index=list(range(args.train_end))),
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
    )
    val_loader = DataLoader(
        HeatDataset(index=list(range(args.train_end, args.val_end))),
        batch_size=args.batch_size,
        num_workers=4,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    history = []

    for epoch in range(args.epochs):
        net.train()
        train_loss, train_num = 0.0, 0
        for i, (inputs, outputs) in enumerate(train_loader):
            inputs, outputs = inputs.cuda(), outputs.cuda()
            loss, parts, _ = compute_loss(net, inputs, outputs, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tb_writer.add_scalar('train/total', parts['total'].item(), i + epoch * len(train_loader))
            train_loss += parts['total'].item() * inputs.size(0)
            train_num += inputs.size(0)

        train_loss /= train_num
        scheduler.step()

        net.eval()
        val_loss, val_num = 0.0, 0
        val_field = val_grad = val_sdf = val_ssim = 0.0
        last_outputs, last_pre, last_coarse = None, None, None
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs, outputs = inputs.cuda(), outputs.cuda()
                _, parts, aux = compute_loss(net, inputs, outputs, args)
                val_loss += parts['total'].item() * inputs.size(0)
                val_field += parts['field'].item() * inputs.size(0)
                val_grad += parts['grad'].item() * inputs.size(0)
                val_sdf += parts['sdf'].item() * inputs.size(0)
                val_ssim += parts['ssim'].item() * inputs.size(0)
                val_num += inputs.size(0)
                last_outputs, last_pre, last_coarse = outputs, aux['field'], aux['coarse']

        val_loss /= val_num
        val_field /= val_num
        val_grad /= val_num
        val_sdf /= val_num
        val_ssim /= val_num
        tb_writer.add_scalar('val/total', val_loss, epoch)
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_field': val_field,
            'val_grad': val_grad,
            'val_sdf': val_sdf,
            'val_ssim': val_ssim,
        })

        msg = (
            f'Epoch {epoch:02d}/{args.epochs - 1} | train={train_loss:.6f} | val={val_loss:.6f} '
            f'| field={val_field:.6f} grad={val_grad:.6f} sdf={val_sdf:.6f} ssim={val_ssim:.6f}'
        )
        print(msg, flush=True)
        logging.info(msg)

        if val_loss < args.best_record['loss']:
            save_model(args, epoch, val_loss, net)

        if last_outputs is not None:
            plot3x1(
                last_outputs[-1, 0].cpu().numpy(),
                last_pre[-1, 0].cpu().numpy(),
                os.path.join(args.fig_path, f'epoch{epoch}_field.png'),
            )
            plot3x1(
                last_outputs[-1, 0].cpu().numpy(),
                last_coarse[-1, 0].cpu().numpy(),
                os.path.join(args.fig_path, f'epoch{epoch}_coarse.png'),
            )

    tb_writer.close()
    return args, history, net


def visualize_test(args, history):
    ckpts = sorted(glob.glob(os.path.join(args.exp_path, 'best_epoch_*.pth')))
    if not ckpts:
        raise FileNotFoundError('no checkpoint saved')
    ckpt_path = ckpts[-1]
    print(f'加载最优模型: {ckpt_path}', flush=True)

    net = build_model(args)
    net.load_state_dict(torch.load(ckpt_path, map_location='cuda', weights_only=False)['state_dict'])
    net.eval()

    test_idx = 5500
    ds = HeatDataset(index=[test_idx])
    inputs, target_norm = ds[0]
    inputs = torch.from_numpy(np.asarray(inputs)).float().unsqueeze(0).cuda()

    with torch.no_grad():
        aux = net(inputs, return_aux=True)
        pred_norm = aux['field'][0, 0].cpu().numpy()
        coarse_norm = aux['coarse'][0, 0].cpu().numpy()
        sdf_pred = aux['sdf_pred'][0].cpu().numpy()
    target_norm = target_norm[0].numpy()

    target = denorm(target_norm)
    pred = denorm(pred_norm)
    coarse = denorm(coarse_norm)
    err = pred - target

    out_dir = args.fig_path
    os.makedirs(out_dir, exist_ok=True)
    plot3x1(target, pred, os.path.join(out_dir, 'test_truth_pred_error.png'))
    plot3x1(target, coarse, os.path.join(out_dir, 'test_truth_coarse_error.png'))

    h, w = target.shape
    x = np.linspace(0, w / 100.0, w)
    y = np.linspace(h / 100.0, 0, h)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    cf = axes[0].contourf(X, Y, target, levels=50, cmap='coolwarm')
    axes[0].set_title('Truth (K)')
    plt.colorbar(cf, ax=axes[0], fraction=0.046)
    cf = axes[1].contourf(X, Y, pred, levels=50, cmap='coolwarm')
    axes[1].set_title('IsoRecFNO (K)')
    plt.colorbar(cf, ax=axes[1], fraction=0.046)
    cf = axes[2].contourf(X, Y, np.abs(err), levels=50, cmap='viridis')
    axes[2].set_title('|Error| (K)')
    plt.colorbar(cf, ax=axes[2], fraction=0.046)
    axes[3].imshow(sdf_pred.mean(axis=0), cmap='RdBu_r', origin='upper')
    axes[3].set_title('Mean predicted SDF')
    for ax in axes[:3]:
        ax.set_aspect('equal')
    plt.tight_layout()
    iso_path = os.path.join(out_dir, 'test_isotherm_sdf.png')
    plt.savefig(iso_path, dpi=150, bbox_inches='tight')
    plt.close()

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_ae = float(np.max(np.abs(err)))
    summary = f'test sample {test_idx} | MAE={mae:.4f} K | RMSE={rmse:.4f} K | MaxAE={max_ae:.4f} K'
    print(summary, flush=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ep = [h['epoch'] for h in history]
    ax.plot(ep, [h['train_loss'] for h in history], 'o-', label='train total')
    ax.plot(ep, [h['val_loss'] for h in history], 's-', label='val total')
    ax.plot(ep, [h['val_field'] for h in history], '^--', label='val field', alpha=0.7)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
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
        print(
            f"  epoch {h['epoch']}: train={h['train_loss']:.6f}, val={h['val_loss']:.6f}, "
            f"field={h['val_field']:.6f}, sdf={h['val_sdf']:.6f}",
            flush=True,
        )
    ckpt, iso_path, hist_path, summary = visualize_test(args, history)
    print(f'\n可视化已保存至: {args.fig_path}', flush=True)
    print(f'  - loss 曲线: {hist_path}', flush=True)
    print(f'  - 等温线/SDF: {iso_path}', flush=True)
    print(f'  - {summary}', flush=True)
