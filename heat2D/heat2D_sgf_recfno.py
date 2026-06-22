# -*- coding: utf-8 -*-
"""Train Self-Geometry Feedback RecFNO on 2D heat conduction data."""
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(filename)

from data.dataset import HeatDataset
from model.sgf_recfno import SGFRecFNO
from utils.misc import prep_experiment, save_model
from utils.options import parse_sgf_args
from utils.sgf_loss import sgf_recfno_loss
from utils.visualization import plot3x1


def build_model(args):
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


def train():
    args = parse_sgf_args()
    print(args)
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

    for epoch in range(args.epochs):
        net.train()
        train_loss, train_num = 0.0, 0
        train_parts = {'field': 0.0, 'grad': 0.0, 'sdf': 0.0, 'ssim': 0.0}

        for i, (inputs, outputs) in enumerate(train_loader):
            inputs = inputs.cuda()
            outputs = outputs.cuda()
            loss, parts, _ = compute_loss(net, inputs, outputs, args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tb_writer.add_scalar('train/total', parts['total'].item(), i + epoch * len(train_loader))
            train_loss += parts['total'].item() * inputs.size(0)
            for key in train_parts:
                train_parts[key] += parts[key].item() * inputs.size(0)
            train_num += inputs.size(0)

        train_loss /= train_num
        for key in train_parts:
            train_parts[key] /= train_num
        scheduler.step()

        if epoch % args.val_interval == 0:
            net.eval()
            val_loss, val_num = 0.0, 0
            val_parts = {'field': 0.0, 'grad': 0.0, 'sdf': 0.0, 'ssim': 0.0}
            last_outputs, last_pre, last_coarse = None, None, None

            with torch.no_grad():
                for inputs, outputs in val_loader:
                    inputs = inputs.cuda()
                    outputs = outputs.cuda()
                    _, parts, aux = compute_loss(net, inputs, outputs, args)
                    val_loss += parts['total'].item() * inputs.size(0)
                    for key in val_parts:
                        val_parts[key] += parts[key].item() * inputs.size(0)
                    val_num += inputs.size(0)
                    last_outputs, last_pre, last_coarse = outputs, aux['field'], aux['coarse']

            val_loss /= val_num
            for key in val_parts:
                val_parts[key] /= val_num
                tb_writer.add_scalar(f'val/{key}', val_parts[key], epoch)
            tb_writer.add_scalar('val/total', val_loss, epoch)

            logging.info(
                'Epoch {} | train {:.6f} | val {:.6f} | field {:.6f} grad {:.6f} sdf {:.6f} ssim {:.6f}'.format(
                    epoch,
                    train_loss,
                    val_loss,
                    val_parts['field'],
                    val_parts['grad'],
                    val_parts['sdf'],
                    val_parts['ssim'],
                )
            )

            if val_loss < args.best_record['loss']:
                save_model(args, epoch, val_loss, net)

            if epoch % args.plot_freq == 0 and last_outputs is not None:
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

            net.train()

    tb_writer.close()


if __name__ == '__main__':
    train()
