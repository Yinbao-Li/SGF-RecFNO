# -*- coding: utf-8 -*-
"""Unified evaluation metrics for field reconstruction benchmarks."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from utils.iso_geometry import spatial_gradients, ssim_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def relative_l2(pred, target, eps=1e-8):
    diff = (pred - target).flatten(1)
    denom = target.flatten(1).norm(dim=1).clamp_min(eps)
    return (diff.norm(dim=1) / denom).mean()


def mse_metric(pred, target):
    return F.mse_loss(pred, target)


def mae_metric_k(pred, target, std=50.0):
    return torch.mean(torch.abs((pred - target) * std))


def psnr_metric(pred, target, data_range=1.0, eps=1e-10):
    mse = F.mse_loss(pred, target).clamp_min(eps)
    return 10.0 * torch.log10(torch.tensor(data_range ** 2, device=pred.device) / mse)


def ssim_metric(pred, target):
    return 1.0 - ssim_loss(pred, target)


def gradient_error(pred, target):
    pgx, pgy = spatial_gradients(pred)
    tgx, tgy = spatial_gradients(target)
    return F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)


def radial_spectrum(field):
    """Mean radial magnitude spectrum of 2D rFFT."""
    if field.dim() == 4:
        field = field[:, 0]
    elif field.dim() == 2:
        field = field.unsqueeze(0)
    spec = torch.fft.rfft2(field, norm='ortho')
    mag = torch.abs(spec)
    h, w = mag.shape[-2:]
    cy = h // 2
    yy, xx = torch.meshgrid(
        torch.arange(h, device=field.device),
        torch.arange(w, device=field.device),
        indexing='ij',
    )
    rr = torch.sqrt((yy - cy).float() ** 2 + xx.float() ** 2)
    max_r = int(rr.max().item()) + 1
    ri = rr.flatten().long()
    mag_mean = mag.mean(dim=0).flatten()
    radial = torch.zeros(max_r, device=field.device)
    counts = torch.zeros(max_r, device=field.device)
    radial.scatter_add_(0, ri, mag_mean)
    counts.scatter_add_(0, ri, torch.ones_like(mag_mean))
    return radial / counts.clamp_min(1)


def fourier_spectrum_error(pred, target):
    rp = radial_spectrum(pred)
    rt = radial_spectrum(target)
    n = min(rp.numel(), rt.numel())
    return F.mse_loss(rp[:n], rt[:n])


def batch_fourier_spectrum_curve(pred, target):
    """Average radial spectrum curves over batch."""
    if pred.dim() == 4:
        pred = pred[:, 0]
        target = target[:, 0]
    sp = radial_spectrum(pred).cpu().numpy()
    st = radial_spectrum(target).cpu().numpy()
    n = min(sp.size, st.size)
    sp, st = sp[:n], st[:n]
    err = (sp - st) ** 2
    return sp, st, err


@torch.no_grad()
def measure_inference(model, forward_fn, sample_input, warmup=20, repeats=100):
    model.eval()
    device = sample_input.device
    for _ in range(warmup):
        forward_fn(model, sample_input)
    torch.cuda.synchronize()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        forward_fn(model, sample_input)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / repeats
    mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0.0
    return ms, mem_mb


def compute_field_metrics(pred, target, std=50.0):
    return {
        'relative_l2': float(relative_l2(pred, target).item()),
        'mse': float(mse_metric(pred, target).item()),
        'mae_k': float(mae_metric_k(pred, target, std=std).item()),
        'psnr': float(psnr_metric(pred, target).item()),
        'ssim': float(ssim_metric(pred, target).item()),
        'grad_error': float(gradient_error(pred, target).item()),
        'spectrum_error': float(fourier_spectrum_error(pred, target).item()),
    }
