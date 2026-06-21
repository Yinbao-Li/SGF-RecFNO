# -*- coding: utf-8 -*-
"""Isotherm geometry metrics: Chamfer and Hausdorff distances."""
import numpy as np
import torch

from utils.iso_geometry import DEFAULT_QUANTILES, temperature_quantile_levels


def _extract_contour_points(field2d, level):
    """Extract isotherm contour points from a 2D field."""
    from scipy.ndimage import binary_erosion

    mask = field2d >= level
    if not mask.any():
        return np.zeros((0, 2), dtype=np.float32)
    boundary = mask & ~binary_erosion(mask)
    ys, xs = np.where(boundary)
    if len(xs) == 0:
        ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    idx = np.linspace(0, len(xs) - 1, min(512, len(xs))).astype(int)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)


def _directed_hausdorff(a, b):
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    diff = a[:, None, :] - b[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return float(dist.min(axis=1).max())


def hausdorff_distance(a, b):
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    return max(_directed_hausdorff(a, b), _directed_hausdorff(b, a))


def chamfer_distance(a, b):
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    diff = a[:, None, :] - b[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return 0.5 * (dist.min(axis=1).mean() + dist.min(axis=0).mean())


def isotherm_geometry_metrics(pred, target, quantiles=DEFAULT_QUANTILES, std=50.0):
    """Compute per-quantile Chamfer/Hausdorff on physical temperature fields."""
    pred_np = pred[0, 0].detach().cpu().numpy() * std
    tgt_np = target[0, 0].detach().cpu().numpy() * std
    levels = temperature_quantile_levels(target, quantiles)[0].detach().cpu().numpy()

    out = {}
    chamfers, hausdorffs = [], []
    for q, lv in zip(quantiles, levels):
        pp = _extract_contour_points(pred_np, lv)
        tp = _extract_contour_points(tgt_np, lv)
        cd = chamfer_distance(pp, tp)
        hd = hausdorff_distance(pp, tp)
        out[f'chamfer_q{int(q * 100)}'] = cd
        out[f'hausdorff_q{int(q * 100)}'] = hd
        if not np.isnan(cd):
            chamfers.append(cd)
        if not np.isnan(hd):
            hausdorffs.append(hd)
    out['chamfer_mean'] = float(np.nanmean(chamfers)) if chamfers else float('nan')
    out['hausdorff_mean'] = float(np.nanmean(hausdorffs)) if hausdorffs else float('nan')
    return out
