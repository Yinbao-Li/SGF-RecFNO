# -*- coding: utf-8 -*-
"""Losses for Self-Geometry Feedback RecFNO."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.field_geometry import (
    FieldGeometryConfig,
    levelset_sdf_gt,
    sdf_channel_slices,
    sdf_loss_weight,
)
from utils.iso_geometry import (
    DEFAULT_QUANTILES,
    gradient_loss,
    ssim_loss,
)


def relative_l2_loss(pred, target, eps=1e-8):
    diff = (pred - target).flatten(1)
    denom = target.flatten(1).norm(dim=1).clamp_min(eps)
    return (diff.norm(dim=1) / denom).mean()


def _spatial_field_loss(pred, target, geometry: FieldGeometryConfig | None, field_loss: str):
    """Field loss with optional downstream wake column weighting."""
    if geometry is None or geometry.wake_col_start is None or geometry.wake_field_weight == 1.0:
        if field_loss == 'relative_l2':
            return relative_l2_loss(pred, target)
        if field_loss == 'mse':
            return F.mse_loss(pred, target)
        return F.l1_loss(pred, target)

    w = torch.ones_like(pred)
    w[:, :, :, geometry.wake_col_start:] = geometry.wake_field_weight
    if field_loss == 'relative_l2':
        diff = (pred - target) * w
        denom = (target * w).flatten(1).norm(dim=1).clamp_min(1e-8)
        return (diff.flatten(1).norm(dim=1) / denom).mean()
    if field_loss == 'mse':
        return ((w * (pred - target) ** 2)).mean()
    return (w * (pred - target).abs()).mean()


def _composite_sdf_loss(sdf_pred, sdf_gt, geometry: FieldGeometryConfig, k: int):
    parts = {}
    total = sdf_pred.new_tensor(0.0)
    for name, start, end in sdf_channel_slices(geometry, k):
        if end <= start:
            continue
        term = F.l1_loss(sdf_pred[:, start:end], sdf_gt[:, start:end])
        w = sdf_loss_weight(geometry, name)
        parts[f'sdf_{name}'] = term
        total = total + w * term
    parts['sdf'] = total
    return total, parts


def sgf_recfno_loss(
    pred,
    target,
    sdf_self,
    quantiles=DEFAULT_QUANTILES,
    field_loss='l1',
    lambda_field=1.0,
    lambda_grad=0.1,
    lambda_sdf=0.5,
    lambda_ssim=0.1,
    sdf_scale=5.0,
    geometry: FieldGeometryConfig | None = None,
    k: int | None = None,
):
    """L = λ_f L_field + λ_g L_grad + weighted L_sdf + λ_s L_ssim."""
    geom = geometry
    k = k or len(quantiles)

    loss_field = _spatial_field_loss(pred, target, geom, field_loss)

    lam_grad = geom.lambda_grad if geom is not None else lambda_grad
    loss_grad = gradient_loss(pred, target)

    sdf_gt, _ = levelset_sdf_gt(
        target, geometry=geom, quantiles=quantiles, sdf_scale=sdf_scale, k=k,
    )
    if geom is None or geom.mode == 'single':
        loss_sdf = F.l1_loss(sdf_self, sdf_gt)
        sdf_parts = {'sdf': loss_sdf}
    else:
        loss_sdf, sdf_parts = _composite_sdf_loss(sdf_self, sdf_gt, geom, k)

    loss_ssim = ssim_loss(pred, target)

    total = (
        lambda_field * loss_field
        + lam_grad * loss_grad
        + loss_sdf
        + lambda_ssim * loss_ssim
    )
    parts = {
        'total': total,
        'field': loss_field,
        'grad': loss_grad,
        'ssim': loss_ssim,
        **sdf_parts,
    }
    return total, parts, sdf_gt


def iso_recfno_geometry_loss(
    pred,
    target,
    sdf_pred,
    quantiles=DEFAULT_QUANTILES,
    field_loss='l1',
    lambda_field=1.0,
    lambda_grad=0.1,
    lambda_sdf=0.5,
    lambda_ssim=0.1,
    sdf_scale=5.0,
    geometry: FieldGeometryConfig | None = None,
    k: int | None = None,
):
    """IsoRecFNO loss with task-specific GT geometry (same targets as SGF)."""
    geom = geometry
    k = k or len(quantiles)

    loss_field = _spatial_field_loss(pred, target, geom, field_loss)
    lam_grad = geom.lambda_grad if geom is not None else lambda_grad
    loss_grad = gradient_loss(pred, target)

    sdf_gt, _ = levelset_sdf_gt(
        target, geometry=geom, quantiles=quantiles, sdf_scale=sdf_scale, k=k,
    )
    if geom is None or geom.mode == 'single':
        lam_sdf = geom.lambda_sdf_field if geom is not None else lambda_sdf
        loss_sdf = F.l1_loss(sdf_pred, sdf_gt)
        sdf_parts = {'sdf': loss_sdf}
        sdf_total = lam_sdf * loss_sdf
    else:
        loss_sdf, sdf_parts = _composite_sdf_loss(sdf_pred, sdf_gt, geom, k)
        sdf_total = loss_sdf

    loss_ssim = ssim_loss(pred, target)
    total = lambda_field * loss_field + lam_grad * loss_grad + sdf_total + lambda_ssim * loss_ssim
    parts = {
        'total': total,
        'field': loss_field,
        'grad': loss_grad,
        'ssim': loss_ssim,
        **sdf_parts,
    }
    return total, parts
