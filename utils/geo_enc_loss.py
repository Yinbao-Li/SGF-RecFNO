# -*- coding: utf-8 -*-
"""Losses for GeoEnc RecFNO (multi-channel field + SDF encoding)."""
import torch
import torch.nn.functional as F

from utils.iso_geometry import (
    DEFAULT_QUANTILES,
    gradient_loss,
    soft_isotherm_sdf,
    temperature_quantile_levels,
)


def relative_l2_loss(pred, target, eps=1e-8):
    diff = (pred - target).flatten(1)
    denom = target.flatten(1).norm(dim=1).clamp_min(eps)
    return (diff.norm(dim=1) / denom).mean()


def geo_enc_fno_loss(
    pred,
    target,
    sdf_pred,
    quantiles=DEFAULT_QUANTILES,
    field_loss='l1',
    lambda_grad=0.1,
    lambda_sdf=0.5,
    sdf_scale=5.0,
):
    """L = L_field + lambda_grad * L_grad + lambda_sdf * L_sdf."""
    if field_loss == 'relative_l2':
        loss_field = relative_l2_loss(pred, target)
    elif field_loss == 'mse':
        loss_field = F.mse_loss(pred, target)
    else:
        loss_field = F.l1_loss(pred, target)

    loss_grad = gradient_loss(pred, target)

    levels = temperature_quantile_levels(target, quantiles)
    sdf_gt = soft_isotherm_sdf(target, levels, scale=sdf_scale)
    loss_sdf = F.l1_loss(sdf_pred, sdf_gt)

    total = loss_field + lambda_grad * loss_grad + lambda_sdf * loss_sdf
    parts = {
        'total': total,
        'field': loss_field,
        'grad': loss_grad,
        'sdf': loss_sdf,
    }
    return total, parts, sdf_gt
