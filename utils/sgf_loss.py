# -*- coding: utf-8 -*-
"""Losses for Self-Geometry Feedback RecFNO."""
import torch
import torch.nn.functional as F

from utils.iso_geometry import (
    DEFAULT_QUANTILES,
    gradient_loss,
    soft_isotherm_sdf,
    ssim_loss,
    temperature_quantile_levels,
)


def relative_l2_loss(pred, target, eps=1e-8):
    diff = (pred - target).flatten(1)
    denom = target.flatten(1).norm(dim=1).clamp_min(eps)
    return (diff.norm(dim=1) / denom).mean()


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
):
    """L = λ_f L_field + λ_g L_grad + λ_s L_sdf + λ_s L_ssim."""
    if field_loss == 'relative_l2':
        loss_field = relative_l2_loss(pred, target)
    elif field_loss == 'mse':
        loss_field = F.mse_loss(pred, target)
    else:
        loss_field = F.l1_loss(pred, target)

    loss_grad = gradient_loss(pred, target)

    levels_gt = temperature_quantile_levels(target, quantiles)
    sdf_gt = soft_isotherm_sdf(target, levels_gt, scale=sdf_scale)
    loss_sdf = F.l1_loss(sdf_self, sdf_gt)

    loss_ssim = ssim_loss(pred, target)

    total = (
        lambda_field * loss_field
        + lambda_grad * loss_grad
        + lambda_sdf * loss_sdf
        + lambda_ssim * loss_ssim
    )
    parts = {
        'total': total,
        'field': loss_field,
        'grad': loss_grad,
        'sdf': loss_sdf,
        'ssim': loss_ssim,
    }
    return total, parts, sdf_gt
