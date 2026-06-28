# -*- coding: utf-8 -*-
"""Geometry utilities and composite losses for IsoRecFNO."""
import torch
import torch.nn.functional as F


DEFAULT_QUANTILES = (0.2, 0.4, 0.6, 0.8)


def field_quantile_levels(field, quantiles=DEFAULT_QUANTILES):
    """Per-sample quantile levels from a scalar field.

    Args:
        field: (B, 1, H, W)
        quantiles: sequence of quantile values in (0, 1)

    Returns:
        levels: (B, K)
    """
    flat = field.reshape(field.shape[0], -1)
    q = torch.tensor(quantiles, device=field.device, dtype=field.dtype)
    return torch.quantile(flat, q, dim=1).T.contiguous()


def spatial_gradients(field):
    """Central-difference gradients with edge replication.

    Args:
        field: (B, C, H, W)

    Returns:
        gx, gy with the same shape as field
    """
    left = F.pad(field, (1, 0, 0, 0))[:, :, :, :-1]
    right = F.pad(field, (0, 1, 0, 0))[:, :, :, 1:]
    gx = 0.5 * (right - left)

    top = F.pad(field, (0, 0, 1, 0))[:, :, :-1, :]
    bottom = F.pad(field, (0, 0, 0, 1))[:, :, 1:, :]
    gy = 0.5 * (bottom - top)
    return gx, gy


def temperature_quantile_levels(field, quantiles=DEFAULT_QUANTILES):
    """Alias for :func:`field_quantile_levels` (heat / isotherm tasks)."""
    return field_quantile_levels(field, quantiles)


def soft_levelset_sdf(field, levels, scale=5.0, eps=1e-3):
    """Differentiable signed-distance proxy to scalar-field level sets.

    Uses the normalized level-set function phi = (phi - c) / |grad phi|, which
    approximates signed distance in the direction normal to each contour.
    Values are squashed to (-1, 1) for stable regression.

    Args:
        field: (B, 1, H, W)
        levels: (B, K)

    Returns:
        sdf: (B, K, H, W)
    """
    gx, gy = spatial_gradients(field)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + eps)
    sdfs = []
    for k in range(levels.shape[1]):
        lv = levels[:, k].view(-1, 1, 1, 1)
        phi = (field - lv) / grad_mag
        sdfs.append(torch.tanh(phi / scale))
    return torch.cat(sdfs, dim=1)


def soft_isotherm_sdf(field, levels, scale=5.0, eps=1e-3):
    """Alias for :func:`soft_levelset_sdf` (heat / isotherm tasks)."""
    return soft_levelset_sdf(field, levels, scale=scale, eps=eps)


def gradient_loss(pred, target):
    pgx, pgy = spatial_gradients(pred)
    tgx, tgy = spatial_gradients(target)
    return F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)


def _gaussian_kernel(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] @ g[None, :]
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel


def ssim_loss(pred, target, window_size=11, sigma=1.5, c1=0.01 ** 2, c2=0.03 ** 2):
    """1 - SSIM, averaged over batch."""
    if pred.shape[1] != 1:
        pred = pred.mean(dim=1, keepdim=True)
        target = target.mean(dim=1, keepdim=True)

    channels = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, channels, pred.device, pred.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=pad, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_x2
    sigma_y = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - ssim_map.mean()


def iso_recfno_loss(
    pred,
    target,
    sdf_pred,
    quantiles=DEFAULT_QUANTILES,
    field_loss='l1',
    lambda_grad=0.1,
    lambda_sdf=0.5,
    lambda_ssim=0.1,
    sdf_scale=5.0,
):
    """Composite IsoRecFNO training objective.

    L = L_field + lambda_grad * L_grad + lambda_sdf * L_sdf + lambda_ssim * L_ssim
    """
    if field_loss == 'mse':
        loss_field = F.mse_loss(pred, target)
    else:
        loss_field = F.l1_loss(pred, target)

    loss_grad = gradient_loss(pred, target)

    levels = field_quantile_levels(target, quantiles)
    sdf_gt = soft_levelset_sdf(target, levels, scale=sdf_scale)
    loss_sdf = F.l1_loss(sdf_pred, sdf_gt)

    loss_ssim = ssim_loss(pred, target)

    total = (
        loss_field
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
    return total, parts
