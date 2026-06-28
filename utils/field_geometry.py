# -*- coding: utf-8 -*-
"""Task-specific level-set geometry extraction for SGF-RecFNO."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.iso_geometry import (
    DEFAULT_QUANTILES,
    field_quantile_levels,
    soft_levelset_sdf,
    spatial_gradients,
)


@dataclass(frozen=True)
class FieldGeometryConfig:
    """Geometry preset for a scalar reconstruction task."""

    kind: str
    label: str
    mode: str = 'single'  # single | dual | wake
    quantiles: tuple = DEFAULT_QUANTILES
    sdf_scale: float = 5.0
    grad_log1p: bool = False
    lambda_sdf_field: float = 0.5
    lambda_sdf_grad: float = 0.0
    lambda_sdf_zero: float = 0.0
    lambda_grad: float = 0.1
    include_zero_level: bool = True
    wake_col_start: int | None = None
    wake_field_weight: float = 1.0

    def num_sdf_channels(self, k: int) -> int:
        if self.mode == 'single':
            return k
        if self.mode == 'dual':
            return k
        if self.mode == 'wake':
            return k + 1 if self.include_zero_level else k
        raise ValueError(f'unknown geometry mode: {self.mode}')

    def split_quantiles(self, k: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """First/second half of K quantile levels (for dual / wake branches)."""
        q = tuple(_quantiles_for_k(k))
        half = k // 2
        return q[:half], q[half:]


def _quantiles_for_k(k: int) -> tuple[float, ...]:
    return tuple(i / (k + 1) for i in range(1, k + 1))


GEOMETRY_PRESETS = {
    'heat': FieldGeometryConfig('isotherm', 'Isotherm', mode='single'),
    'cylinder': FieldGeometryConfig(
        'wake_vorticity_r1',
        'Wake R1: |ω| + |∇ω|, wake-weighted',
        mode='wake',
        include_zero_level=False,
        lambda_sdf_field=0.1,
        lambda_sdf_grad=0.1,
        lambda_sdf_zero=0.0,
        lambda_grad=0.4,
        wake_col_start=60,
        wake_field_weight=3.0,
    ),
    'darcy': FieldGeometryConfig(
        'dual_pressure',
        'Dual: p + |∇p|',
        mode='dual',
        grad_log1p=True,
        lambda_sdf_field=0.15,
        lambda_sdf_grad=0.4,
        lambda_grad=0.25,
    ),
}


def grad_magnitude(field: torch.Tensor, log1p: bool = False) -> torch.Tensor:
    gx, gy = spatial_gradients(field)
    gm = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    if log1p:
        return torch.log1p(gm)
    return gm


def _fixed_level_sdf(field: torch.Tensor, level: float, scale: float) -> torch.Tensor:
    b = field.shape[0]
    levels = torch.full(
        (b, 1), level, device=field.device, dtype=field.dtype,
    )
    return soft_levelset_sdf(field, levels, scale=scale)


def _branch_sdf(
    scalar_field: torch.Tensor,
    quantiles: tuple[float, ...],
    scale: float,
) -> torch.Tensor:
    if not quantiles:
        return scalar_field.new_zeros(scalar_field.shape[0], 0, *scalar_field.shape[-2:])
    levels = field_quantile_levels(scalar_field, quantiles)
    return soft_levelset_sdf(scalar_field, levels, scale=scale)


def extract_self_geometry(
    field: torch.Tensor,
    geometry: FieldGeometryConfig | None = None,
    quantiles=None,
    sdf_scale=None,
    k: int | None = None,
):
    """Build multi-level SDF maps from a predicted coarse field only."""
    geom = geometry or GEOMETRY_PRESETS['heat']
    scale = sdf_scale if sdf_scale is not None else geom.sdf_scale
    k = k or len(quantiles or geom.quantiles)

    if geom.mode == 'single':
        q = tuple(quantiles) if quantiles is not None else _quantiles_for_k(k)
        levels = field_quantile_levels(field, q)
        sdf = soft_levelset_sdf(field, levels, scale=scale)
        return sdf, levels

    q_field, q_grad = geom.split_quantiles(k)

    if geom.mode == 'dual':
        sdf_p = _branch_sdf(field, q_field, scale)
        sdf_g = _branch_sdf(
            grad_magnitude(field, log1p=geom.grad_log1p), q_grad, scale,
        )
        sdf = torch.cat([sdf_p, sdf_g], dim=1)
        return sdf, None

    if geom.mode == 'wake':
        sdf_abs = _branch_sdf(field.abs(), q_field, scale)
        sdf_grad = _branch_sdf(grad_magnitude(field), q_grad, scale)
        parts = [sdf_abs, sdf_grad]
        if geom.include_zero_level:
            parts.append(_fixed_level_sdf(field, 0.0, scale))
        sdf = torch.cat(parts, dim=1)
        return sdf, None

    raise ValueError(f'unknown geometry mode: {geom.mode}')


def levelset_sdf_gt(
    target: torch.Tensor,
    geometry=None,
    quantiles=None,
    sdf_scale=None,
    k: int | None = None,
):
    """Ground-truth level-set SDF maps (SGF training loss)."""
    return extract_self_geometry(
        target,
        geometry=geometry,
        quantiles=quantiles,
        sdf_scale=sdf_scale,
        k=k,
    )


def sdf_channel_slices(geometry: FieldGeometryConfig, k: int) -> list[tuple[str, int, int]]:
    """Named channel ranges in the concatenated SDF tensor."""
    if geometry.mode == 'single':
        return [('field', 0, k)]
    half = k // 2
    if geometry.mode == 'dual':
        return [('field', 0, half), ('grad', half, k)]
    if geometry.mode == 'wake':
        slices = [
            ('abs', 0, half),
            ('grad', half, half * 2),
        ]
        if geometry.include_zero_level:
            slices.append(('zero', half * 2, half * 2 + 1))
        return slices
    raise ValueError(geometry.mode)


def sdf_loss_weight(geometry: FieldGeometryConfig, branch: str) -> float:
    if geometry.mode == 'single':
        return geometry.lambda_sdf_field
    if branch in ('field', 'abs'):
        return geometry.lambda_sdf_field
    if branch == 'grad':
        return geometry.lambda_sdf_grad
    if branch == 'zero':
        return geometry.lambda_sdf_zero
    raise KeyError(branch)
