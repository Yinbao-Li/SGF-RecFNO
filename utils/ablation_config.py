# -*- coding: utf-8 -*-
"""Configurations for SGF-RecFNO ablation studies (loss terms & SDF depth K)."""
from __future__ import annotations

# Default loss weights (full model)
DEFAULT_LAMBDA_FIELD = 1.0
DEFAULT_LAMBDA_GRAD = 0.1
DEFAULT_LAMBDA_SDF = 0.5
DEFAULT_LAMBDA_SSIM = 0.1

# Loss-component ablations: remove one term at a time (300 epochs each).
LOSS_ABLATIONS = {
    'full': {
        'label': 'Full loss',
        'lambda_field': DEFAULT_LAMBDA_FIELD,
        'lambda_grad': DEFAULT_LAMBDA_GRAD,
        'lambda_sdf': DEFAULT_LAMBDA_SDF,
        'lambda_ssim': DEFAULT_LAMBDA_SSIM,
    },
    'no_field': {
        'label': 'w/o Field',
        'lambda_field': 0.0,
        'lambda_grad': DEFAULT_LAMBDA_GRAD,
        'lambda_sdf': DEFAULT_LAMBDA_SDF,
        'lambda_ssim': DEFAULT_LAMBDA_SSIM,
    },
    'no_grad': {
        'label': 'w/o Grad',
        'lambda_field': DEFAULT_LAMBDA_FIELD,
        'lambda_grad': 0.0,
        'lambda_sdf': DEFAULT_LAMBDA_SDF,
        'lambda_ssim': DEFAULT_LAMBDA_SSIM,
    },
    'no_sdf': {
        'label': 'w/o SDF',
        'lambda_field': DEFAULT_LAMBDA_FIELD,
        'lambda_grad': DEFAULT_LAMBDA_GRAD,
        'lambda_sdf': 0.0,
        'lambda_ssim': DEFAULT_LAMBDA_SSIM,
    },
    'no_ssim': {
        'label': 'w/o SSIM',
        'lambda_field': DEFAULT_LAMBDA_FIELD,
        'lambda_grad': DEFAULT_LAMBDA_GRAD,
        'lambda_sdf': DEFAULT_LAMBDA_SDF,
        'lambda_ssim': 0.0,
    },
}

QUANTILE_K_VALUES = (1, 2, 4, 8)

# Pre-trained main model (300 ep, full loss, K=4) — reuse instead of retraining.
MAIN_SGF_EXP = 'benchmark_sgf-recfno_300'

# Variants that actually need new training (full / K=4 duplicate the main model).
LOSS_TRAIN_VARIANTS = ('no_field', 'no_grad', 'no_sdf', 'no_ssim')
QUANTILE_TRAIN_K = (1, 2, 8)


def quantiles_for_k(k: int) -> tuple[float, ...]:
    """Evenly spaced quantile levels for K SDF channels: q_i = i / (K + 1)."""
    if k < 1:
        raise ValueError('K must be >= 1')
    return tuple(i / (k + 1) for i in range(1, k + 1))


def loss_ablation_exp_name(variant: str) -> str:
    return f'ablation_loss_{variant}'


def quantile_ablation_exp_name(k: int) -> str:
    return f'ablation_k{k}'
