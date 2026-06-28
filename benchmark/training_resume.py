# -*- coding: utf-8 -*-
"""Helpers to resume benchmark training from existing checkpoints."""
from __future__ import annotations

import glob
import os
import re

import torch

from benchmark.config import CKPT_ROOT, TRAIN_CKPT_ROOT

_CKPT_RE = re.compile(r'best_epoch_(\d+)_loss_([0-9.eE+-]+)\.pth$')


def find_best_checkpoint(exp_name: str) -> str | None:
    paths = []
    for root in (CKPT_ROOT, TRAIN_CKPT_ROOT):
        paths.extend(glob.glob(os.path.join(root, exp_name, 'best_epoch_*.pth')))
    if not paths:
        return None

    def _loss(p: str) -> float:
        m = _CKPT_RE.search(p)
        if m:
            return float(m.group(2))
        return float('inf')

    return min(paths, key=_loss)


def parse_checkpoint_meta(ckpt_path: str) -> tuple[int, float]:
    m = _CKPT_RE.search(ckpt_path)
    if not m:
        raise ValueError(f'Cannot parse checkpoint filename: {ckpt_path}')
    return int(m.group(1)), float(m.group(2))


def load_model_weights(model, ckpt_path: str, device=None) -> tuple[int, float]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload.get('state_dict', payload)
    state = {k: v for k, v in state.items() if not k.endswith('_metadata')}
    model.load_state_dict(state, strict=False)
    if 'epoch' in payload:
        return int(payload['epoch']), parse_checkpoint_meta(ckpt_path)[1]
    return parse_checkpoint_meta(ckpt_path)


def align_scheduler(optimizer, scheduler, start_epoch: int) -> None:
    """Match LR schedule as if ``start_epoch`` full steps already ran."""
    for _ in range(start_epoch):
        scheduler.step()
