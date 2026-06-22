# -*- coding: utf-8 -*-
"""Model registry for unified heat-reconstruction benchmark."""
import glob
import os

import torch

from benchmark.config import (
    CKPT_ROOT, FC_SIZE, MODES, OUT_SIZE, SENSOR_NUM, TRAIN_CKPT_ROOT, WIDTH,
)
from model.fno import FNORecon
from model.iso_recfno import IsoRecFNO
from model.sgf_recfno import SGFRecFNO


def _find_ckpt(exp_name):
    paths = []
    for root in (CKPT_ROOT, TRAIN_CKPT_ROOT):
        paths.extend(glob.glob(os.path.join(root, exp_name, 'best_epoch_*.pth')))
    if not paths:
        return None
    # Pick checkpoint with lowest val loss encoded in filename
    def _loss(p):
        try:
            return float(p.split('_loss_')[-1].replace('.pth', ''))
        except ValueError:
            return float('inf')
    return min(paths, key=_loss)


def inference_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_state(model, ckpt):
    dev = inference_device()
    state = torch.load(ckpt, map_location=dev, weights_only=False)['state_dict']
    state = {k: v for k, v in state.items() if not k.endswith('_metadata')}
    model.load_state_dict(state, strict=False)
    model.to(dev).eval()
    return model


def _build_gino():
    from benchmark.external_models import GINOHeatRecon
    return GINOHeatRecon()


def _build_geofno():
    from benchmark.external_models import GeoFNOHeatRecon
    return GeoFNOHeatRecon()


def _build_pino():
    from benchmark.external_models import PINOHeatRecon
    return PINOHeatRecon()


MODEL_SPECS = {
    'SGF-RecFNO': {
        'type': 'sensor',
        'source': 'Yinbao Li — Self-Geometry Feedback RecFNO (this repo)',
        'exp': 'benchmark_sgf-recfno_300',
        'build': lambda: SGFRecFNO(SENSOR_NUM, FC_SIZE, OUT_SIZE, modes1=MODES, modes2=MODES, width=WIDTH),
        'forward': lambda m, x: m(x, return_aux=True)['field'],
        'geometry_model': True,
        'primary': True,
    },
    'IsoRecFNO': {
        'type': 'sensor',
        'source': 'Yinbao Li — geometry-aware RecFNO (this repo)',
        'exp': 'benchmark_isorecfno_300',
        'build': lambda: IsoRecFNO(SENSOR_NUM, FC_SIZE, OUT_SIZE, modes1=MODES, modes2=MODES, width=WIDTH),
        'forward': lambda m, x: m(x, return_aux=True)['field'],
    },
    'RecFNO': {
        'type': 'sensor',
        'source': 'Zhao et al. (2023) — https://github.com/zhaoxiaoyu1995/RecFNO',
        'exp': 'benchmark_recfno_300',
        'build': lambda: FNORecon(SENSOR_NUM, FC_SIZE, OUT_SIZE, modes1=MODES, modes2=MODES, width=WIDTH),
        'forward': lambda m, x: m(x),
    },
    'GINO': {
        'type': 'sensor',
        'source': 'https://github.com/neuraloperator/neuraloperator',
        'exp': 'benchmark_gino_300',
        'build': _build_gino,
        'forward': lambda m, x: m(x),
    },
    'Geo-FNO': {
        'type': 'sensor',
        'source': 'https://github.com/neuraloperator/Geo-FNO',
        'exp': 'benchmark_geofno_300',
        'build': _build_geofno,
        'forward': lambda m, x: m(x),
    },
    'PINO': {
        'type': 'grid',
        'source': 'https://github.com/neuraloperator/physics_informed',
        'exp': 'benchmark_pino_300',
        'build': _build_pino,
        'forward': lambda m, x: m(x),
    },
}


def load_model(name):
    spec = MODEL_SPECS[name]
    ckpt = _find_ckpt(spec['exp'])
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint for {name} ({spec["exp"]})')
    model = spec['build']()
    _load_state(model, ckpt)
    return model, ckpt, spec
