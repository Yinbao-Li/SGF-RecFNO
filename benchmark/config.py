# -*- coding: utf-8 -*-
"""Shared benchmark configuration."""
import os

from data.paths import heat2d_log_root

# Dataset split (temperature6000.h5, 6000 samples)
TRAIN_INDEX = list(range(4000))
VAL_INDEX = list(range(4000, 5000))
TEST_INDEX = list(range(5000, 6000))
VIS_INDEX = 5500

# Model / field geometry
SENSOR_NUM = 25
FC_SIZE = (12, 12)
OUT_SIZE = (200, 200)
MODES = 50
WIDTH = 32
FIELD_STD = 50.0  # temperature normalization scale (K)

# Training
EPOCHS = 300
RESUME_EPOCHS = 200
TOTAL_EPOCHS = EPOCHS + RESUME_EPOCHS
DEFAULT_BATCH = 8

# Paths (relative to heat2D/ working directory)
CKPT_ROOT = os.path.join('..', 'checkpoints')
TRAIN_CKPT_ROOT = os.path.join('logs', 'ckpt')  # new training outputs
BENCHMARK_LOG_DIR = os.path.join('logs', 'benchmark_300epoch')
COMPARISON_OUT_DIR = os.path.join('logs', 'benchmark_comparison')

# Absolute log root (for tooling outside heat2D/)
HEAT2D_LOG_ROOT = str(heat2d_log_root())

# Model groups — SGF-RecFNO first (primary contribution, Yinbao Li)
OUR_MODELS = ['SGF-RecFNO', 'IsoRecFNO']
BASELINE_RECFNO = ['RecFNO']  # Zhao et al. (2023)
EXTERNAL_MODELS = ['PINO', 'Geo-FNO', 'GINO']
ALL_MODELS = OUR_MODELS + BASELINE_RECFNO + EXTERNAL_MODELS
COMPARISON_MODELS = ['SGF-RecFNO', 'SGF-RecFNO (K=8)'] + ['IsoRecFNO'] + BASELINE_RECFNO + EXTERNAL_MODELS
PRIMARY_MODEL = 'SGF-RecFNO'

# Legacy alias
RECFNO_MODELS = OUR_MODELS + BASELINE_RECFNO
