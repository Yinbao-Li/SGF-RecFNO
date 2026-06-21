# -*- coding: utf-8 -*-
"""CLI entry points (also exposed via pyproject [project.scripts])."""
from __future__ import annotations

import runpy
import sys

from utils.bootstrap import ensure_repo_context


def _exec(script_relative: str):
    root = ensure_repo_context(heat2d_cwd=True)
    script = root / script_relative
    runpy.run_path(str(script), run_name='__main__')


def train_sgf_recfno():
    """Train SGF-RecFNO only (primary method)."""
    _exec('heat2D/heat2D_sgf_recfno.py')


def train_recfno_benchmark():
    """Train SGF-RecFNO, IsoRecFNO, and RecFNO baseline (300 epochs)."""
    _exec('heat2D/run_benchmark_300epoch.py')


def train_external_baselines():
    _exec('benchmark/train_external.py')


def run_comparison():
    _exec('benchmark/run_comparison.py')


def plot_case_comparison():
    _exec('benchmark/plot_case_comparison.py')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='SGF-RecFNO benchmark CLI')
    p.add_argument('command', choices=[
        'train-sgf', 'train-all', 'train-external', 'compare', 'plot-case',
    ])
    args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    {
        'train-sgf': train_sgf_recfno,
        'train-all': train_recfno_benchmark,
        'train-external': train_external_baselines,
        'compare': run_comparison,
        'plot-case': plot_case_comparison,
    }[args.command]()
