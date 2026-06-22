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


def plot_recfno_2x3():
    _exec('benchmark/plot_recfno_2x3.py')


def plot_three_cases_six_models():
    _exec('benchmark/plot_three_cases_six_models.py')


def plot_test_metrics():
    _exec('benchmark/plot_test_metrics.py')


def plot_problem_setup():
    _exec('benchmark/plot_problem_setup.py')


def plot_sgf_pipeline():
    _exec('benchmark/plot_sgf_pipeline.py')


def plot_isotherm_comparison():
    _exec('benchmark/plot_isotherm_comparison.py')


def plot_mae_histogram():
    _exec('benchmark/plot_mae_histogram.py')


def plot_sdf_ablation():
    _exec('benchmark/plot_sdf_ablation.py')


def train_sgf_ablations():
    _exec('heat2D/run_sgf_ablations.py')


def plot_ablation_studies():
    _exec('benchmark/plot_ablation_studies.py')


def plot_all_figures():
    _exec('benchmark/plot_all_figures.py')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='SGF-RecFNO benchmark CLI')
    p.add_argument('command', choices=[
        'train-sgf', 'train-all', 'train-external', 'compare', 'plot-case',
        'plot-recfno-2x3', 'plot-3cases-6models', 'plot-test-metrics',
        'plot-problem-setup', 'plot-sgf-pipeline', 'plot-isotherm-comparison',
        'plot-mae-histogram',
        'plot-sdf-ablation',
        'train-sgf-ablations',
        'plot-ablation-studies',
        'plot-all-figures',
    ])
    args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    {
        'train-sgf': train_sgf_recfno,
        'train-all': train_recfno_benchmark,
        'train-external': train_external_baselines,
        'compare': run_comparison,
        'plot-case': plot_case_comparison,
        'plot-recfno-2x3': plot_recfno_2x3,
        'plot-3cases-6models': plot_three_cases_six_models,
        'plot-test-metrics': plot_test_metrics,
        'plot-problem-setup': plot_problem_setup,
        'plot-sgf-pipeline': plot_sgf_pipeline,
        'plot-isotherm-comparison': plot_isotherm_comparison,
        'plot-mae-histogram': plot_mae_histogram,
        'plot-sdf-ablation': plot_sdf_ablation,
        'train-sgf-ablations': train_sgf_ablations,
        'plot-ablation-studies': plot_ablation_studies,
        'plot-all-figures': plot_all_figures,
    }[args.command]()
