#!/usr/bin/env python3
"""Regenerate all README figures and copy them into ``figures/``."""
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_HEAT2D = os.path.join(_ROOT, 'heat2D')
_SCRIPT = os.path.join(_ROOT, 'benchmark')


def _run(name, script, extra=None):
    cmd = [sys.executable, os.path.join(_SCRIPT, script), '--copy-figures']
    if extra:
        cmd.extend(extra)
    print(f'\n>>> {name}', flush=True)
    env = os.environ.copy()
    if 'RECFNO_DATA_ROOT' not in env:
        env['RECFNO_DATA_ROOT'] = os.path.join(_ROOT, 'data')
    subprocess.check_call(cmd, cwd=_HEAT2D, env=env)


def main():
    jobs = [
        ('Problem setup', 'plot_problem_setup.py', None),
        ('3 cases × 6 models', 'plot_three_cases_six_models.py', None),
        ('Isotherm overlay', 'plot_isotherm_comparison.py', None),
        ('Test metrics', 'plot_test_metrics.py', None),
        ('MAE distribution', 'plot_mae_histogram.py', ['--use-cache']),
        ('SGF pipeline', 'plot_sgf_pipeline.py', ['--sample-idx', '5500']),
        ('SDF inference ablation', 'plot_sdf_ablation.py', None),
        ('Training ablations', 'plot_ablation_studies.py', ['--study', 'all']),
    ]
    for name, script, extra in jobs:
        _run(name, script, extra)
    print('\nAll figures copied under figures/. See figures/README.md', flush=True)


if __name__ == '__main__':
    main()
