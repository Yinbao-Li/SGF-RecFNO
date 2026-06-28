#!/usr/bin/env python3
"""Export fluid benchmark tables and figures (wrapper)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    p = argparse.ArgumentParser(description='Export fluid benchmark (tables + figures)')
    p.add_argument('--compare', action='store_true', help='re-run evaluation')
    p.add_argument('--copy-figures', action='store_true')
    p.add_argument('--all', action='store_true')
    args = p.parse_args()
    if args.all:
        args.compare = args.copy_figures = True

    cmd = [sys.executable, os.path.join(_ROOT, 'benchmark', 'plot_all_fluid_figures.py')]
    if not args.compare:
        cmd.append('--skip-eval')
    if args.copy_figures:
        cmd.append('--copy-figures')

    heat2d = os.path.join(_ROOT, 'heat2D')
    env = os.environ.copy()
    if 'RECFNO_DATA_ROOT' not in env:
        env['RECFNO_DATA_ROOT'] = os.path.join(_ROOT, '..', 'data')
    subprocess.check_call(cmd, cwd=heat2d, env=env)


if __name__ == '__main__':
    main()
