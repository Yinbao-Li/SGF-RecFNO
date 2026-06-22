# -*- coding: utf-8 -*-
"""Canonical paths for README / paper figures under ``figures/``."""
from __future__ import annotations

import os
import shutil

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGURES_ROOT = os.path.join(_REPO_ROOT, 'figures')

# category -> subdirectory name
FIGURE_CATEGORIES = {
    'setup': 'setup',
    'benchmark': 'benchmark',
    'method': 'method',
    'ablation': 'ablation',
}


def figures_dir(category: str = 'benchmark') -> str:
    sub = FIGURE_CATEGORIES.get(category, category)
    path = os.path.join(FIGURES_ROOT, sub)
    os.makedirs(path, exist_ok=True)
    return path


def copy_to_figures(src: str, filename: str, category: str = 'benchmark') -> str:
    """Copy a generated PNG into ``figures/<category>/``."""
    dst = os.path.join(figures_dir(category), filename)
    shutil.copy2(src, dst)
    return dst
