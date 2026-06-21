# -*- coding: utf-8 -*-
"""Ensure imports and working directory for RecFNO scripts."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_repo_context(heat2d_cwd: bool = False) -> Path:
    """
    Add repository root to sys.path.
    If heat2d_cwd=True, also chdir to heat2D/ (required for logs/ relative paths).
    """
    root = Path(__file__).resolve().parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if heat2d_cwd:
        heat2d = root / 'heat2D'
        if os.getcwd() != str(heat2d):
            os.chdir(heat2d)
    return root
