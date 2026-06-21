# -*- coding: utf-8 -*-
"""Paths to official third-party repos used in the heat benchmark."""
from __future__ import annotations

import os
from pathlib import Path

from data.paths import external_root, repo_root

EXTERNAL_ROOT = str(external_root())
REPO_ROOT = str(repo_root())

REPOS = {
    'neuraloperator': os.path.join(EXTERNAL_ROOT, 'neuraloperator'),
    'geofno': os.path.join(EXTERNAL_ROOT, 'Geo-FNO'),
    'physics_informed': os.path.join(EXTERNAL_ROOT, 'physics_informed'),
}

REPO_URLS = {
    'GINO': 'https://github.com/neuraloperator/neuraloperator',
    'Geo-FNO': 'https://github.com/neuraloperator/Geo-FNO',
    'PINO': 'https://github.com/neuraloperator/physics_informed',
}


def ensure_repos():
    missing = [name for name, path in REPOS.items() if not os.path.isdir(path)]
    if missing:
        setup_script = Path(REPO_ROOT) / 'scripts' / 'setup_external.sh'
        lines = [
            'Missing external repos. Run:',
            f'  bash {setup_script}',
            '',
            'Or clone manually:',
        ]
        for name in missing:
            url = {
                'neuraloperator': REPO_URLS['GINO'],
                'geofno': REPO_URLS['Geo-FNO'],
                'physics_informed': REPO_URLS['PINO'],
            }[name]
            lines.append(f'  git clone {url} {REPOS[name]}')
        raise FileNotFoundError('\n'.join(lines))
