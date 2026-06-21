# -*- coding: utf-8 -*-
"""DEPRECATED: use heat2D_sgf_recfno.py instead."""
import warnings

warnings.warn(
    'heat2D_geo_enc_fno.py is deprecated; use heat2D_sgf_recfno.py',
    DeprecationWarning,
    stacklevel=1,
)

from heat2D_sgf_recfno import *  # noqa: F401,F403

if __name__ == '__main__':
    from heat2D_sgf_recfno import train

    train()
