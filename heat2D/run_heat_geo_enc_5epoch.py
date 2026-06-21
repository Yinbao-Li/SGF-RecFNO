#!/usr/bin/env python3
"""DEPRECATED: use run_heat_sgf_5epoch.py instead."""
import warnings

warnings.warn(
    'run_heat_geo_enc_5epoch.py is deprecated; use run_heat_sgf_5epoch.py',
    DeprecationWarning,
    stacklevel=1,
)

from run_heat_sgf_5epoch import *  # noqa: F401,F403

if __name__ == '__main__':
    from run_heat_sgf_5epoch import train_5_epochs, visualize_test

    args, history, _ = train_5_epochs()
    print('\n=== 训练历史 ===', flush=True)
    for h in history:
        print(
            f"  epoch {h['epoch']}: train={h['train_loss']:.6f}, val={h['val_loss']:.6f}, "
            f"field={h['val_field']:.6f}, sdf={h['val_sdf']:.6f}",
            flush=True,
        )
    ckpt, vis_path, hist_path, summary = visualize_test(args, history)
    print(f'\n可视化: {args.fig_path}', flush=True)
    print(f'  - {hist_path}', flush=True)
    print(f'  - {vis_path}', flush=True)
    print(f'  - {summary}', flush=True)
