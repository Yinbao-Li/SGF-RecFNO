# Pre-trained checkpoints (300 epochs, heat conduction)

Best validation checkpoints for the unified benchmark. Loaded automatically by `benchmark/run_comparison.py`.

| Directory | Model | Test MAE (K) |
|-----------|-------|--------------|
| `benchmark_sgf-recfno_300/` | **SGF-RecFNO** (primary) | 0.00346 |
| `benchmark_isorecfno_300/` | IsoRecFNO | 0.00512 |
| `benchmark_recfno_300/` | RecFNO (Zhao et al.) | 0.00727 |
| `benchmark_pino_300/` | PINO | 0.00351 |
| `benchmark_geofno_300/` | Geo-FNO | 0.0373 |
| `benchmark_gino_300/` | GINO | 0.3658 |

> Files over 100 MB are stored with **Git LFS**. After cloning:
> ```bash
> git lfs install
> git lfs pull
> ```

New training runs save to `heat2D/logs/ckpt/` by default.
