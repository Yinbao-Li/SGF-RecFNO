# Repository structure

> **Maintainer:** Yinbao Li  
> **Primary method:** SGF-RecFNO  
> **Task:** 2D steady heat conduction (sparse sensors → full field)

## Layout

| Path | Description |
|------|-------------|
| `checkpoints/` | Pre-trained 300-epoch weights for all 6 benchmark models (Git LFS) |
| `model/sgf_recfno.py` | **SGF-RecFNO** (primary) |
| `model/iso_recfno.py` | **IsoRecFNO** |
| `model/fno.py` | RecFNO backbone (Zhao et al.) |
| `data/dataset.py` | `HeatDataset`, `HeatInterpolDataset` |
| `benchmark/` | Evaluation vs. RecFNO / PINO / Geo-FNO / GINO |
| `heat2D/` | Training entry scripts |

## Workflow

```
temperature6000.h5
       │
       ▼
 HeatDataset (25 sensors → 200×200)
       │
       ├── heat2D/heat2D_sgf_recfno.py   → train SGF-RecFNO
       ├── heat2D/run_benchmark_300epoch.py
       └── benchmark/run_comparison.py   → load checkpoints/ → metrics
```

## Checkpoints

Evaluation loads from `checkpoints/` (see `benchmark/config.py`: `CKPT_ROOT = '../checkpoints'`).

New training saves to `heat2D/logs/ckpt/` (gitignored).

## External baselines

```bash
bash scripts/setup_external.sh
```

See [CITATION.md](../CITATION.md) for attribution.
