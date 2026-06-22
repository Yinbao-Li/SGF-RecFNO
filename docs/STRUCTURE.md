# Repository structure

> **Maintainer:** Yinbao Li  
> **Primary method:** SGF-RecFNO  
> **Task:** 2D steady heat conduction (sparse sensors → full field)

## Layout

| Path | Description |
|------|-------------|
| `checkpoints/` | Pre-trained 300-epoch weights for all 6 benchmark models (Git LFS) |
| `figures/` | README figures (`setup/`, `benchmark/`, `method/`, `ablation/`) |
| `model/sgf_recfno.py` | **SGF-RecFNO** (primary) |
| `model/iso_recfno.py` | **IsoRecFNO** |
| `model/fno.py` | RecFNO backbone (Zhao et al.) |
| `data/dataset.py` | `HeatDataset`, `HeatInterpolDataset` |
| `benchmark/` | Evaluation, plotting, ablation (`ablation.py`, `visualize.py`) |
| `heat2D/` | Training entry scripts (`run_sgf_ablations.py`) |
| `utils/ablation_config.py` | Loss & quantile-K ablation configs |

## Workflow

```
temperature6000.h5  (or train/val/test.h5)
       │
       ▼
 HeatDataset (25 sensors → 200×200)
       │
       ├── heat2D/heat2D_sgf_recfno.py      → train SGF-RecFNO
       ├── heat2D/run_benchmark_300epoch.py
       ├── heat2D/run_sgf_ablations.py      → ablation training
       └── benchmark/run_comparison.py      → load checkpoints/ → metrics
```

## Checkpoints

Evaluation loads from `checkpoints/` (see `benchmark/config.py`: `CKPT_ROOT = '../checkpoints'`).

New training saves to `heat2D/logs/ckpt/` (gitignored).

## Figures

Regenerate committed figures with `make plot-figures` or `benchmark/plot_all_figures.py`.  
See [figures/README.md](../figures/README.md).

## External baselines

```bash
bash scripts/setup_external.sh
```

See [CITATION.md](../CITATION.md) for attribution.
