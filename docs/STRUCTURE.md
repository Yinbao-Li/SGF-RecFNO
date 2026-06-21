# Repository structure

> **Maintainer:** Yinbao Li  
> **Primary method:** SGF-RecFNO (`model/sgf_recfno.py`)  
> **Based on:** [RecFNO](https://github.com/zhaoxiaoyu1995/RecFNO) (Zhao et al., 2023)

## Core packages

| Directory | Role |
|-----------|------|
| `model/sgf_recfno.py` | **SGF-RecFNO** — self-geometry feedback refinement (primary method) |
| `model/iso_recfno.py` | **IsoRecFNO** — isotherm geometry-aware branch |
| `model/fno.py` | RecFNO backbone (original implementation by Zhao et al.) |
| `data/` | Dataset loaders and path configuration |
| `utils/sgf_loss.py`, `iso_geometry.py` | SGF / isotherm losses and geometry utilities |
| `benchmark/` | Unified 6-model comparison (SGF-RecFNO listed first) |

## Experiment directories

| Directory | Task | SGF-RecFNO entry point |
|-----------|------|------------------------|
| `heat2D/` | 2D steady heat conduction | **`heat2D_sgf_recfno.py`** |
| `cylinder2D/` | 2D cylinder wake | Original RecFNO scripts (SGF not extended) |
| `darcy/` | Darcy flow | Same as above |
| `NOAA/` | Sea surface temperature | Same as above |

Benchmarks and primary experiments currently focus on the **heat2D/** heat conduction task.

## Benchmark pipeline

```
temperature6000.h5
        │
        ▼
  HeatDataset (25 sensors → 200×200)
        │
        ├── heat2D/heat2D_sgf_recfno.py     → SGF-RecFNO  ★
        ├── heat2D/heat2D_iso_recfno.py     → IsoRecFNO
        ├── heat2D/heat2D_fno.py            → RecFNO (baseline)
        └── benchmark/train_external.py     → PINO / Geo-FNO / GINO
        │
        ▼
  benchmark/run_comparison.py  →  comparison_table.csv / figures
```

## Model priority in code

`benchmark/config.py`:

```python
OUR_MODELS = ['SGF-RecFNO', 'IsoRecFNO']      # Yinbao Li
BASELINE_RECFNO = ['RecFNO']                   # Zhao et al.
EXTERNAL_MODELS = ['PINO', 'Geo-FNO', 'GINO']
PRIMARY_MODEL = 'SGF-RecFNO'
```

Evaluation and plotting iterate models in this order by default.

## Benchmark modules

| File | Purpose |
|------|---------|
| `registry.py` | Model registry and checkpoint loading |
| `config.py` | Shared constants (splits, geometry, paths) |
| `heat_adapters.py` | Input format conversion for external models |
| `external_models.py` | Wrappers around official GINO / Geo-FNO / PINO code |
| `external_loaders.py` | Dynamic import from `external/` repos |
| `metrics.py`, `isotherm_metrics.py` | Evaluation metrics |
| `visualize.py`, `plot_case_comparison.py` | Figures |
| `cli.py` | CLI entry points |

## External baselines

Third-party repos are **not** vendored in git. Clone them with:

```bash
bash scripts/setup_external.sh
```

Expected layout:

```
external/
├── neuraloperator/      # GINO
├── Geo-FNO/             # Geo-FNO
└── physics_informed/    # PINO
```

See [CITATION.md](../CITATION.md) for attribution.

## What is gitignored

- `**/logs/` — training outputs, checkpoints (`.pth`)
- `data/**/*.h5` — large datasets (`.gitkeep` is tracked)
- `external/` — cloned repos
- `__pycache__/`, virtual environments

Paper figures under `figures/` remain tracked.
