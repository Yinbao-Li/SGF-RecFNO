# Heat benchmark

Unified comparison centered on **SGF-RecFNO**, against IsoRecFNO, RecFNO, and the external baselines GINO / Geo-FNO / PINO.

## Quick commands

```bash
make train-sgf          # train SGF-RecFNO only (recommended)
make train-all          # SGF-RecFNO + IsoRecFNO + RecFNO (300 epochs each)
make setup-external     # clone official baseline repos
make train-external     # train GINO / Geo-FNO / PINO
make compare            # evaluate all models on test set (5000–5999)
make plot-case          # single-case error visualization
make plot-figures       # regenerate all README figures
make train-ablations    # loss & SDF-depth ablations (7×300 ep; full/K=4 reuse main ckpt)
make plot-ablations     # evaluate ablation checkpoints & plot MAE
```

> Run benchmark scripts from the `heat2D/` working directory, or use the Makefile / CLI (they set cwd automatically).

## Plotting scripts

| Script | Output | `--copy-figures` → |
|--------|--------|---------------------|
| `plot_problem_setup.py` | Benchmark geometry schematic | `figures/setup/` |
| `plot_three_cases_six_models.py` | 3×6 prediction + error grid | `figures/benchmark/` |
| `plot_isotherm_comparison.py` | Isotherm contour overlays | `figures/benchmark/` |
| `plot_test_metrics.py` | Test metric bar charts + CSV | `figures/benchmark/` |
| `plot_mae_histogram.py` | Per-sample MAE ECDF + ridge | `figures/benchmark/` |
| `plot_sgf_pipeline.py` | SGF 2×5 pipeline figure | `figures/benchmark/` |
| `plot_sdf_ablation.py` | Inference-time SDF ablation | `figures/ablation/` |
| `plot_ablation_studies.py` | Training ablation MAE plots | `figures/ablation/` |
| `plot_all_figures.py` | Run all of the above | all categories |

Example (from `heat2D/`):

```bash
RECFNO_DATA_ROOT=../data python ../benchmark/plot_sgf_pipeline.py --sample-idx 5500 --copy-figures
```

## Models

| Model | Type | Input | Source |
|-------|------|-------|--------|
| **SGF-RecFNO** | sensor | 25 sensor values | Yinbao Li (this repo) |
| **IsoRecFNO** | sensor | 25 sensor values | Yinbao Li (this repo) |
| RecFNO | sensor | 25 sensor values | Zhao et al. (2023) |
| PINO | grid | 4-channel interpolated grid | [physics_informed](https://github.com/neuraloperator/physics_informed) |
| Geo-FNO | sensor | 25 sensors + coordinates | [Geo-FNO](https://github.com/neuraloperator/Geo-FNO) |
| GINO | sensor | 25 sensors + coordinates | [neuraloperator](https://github.com/neuraloperator/neuraloperator) |

## Ablation studies

**Loss components** (`heat2D/run_sgf_ablations.py --study loss`): full / w/o Field / w/o Grad / w/o SDF / w/o SSIM.

**SDF depth K** (`--study quantile`): K = 1, 2, 4, 8 isotherm levels; quantiles \(q_i = i/(K+1)\).

Checkpoints: `heat2D/logs/ckpt/ablation_loss_*` and `ablation_k*`. Evaluation: `make plot-ablations`.

## Outputs

Runtime results: `heat2D/logs/benchmark_comparison/` (heat), `heat2D/logs/fluid_benchmark/` (fluid).

## Fluid benchmark

```bash
make train-fluid-resume     # 7 models × 2 tasks → 500 epochs
make compare-fluid
make plot-fluid-figures
```

| Script | Output |
|--------|--------|
| `plot_all_fluid_figures.py` | All `figures/fluid/*` + `logs/fluid_benchmark/` tables |
| `fluid2D/run_fluid_benchmark.py` | Train + evaluate |
| `fluid2D/run_fluid_resume.py` | Resume 300→500 (Iso/K8 fresh) |

Geometry presets: `benchmark/fluid_config.py` · Cylinder: wake R1 · Darcy: dual \(p + |\nabla p|\).

## Further reading

- [README](../README.md) — overview and quick start  
- [docs/STRUCTURE.md](../docs/STRUCTURE.md) — module layout  
- [CITATION.md](../CITATION.md) — how to cite this work
