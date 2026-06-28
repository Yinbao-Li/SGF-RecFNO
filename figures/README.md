# Figures

Paper-ready figures for the README. Regenerate from `heat2D/` after `pip install -e .` and `git lfs pull`.

## Heat (`figures/benchmark/`, `figures/setup/`, `figures/ablation/`)

| File | Description | Script |
|------|-------------|--------|
| `setup/benchmark_problem_setup.png` | Sensors, BCs, sample field | `plot_problem_setup.py --copy-figures` |
| `benchmark/three_cases_six_models.png` | 3 cases × 6 models | `plot_three_cases_six_models.py --copy-figures` |
| `benchmark/three_cases_isotherm_overlay.png` | Isotherm contours vs GT | `plot_isotherm_comparison.py --copy-figures` |
| `benchmark/test_metrics_comparison.png` | Test metric bar charts | `plot_test_metrics.py --copy-figures` |
| `benchmark/test_mae_distribution.png` | Per-sample MAE ECDF | `plot_mae_histogram.py --copy-figures` |
| `benchmark/sgf_pipeline_2x5.png` | SGF pipeline (sample #5500) | `plot_sgf_pipeline.py --copy-figures` |
| `ablation/sdf_ablation.png` | Inference-time SDF ablation | `plot_sdf_ablation.py --copy-figures` |
| `ablation/ablation_loss_mae.png` | Loss ablation | `plot_ablation_studies.py --study loss --copy-figures` |
| `ablation/ablation_quantile_k_mae.png` | SDF depth K ablation | `plot_ablation_studies.py --study quantile --copy-figures` |

```bash
make plot-figures    # heat only
```

## Fluid (`figures/fluid/`)

| File | Description |
|------|-------------|
| `{cylinder,darcy}_problem_setup.png` | Sensors + sample GT field |
| `{cylinder,darcy}_three_cases.png` | 3 test cases × 5 models (pred + error) |
| `{cylinder,darcy}_geometry_overlay.png` | Task-specific level-set contours |
| `{cylinder,darcy}_test_metrics.png` | MAE / PSNR / SSIM bar charts |
| `{cylinder,darcy}_mae_distribution.png` | Per-sample MAE ECDF |
| `{cylinder,darcy}_sgf_vs_recfno.png` | SGF vs RecFNO qualitative |

```bash
export RECFNO_DATA_ROOT=../../data
make plot-fluid-figures
```

Tables: `heat2D/logs/fluid_benchmark/comparison_table.csv`

## Method (`figures/method/`)

Architecture diagrams (manual / PDF export).
