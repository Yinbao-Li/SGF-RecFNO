# Figures

Paper-ready figures for the README and documentation. Regenerate from `heat2D/` after
`pip install -e .` and `git lfs pull` (checkpoints + optional data splits).

| File | Description | Script |
|------|-------------|--------|
| `setup/benchmark_problem_setup.png` | Sensors, BCs, sample temperature field | `plot_problem_setup.py --copy-figures` |
| `benchmark/three_cases_six_models.png` | 3 cases × 6 models (pred + error) | `plot_three_cases_six_models.py --copy-figures` |
| `benchmark/three_cases_isotherm_overlay.png` | Isotherm contours vs. GT (3 cases) | `plot_isotherm_comparison.py --copy-figures` |
| `benchmark/test_metrics_comparison.png` | Test-set metric bar charts | `plot_test_metrics.py --copy-figures` |
| `benchmark/test_mae_distribution.png` | Per-sample MAE ECDF + ridge plot | `plot_mae_histogram.py --copy-figures` |
| `method/sgf_pipeline_2x5.png` | SGF-RecFNO pipeline (2×5) | `plot_sgf_pipeline.py --copy-figures` |
| `method/sdf_ablation.png` | Inference-time SDF ablation | `plot_sdf_ablation.py --copy-figures` |
| `ablation/ablation_loss_mae.png` | Loss-component ablation (test MAE) | `plot_ablation_studies.py --study loss --copy-figures` |
| `ablation/ablation_quantile_k_mae.png` | SDF depth K ablation curve | `plot_ablation_studies.py --study quantile --copy-figures` |

One-shot (from repo root):

```bash
make plot-figures
```

Or from `heat2D/`:

```bash
export RECFNO_DATA_ROOT=../data   # if data live outside repo
make -C .. plot-figures
```
