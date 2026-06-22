# SGF-RecFNO

**Self-Geometry Feedback RecFNO** — operator learning for heat field reconstruction from sparse observations.

> This repository extends [RecFNO](https://github.com/zhaoxiaoyu1995/RecFNO) (Zhao et al., 2023). The **primary contributions** by **Yinbao Li** are **IsoRecFNO** and **SGF-RecFNO**. **SGF-RecFNO** is the recommended method.

[![GitHub](https://img.shields.io/github/stars/Yinbao-Li/SGF-RecFNO?style=social)](https://github.com/Yinbao-Li/SGF-RecFNO)

---

## Methods

| Method | Source | Description |
|--------|--------|-------------|
| **SGF-RecFNO** | **Yinbao Li (this repo)** | Self-geometry feedback: multi-level isotherm SDFs from the predicted field, refined via a lightweight Fourier block |
| **IsoRecFNO** | **Yinbao Li (this repo)** | Geometry-aware branch with joint isotherm supervision |
| RecFNO | Zhao et al. (2023) | Original FNO reconstruction baseline |
| PINO / Geo-FNO / GINO | Third-party baselines | Official implementations for comparison |

Implementation: `model/sgf_recfno.py` · Loss: `utils/sgf_loss.py`

---

## Benchmark results (test 5000–5999, n=1000)

| Model | Test MAE ↓ | Notes |
|-------|-----------|-------|
| **SGF-RecFNO** | **0.00346 K** | **Best — pre-trained weights included** |
| PINO | 0.00351 K | Pre-trained |
| IsoRecFNO | 0.00512 K | Pre-trained |
| RecFNO | 0.00727 K | Pre-trained |
| Geo-FNO | 0.0373 K | Pre-trained |
| GINO | 0.3658 K | Pre-trained |

### Problem setup

25 sensors on a 200×200 field; Dirichlet / adiabatic boundaries.

![Problem setup](figures/setup/benchmark_problem_setup.png)

### Six-model comparison (3 test cases)

![Three cases × six models](figures/benchmark/three_cases_six_models.png)

### Isotherm geometry (contours vs. ground truth)

![Isotherm overlay](figures/benchmark/three_cases_isotherm_overlay.png)

### Test-set metrics & MAE distribution

| Bar charts (MAE, RMSE, PSNR, …) | Per-sample MAE (ECDF + ridge) |
|--------------------------------|-------------------------------|
| ![Metrics](figures/benchmark/test_metrics_comparison.png) | ![MAE distribution](figures/benchmark/test_mae_distribution.png) |

### SGF-RecFNO pipeline & SDF ablation

| Pipeline (sample #5500) | Inference-time SDF ablation |
|-------------------------|----------------------------|
| ![Pipeline](figures/method/sgf_pipeline_2x5.png) | ![SDF ablation](figures/method/sdf_ablation.png) |

More figures and regeneration commands: [figures/README.md](figures/README.md).

---

## Quick start

### 1. Clone (with checkpoints via Git LFS)

```bash
sudo apt install git-lfs    # once per machine
git lfs install
git clone https://github.com/Yinbao-Li/SGF-RecFNO.git
cd SGF-RecFNO
git lfs pull
```

Verify weights downloaded (each `.pth` should be **MB**, not ~130 bytes):

```bash
ls -lh checkpoints/*/*.pth
```

### 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 3. Data

**Option A — bundled splits (after `git lfs pull`):**

```bash
ls data/heat/train.h5 data/heat/val.h5 data/heat/test.h5
```

**Option B — download or generate:**

```bash
python scripts/generate_temperature6000.py
# or download from SharePoint (see data/heat/README.md)
```

See [data/heat/README.md](data/heat/README.md) for split indices (train 0–3999, val 4000–4999, test 5000–5999).

### 4. Evaluate pre-trained models (no training required)

```bash
make compare
make plot-case
make plot-figures    # regenerate README figures (needs GPU + data)
```

### 5. Train from scratch

```bash
make train-sgf        # SGF-RecFNO only
make train-all        # SGF-RecFNO + IsoRecFNO + RecFNO
make setup-external && make train-external   # external baselines
make train-ablations  # loss & SDF-depth ablations (9×300 ep)
```

---

## Repository layout

```
SGF-RecFNO/
├── checkpoints/           ← pre-trained weights (300 epochs, Git LFS)
├── figures/               ← README figures (setup / benchmark / method / ablation)
├── model/                 ← SGF-RecFNO, IsoRecFNO, RecFNO backbone
├── data/                  ← HeatDataset loaders
├── benchmark/             ← evaluation, plotting, ablation tools
├── heat2D/                ← training scripts
├── scripts/               ← data generation & setup
└── docs/                  ← structure notes
```

See [docs/STRUCTURE.md](docs/STRUCTURE.md), [benchmark/README.md](benchmark/README.md), and [checkpoints/README.md](checkpoints/README.md).

---

## Citation

See [CITATION.md](CITATION.md). Please cite RecFNO (Zhao et al.) when using the backbone, and credit SGF-RecFNO / IsoRecFNO (Yinbao Li) for the extensions.

Maintainer: **Yinbao Li** · [GitHub](https://github.com/Yinbao-Li)
