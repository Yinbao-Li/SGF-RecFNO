# SGF-RecFNO

**Self-Geometry Feedback RecFNO** — operator learning for heat field reconstruction from sparse observations.

> This repository extends [RecFNO](https://github.com/zhaoxiaoyu1995/RecFNO) (Zhao et al., 2023). The **primary contributions** by **Yinbao Li** are **IsoRecFNO** and **SGF-RecFNO**. **SGF-RecFNO** is the recommended method.

[![GitHub](https://img.shields.io/github/stars/Yinbao-Li/RecFNO?style=social)](https://github.com/Yinbao-Li/RecFNO)

---

## Methods

| Method | Source | Description |
|--------|--------|-------------|
| **SGF-RecFNO** | **Yinbao Li (this repo)** | Self-geometry feedback: multi-level isotherm SDFs extracted from the predicted field, refined via a lightweight Fourier block in a closed loop |
| **IsoRecFNO** | **Yinbao Li (this repo)** | Geometry-aware branch: predicts isotherm geometry from a coarse field with joint supervision |
| RecFNO | Zhao et al. (2023) | Original Fourier neural operator reconstruction baseline |
| PINO / Geo-FNO / GINO | Official third-party implementations | External baselines for comparison |

### SGF-RecFNO at a glance

1. **RecFNO backbone** — 25 sensors → coarse 200×200 temperature field  
2. **Self-geometry extraction** — multi-level isotherm SDFs computed from the prediction only (no ground-truth geometry)  
3. **Fourier refinement** — `[coarse field, SDF]` fed into a lightweight spectral block for closed-loop correction  

Implementation: `model/sgf_recfno.py` · Loss: `utils/sgf_loss.py`

---

## Heat conduction benchmark (test indices 5000–5999)

Steady 2D heat conduction: 6000 samples, 25 sensors, 200×200 grid, 300 training epochs.

| Model | Test MAE ↓ | Inference (ms) | Notes |
|-------|-----------|----------------|-------|
| **SGF-RecFNO** | **0.00346 K** | 2.89 | **Best in this repo** |
| PINO | 0.00351 K | 1.05 | External baseline |
| IsoRecFNO | 0.00512 K | 3.41 | This repo |
| RecFNO | 0.00727 K | 1.64 | Original paper method |
| Geo-FNO | 0.0373 K | 2.38 | External baseline |
| GINO | 0.3658 K | 10.31 | External baseline |

Single-case error comparison (sample #5500):

![SGF-RecFNO benchmark](figures/sgf_benchmark_case5500.png)

---

## Quick start

### 1. Install

```bash
git clone https://github.com/Yinbao-Li/RecFNO.git
cd RecFNO
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. Data

Download the [heat conduction dataset](https://nudteducn-my.sharepoint.com/:f:/g/personal/zhaoxiaoyu13_nudt_edu_cn/ElHePUBS_gpIjr240jcrdZ4BhMKsA3DBeYWLS6Roq_52TA?e=RZKOh5) or generate it locally:

```bash
python scripts/generate_temperature6000.py
# writes to $RECFNO_DATA_ROOT/heat/temperature6000.h5 by default
```

### 3. Train SGF-RecFNO (recommended)

```bash
cd heat2D
python heat2D_sgf_recfno.py       # single-model full training
# or batch training (SGF-RecFNO + IsoRecFNO + RecFNO baseline):
python run_benchmark_300epoch.py
```

Equivalent Makefile targets:

```bash
make train-sgf    # SGF-RecFNO only
make train-all    # SGF-RecFNO + IsoRecFNO + RecFNO
```

### 4. Compare against external baselines

```bash
make setup-external   # clone GINO / Geo-FNO / PINO repos
make train-external
make compare
make plot-case
```

CLI entry points (after `pip install -e .`):

```bash
sgf-recfno-train
sgf-recfno-compare --out-dir logs/benchmark_comparison
sgf-recfno-plot-case
```

Results are written to `heat2D/logs/benchmark_comparison/`.

---

## Repository layout

```
RecFNO/
├── model/
│   ├── sgf_recfno.py      ← SGF-RecFNO (primary)
│   ├── iso_recfno.py      ← IsoRecFNO
│   └── fno.py             ← RecFNO backbone (Zhao et al.)
├── utils/
│   ├── sgf_loss.py        ← SGF loss
│   └── iso_geometry.py    ← isotherm geometry utilities
├── benchmark/             ← unified 6-model comparison framework
├── heat2D/                ← heat conduction experiments
└── …                      ← original RecFNO scripts (cylinder2D / darcy / NOAA)
```

See [docs/STRUCTURE.md](docs/STRUCTURE.md) for module relationships.

---

## Acknowledgements & citation

### Contributions in this repository (Yinbao Li)

- **SGF-RecFNO** — self-geometry feedback Fourier operator  
- **IsoRecFNO** — isotherm geometry-aware RecFNO  
- Unified heat-conduction benchmark (vs. RecFNO / PINO / Geo-FNO / GINO)

### Prior work

This code builds on the [official RecFNO implementation](https://github.com/zhaoxiaoyu1995/RecFNO) and the [Fourier Neural Operator](https://github.com/neuraloperator/neuraloperator). Please cite RecFNO when using any component of this codebase:

```bibtex
@misc{zhao2023recfno,
  doi = {10.48550/ARXIV.2302.09808},
  author = {Zhao, Xiaoyu and Chen, Xiaoqian and Gong, Zhiqiang and Zhou, Weien and Yao, Wen and Zhang, Yunyang},
  title = {RecFNO: a resolution-invariant flow and heat field reconstruction method from sparse observations via Fourier neural operator},
  year = {2023},
}
```

For SGF-RecFNO / IsoRecFNO, see [CITATION.md](CITATION.md). A formal preprint link will be added upon publication.

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `RECFNO_DATA_ROOT` | Dataset root directory |
| `RECFNO_EXTERNAL_ROOT` | External baseline repos directory |
| `RECFNO_REPO_ROOT` | Repository root (auto-detected if unset) |

Copy `.env.example` to `.env` and adjust paths as needed.

---

## License

Follows the license of the original RecFNO repository. New modules (IsoRecFNO, SGF-RecFNO, benchmark) are released under the same open-source terms.

Maintainer: **Yinbao Li** · [GitHub](https://github.com/Yinbao-Li)
