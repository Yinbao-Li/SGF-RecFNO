# Heat benchmark

Unified comparison centered on **SGF-RecFNO**, against IsoRecFNO, RecFNO, and the external baselines GINO / Geo-FNO / PINO.

## Quick commands

```bash
make train-sgf        # train SGF-RecFNO only (recommended)
make train-all        # SGF-RecFNO + IsoRecFNO + RecFNO (300 epochs each)
make setup-external   # clone official baseline repos
make train-external   # train GINO / Geo-FNO / PINO
make compare          # evaluate all models on test set (5000–5999)
make plot-case        # single-case error visualization
```

Or from repo root after `pip install -e .`:

```bash
sgf-recfno-train
sgf-recfno-compare --out-dir logs/benchmark_comparison
sgf-recfno-plot-case
```

> Run benchmark scripts from the `heat2D/` working directory, or use the Makefile / CLI above (they set the cwd automatically).

## Models

| Model | Type | Input | Source |
|-------|------|-------|--------|
| **SGF-RecFNO** | sensor | 25 sensor values | Yinbao Li (this repo) |
| **IsoRecFNO** | sensor | 25 sensor values | Yinbao Li (this repo) |
| RecFNO | sensor | 25 sensor values | Zhao et al. (2023) |
| PINO | grid | 4-channel interpolated grid | [physics_informed](https://github.com/neuraloperator/physics_informed) |
| Geo-FNO | sensor | 25 sensors + coordinates | [Geo-FNO](https://github.com/neuraloperator/Geo-FNO) |
| GINO | sensor | 25 sensors + coordinates | [neuraloperator](https://github.com/neuraloperator/neuraloperator) |

## Outputs

Results are written to `heat2D/logs/benchmark_comparison/`:

- `comparison_table.csv` — quantitative metrics
- `comparison_table.tex` — LaTeX table
- `comparison_results.json` — full results
- Spectrum / bar charts and per-model field visualizations

## Further reading

- [README](../README.md) — overview and quick start  
- [docs/STRUCTURE.md](../docs/STRUCTURE.md) — module layout  
- [CITATION.md](../CITATION.md) — how to cite this work
