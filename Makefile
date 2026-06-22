.PHONY: install setup-external train-sgf train-all train-external train-ablations plot-ablations plot-figures compare plot-case clean-pyc help

install:
	pip install -e .

setup-external:
	bash scripts/setup_external.sh

train-sgf:
	cd heat2D && python heat2D_sgf_recfno.py

train-all:
	cd heat2D && python run_benchmark_300epoch.py

train-external:
	cd heat2D && python ../benchmark/train_external.py

train-ablations:
	cd heat2D && python run_sgf_ablations.py --study all

plot-ablations:
	cd heat2D && RECFNO_DATA_ROOT=../data python ../benchmark/plot_ablation_studies.py --copy-figures

plot-figures:
	cd heat2D && RECFNO_DATA_ROOT=../data python ../benchmark/plot_all_figures.py

compare:
	cd heat2D && python ../benchmark/run_comparison.py --out-dir logs/benchmark_comparison

plot-case:
	cd heat2D && python ../benchmark/plot_case_comparison.py

clean-pyc:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "SGF-RecFNO Makefile targets:"
	@echo "  make install          - editable pip install"
	@echo "  make train-sgf        - train SGF-RecFNO (recommended)"
	@echo "  make train-all        - train SGF / Iso / RecFNO baselines (300 ep)"
	@echo "  make setup-external   - clone GINO/Geo-FNO/PINO repos"
	@echo "  make train-external   - train external baselines (300 ep)"
	@echo "  make train-ablations  - loss & quantile-K ablations (7×300 ep)"
	@echo "  make plot-ablations   - evaluate ablations & plot MAE figures"
	@echo "  make plot-figures     - regenerate all README figures"
	@echo "  make compare          - run unified benchmark evaluation"
	@echo "  make plot-case        - single-case error visualization"

# backward-compatible alias
train-recfno: train-all
