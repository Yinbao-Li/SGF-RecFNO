.PHONY: install setup-external train-sgf train-all train-external train-ablations train-resume plot-ablations plot-figures compare plot-benchmark-figures train-fluid train-fluid-resume compare-fluid plot-fluid-figures export-fluid plot-case clean-pyc help

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

train-resume:
	cd heat2D && RECFNO_DATA_ROOT=../data python run_benchmark_resume.py

plot-ablations:
	cd heat2D && RECFNO_DATA_ROOT=../data python ../benchmark/plot_ablation_studies.py --copy-figures

plot-figures:
	cd heat2D && RECFNO_DATA_ROOT=../data python ../benchmark/plot_all_figures.py

compare:
	cd heat2D && RECFNO_DATA_ROOT=../data python ../benchmark/run_comparison.py --out-dir logs/benchmark_comparison
	cd heat2D && RECFNO_DATA_ROOT=../data python ../benchmark/plot_spectrum_error.py --out-dir logs/benchmark_comparison --use-cache
	cd heat2D && RECFNO_DATA_ROOT=../data python ../benchmark/plot_mae_histogram.py --out-dir logs/benchmark_comparison

plot-benchmark-figures:
	cd heat2D && RECFNO_DATA_ROOT=../data python ../benchmark/plot_spectrum_error.py --out-dir logs/benchmark_comparison --use-cache --copy-figures
	cd heat2D && RECFNO_DATA_ROOT=../data python ../benchmark/plot_mae_histogram.py --out-dir logs/benchmark_comparison --copy-figures

train-fluid:
	cd fluid2D && RECFNO_DATA_ROOT=../../data python run_fluid_benchmark.py --task all --skip-existing

train-fluid-resume:
	cd fluid2D && RECFNO_DATA_ROOT=../../data python run_fluid_resume.py --task all

compare-fluid:
	cd fluid2D && RECFNO_DATA_ROOT=../../data python run_fluid_benchmark.py --task all --compare-only

plot-fluid-figures:
	cd heat2D && RECFNO_DATA_ROOT=../../data python ../benchmark/plot_all_fluid_figures.py --copy-figures

export-fluid:
	cd heat2D && RECFNO_DATA_ROOT=../../data python ../benchmark/export_fluid_benchmark.py --all

plot-case:
	cd heat2D && python ../benchmark/plot_case_comparison.py

clean-pyc:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "SGF-RecFNO Makefile targets:"
	@echo "  Heat: train-sgf, train-all, compare, plot-figures, plot-case"
	@echo "  Fluid: train-fluid, train-fluid-resume, compare-fluid, plot-fluid-figures"
	@echo "  Ablation: train-ablations, plot-ablations"

train-recfno: train-all
