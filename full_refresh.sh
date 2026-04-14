#!/usr/bin/env bash
# Full clean rebuild: clear results/figures, rebuild C++ binary, run experiments, plot.
# Run from repo root (BA-Trees) or from BA-Trees/src (cwd must be one of those).

set -euo pipefail

# Resolve project root: if we are in src/, go up one level.
if [ -d "born_again_dp" ]; then
  cd ..
fi
PROJECT_ROOT=$(pwd)
echo "--- Starting full refresh at: $PROJECT_ROOT"

echo "--- Clearing old results and figures..."
for dir in \
  "${PROJECT_ROOT}/src/born_again_dp/results" \
  "${PROJECT_ROOT}/src/analysis_plots" \
  "${PROJECT_ROOT}/docs/mie424-figs"; do
  mkdir -p "$dir"
  find "$dir" -mindepth 1 -delete
done

echo "--- Rebuilding C++ solver..."
cd "${PROJECT_ROOT}/src/born_again_dp"
make clean && make

cd "${PROJECT_ROOT}"
# CVD-1 is already in datasets.dataset_names; run_experiments.py writes the full summary.csv.
echo "--- Running full experiment suite (all datasets, including CVD-1)..."
python3 src/run_experiments.py

echo "--- Generating analysis plots..."
python3 src/analyze_results.py

echo "--- Generating final paper figures..."
python3 src/plot_paper_figures.py

echo "--- DONE! Check docs/mie424-figs/ and src/analysis_plots/."
