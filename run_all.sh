#!/bin/sh
# Reproduce every number, table and figure. Runs one single-threaded job per
# core (set JOBS to the number of cores to use).
set -e
cd "$(dirname "$0")/src"
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 test_gradients.py
python3 make_jobs.py
xargs -P "${JOBS:-8}" -L 1 python3 bench.py < ../results/jobs.txt > ../results/run_log.txt
python3 aggregate.py
python3 make_figures.py
python3 gen_tex.py
