# Continual Learning for Multi-Band Antenna Surrogate Modelling

Code, data and results for a benchmark study of continual learning applied to machine-learning surrogates of microstrip patch antennas.

Published antenna surrogates are trained on pooled datasets covering every operating band at once. Real design campaigns do not produce data that way — simulations arrive one band, and one geometric corner, at a time. This repository formulates that setting as a continual-learning problem, provides a benchmark built on 27,026 CST simulations, and evaluates seven methods over five random seeds.

## Headline result

Every standard surrogate catastrophically forgets under sequential multi-band training. Refit tree ensembles achieve the *highest* per-task accuracy of any method tested (R² ≈ 0.90) while retaining nothing (backward transfer −4.67). The canonical continual-learning remedies do not rescue them: Elastic Weight Consolidation degrades to −2.84 ± 1.92, and a 10 % replay buffer to −1.70 ± 0.30.

Progressive Networks solve the problem, reaching **R² = 0.9051 ± 0.0018 with exact zero forgetting** — matching tree-ensemble per-task accuracy while preserving every band seen so far. Their lateral connections grow quadratically in task count. A nested-learning alternative with a frozen shared prior (NestAnt-CMS) reaches 0.8606 ± 0.0061 at strictly linear growth, and is shown to accelerate convergence without improving the converged optimum.


| Method                   | BWT ↑               | Final R² ↑          | Diagonal R² ↑       | Params (T=5) |
| ------------------------ | ------------------- | ------------------- | ------------------- | ------------ |
| **Progressive Networks** | **0.0000 ± 0.0000** | **0.9051 ± 0.0018** | **0.9051 ± 0.0018** | 120,655      |
| NestAnt-CMS              | −0.0010 ± 0.0009    | 0.8606 ± 0.0061     | 0.8614 ± 0.0064     | **42,719**   |
| Independent per-task MLP | 0.0000 ± 0.0000     | 0.8590 ± 0.0071     | 0.8590 ± 0.0071     | 56,655       |
| Replay (10 % buffer)     | −1.7037 ± 0.3025    | −0.6271 ± 0.2404    | 0.7358 ± 0.0266     | 42,719       |
| EWC (λ = 10⁶)            | −2.8371 ± 1.9244    | −2.3028 ± 1.4294    | −0.0331 ± 0.2751    | 42,719       |
| Random Forest (refit)    | −4.6614 ± 0.0097    | −2.8292 ± 0.0076    | 0.9000 ± 0.0004     | —            |
| XGBoost (refit)          | −4.6714 ± 0.0000    | −2.8407 ± 0.0000    | 0.8963 ± 0.0000     | —            |


Five tasks, five seeds, 300 epochs per task. R² is measured on S₁₁.



Reading down a column of the matrices below shows what happens to a band as later bands arrive. For the two architectural methods each column is constant — a band, once learned, is never disturbed. For the refit ensemble every off-diagonal entry is negative.



## Benchmark definition

**Data.** The AntennaCAT microstrip-patch dataset (27,026 CST simulations, GPL-2.0). Inputs are patch width, patch length, ground-plane side and feed-notch depth; substrate height is fixed at 1.6 mm. Targets are S₁₁ at the design frequency (dB), peak total gain (dB) and local resonance frequency (log₁₀ Hz).

**Task sequence.** The 1.5 GHz and 5.8 GHz bands are each split at the patch-length median into small and large geometry sub-tasks; 2.4 GHz remains a single task.


| Task | Band    | Geometry | Train samples |
| ---- | ------- | -------- | ------------- |
| T₁   | 1.5 GHz | small    | 3,840         |
| T₂   | 1.5 GHz | large    | 4,224         |
| T₃   | 2.4 GHz | —        | 4,926         |
| T₄   | 5.8 GHz | small    | 3,839         |
| T₅   | 5.8 GHz | large    | 4,791         |


Every transition changes the frequency, the geometric range, or both.

**Metrics.** After each stage *i*, R² is recorded on every test subset *j*, giving a 5×5 accuracy matrix. From it: backward transfer (BWT), forgetting measure (FM), diagonal mean (per-task fitting capacity) and final-row mean (retained cross-task accuracy). Definitions follow Lopez-Paz & Ranzato (2017) and Chaudhry et al. (2018).

## Installation

Python 3.10 or newer. No GPU and no deep-learning framework required.

```bash
git clone <repository-url> && cd nestant
python3 -m venv venv && source venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib xgboost

```

## Reproducing the results

Verify the hand-derived gradients first:

```bash
cd src
python3 test_nestant_grads.py    # relative error < 3e-7
python3 test_cms_grads.py        # relative error < 3e-8

```

Run the full benchmark — seven methods, five seeds, 300 epochs per task:

```bash
for s in 0 1 2 3 4; do
  for m in cms_freeze indep replay ewc; do
    python3 continual5_runner.py --method $m --seed $s --epochs 300
  done
  python3 progressive_nets.py --seed $s --epochs 300
  python3 continual5_runner.py --method xgb --seed $s
  python3 continual5_runner.py --method rf  --seed $s
done

```

Aggregate and regenerate every figure:

```bash
python3 aggregate_canonical.py    # -> results/multiseed_summary_5task.json
python3 make_figures_v2.py        # -> figures/fig1 .. fig5
python3 mk_heat.py                # -> figures/fig6_heatmaps.png

```

Approximately 25 minutes on a single CPU core.

### Supplementary studies

```bash
# Does the benefit of the shared prior survive longer training?
python3 convergence_study.py --epochs 30 100 300 --seeds 0 1 2

# Does it matter which task the frozen prior is learned from?
python3 ordering_ablation.py --n_orders 6 --seeds 0 1 2 --epochs 300

```

## Repository layout

```
data/       AntennaCAT dataset (GPL-2.0) with upstream licence and README
src/        Implementations, runners, ablations, gradient checks, plotting
results/    Per-seed 5x5 accuracy matrices and aggregated JSON
figures/    Generated figures

```

### Source files


| File                               | Purpose                                          |
| ---------------------------------- | ------------------------------------------------ |
| `data.py`, `data5.py`              | Loading, preprocessing, task partitioning        |
| `metrics.py`                       | R², BWT, forgetting measure                      |
| `nestant.py`                       | Two-level nested architecture                    |
| `nestant_cms.py`                   | NestAnt-CMS with per-task fast heads             |
| `cl_baselines.py`                  | EWC, experience replay, independent per-task MLP |
| `progressive_nets.py`              | Progressive Networks (Rusu et al., 2016)         |
| `baselines.py`                     | Tree ensembles and plain MLP                     |
| `continual5_runner.py`             | Main five-task experiment driver                 |
| `convergence_study.py`             | Training-budget sweep                            |
| `ordering_ablation.py`             | Task-permutation sensitivity                     |
| `aggregate_canonical.py`           | Per-seed CSVs → summary JSON                     |
| `make_figures_v2.py`, `mk_heat.py` | Figure generation                                |
| `test_*_grads.py`                  | Finite-difference gradient checks                |


### Results files

`forgetting5_<method>_seed<n>_v3.csv` holds one 5×5 accuracy matrix: row *i* is the state after training stage *i*, column *j* is the R² on task *j*'s test set. `multiseed_summary_5task.json` aggregates these across seeds. `convergence_study.json` and `ordering_ablation.json` hold the two supplementary studies, with `ordering_A_o<k>_seed<n>.csv` giving the per-permutation matrices behind the second.

## Implementation notes

All neural methods are implemented in NumPy with hand-derived gradients verified against finite differences, so results are not tied to any framework version. They share the same backbone capacity (128 hidden units, 80-dimensional embedding, GELU), optimiser (Adam), batch size (1024) and weight decay (1e-5); only the structural mechanism differs.

Training budget materially affects conclusions. At 30 epochs the methods' relative ordering differs from their converged ordering, so comparisons at a small fixed budget measure optimisation speed rather than representational capacity. All headline numbers use 300 epochs per task.

One exception to note: the task-ordering study (`results/ordering_ablation.json`, and `figures/fig5_ordering.png`) was run at 30 epochs, unlike everything else. Its qualitative conclusion — that backward transfer is order-independent — holds, but the absolute R² values are not comparable with the main table. Re-run with `--epochs 300` for a like-for-like comparison.

Each seed jointly controls weight initialisation, minibatch order and the train/test partition, so reported deviations capture data-split variability as well as optimisation noise. Every individual 5×5 accuracy matrix is included in `results/`, so any reported quantity can be recomputed independently.

## Licence

Code is released under the MIT Licence (see `LICENSE`).

The dataset in `data/` is redistributed from the AntennaCAT project under **GPL-2.0**; its original licence and README are preserved in that directory and apply to the data regardless of the code licence.

## Acknowledgements

The AntennaCAT project (Linkous & Topsakal, USNC-URSI NRSM 2024) for releasing the underlying simulations publicly.