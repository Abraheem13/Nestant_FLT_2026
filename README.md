# Continual learning for multi-band antenna surrogates

Code, splits and per-seed results for the FIT 2026 paper *Overcoming Catastrophic Forgetting in Sequential Multi-Band Antenna Design*.

A surrogate is trained on one task at a time (a frequency band, or a range of patch lengths within a band) and is never shown the data of earlier tasks again. After every task it is tested on all tasks. The repository contains the benchmark definition, eight methods, two pooled-training references and every result matrix used in the paper.

## Data

`data/colab_dataset_microstrip.csv` is the AntennaCAT microstrip-patch dataset (Linkous and Topsakal, USNC-URSI NRSM 2024), redistributed under GPL-2.0 with its original licence and README (`data/UPSTREAM_*`).

Things to know before using it:

* 27,026 rows but only 6,848 distinct input geometries. Each geometry appears 2 to 5 times with different outputs, and the release does not explain why. Splitting by row would leak test geometries into training, so every split here is grouped by geometry (`src/data.py`).
* The rows come from an unfiltered parameter sweep. Only about 3 % of them have S11 below -10 dB at the design frequency, and 3 rows report S11 above 0 dB. Nothing is removed.
* Substrate height is constant (1.6 mm) and is dropped. Inputs: width, length, ground-plane side, feed-notch depth, log10 design frequency. Targets: S11 at the design frequency (dB), peak total gain (dB), log10 of the resonance frequency nearest the design frequency.

## Task sequences

| T  | Tasks (in training order) |
|----|---------------------------|
| 5  | 1.5 GHz with L < 47 mm, 1.5 GHz with L >= 47 mm, 2.4 GHz, 5.8 GHz with L < 11 mm, 5.8 GHz with L >= 11 mm |
| 10 | 1.5 GHz in four length bins (37-41, 42-46, 47-51, 52-57 mm), 2.4 GHz in two (25-29, 30-35 mm), 5.8 GHz in four (7-8.5, 9-10.5, 11-13, 13.5-15.5 mm) |
| 20 | every T=10 bin split in two (two or three patch lengths per task, 770 to 1,439 training rows) |

For each seed, 20 % of the geometries of every task (with all their replicates) are held out for testing. The seed also sets the initialisation and the minibatch order.

## Methods

All neural models are written in NumPy with hand-derived gradients; `src/test_gradients.py` checks each against central finite differences at the sizes used in the runs.

| Key | Method |
|-----|--------|
| `xgb`, `rf` | XGBoost / random forest, refit from scratch on each task |
| `finetune` | one MLP (5-128-80-3, GELU) trained on the tasks in order |
| `ewc` | online EWC with the exact diagonal empirical Fisher; `--lam` sets the penalty |
| `replay` | MLP with a replay buffer; `--replay_frac` sets the stored and replayed fraction |
| `indep` | a fresh MLP for every task |
| `prognet` | Progressive Networks: one MLP column per task with tanh lateral inputs from frozen earlier columns |
| `cms` | NestAnt-CMS: an encoder shared by all tasks (frozen after task 1), a learned code and a head per task |
| `joint_mlp`, `joint_xgb` | pooled references trained once on all tasks (not sequential) |

## Reproducing everything

Python 3.10+ with `numpy pandas scipy scikit-learn matplotlib xgboost`.

```bash
JOBS=8 ./run_all.sh
```

This runs the gradient checks, all 445 training runs (`results/jobs.txt`), the aggregation, the figures and the generated LaTeX numbers and tables (`results/tex/`). A single run is

```bash
cd src
python3 bench.py --method prognet --seed 0 --tasks 5 --epochs 300
```

## Results layout

* `results/runs/T{tasks}_e{epochs}[_tag]/{method}_seed{n}.json`: one file per run, with the T x T matrices (row = after training stage i, column = test task j) for S11 R2, S11 MAE, S11 MAE on matched designs, gain R2 and MAE, resonance-frequency R2 and relative error, plus run time and parameter count.
* `results/summary.json`: means, sample standard deviations, paired t-tests, the EWC sweep and the replicate limit.
* `figures/`: the paper figures.

## Licence

Code: MIT (see `LICENSE`). Data: GPL-2.0, which applies to the files in `data/` whatever the code licence.
