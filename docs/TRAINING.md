# Training — MarsLandmark-AI

Status: **Training loop implemented and smoke-tested** (`src/training/train.py`,
`tests/test_train.py`, tiny synthetic data, CPU); **no real training run has
happened yet**. This document describes the mechanism, not a result.

## Where training actually runs

Local execution on this machine's GPU was **explicitly declined by the
project owner** for this run (see `docs/PROJECT_OVERVIEW.md`). Real
training happens in `notebooks/colab_train.ipynb`, run by the user in
Google Colab. `src/training/train.py` is the same code the notebook
calls — both paths share the exact same `fit()`/`evaluate()` functions so
results are reproducible outside Colab too, on any machine with a GPU.

## Loop mechanics (`src/training/train.py`)

- `train_one_epoch`: standard forward/backward/step over the train loader.
- `evaluate`: no-grad forward pass over a loader, returns loss + full
  `compute_metrics()` output (accuracy, macro/weighted F1, per-class,
  confusion matrix) — never just accuracy (project rule §14).
- `fit`: runs `epochs` rounds, keeps the best-val-macro-F1 checkpoint
  in memory (model selection by macro F1, not accuracy, given the
  measured class imbalance), supports early stopping.
- `log_experiment`: appends one row to `experiments/experiments.csv`
  matching its existing header exactly. Any field not explicitly passed
  is written empty, never fabricated or defaulted to a guessed value.

## What is chosen vs. what is measured

- Loss function: cross-entropy — chosen because this is single-label
  multi-class classification; **not yet compared** against a
  class-weighted variant (Phase 07 — the measured 83.6%/0.3% imbalance,
  `docs/DATASET.md`, is the motivating reason to test this, not a reason
  to assume it's needed).
- Optimizer: AdamW, per `configs/config.yaml` defaults — these are
  engineering defaults, explicitly not the result of a hyperparameter
  search (see the comment already in `configs/config.yaml`).
- Model selection metric: validation macro F1, not accuracy — chosen
  because accuracy alone is misleading on this dataset (a
  majority-class-only predictor would score ~83.6%, see
  `docs/MODEL_ARCHITECTURE.md`).

## Test-set discipline

`src/training/train.py`'s CLI only ever loads the `train`/`val` splits —
it has no code path that touches the `test` split. Final test evaluation
is a deliberately separate, explicit step (Phase 08), to prevent the
test set from being used for repeated model selection (project rule §6).

## Reproducibility

`fit()`/`train_one_epoch()` do not themselves set the random seed — the
caller must (`torch.manual_seed(...)`, done once in `train.py`'s CLI
`main()` from `configs/config.yaml: seed`). Every field written by
`log_experiment` — git commit, dataset version, hyperparameters, hardware
— is intended to make an experiment row enough to reproduce that run
without consulting this document.
