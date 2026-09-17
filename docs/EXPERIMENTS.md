# Experiment Log — MarsLandmark-AI

Every entry below corresponds to a row in `experiments/experiments.csv`
and an actual training run — nothing here is projected or estimated.
Run 2026-09-17, in Google Colab (T4 GPU) via `notebooks/colab_train.ipynb`,
by the project owner, against commit `c16c102`.

## Format

Each experiment entry must record: experiment ID, date, git commit,
dataset version/checksum, model, image size, batch size, optimizer,
learning rate, scheduler, epochs, seed, augmentation, loss function,
hardware, training time, validation metrics, test metrics (final
evaluation only), notes, and failure modes.

## Log

| Experiment | Model | Epochs | Val Acc | Val Macro F1 | Test Acc | Test Macro F1 | Time (min) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| exp_baseline_simplecnn | SimpleCNN (scratch) | 10 | 0.7766 | 0.2978 | — | — | 22.04 | COMPLETED |
| exp_resnet18_frozen_run1 | ResNet18 frozen | 5 | 0.8509 | 0.5476 | — | — | 10.68 | COMPLETED |
| exp_resnet18_frozen_run2 | ResNet18 frozen | 5 | 0.8554 | 0.5498 | — | — | 10.56 | COMPLETED |
| exp_resnet18_finetuned | ResNet18 fine-tuned | 15 | 0.9013 | 0.7188 | **0.9311** | **0.7156** | 52.9 | COMPLETED |
| exp_resnet18_frozen_run3 | ResNet18 frozen | 5 | 0.8451 | 0.5761 | — | — | 10.53 | COMPLETED |
| exp_resnet18_frozen_run4 | ResNet18 frozen | 5 | 0.8461 | 0.5195 | — | — | 10.52 | COMPLETED |

Full per-run detail (hyperparameters, checksums, git commit) is in
`experiments/experiments.csv`. Full per-class final-test metrics and the
confusion matrix are in `reports/final_test_evaluation.json` and
`docs/RESULTS.md`.

## Observations from this run

- **ResNet18 fine-tuned clearly wins** on val macro F1 (0.7188) over both
  the from-scratch baseline (0.2978) and the frozen-backbone variant
  (0.52–0.58 across 4 runs) — consistent with the project's expectation
  that transfer learning should help, but only actually confirmed by
  running it (project rule: never assume).
- **The frozen-backbone cell was run 4 times**, producing macro F1 values
  ranging 0.5195–0.5761 despite `seed=42` being set in every run. This is
  a genuine, measured non-determinism, not an error in recording — see
  `docs/REPRODUCIBILITY.md` for the likely cause (cuDNN non-determinism,
  not disabled by this notebook) and what would need to change to close
  the gap.
- **SimpleCNN's macro F1 (0.2978) is barely above what a reasonable
  majority-heavy classifier would score** given the measured 85%+ "other"
  share of the training split (`docs/DATA_SPLIT.md`) — from-scratch
  training on this dataset size/imbalance does not learn the minority
  classes well without help (weighting, more data, or transfer learning).
- See `docs/ERROR_ANALYSIS.md` for the final-test failure modes (complete
  "spider" class failure, "swiss cheese" recall problem).
