# Experiment Log — MarsLandmark-AI

Every entry below corresponds to a row in `experiments/experiments.csv`
and an actual training run — nothing here is projected or estimated.
Two full sessions, both 2026-09-17, in Google Colab (T4 GPU) via
`notebooks/colab_train.ipynb`, by the project owner: session 1 against
commit `c16c102`, session 2 (which also added the Phase 07/09 cells)
against commit `718499b`.

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
| **Session 2** (fresh Colab runtime, full re-run) | | | | | | | | |
| exp_baseline_simplecnn_2ndsession | SimpleCNN (scratch) | 10 | 0.7838 | 0.3123 | — | — | 22.6 | COMPLETED |
| exp_resnet18_frozen_2ndsession_run1 | ResNet18 frozen | 5 | 0.8509 | 0.5476 | — | — | 10.65 | COMPLETED |
| exp_resnet18_frozen_2ndsession_run2 | ResNet18 frozen | 5 | 0.8554 | 0.5498 | — | — | 10.59 | COMPLETED |
| exp_resnet18_finetuned_2ndsession | ResNet18 fine-tuned | 6 (early-stopped) | 0.9013 | 0.6926 | 0.9363 | 0.7356 | 19.22 | COMPLETED |
| exp_resnet18_finetuned_classweighted | ResNet18 fine-tuned, weighted loss | 15 | 0.8442 | **0.7153** | — | — | 48.1 | COMPLETED |

Full per-run detail (hyperparameters, checksums, git commit) is in
`experiments/experiments.csv`. Full per-class final-test metrics and
confusion matrices are in `reports/final_test_evaluation.json` (session 1),
`reports/final_test_evaluation_run2.json` (session 2), and
`docs/RESULTS.md`. Calibration: `reports/calibration_report.json`.
Grad-CAM figures: `reports/figures/gradcam_classweighted_run2.png`.

## Observations

- **ResNet18 fine-tuned clearly wins** over both the from-scratch baseline
  (macro F1 0.2978–0.3123 across sessions) and the frozen-backbone variant
  (0.52–0.58 across 6 runs) — consistent with the project's expectation
  that transfer learning should help, but only actually confirmed by
  running it (project rule: never assume).
- **Non-determinism is real and measurable at every level of this
  pipeline, not just the frozen-backbone cell.** Frozen-backbone runs were
  tightly reproducible (6 runs across 2 sessions, macro F1 0.5195–0.5761,
  and 2 of those runs matched to 4 decimal places). Full fine-tuning was
  not: session 1's `exp_resnet18_finetuned` ran the full 15 epochs and
  reached val macro F1 0.7188; session 2's `exp_resnet18_finetuned_2ndsession`
  early-stopped at epoch 6 and reached only 0.6926 — same code, same
  seed, same data. This propagated all the way to the **final test
  evaluation**: session 1 measured 93.11% test accuracy / 71.56% macro F1
  (`reports/final_test_evaluation.json`); session 2's separately-trained
  model measured 93.63% / 73.56% (`reports/final_test_evaluation_run2.json`).
  Both are real, both are kept — see `docs/REPRODUCIBILITY.md`.
- **Phase 07 class-weighted loss result: inconclusive on macro F1, and a
  clear non-fix for the motivating problem.** `exp_resnet18_finetuned_classweighted`
  reached val macro F1 0.7153 — essentially indistinguishable from the two
  unweighted runs (0.7188, 0.6926) given the measured run-to-run noise
  above. It did **not** fix "spider": val-set spider precision/recall/F1
  were all 0.0, identical to the unweighted runs. Grad-CAM
  (`docs/ERROR_ANALYSIS.md`) shows why: the class-weighted model still
  never predicts "spider" — it shifted its wrong guess from "other" to
  "impact ejecta," but the underlying feature it attends to for spider
  images looks like the correct terrain feature, not noise. Simple
  inverse-frequency weighting does not appear to be the fix here; a
  volume problem (spider is one of the rarest classes across all splits,
  `docs/DATA_SPLIT.md`) may need a volume solution (targeted augmentation,
  more source data) rather than a loss-weighting one — untested.
- **SimpleCNN's macro F1 (0.2978–0.3123) is barely above what a
  reasonable majority-heavy classifier would score** given the measured
  85%+ "other" share of the training split (`docs/DATA_SPLIT.md`) —
  from-scratch training on this dataset size/imbalance does not learn the
  minority classes well without help.
- **Calibration**: ECE before scaling 0.1131, after temperature scaling
  (T=1.742) 0.0644 — a real, meaningful improvement, measured on the
  class-weighted model's val-set logits. See `docs/EXPLAINABILITY.md`.
- See `docs/ERROR_ANALYSIS.md` for the final-test failure modes (complete
  "spider" class failure, "swiss cheese" recall problem) and the Grad-CAM
  qualitative findings.
