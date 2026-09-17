# Results — MarsLandmark-AI

Status: **Measured.** Final test evaluation ran once, 2026-09-17, on the
model selected by validation macro F1 (ResNet18, fully fine-tuned). Raw
output: `reports/final_test_evaluation.json`. Full training history:
`docs/EXPERIMENTS.md`, `experiments/experiments.csv`.

## Performance target dashboard

```text
Minimum desired genuine performance: 85%
Upper target boundary:               95%
Actual measured performance:         93.11% (test accuracy)
```

The measured result falls inside the 85–95% target band. Per project rule
§35, this is reported because it's what was honestly measured — no
dataset reduction, undertraining, split manipulation, or cherry-picking
was used to land inside this range. See `docs/DATASET.md` (unmodified
dataset), `docs/DATA_SPLIT.md` (split fixed before any model was trained),
and `docs/EXPERIMENTS.md` (every run logged, including the ones that
didn't get selected).

## Final test metrics (test set, n=10,948, touched exactly once)

| Metric | Value |
|---|---:|
| Accuracy | 0.9311 |
| Macro F1 | 0.7156 |
| Weighted F1 | 0.9309 |

**FACT:** accuracy (93.1%) and macro F1 (71.6%) diverge sharply. This is
the class-imbalance effect the project's evaluation methodology was
specifically designed to surface (§14) — a model that is excellent on the
dominant class can still score a high blended accuracy while failing
badly on rare classes. See per-class breakdown below and
`docs/ERROR_ANALYSIS.md`.

### Per-class test metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| other | 0.9611 | 0.9719 | 0.9665 | 9,100 |
| crater | 0.6790 | 0.8231 | 0.7442 | 735 |
| dark dune | 0.4877 | 0.7054 | 0.5766 | 112 |
| slope streak | 0.8983 | 0.7970 | 0.8446 | 133 |
| bright dune | 1.0000 | 0.8387 | 0.9123 | 217 |
| impact ejecta | 1.0000 | 0.9048 | 0.9500 | 21 |
| swiss cheese | 0.9972 | 0.5762 | 0.7304 | 623 |
| **spider** | **0.0000** | **0.0000** | **0.0000** | 7 |

**INTERPRETATION:** performance correlates with support (sample count) —
the three classes with the fewest test examples (impact ejecta n=21,
spider n=7, and to a lesser extent dark dune n=112) show either the
weakest or most fragile scores. Spider's complete failure (n=7) is
consistent with too few examples for the model to learn a distinguishing
representation, not necessarily with the class being visually
indistinguishable — see `docs/ERROR_ANALYSIS.md` for the confusion
pattern.

## Benchmark table (all experiments)

| Experiment | Model | Params | Image Size | Epochs | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Train Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp_baseline_simplecnn | SimpleCNN | ~0.1M | 227 | 10 | 0.7766 | 0.2978 | NOT MEASURED | NOT MEASURED | 22.04 min |
| exp_resnet18_frozen (x4 runs) | ResNet18 (frozen) | 11.7M | 227 | 5 | 0.845–0.855 | 0.520–0.576 | NOT MEASURED | NOT MEASURED | ~10.6 min |
| exp_resnet18_finetuned | ResNet18 (fine-tuned) | 11.7M | 227 | 15 | 0.9013 | 0.7188 | **0.9311** | **0.7156** | 52.9 min |

`NOT MEASURED` for baseline/frozen: the project rules (§18) require the
test set be evaluated exactly once, on the model that has already been
selected — it was not separately evaluated for the two candidates that
weren't chosen, and will not be, to preserve test-set integrity.

## What this does NOT establish

- Whether class-weighted loss or resampling would close the macro-F1 gap
  — `NOT YET MEASURED`, Phase 07.
- Whether the model generalizes to HiRISE imagery outside this dataset's
  180 source strips — untested, out of scope for this dataset alone.
- Calibration of the model's confidence scores — `NOT YET MEASURED`,
  Phase 09 (Explainability & Robustness), not yet started.
