# Results — MarsLandmark-AI

Status: **Measured, two independent runs.** Two separate full Colab
sessions each trained and once-evaluated a ResNet18 fully-fine-tuned
model (2026-09-17). Both are kept, not merged or cherry-picked — see
`docs/REPRODUCIBILITY.md` for why they differ. Raw output:
`reports/final_test_evaluation.json` (session 1, the primary headline
number below) and `reports/final_test_evaluation_run2.json` (session 2).
Full training history: `docs/EXPERIMENTS.md`, `experiments/experiments.csv`.

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

## Final test metrics (test set, n=10,948, each model touched exactly once)

| Metric | Session 1 (headline) | Session 2 (independent re-run) |
|---|---:|---:|
| Accuracy | 0.9311 | 0.9363 |
| Macro F1 | 0.7156 | 0.7356 |
| Weighted F1 | 0.9309 | 0.9320 |

Session 2 trained a separately-initialized model with the identical
config — not the same weights re-evaluated (that would violate test-set
integrity, §6). Both numbers are real; the difference (accuracy ±0.5pp,
macro F1 ±2pp) is the measured non-determinism documented in
`docs/REPRODUCIBILITY.md`. Session 1's numbers are used as this project's
primary reference throughout the rest of this document, arbitrarily by
being first, not because it's "more correct."

**FACT:** accuracy (93.1%) and macro F1 (71.6%) diverge sharply. This is
the class-imbalance effect the project's evaluation methodology was
specifically designed to surface (§14) — a model that is excellent on the
dominant class can still score a high blended accuracy while failing
badly on rare classes. See per-class breakdown below and
`docs/ERROR_ANALYSIS.md`.

### Per-class test metrics (session 1)

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

## Benchmark table (all experiments, both sessions)

| Experiment | Model | Params | Image Size | Epochs | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Train Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp_baseline_simplecnn | SimpleCNN | ~0.1M | 227 | 10 | 0.7766–0.7838 | 0.2978–0.3123 | NOT MEASURED | NOT MEASURED | ~22.3 min |
| exp_resnet18_frozen (x6 runs) | ResNet18 (frozen) | 11.7M | 227 | 5 | 0.845–0.855 | 0.520–0.576 | NOT MEASURED | NOT MEASURED | ~10.6 min |
| exp_resnet18_finetuned (session 1) | ResNet18 (fine-tuned) | 11.7M | 227 | 15 | 0.9013 | 0.7188 | **0.9311** | **0.7156** | 52.9 min |
| exp_resnet18_finetuned (session 2) | ResNet18 (fine-tuned) | 11.7M | 227 | 6 | 0.9013 | 0.6926 | 0.9363 | 0.7356 | 19.2 min |
| exp_resnet18_finetuned_classweighted | ResNet18 (fine-tuned, weighted loss) | 11.7M | 227 | 15 | 0.8442 | 0.7153 | NOT MEASURED | NOT MEASURED | 48.1 min |

`NOT MEASURED` for baseline/frozen/class-weighted: the project rules
(§18) require the test set be evaluated exactly once, on the model
that's actually selected — it was not separately evaluated for
candidates that weren't chosen, to preserve test-set integrity. The
class-weighted model's val macro F1 (0.7153) did not clearly beat the
unweighted models (0.7188, 0.6926 — see `docs/REPRODUCIBILITY.md` for why
that comparison is noisy), so it was not promoted to final test
evaluation.

## Phase 07 — did class-weighted loss help? (measured, not assumed)

**No — not on macro F1, and not on the specific problem that motivated
it.** `exp_resnet18_finetuned_classweighted`'s val macro F1 (0.7153) sits
inside the noise band of the two unweighted runs (0.6926–0.7188). More
directly: **val-set "spider" precision/recall/F1 were all 0.0 with
class-weighted loss too** — identical to the unweighted models. Balanced
inverse-frequency weighting did not fix the problem it was tried for.
Grad-CAM (`docs/ERROR_ANALYSIS.md`) suggests why: the model reliably
attends to a real feature for spider images, but confuses it with
"impact ejecta" — a visual-similarity problem that reweighting the loss
function doesn't address. Full numbers: `docs/EXPERIMENTS.md`.

## What this does NOT establish

- Whether a different imbalance strategy (targeted augmentation for
  spider specifically, oversampling, focal loss) would help — untested.
- Whether the model generalizes to HiRISE imagery outside this dataset's
  180 source strips — untested, out of scope for this dataset alone.
- Whether the class-weighted model's calibration (measured,
  `docs/EXPLAINABILITY.md`) would differ for the unweighted Phase 08
  model — not cross-checked.
