# Reproducibility — MarsLandmark-AI

## FACT: fixed seed did not produce identical results across runs

`experiments/experiments.csv` records 6 separate runs of the identical
ResNet18-frozen-backbone cell (across two full Colab sessions), same
`seed=42`, same code, same data, same hardware class (Tesla T4): val
macro F1 came back as 0.5476, 0.5498, 0.5761, 0.5195, 0.5476, 0.5498 — a
spread of 0.057 (about 10% relative) across the 4 distinct values, with 2
runs (one per session) matching to 4 decimal places. Frozen-backbone runs
are evidently far more reproducible than full fine-tunes (below), likely
because far fewer parameters (only `fc`) are actually being optimized.

**FACT: full fine-tuning is much less reproducible, and the divergence
propagates all the way to the final test evaluation.** Two independent
full sessions each ran `exp_resnet18_finetuned` from the same starting
config (pretrained ImageNet init, lr=1e-4, 15-epoch budget, patience=5,
seed=42):

| | Session 1 | Session 2 |
|---|---:|---:|
| Epochs completed | 15 (ran full budget) | 6 (early-stopped) |
| Val macro F1 (best) | 0.7188 | 0.6926 |
| Test accuracy | 93.11% | 93.63% |
| Test macro F1 | 71.56% | 73.56% |

Both are genuine, separately-trained models — not the same weights
evaluated twice (which would violate test-set integrity, project rule
§6). Both final test evaluations are kept and documented
(`reports/final_test_evaluation.json` and
`reports/final_test_evaluation_run2.json`) rather than one being
discarded or silently overwritten. This is the concrete evidence behind
this project's repeated caution that its headline numbers describe one
measured run, not a guaranteed constant.

**INTERPRETATION:** `torch.manual_seed(SEED)` (and `cuda.manual_seed_all`)
seeds PyTorch's own RNG, but does not by itself make cuDNN's GPU
convolution kernels deterministic — cuDNN selects among multiple
algorithm implementations for performance, and by default that selection
and some of the kernels themselves are non-deterministic across runs.
The notebook does not set
`torch.backends.cudnn.deterministic = True` /
`torch.backends.cudnn.benchmark = False`, so this is expected, not a bug
in the seeding code.

**HYPOTHESIS (not yet tested):** setting those two flags would reduce
(not necessarily eliminate — dataloader worker ordering with
`num_workers > 0` is a second, independent source of non-determinism)
run-to-run variance. This has a real cost: `cudnn.benchmark = False`
typically makes training slower, since it disables cuDNN's
runtime-autotuned kernel selection. Whether that tradeoff is worth it for
this project has not been evaluated.

## Environment (as actually run)

| Field | Value |
|---|---|
| Platform | Google Colab (Linux), Tesla T4 GPU |
| Python / PyTorch | Whatever Colab's default runtime provided at run time (2026-09-17) — **not pinned or recorded by the notebook**; this is a known gap, see below |
| Dataset | `hirise-map-proj-v3.zip`, MD5 `cab4aeb474f76d82b7188a8f342a608b`, verified at acquisition time (`docs/DATASET.md`) |
| Seed | 42 (`random`, `numpy`, `torch`, `torch.cuda`) — insufficient alone for bit-exact reproduction, see above |
| Split | Deterministic (no RNG), `src/data/split.py` / notebook cell 6 — same code, same input, always produces the same `assignment` |
| Git commit at run time | `c16c102bd19514a196fb9813e337c6693ad88923` |

## Known reproducibility gaps

1. **cuDNN non-determinism** (above) — measured, not fixed yet.
2. **Colab's Python/PyTorch/CUDA versions are not pinned or printed to a
   saved artifact.** Cell 1 does print them at run time, but that output
   wasn't captured into `experiments.csv`/a report file. A future run
   should redirect that printed info into the experiment log.
3. **DataLoader worker ordering** (`num_workers=2`, `shuffle=True` for
   train) is itself a secondary source of run-to-run variance beyond
   cuDNN, un-addressed by the current seeding.

## How to reproduce this project's reported result

```bash
# 1. Get the exact code
git clone https://github.com/PulkitTiwari87/MarsLandmark-AI.git
git checkout c16c102bd19514a196fb9813e337c6693ad88923

# 2. Get the exact data (verify against the checksum in docs/DATASET.md)
# See src/data/download.py, or notebooks/colab_train.ipynb cell 4.

# 3. Run notebooks/colab_train.ipynb in Colab (T4 GPU), or
#    python -m src.training.train --config configs/config.yaml \
#      --model resnet18 --experiment-id repro_run
```

Exact metric reproduction is **not guaranteed** given the cuDNN gap above
— expect results in the same range as `docs/EXPERIMENTS.md`, not
bit-identical numbers.
