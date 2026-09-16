# Data Split Policy — MarsLandmark-AI

Status: **design decided, not yet executed against real data** (blocked on
acquisition — see `docs/DATASET.md`).

## Policy

The HiRISE landmark dataset is reported (see `docs/DATASET_SELECTION.md`)
to ship an official train/validation/test split of 6,997 / 2,025 / 1,793
crops, described as grouped by source HiRISE image. Per project rule 0.6
("If the NASA dataset already provides official splits, investigate and
preferably preserve them"), this project's default is to **use the
official split as-is** rather than re-deriving one, once its existence and
grouping methodology are confirmed against the real downloaded files.

## Why source-image grouping matters here

HiRISE landmark crops are extracted from a smaller number of larger source
images. Multiple crops from the same source image can be visually and
statistically correlated (shared lighting, shared terrain context,
sometimes overlapping regions). Randomly shuffling crops into train/test
without accounting for their source image risks leaking information: the
model could learn to recognize a specific source image's characteristics
rather than the landmark class in general, inflating test performance.

## Implementation

`src/data/splits.py` provides:

- `load_split(data_dir, split)` — loads whichever label file exists for a
  named split (`train`, `val`, `test`), rather than assuming a filename
  convention that hasn't been confirmed yet.
- `check_filename_leakage(splits)` — a mechanical check that no exact
  filename appears in more than one split. This is necessary but not
  sufficient for leakage prevention.

## Outstanding work (Phase 03, once real data exists)

1. Confirm whether the official split's label files are shipped with the
   dataset archive, and under what filenames — update
   `src/data/splits.SPLIT_FILENAMES` accordingly.
2. Determine the real filename convention (does it encode a source-image
   ID?) and, if so, implement a `group_key_fn` for a stronger
   source-grouped leakage check beyond exact-filename matching.
3. Run `check_filename_leakage` against the real, loaded splits and record
   the result here as a fact, not an assumption.
4. If the official split turns out not to be grouped as reported, or
   cannot be confirmed, escalate per project rule 0.7/39 rather than
   silently proceeding with an ungrouped split.

No split has been executed yet. This document will be updated with
`CONFIRMED` results once Phase 03 runs against the real dataset.
