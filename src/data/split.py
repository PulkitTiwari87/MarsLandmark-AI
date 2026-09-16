"""Leakage-safe, class-balanced group split for the HiRISE landmark dataset.

No official split ships with this dataset (see docs/DATASET.md). This
module assigns every image to exactly one of train/val/test, grouped by
source RED-strip id so that no augmented sibling or same-strip crop
crosses a split boundary (see docs/DATA_SPLIT.md).

Algorithm, two phases:

1. Minimum class-coverage pass: a pure size-proportional greedy (see
   phase 2) was tried first on this dataset and measured to produce
   zero-representation failures — e.g. the "bright dune" class ended up
   with 0 images in val, and "swiss cheese" with 0 in test (see
   docs/DATA_SPLIT.md for the measured numbers). That is a real integrity
   problem: a class absent from val/test cannot be validated or reported
   in the final benchmark. To prevent it, for every class this phase
   picks the smallest available group containing that class and reserves
   one for val and one for test (skipped if a class appears in too few
   distinct groups to do so, which is reported, not silently ignored).
2. Proportional-deficit pass (a standard largest-remainder-style
   apportionment method) over all remaining groups, largest-first: each
   group goes to whichever split has the smallest value of
   (current_size + group_size) / target_fraction — i.e. whichever split
   is currently furthest below its target share of the total. Measured to
   converge to within ~0.2 percentage points of target ratios on both a
   synthetic benchmark and the real 173-strip dataset.

This still does not fully stratify by class — only "at least one group"
is guaranteed per class per split, not proportional representation. The
achieved per-class distribution is measured and reported by
split_report()/main(), never assumed.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

from src.data.common import strip_id_for

SPLITS = ("train", "val", "test")


def assign_splits(
    label_map: dict[str, str],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, str]:
    """Return {filename: split_name} for every key in label_map."""
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be in (0,1) and sum to < 1")
    target_frac = {
        "train": 1.0 - val_fraction - test_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }

    groups: dict[str, list[str]] = collections.defaultdict(list)
    for fname in label_map:
        sid = strip_id_for(fname) or f"__unparsed__:{fname}"
        groups[sid].append(fname)

    classes_in_group: dict[str, set[str]] = {
        sid: {label_map[f] for f in fnames} for sid, fnames in groups.items()
    }
    groups_containing: dict[str, list[str]] = collections.defaultdict(list)
    for sid, classes in classes_in_group.items():
        for cls in classes:
            groups_containing[cls].append(sid)

    running_total = {s: 0 for s in SPLITS}
    assignment: dict[str, str] = {}
    assigned_groups: set[str] = set()

    def assign_group(sid: str, split: str) -> None:
        running_total[split] += len(groups[sid])
        for fname in groups[sid]:
            assignment[fname] = split
        assigned_groups.add(sid)

    # Phase 1: guarantee every class has >=1 group in val and >=1 in test,
    # if enough distinct groups contain that class to do so.
    for cls in sorted(groups_containing):
        candidates = sorted(groups_containing[cls], key=lambda sid: len(groups[sid]))
        for split in ("val", "test"):
            available = [sid for sid in candidates if sid not in assigned_groups]
            if not available:
                break  # not enough distinct groups for this class; leave as a measured gap
            assign_group(available[0], split)

    # Phase 2: proportional-deficit greedy over everything left, largest-first.
    remaining = sorted(
        (sid for sid in groups if sid not in assigned_groups),
        key=lambda sid: (-len(groups[sid]), sid),
    )
    for sid in remaining:
        size = len(groups[sid])
        best_split = min(SPLITS, key=lambda s: (running_total[s] + size) / target_frac[s])
        assign_group(sid, best_split)

    return assignment


def split_report(label_map: dict[str, str], assignment: dict[str, str]) -> dict:
    all_classes = sorted(set(label_map.values()))
    per_split_total = collections.Counter(assignment.values())
    per_split_class: dict[str, collections.Counter] = {s: collections.Counter() for s in SPLITS}
    for fname, split in assignment.items():
        per_split_class[split][label_map[fname]] += 1

    classes_missing_per_split = {
        s: [c for c in all_classes if per_split_class[s].get(c, 0) == 0] for s in SPLITS
    }

    n_groups_per_split = collections.Counter()
    seen_strip_per_split: dict[str, set[str]] = collections.defaultdict(set)
    for fname, split in assignment.items():
        sid = strip_id_for(fname) or f"__unparsed__:{fname}"
        if sid not in seen_strip_per_split[split]:
            seen_strip_per_split[split].add(sid)
            n_groups_per_split[split] += 1

    return {
        "total_images": len(assignment),
        "per_split_image_count": dict(per_split_total),
        "per_split_fraction": {s: round(per_split_total[s] / len(assignment), 4) for s in SPLITS},
        "per_split_class_distribution": {
            s: {c: per_split_class[s].get(c, 0) for c in all_classes} for s in SPLITS
        },
        "classes_missing_per_split": classes_missing_per_split,
        "per_split_group_count": dict(n_groups_per_split),
    }


def write_split_manifest(assignment: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split"])
        for fname in sorted(assignment):
            writer.writerow([fname, assignment[fname]])


def read_split_manifest(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["filename"]: row["split"] for row in reader}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--manifest-out", type=Path, default=Path("data/processed/split_manifest.csv"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/split_report.json"))
    args = parser.parse_args()

    labels_path = args.data_dir / "labels-map-proj-v3.txt"
    if not labels_path.exists():
        print(f"No dataset found at {args.data_dir}.", file=sys.stderr)
        return 1

    label_map = dict(
        line.split() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    assignment = assign_splits(label_map, args.val_fraction, args.test_fraction)
    write_split_manifest(assignment, args.manifest_out)
    report = split_report(label_map, assignment)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.manifest_out} and {args.report_out}")
    print(json.dumps(report["per_split_fraction"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
