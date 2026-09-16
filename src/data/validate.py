"""Automated dataset validation for the HiRISE landmark dataset.

Reads the extracted archive under --data-dir (default data/raw), checks
image integrity, dimension/mode consistency, exact-duplicate content, and
label/image consistency, and writes a measured (never fabricated) report.

Usage:
    python -m src.data.validate --data-dir data/raw --output reports/data_validation.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

from PIL import Image

STRIP_ID_PATTERN = re.compile(r"^([A-Z]+_\d+_\d+_RED)-(\d+)(.*)\.jpg$")


def validate_dataset(data_dir: Path) -> dict:
    images_dir = data_dir / "map-proj-v3"
    labels_path = data_dir / "labels-map-proj-v3.txt"

    label_lines = [l for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    label_map = dict(l.split() for l in label_lines)
    class_counts = collections.Counter(v for v in label_map.values())

    image_files = sorted(p.name for p in images_dir.glob("*.jpg"))
    image_set = set(image_files)
    label_set = set(label_map.keys())

    corrupt = []
    dims = collections.Counter()
    modes = collections.Counter()
    content_hashes: dict[str, list[str]] = collections.defaultdict(list)
    strip_ids: set[str] = set()
    unparseable_filenames = []

    for name in image_files:
        path = images_dir / name
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                dims[str(im.size)] += 1
                modes[im.mode] += 1
        except Exception as exc:  # noqa: BLE001 - record and continue
            corrupt.append({"file": name, "error": str(exc)})
            continue

        content_hashes[hashlib.md5(path.read_bytes()).hexdigest()].append(name)

        m = STRIP_ID_PATTERN.match(name)
        if m:
            strip_ids.add(m.group(1))
        else:
            unparseable_filenames.append(name)

    exact_duplicate_groups = {h: v for h, v in content_hashes.items() if len(v) > 1}

    report = {
        "dataset_version": "hirise-map-proj-v3",
        "data_dir": str(data_dir),
        "total_images_on_disk": len(image_files),
        "total_label_lines": len(label_lines),
        "labels_without_matching_image": sorted(label_set - image_set),
        "images_without_matching_label": sorted(image_set - label_set),
        "corrupted_or_unreadable_images": corrupt,
        "dimension_distribution": dict(dims),
        "color_mode_distribution": dict(modes),
        "exact_duplicate_content_groups": len(exact_duplicate_groups),
        "files_in_exact_duplicate_groups": sum(len(v) for v in exact_duplicate_groups.values()),
        "class_distribution": dict(sorted(class_counts.items(), key=lambda kv: int(kv[0]))),
        "unique_source_strip_ids": len(strip_ids),
        "unparseable_filenames": unparseable_filenames,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("reports/data_validation.json"))
    args = parser.parse_args()

    if not (args.data_dir / "labels-map-proj-v3.txt").exists():
        print(f"No dataset found at {args.data_dir} — nothing to validate.", file=sys.stderr)
        return 1

    report = validate_dataset(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
