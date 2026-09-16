"""Exploratory data analysis for the HiRISE landmark dataset.

Reads the extracted archive under --data-dir (default data/raw), computes
class distribution, per-source-strip crop counts, and pixel-intensity
(brightness) distribution overall and per class, saves figures under
--figures-dir (default reports/figures), and writes a measured summary
JSON. All numbers are computed directly from the files on disk — nothing
here is estimated or assumed.

Usage:
    python -m src.data.eda --data-dir data/raw --figures-dir reports/figures
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

STRIP_ID_PATTERN = re.compile(r"^([A-Z]+_\d+_\d+_RED)-(\d+)(.*)\.jpg$")

CLASS_NAMES = {
    "0": "other",
    "1": "crater",
    "2": "dark dune",
    "3": "slope streak",
    "4": "bright dune",
    "5": "impact ejecta",
    "6": "swiss cheese",
    "7": "spider",
}


def run_eda(data_dir: Path, figures_dir: Path, seed: int = 42, brightness_sample_size: int = 6000) -> dict:
    images_dir = data_dir / "map-proj-v3"
    labels_path = data_dir / "labels-map-proj-v3.txt"
    figures_dir.mkdir(parents=True, exist_ok=True)

    lines = [l.split() for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    class_counts = collections.Counter(cls for _, cls in lines)

    ordered = sorted(class_counts.items(), key=lambda kv: int(kv[0]))
    names = [CLASS_NAMES[c] for c, _ in ordered]
    counts = [n for _, n in ordered]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, counts)
    ax.set_ylabel("Image count (includes augmentation)")
    ax.set_title("Class distribution - HiRISE landmark dataset v3 (n=%d)" % sum(counts))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=120)
    plt.close(fig)

    strip_counts: collections.Counter = collections.Counter()
    for fname, _ in lines:
        m = STRIP_ID_PATTERN.match(fname)
        if m:
            strip_counts[m.group(1)] += 1
    strip_values = sorted(strip_counts.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(strip_values, bins=30)
    ax.set_xlabel("Crops (incl. augmentation) per source RED strip")
    ax.set_ylabel("Number of strips")
    ax.set_title("Crop count per source HiRISE strip (n_strips=%d)" % len(strip_counts))
    fig.tight_layout()
    fig.savefig(figures_dir / "crops_per_source_strip.png", dpi=120)
    plt.close(fig)

    rng = np.random.default_rng(seed)
    sample_size = min(brightness_sample_size, len(lines))
    sample_idx = rng.choice(len(lines), size=sample_size, replace=False)
    means_by_class: dict[str, list[float]] = collections.defaultdict(list)
    overall_means = []
    for i in sample_idx:
        fname, cls = lines[i]
        with Image.open(images_dir / fname) as im:
            arr = np.asarray(im, dtype=np.float32)
        m = float(arr.mean())
        overall_means.append(m)
        means_by_class[cls].append(m)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(overall_means, bins=50)
    ax.set_xlabel("Mean pixel intensity (0-255)")
    ax.set_ylabel("Image count")
    ax.set_title("Brightness distribution - random sample (n=%d, seed=%d)" % (sample_size, seed))
    fig.tight_layout()
    fig.savefig(figures_dir / "brightness_distribution.png", dpi=120)
    plt.close(fig)

    summary = {
        "sample_size_for_brightness": sample_size,
        "brightness_sample_seed": seed,
        "class_distribution": {c: n for c, n in ordered},
        "class_distribution_pct": {c: round(100 * n / sum(counts), 2) for c, n in ordered},
        "strips_total": len(strip_counts),
        "crops_per_strip_min": min(strip_values) if strip_values else None,
        "crops_per_strip_max": max(strip_values) if strip_values else None,
        "crops_per_strip_median": float(np.median(strip_values)) if strip_values else None,
        "overall_brightness_mean": float(np.mean(overall_means)),
        "overall_brightness_std": float(np.std(overall_means)),
        "brightness_mean_by_class": {
            c: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for c, v in sorted(means_by_class.items(), key=lambda kv: int(kv[0]))
        },
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--output", type=Path, default=Path("reports/eda_summary.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (args.data_dir / "labels-map-proj-v3.txt").exists():
        print(f"No dataset found at {args.data_dir} - nothing to analyze.", file=sys.stderr)
        return 1

    summary = run_eda(args.data_dir, args.figures_dir, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} and figures under {args.figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
