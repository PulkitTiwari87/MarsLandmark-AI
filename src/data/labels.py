"""Label-file parsing for the HiRISE landmark dataset.

The class index -> class name mapping is NOT hardcoded as fact: the real
dataset ships its own class-map file (per the reference implementation,
something like ``landmarks_map-proj-v3_2.txt``). ``DEFAULT_CLASS_NAMES``
below is a placeholder ordering assembled from secondary sources during
Phase 01 research (see docs/DATASET.md) and MUST be confirmed against the
actual class-map file once the real dataset is downloaded — an unconfirmed
ordering would silently mislabel every class.
"""

from __future__ import annotations

from pathlib import Path

# PLACEHOLDER ordering — confirm against the dataset's own class-map file
# before training (see docs/DATASET.md, "Outstanding actions").
DEFAULT_CLASS_NAMES = [
    "other",
    "crater",
    "dark_dune",
    "slope_streak",
    "bright_dune",
    "impact_ejecta",
    "swiss_cheese",
    "spider",
]


def load_class_map(path: Path | None) -> dict[int, str]:
    """Load an index->name class map from ``path`` (format: ``index,name`` or
    ``index name`` per line). Falls back to the placeholder ordering if
    ``path`` is None or missing.
    """
    if path is None or not Path(path).exists():
        return dict(enumerate(DEFAULT_CLASS_NAMES))

    class_map: dict[int, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError(f"Malformed class-map line: {line!r}")
        idx_str, name = parts
        class_map[int(idx_str)] = name
    return class_map


def parse_label_file(path: Path) -> list[tuple[str, int]]:
    """Parse a ``filename label_idx`` per-line label file into samples.

    Blank lines and lines starting with ``#`` are skipped.
    """
    samples: list[tuple[str, int]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Malformed label line: {line!r}")
        filename, label_str = parts
        samples.append((filename, int(label_str)))
    return samples
