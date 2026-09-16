"""Shared helpers for parsing HiRISE landmark dataset filenames.

Filenames look like: ESP_011623_2100_RED-0069.jpg or
ESP_011623_2100_RED-0069-r90.jpg (augmented variant). The part before the
crop id identifies the source HiRISE "RED strip" image.
"""

from __future__ import annotations

import re

STRIP_ID_PATTERN = re.compile(r"^([A-Z]+_\d+_\d+_RED)-(\d+)(.*)\.jpg$")


def strip_id_for(filename: str) -> str | None:
    """Return the source RED-strip id for a crop filename, or None if unparseable."""
    m = STRIP_ID_PATTERN.match(filename)
    return m.group(1) if m else None
