"""Reproducible acquisition of the HiRISE landmark dataset (DeepMars).

Primary source: Zenodo record 2538136, "Mars orbital image (HiRISE)
labeled data set version 3" (Wagstaff et al., DOI 10.5281/zenodo.2538136).
Confirmed via NASA's Open Data Portal listing (data.nasa.gov), which links
directly to this record, and cross-checked against the archive's own
bundled README.txt. See docs/DATASET.md for full provenance.

Note: v3.2 (Zenodo record 4002935) is a *different, later* record with a
different DOI — do not conflate the two. This project uses v3.

Usage:
    python -m src.data.download --output-dir data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from urllib.request import urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PRIMARY_URL = "https://zenodo.org/records/2538136/files/hirise-map-proj-v3.zip"
# Verified by this project against a copy of the archive obtained 2026-09-17;
# see docs/DATASET.md "Confirmed measurements" for how this was computed.
EXPECTED_SHA256: str | None = "e22ee769a61986082f76d762182c572ce2b9ae6b7a142b23a14e46631c9e4d06"


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    if dest.exists():
        logger.info("File already exists, skipping download: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s -> %s", url, dest)
    with urlopen(url) as response, open(tmp_dest, "wb") as f:
        while chunk := response.read(chunk_size):
            f.write(chunk)
    tmp_dest.rename(dest)


def verify_checksum(path: Path, expected_sha256: str) -> bool:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected_sha256:
        logger.error("Checksum mismatch for %s: expected %s, got %s", path, expected_sha256, actual)
        return False
    logger.info("Checksum verified for %s", path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--url", type=str, default=PRIMARY_URL)
    args = parser.parse_args()

    if "REPLACE_WITH" in args.url:
        logger.error(
            "PRIMARY_URL is still a placeholder — the canonical dataset DOI/version "
            "has not been verified yet (see docs/DATASET.md, 'Outstanding actions'). "
            "Refusing to download an unverified/guessed URL."
        )
        return 1

    dest = args.output_dir / Path(args.url).name
    download_file(args.url, dest)

    if EXPECTED_SHA256:
        if not verify_checksum(dest, EXPECTED_SHA256):
            return 1
    else:
        logger.warning(
            "No EXPECTED_SHA256 set yet — checksum was not verified. "
            "Record the checksum of this download and update EXPECTED_SHA256."
        )

    logger.info("Download complete: %s", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
