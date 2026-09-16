# Dataset Provenance — MarsLandmark-AI

Status: **ACQUIRED AND VERIFIED (2026-09-17).** The archive was obtained by
the user directly (this repository's automated downloader could not reach
`zenodo.org` from the cloud sandbox it was originally written in — see
"Acquisition method" below) and has now been inspected directly: extracted,
checksummed, and every figure below either measured from the actual files
on disk or corroborated against NASA's own Open Data Portal. Fields that
remain genuinely unresolved are still marked `PENDING VERIFICATION`, but
that list is now short.

## Acquisition method

- The archive `hirise-map-proj-v3.zip` was supplied directly by the user
  into this local session's `dataset/` directory (not downloaded by
  `src/data/download.py`), because Zenodo is unreachable from the cloud
  sandbox where Phase 00/01 were originally scaffolded.
- This local session independently re-checked network reachability:
  `zenodo.org` returned a **504 Gateway Timeout** consistently across three
  different tools (`curl`, `WebFetch`, and the browser pane), while
  `data.nasa.gov` and `doi.org` both resolved normally. This indicates a
  genuine upstream Zenodo outage at the time of this session, not a local
  network restriction — but it means the primary Zenodo record page itself
  still could not be read directly.
- `src/data/download.py` has been updated with the confirmed real URL and
  checksum (below) so that acquisition is reproducible from any environment
  with working access to Zenodo, but that script's HTTP path has still not
  been exercised end-to-end in *this* project (the file already in hand was
  used instead, after verifying its checksum matches what's recorded here).

## Dataset identity

| Field | Value | Verification status |
|---|---|---|
| Official name | Mars orbital image (HiRISE) labeled data set **version 3** | **CONFIRMED** — matches archive filenames (`hirise-map-proj-v3.zip`, `labels-map-proj-v3.txt`, `map-proj-v3/`), the archive's own bundled `README.txt`, and the NASA Open Data Portal listing title |
| Common/project name | DeepMars — HiRISE landmark classification dataset | Confirmed |
| NASA organization/mission | NASA / JPL, Mars Reconnaissance Orbiter (MRO) | Confirmed |
| Instrument | HiRISE (High Resolution Imaging Science Experiment) | Confirmed |
| Originating publication | Wagstaff, K.L., Lu, Y., Stanboli, A., Grimes, K., Gowda, T., Padams, J. "Deep Mars: CNN Classification of Mars Imagery for the PDS Imaging Atlas." IAAI 2018. | Confirmed |
| Follow-up publication | Wagstaff et al., "Mars Image Content Classification: Three Years of NASA Deployment and Recent Advances," 2021 (arXiv:2102.05011) | Confirmed to exist via search; full text not independently re-read |
| DOI | `10.5281/zenodo.2538136` | **CONFIRMED** — (a) printed verbatim in the archive's own `README.txt` ("please cite this DOI: 10.5281/zenodo.2538136"), (b) `https://doi.org/10.5281/zenodo.2538136` successfully resolves to `zenodo.org/record/2538136`, (c) NASA's Open Data Portal listing for "version 3" links directly to `https://zenodo.org/record/2538136` |
| Hosting | Zenodo record 2538136, mirrored/listed on NASA's Open Data Portal (`data.nasa.gov`, resource id `c93bf426-1eae-4d3b-8afd-548add5e24ce`, organization = NASA) | Confirmed — data.nasa.gov page loaded and read directly this session |
| Dataset version acquired | **Version 3** (this archive) — NOT version 3.2 | **CORRECTED.** Earlier scaffolding assumed "v3.2" from a third-party reimplementation reference. Version 3.2 is a *separate* Zenodo record (4002935) with a different DOI. The file actually in hand (`hirise-map-proj-v3.zip`) is version 3. `configs/config.yaml` has been corrected accordingly. |
| License / usage terms | NASA's Open Data Portal resource page explicitly shows **"License not specified"** for this resource (directly observed, 2026-09-17). A secondary web-search aggregator summary claimed "Creative Commons Attribution," but this could not be independently confirmed against the primary Zenodo record itself, which returned 504 errors throughout this session. | **PENDING VERIFICATION** — treat as "license unconfirmed" for any redistribution/commercial-use decision until the primary Zenodo page is readable, or contact NASA/JPL directly. Do not rely on the secondhand CC-BY claim. |

## Task and annotations — confirmed by direct measurement

The figures below were computed directly from the extracted archive
(`data/raw/`), not taken from secondary sources.

| Field | Value | Method |
|---|---|---|
| Primary task | Multi-class image classification | Confirmed by structure: one label integer per image file |
| Classes (8) | 0=other, 1=crater, 2=dark dune, 3=slope streak, 4=bright dune, 5=impact ejecta, 6=swiss cheese, 7=spider | Read directly from `landmarks_map-proj-v3_classmap.csv` |
| Total labeled images | **73,031** | `len(open('labels-map-proj-v3.txt').readlines())`, and confirmed equal to the file count in `map-proj-v3/` on disk |
| Label ↔ image match | 100% — every label has a matching image file and vice versa | Set difference computed both directions; 0 mismatches |
| Original landmark crops (pre-augmentation) | **10,433**, from **180** source HiRISE browse images | Stated by the archive's own `README.txt`; corroborated by NASA Open Data Portal description text (verbatim match) and by the search-aggregator summary. NOTE: the earlier scaffolding's figure of "10,815 originals" and "6,997/2,025/1,793 split" does not appear anywhere in this archive's own files and could not be corroborated — likely confusion with a different dataset version or a split defined only in the paper, not shipped in this archive. Treat the 10,815/split figures as **unconfirmed and probably wrong for v3**. |
| Augmented crop total | **62,598** (6 augmentations per original: 90°/180°/270° rotation, horizontal flip, vertical flip, random brightness) | Matches `10,433 × 6 = 62,598`; `10,433 + 62,598 = 73,031` ties out exactly against the measured total |
| Image resolution | **227×227 px, single-channel (grayscale, PIL mode `L`)** | Measured directly: opened and verified all 73,031 images; dimension distribution is `{(227,227): 73031}` with zero exceptions; mode distribution is `{'L': 73031}` — **not** 3-channel RGB, which matters for any pretrained-ImageNet backbone (Phase 06) |
| Corrupted/unreadable images | **0** | `PIL.Image.verify()` run on all 73,031 files; zero failures |
| Exact byte-identical duplicate images | **0** | MD5 of raw file bytes computed for all 73,031 files; zero collisions |
| Official train/validation/test split | **None present in this archive.** No split file, split column, or split directory exists anywhere in the zip — only a single flat label file covering all 73,031 images. | Directly inspected the full archive namelist; confirmed no other top-level files beyond `README.txt`, `labels-map-proj-v3.txt`, `landmarks_map-proj-v3_classmap.csv`, and `map-proj-v3/`. A custom, leakage-safe split must be constructed — see `docs/DATA_SPLIT.md`. |
| Annotation format | Flat text file `labels-map-proj-v3.txt`, one line per image: `<filename>.jpg <class_id>` | Confirmed directly |
| Class distribution (measured, all 73,031 images, includes augmentation) | 0 other: 61,054 · 1 crater: 4,900 · 2 dark dune: 1,141 · 3 slope streak: 2,331 · 4 bright dune: 1,750 · 5 impact ejecta: 231 · 6 swiss cheese: 1,148 · 7 spider: 476 | Counted directly from `labels-map-proj-v3.txt`. Severe imbalance: class 0 ("other") is 83.6% of all images; the rarest class (impact ejecta) is 0.3%. This must inform Phase 08's loss/sampling choice and Phase 14's per-class metric reporting — accuracy alone will be misleading. |
| Source-image grouping (for leakage prevention) | **173** unique source HiRISE "RED strip" IDs derived by regex-parsing every filename (`{MISSION}_{ORBIT}_{ANGLE}_RED-{crop_id}[-{aug_suffix}].jpg` → group key = everything before the crop id). All 73,031 filenames matched the pattern; 0 parse failures. | **Discrepancy flagged, unresolved:** the archive's own README states 180 *source browse images*, but filename-derived grouping yields only 173 unique RED-strip identifiers. Possible explanations (not verified): some browse images share a RED-strip ID under this naming scheme, or some of the 180 source images contributed a crop whose filename doesn't follow the dominant pattern. This does not block using RED-strip grouping as the leakage-prevention key (grouping only needs to keep all crops from the *same* strip together, which it does for 100% of files) — but the exact browse-image count remains unresolved pending direct access to the primary Zenodo metadata. |
| Known limitations | Class imbalance (above) is real and measured, not assumed. Grayscale-only images (not RGB) constrain backbone choice/adaptation. No official split requires a custom, documented grouping strategy. | Directly measured this session, not inferred from literature |

## Archive integrity record

| Field | Value |
|---|---|
| Filename | `hirise-map-proj-v3.zip` |
| SHA-256 | `e22ee769a61986082f76d762182c572ce2b9ae6b7a142b23a14e46631c9e4d06` |
| Size | 985,889,206 bytes (≈940 MiB) |
| Computed | 2026-09-17, this session, via streaming SHA-256 over the full file |
| Extracted to | `data/raw/` (git-ignored; never committed — see `.gitignore`) |

Anyone reproducing this project should download
`https://zenodo.org/records/2538136/files/hirise-map-proj-v3.zip` and
verify it against the SHA-256 above before use — see
`src/data/download.py`, which now encodes this URL and checksum and
refuses to proceed on a mismatch.

## Outstanding actions (genuinely unresolved)

1. **License.** Confirm the actual Zenodo-recorded license once the
   primary record page is reachable (it returned 504 throughout this
   session). Do not redistribute this repository's copy of the dataset or
   make license claims in the README until this is resolved.
2. **180 vs. 173 source images.** Resolve why filename-derived grouping
   yields 173 unique RED-strip IDs against the documented 180 source
   images, if it ever becomes relevant beyond leakage-safe splitting (which
   works correctly today regardless of this count).
3. **The withdrawn 10,815 / 6,997-2,025-1,793 figures.** These appeared in
   earlier (pre-file-access) research summaries and could not be
   corroborated against this archive. If a differently-versioned or
   differently-sourced copy of this dataset does carry an official split,
   it is not the one in this archive — do not silently merge the two.
