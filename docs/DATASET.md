# Dataset Provenance — MarsLandmark-AI

Status: **PENDING FINAL VERIFICATION.** This document records what is known
about the selected dataset from this session's research. Fields marked
`PENDING VERIFICATION` must be confirmed against the primary source before
Phase 02 (acquisition) proceeds, per the project's provenance requirements.

## Environment constraint (read first)

This session's outbound network is restricted by an egress proxy that
**blocks** `zenodo.org`, `data.nasa.gov`, `ntrs.nasa.gov`, `huggingface.co`,
and `arxiv.org` — the actual hosts of the dataset and its documentation.
Only `github.com`/`raw.githubusercontent.com` were reachable. This means:

- All facts in this document come from `WebSearch` result summaries and a
  third-party GitHub reference implementation, not from directly reading the
  primary Zenodo/data.nasa.gov record.
- **Actual dataset download (Phase 02) cannot be executed from inside this
  sandboxed session.** The acquisition script below is written to target the
  documented primary sources and is intended to run in an environment with
  unrestricted network access (a developer's machine, CI, or a differently
  configured cloud environment).
- No training, evaluation, or benchmark numbers can be produced until the
  data is actually downloaded and verified. Any such numbers appearing
  elsewhere in this repository before that point would be fabricated and
  are explicitly disallowed by this project's rules.

## Dataset identity

| Field | Value | Verification status |
|---|---|---|
| Official name | Mars orbital image (HiRISE) labeled data set | Confirmed (multiple independent sources) |
| Common/project name | DeepMars — HiRISE landmark classification dataset | Confirmed |
| NASA organization/mission | NASA / JPL, Mars Reconnaissance Orbiter (MRO) | Confirmed |
| Instrument | HiRISE (High Resolution Imaging Science Experiment) | Confirmed |
| Originating publication | Wagstaff, K.L., Lu, Y., Stanboli, A., Grimes, K., Gowda, T., Padams, J. "Deep Mars: CNN Classification of Mars Imagery for the PDS Imaging Atlas." IAAI 2018. | Confirmed |
| Follow-up publication | Wagstaff et al., "Mars Image Content Classification: Three Years of NASA Deployment and Recent Advances," 2021 (arXiv:2102.05011) | Confirmed to exist; content not independently re-read (arXiv blocked this session) |
| Hosting | Zenodo (DOI `10.5281/zenodo.1048301` for the original release; later versions carry different DOIs seen in search results, e.g. `10.5281/zenodo.2538136`, `10.5281/zenodo.4002935`) and mirrored listing on `data.nasa.gov` | **PENDING VERIFICATION** — which DOI is canonical for "version 3.2" was not resolved this session |
| Dataset version to acquire | Version 3.2 (as referenced by third-party reimplementations) | **PENDING VERIFICATION** |
| License / usage terms | Not confirmed | **PENDING VERIFICATION — must confirm before any redistribution or commercial use** |

## Task and annotations

| Field | Value | Verification status |
|---|---|---|
| Primary task | Multi-class image classification | Confirmed by dataset structure (per-crop single label) |
| Classes (8) | crater, bright dune, dark dune, slope streak, impact ejecta, swiss cheese, spider, other | Confirmed (consistent across multiple search results) |
| Class count | 8 | Confirmed |
| Original landmark crops (pre-augmentation) | 10,815 | Reported consistently across sources; matches the sum of the reported split (6,997 + 2,025 + 1,793 = 10,815) |
| Source HiRISE browse images | Reported as 180 in one source and 232 in another | **PENDING VERIFICATION — conflicting figures found, not resolved this session** |
| Augmented crop total | Reported as ~62,598 in one source and 64,947 in another | **PENDING VERIFICATION — conflicting figures found, not resolved this session** |
| Image resolution | 227×227 px (crop, resized) | Reported by originating paper and reference implementation; **not independently re-derived from raw files this session** |
| Official train/validation/test split | 6,997 / 2,025 / 1,793 (source-image-grouped) | Reported by search result summarizing the primary paper; **PENDING VERIFICATION against primary record** |
| Annotation format | Single class label per crop image, provided as a flat label file (`labels-map-proj.txt` in the reference implementation) mapping filename → class index | Confirmed structurally via `niehusst/HiRISE-Net` GitHub repo, which cites the same Zenodo DOI |
| Known limitations | Documented class imbalance in the source literature ("other" and "crater" reported as dominant classes); imbalance must be re-measured directly from the acquired data in Phase 03, not assumed from secondary sources | Per project rule — do not assume, measure |

## Acquisition plan

See `src/data/download.py`, which targets:

1. Primary: the Zenodo record for "Mars orbital image (HiRISE) labeled data
   set" (DOI to be confirmed — see PENDING VERIFICATION above).
2. Cross-check: the `data.nasa.gov` / `catalog.data.gov` listing for the
   same dataset, to confirm the Zenodo record is the NASA-endorsed source
   and not an unofficial mirror.

The script will not run to completion inside this sandboxed session because
`zenodo.org` and `data.nasa.gov` are network-blocked here. It is written and
tested for correctness of structure (argument parsing, checksum verification
logic, resumability) but the actual HTTP transfer has not been exercised
end-to-end in this session.

## Outstanding actions before Phase 02 can be marked complete

1. From an unrestricted network, open the primary Zenodo record and resolve:
   the canonical DOI for v3.2, exact source-image count, exact augmented
   crop count, and the license.
2. Confirm the `data.nasa.gov` listing points to the same record.
3. Run `src/data/download.py` for real, record the SHA-256 checksums of the
   downloaded archive(s), and update this document with `CONFIRMED` values
   in place of every `PENDING VERIFICATION`.
4. Only after that: proceed to Phase 03 (validation/EDA) using the real,
   downloaded files — never using the numbers in this document as a
   substitute for direct inspection.
