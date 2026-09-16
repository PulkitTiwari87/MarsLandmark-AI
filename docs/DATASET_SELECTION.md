# Dataset Selection — MarsLandmark-AI

Status: **Phase 01 — candidate comparison complete, primary dataset selected.**
This document records the candidates considered, the criteria used, and the
reasoning for the selection. Nothing in this document is a training result —
no model has been trained yet.

## Research method

Candidates were identified via web search against NASA/JPL, NASA Technical
Reports Server (NTRS), NASA's Open Data Portal (data.nasa.gov), Zenodo dataset
records, and the peer-reviewed literature describing them. Direct `WebFetch`
access to `zenodo.org`, `data.nasa.gov`, `ntrs.nasa.gov`, `huggingface.co`, and
`arxiv.org` was **blocked by this session's network egress proxy** (only
`github.com`/`raw.githubusercontent.com` were reachable for verification).
Facts below are therefore drawn from `WebSearch` result summaries and
third-party mirrors/reimplementations that cite the primary sources, not from
directly fetching the primary record pages. Every number below is cited to
where it was found; anything not independently cross-checked against the live
Zenodo/data.nasa.gov record is marked **PENDING VERIFICATION**. This must be
re-verified against the primary source from an unrestricted network before
data acquisition (Phase 02) is finalized. See the "Environment constraint"
note in `docs/DATASET.md`.

## Candidates considered

### 1. AI4Mars — "A Dataset for Terrain-Aware Autonomous Driving on Mars"

- Source: NASA/JPL, built via the Zooniverse "AI4Mars" citizen-science
  crowdsourcing project ([zooniverse.org/projects/hiro-ono/ai4mars](https://www.zooniverse.org/projects/hiro-ono/ai4mars)).
- Publication: Swan et al., "AI4MARS: A Dataset for Terrain-Aware
  Autonomous Driving on Mars," CVPR 2021 Workshops (AI4Space).
  ([NTRS record](https://ntrs.nasa.gov/citations/20220008371), [IEEE Xplore](https://ieeexplore.ieee.org/document/9523149/))
- Task supported: **semantic segmentation** (pixel-level terrain labels).
- Classes: 4 terrain types — soil, sand, bedrock, big rock.
- Reported scale (original release): ~326K crowdsourced full-image
  segmentation labels over ~35K images from Curiosity (MSL) Navcam/Mastcam
  and Spirit/Opportunity (MER) Navcam. ~1.5K additional labels were produced
  by MSL/MER rover planners and scientists for validation.
  A later, expanded release reportedly includes ~425K labels over ~50K
  images and adds Perseverance imagery — **PENDING VERIFICATION** (found via
  search snippet only, could not open the record directly).
- Availability: `data.nasa.gov` / `catalog.data.gov` listing, and a Zenodo
  record (DOI referenced as `10.5281/zenodo.15995036` in search results —
  **PENDING VERIFICATION**, not independently confirmed).
- License: not confirmed from a source this session could reach —
  **PENDING VERIFICATION**.

### 2. HiRISE Orbital Landmark Dataset ("Mars orbital image (HiRISE) labeled data set", DeepMars project)

- Source: NASA/JPL Machine Learning group; imagery from the HiRISE camera
  (Mars Reconnaissance Orbiter). Built for the NASA Planetary Data System (PDS)
  Imaging Atlas.
- Publication: Wagstaff, K.L. et al., "Deep Mars: CNN Classification of Mars
  Imagery for the PDS Imaging Atlas," IAAI 2018; extended in Wagstaff et al.,
  "Mars Image Content Classification: Three Years of NASA Deployment and
  Recent Advances" (2021).
- Task supported: **multi-class image classification** of cropped landmark
  patches.
- Classes (8): crater, bright dune, dark dune, slope streak, impact ejecta,
  swiss cheese, spider, other.
- Reported scale (version 3 / 3.2): 10,815 labeled landmark crops
  (pre-augmentation), extracted from HiRISE browse images; an augmented
  version expands this several-fold. Source-image counts and augmented totals
  varied across search results found for this dataset (180 vs. 232 source
  images; ~62.6K vs. ~64.9K augmented crops) — **PENDING VERIFICATION**
  against the primary Zenodo record before being cited as fact. A released
  **train/validation/test split of 6,997 / 2,025 / 1,793** (summing to
  10,815) was reported and, notably, is described as grouped by source image
  — directly relevant to leakage prevention (see `docs/DATA_SPLIT.md` once
  written).
- Availability: Zenodo record(s) referenced under DOIs `10.5281/zenodo.1048301`
  (original) and later versioned records; also listed on `data.nasa.gov`.
  Multiple DOIs found across versions — **PENDING VERIFICATION** of which is
  canonical for v3.2.
- License: not confirmed from a source this session could reach —
  **PENDING VERIFICATION**.
- Third-party reference implementation exists (`niehusst/HiRISE-Net` on
  GitHub) confirming dataset structure (`map-proj/` image crops +
  `labels-map-proj.txt`) and citing the same Zenodo DOI.

### 3. DeepMars MSL Surface Image Classification Dataset

- Source: Same NASA/JPL group (Wagstaff et al.), imagery from Curiosity
  (MSL) Mastcam/Navcam/MAHLI.
- Task supported: **multi-class scene/content classification** of full rover
  images (e.g. dashed sub-categories such as martian landscape, drill holes,
  wheels — exact class list not verified in this session).
- Availability: Zenodo (DOI referenced as `10.5281/zenodo.1049137` in search
  results — **PENDING VERIFICATION**).
- Scale/class details: **NOT YET VERIFIED** — this session did not confirm
  exact image/class counts before deprioritizing this candidate.

## Comparison against defined criteria

| Criterion | AI4Mars | HiRISE Landmark (DeepMars) | DeepMars MSL Surface |
|---|---|---|---|
| Authenticity (real NASA/JPL data) | Yes | Yes | Yes |
| Task suitability for "Landmark Detection/Classification" | Partial — terrain types, not discrete landmarks | **Strong** — explicit landmark/geological-feature classes | Weak — general scene classification, not landmark-specific |
| Annotation type | Pixel-level segmentation (crowdsourced + expert-validated subset) | Image-level class labels (expert-curated by PDS annotators) | Image-level class labels |
| Class count | 4 | 8 | Not verified this session |
| Dataset size (pre-augmentation) | ~35K images / ~326K labels | 10,815 crops | Not verified this session |
| Official train/val/test split provided | Not confirmed | **Yes**, source-grouped (6,997/2,025/1,793) | Not verified this session |
| Class balance (qualitative, from literature) | Reasonably balanced (4 broad terrain types) | Known to be imbalanced — "other" and "crater" dominate over rare classes like "spider" (per Wagstaff et al.) | Not verified this session |
| Computational feasibility for this project | High (large volume, but segmentation is heavier to train) | High — small resolution (227×227), modest total size, tractable classification training | Not verified this session |
| License clarity (from this session's research) | Pending verification | Pending verification | Pending verification |
| Reproducibility (documented pipeline in prior work) | Yes (Swan et al. 2021) | Yes (Wagstaff et al. 2018/2021), plus public reference implementations | Yes (Wagstaff et al. 2021) |

## Decision

**Primary dataset selected: HiRISE Orbital Landmark Dataset (DeepMars /
Wagstaff et al., NASA PDS Imaging Atlas, HiRISE camera).**

Rationale:

1. It is the only one of the three candidates whose annotation scheme is
   literally landmark classes (crater, dune types, slope streak, impact
   ejecta, swiss cheese terrain, spider terrain) rather than generic terrain
   texture or unrelated scene content — this is the closest real match to
   the project's stated goal (Section 0.4/1 of the project brief:
   "Mars Landmark Detection / Classification").
2. It provides an existing, source-image-grouped official train/val/test
   split, which materially helps satisfy the project's data-leakage
   requirements (Section 0.7) — subject to independently re-verifying that
   split from the primary record.
3. Its image size (227×227, per prior work) and total volume (10,815
   original crops, up to ~65K with augmentation) are computationally
   tractable for this project without requiring large-scale distributed
   training.
4. It is a genuinely difficult, realistic classification problem: known
   class imbalance (Section 14 of the project brief is directly relevant)
   and 8 visually-overlapping geological classes, rather than a trivially
   separable task.

AI4Mars remains a strong **secondary/future-work candidate**: it is real,
well-documented NASA data, but its task (terrain-type segmentation) is a
different ML problem (segmentation, not landmark classification) from the
project's stated primary goal, and it is not force-fit here.

This selection, and every number in this document marked PENDING
VERIFICATION, must be confirmed against the primary Zenodo/data.nasa.gov
record before Phase 02 (data acquisition) is executed for real. See
`docs/DATASET.md`.
