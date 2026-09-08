# Data-Speaks

**How does the mathematical structure of multichannel EEG change during epileptic seizures, and can these changes be used to characterize and distinguish ictal from interictal brain states?**

Data-Speaks is a rigorous, research-capable computational neuroscience project built around the CHB-MIT Scalp EEG Database. It is developed as part of the 3rd-semester course *Mathematics for Data Science* at the **Indian Institute of Technology Dharwad (IIT Dharwad)**, Department of Mathematics and Computing.

> **Scope disclaimer:** This is an educational/research-oriented project. It does **not** claim clinical usefulness or diagnostic validity.

---

## Table of Contents

1. [Research Overview](#1-research-overview)
2. [Dataset](#2-dataset)
3. [Project Architecture](#3-project-architecture)
4. [The Six-Stage Pipeline](#4-the-six-stage-pipeline)
5. [Common Stage 0 — Mandatory Standardization](#5-common-stage-0--mandatory-standardization)
6. [Candidate Preprocessing Pipelines](#6-candidate-preprocessing-pipelines)
7. [Team Collaboration Conventions](#7-team-collaboration-conventions)
8. [Evaluation Philosophy](#8-evaluation-philosophy)
9. [Repository Maintenance Notes](#9-repository-maintenance-notes)
10. [Getting Started](#10-getting-started)
11. [Current Status & Roadmap](#11-current-status--roadmap)
12. [References](#12-references)

---

## 1. Research Overview

The project investigates how the mathematical structure of multichannel EEG changes during epileptic seizures. Instead of a one-off ML exercise, Data-Speaks implements a **reproducible pipeline** that demonstrates concepts from Mathematics for Data Science end to end:

```
Dataset exploration
  → Preprocessing
  → Segmentation
  → Feature extraction
  → Dimensionality reduction
  → Clustering
  → Classification
  → Quantitative evaluation
  → Interpretation
```

The same philosophy governs every stage: **multiple methodologically distinct candidate algorithms are proposed, implemented, and compared under a common evaluation protocol; the empirically strongest method is carried forward.**

We do not claim any algorithm is optimal in advance — we demonstrate *why* a method was selected.

---

## 2. Dataset

**CHB-MIT Scalp EEG Database** — [PhysioNet](https://physionet.org/content/chbmit/1.0.0/), version 1.0.0.

- Multichannel scalp EEG from pediatric patients with intractable epilepsy
- EDF format, seizure annotations included
- ~42.6 GB raw (large but manageable; never committed to Git)

### Local layout

The raw dataset lives at:

```
data/raw/physionet.org/
```

Raw data is **immutable**. All processing writes to `data/processed/`, never back to `data/raw/`.

### Dataset audit (recorded channel counts)

| Channels | Files |
|---|---|
| 22 | 26 |
| 23 | 276 |
| 24 | 54 |
| 25 | 1 |
| 28 | 275 |
| 29 | 14 |
| 31 | 1 |
| 38 | 39 |

**Special montage:** 3 recordings use unipolar/CS2-referenced channel names (e.g., `F7-CS2`, `T7-CS2`, `FP1-CS2`). These are **valid recordings, not corrupted data** — they are handled through a separate special-montage branch and tracked separately in benchmarks.

---

## 3. Project Architecture

```
Data-Speaks/
├── data/
│   ├── raw/physionet.org/          ← CHB-MIT (gitignored)
│   └── processed/
├── pipeline/
│   ├── 01_preprocessing/
│   │   ├── README.md               ← common preprocessing contract
│   │   ├── baseline/
│   │   ├── candidate_algorithms/
│   │   │   ├── filtering/
│   │   │   ├── artifact_removal/
│   │   │   ├── normalization/
│   │   │   └── resampling/
│   │   └── evaluation/             ← shared evaluator
│   ├── 02_segmentation/            ← README + candidates + evaluation
│   ├── 03_feature_extraction/      ← time/frequency/statistical + evaluation
│   ├── 04_dimensionality_reduction/← pca / svd / ica + evaluation
│   ├── 05_clustering/              ← kmeans / hierarchical / dbscan + evaluation
│   └── 06_classification/          ← LR / RF / SVM / KNN + evaluation
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing_comparison.ipynb
│   ├── 03_feature_analysis.ipynb
│   ├── 04_dimensionality_reduction.ipynb
│   ├── 05_clustering_comparison.ipynb
│   └── 06_classification_comparison.ipynb
├── results/
│   ├── figures/
│   ├── tables/
│   └── benchmarks/
├── reports/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 4. The Six-Stage Pipeline

Each stage follows the same cycle:

```
common standardized input
  → candidate method A/B/C/D
  → shared evaluation under identical conditions
  → empirical selection of the winner
  → winner feeds the next stage
```

| Stage | Focus | Candidate families |
|---|---|---|
| 01 Preprocessing | Signal cleaning & standardization | FIR/Butterworth/Chebyshev filters, ICA, wavelets, ASR, robust scaling |
| 02 Segmentation | Windowing | (to be designed) |
| 03 Feature Extraction | Time / frequency / statistical features | (to be designed) |
| 04 Dimensionality Reduction | Linear & subspace methods | PCA, SVD, ICA |
| 05 Clustering | Unsupervised structure | K-Means, hierarchical, DBSCAN |
| 06 Classification | Supervised discrimination | Logistic Regression, Random Forest, SVM, KNN |

Each stage directory contains a `README.md` documenting the stage's **common contract** (inputs, outputs, parameters, policies), plus `candidate_algorithms/` (one folder per method) and a shared `evaluation/` (used by everyone — never customized to favor one implementation).

---

## 5. Common Stage 0 — Mandatory Standardization

Before any candidate preprocessing method is applied, **every** recording passes through the same shared Stage 0:

```
Raw EDF
  → EDF integrity checks
  → Annotation parsing
  → Remove dummy / non-EEG channels
  → Remove ECG, VNS, other auxiliary channels
  → Standardized EEG channel selection
  → Special montage detection
  → Quality checks
  → Standardized EEG input
  → Candidate pipeline A/B/C/D
```

Stage 0 is **shared and immutable** across candidate pipelines. If team members used different channel policies or annotation interpretations, the preprocessing comparison would be invalid.

### Common channel policy

- Keep EEG channels; remove ECG, VNS, and dummy/non-EEG/auxiliary channels
- Preserve original EEG channel names
- Establish a canonical EEG channel set for standard-montage recordings
- Missing channels are **not** automatically treated as corruption
- No interpolation during Stage 0 unless explicitly justified later
- CS2/unipolar recordings are handled in a separate branch, never silently forced into the standard montage

### Common quality-control (QC) policy

QC **flags** problems rather than deleting suspicious data:

| Check | Detects |
|---|---|
| `FLAT` | nearly zero variance |
| `HIGH_AMPLITUDE` | extreme values vs. robust channel distribution |
| `HIGH_VARIANCE` | abnormally large window variance |
| `HIGH_FREQ` | excessive muscle/electrical artifact |
| `CLIPPED` | repeated min/max values |
| `INVALID` | NaN/Inf or invalid numerical values |

Valid status: `GOOD`. MNE-Python utilities may support detection, but the QC policy must remain explicit and documented in code — no undocumented defaults.

### Common windowing policy

- Window length: **4 s**; overlap: **50%**
- Target sampling rate: **256 Hz** → `4 × 256 = 1024` samples per window
- Identical for all pipelines and all downstream comparisons

### Common seizure-labeling policy

For each window, `seizure_overlap = duration(window ∩ seizure_interval)`:

| Condition | Label |
|---|---|
| ≥ 50% overlap with a seizure interval | `1` (ictal) |
| No overlap, sufficiently away from seizure boundaries | `0` (interictal) |
| Overlaps a seizure but below the ictal threshold | `-1` (ambiguous, excluded from supervised training) |

### Leakage prevention

All supervised evaluation uses **patient-wise splits**:

```
Patients_train ∩ Patients_val = ∅
Patients_train ∩ Patients_test = ∅
Patients_val ∩ Patients_test = ∅
```

Windows from the same patient are never scattered across splits — adjacent windows are highly correlated, and patient-specific characteristics would otherwise inflate metrics.

---

## 6. Candidate Preprocessing Pipelines

Four complete, methodologically distinct candidate pipelines are compared. *(4 substeps × 4 choices would be 256 combinations — we deliberately do **not** grid-search; we design four coherent pipelines.)*

### Pipeline A — Conservative / Classical *(reference baseline)*
```
FIR band-pass → ICA artifact removal → per-channel Z-score → 256 Hz
```
Controlled, linear-phase filtering; the most established multichannel EEG artifact-removal approach; simple, interpretable normalization.

### Pipeline B — Adaptive / Time-Frequency
```
Butterworth band-pass → Wavelet denoising → Robust scaling → 256 Hz (polyphase)
```
Wavelet denoising operates in a time-frequency representation (fundamentally different from ICA's independent-source decomposition); robust scaling handles transient extremes; polyphase resampling folds anti-alias filtering into resampling. *(Wavelet family and thresholding strategy must be specified before implementation.)*

### Pipeline C — Subspace-Based / Aggressive
```
Butterworth band-pass → ASR → ICA → per-channel standardization → 256 Hz
```
ASR targets contaminated covariance subspaces via sliding-window PCA reconstruction — a strong mathematical match to the course content. **Caveat:** the ASR → ICA stacking is a candidate ordering to benchmark, not an assumed improvement; over-cleaning that removes seizure-relevant structure must be evaluated.

### Pipeline D — Lightweight / Statistical
```
Chebyshev Type II band-pass → robust statistical artifact detection → per-channel standardization → decimation
```
Tests whether sophisticated artifact-removal justifies its computational cost. Final decimation factor fixed before comparison.

### How pipelines are compared

**Level 1 — Intrinsic quality:** artifact suppression, signal/spectral/waveform preservation, SNR under controlled synthetic contamination, runtime, memory.

**Level 2 — Downstream usefulness:** identical segmentation, features, models, patient-wise splits, and metrics — **only** the preprocessing varies.

Winner is **never** chosen on raw classification accuracy alone, and never on a tiny numerical difference without statistical consideration (per-patient metrics + paired tests / bootstrap CIs).

---

## 7. Team Collaboration Conventions

- **Everyone learns the full pipeline** — no one is permanently siloed into one algorithm.
- Each member implements one **complete** candidate pipeline (filtering → artifact removal → normalization → resampling) from the shared Stage 0 input.
- All evaluation uses the **shared evaluator** under identical conditions.
- Independent implementations append author initials to filenames to avoid merge conflicts:
  ```
  butterworth_AT.py
  wavelet_RK.py
  ica_AS.py
  ```
- Raw dataset and generated processed data are **never** committed (see `.gitignore`). Only small, deliberate artifacts may be committed when needed.
- Empty structural directories are kept in Git via `.gitkeep` files.

---

## 8. Evaluation Philosophy

1. Raw data is immutable.
2. All team members use the same Stage 0.
3. All candidate pipelines receive identical standardized input.
4. All pipelines use the same windowing and labeling policy.
5. All supervised evaluation uses patient-wise splitting.
6. Never compare pipelines with different downstream classifiers or feature sets.
7. Never choose algorithms based only on visual appearance.
8. Never declare a winner on tiny numerical differences without statistics.
9. Never assume more aggressive preprocessing is better.
10. Never process the full dataset until the pipeline is validated on representative recordings.
11. Document every parameter and decision.
12. Treat the three CS2-reference recordings separately — they are not corrupted.

Real EEG has no perfectly clean ground truth, so intrinsic evaluation combines real-data inspection, spectral/waveform preservation analysis, QC metrics, **controlled synthetic contamination** (where an approximate clean reference exists), and downstream validation. *Visual smoothness ≠ better preprocessing.*

---

## 9. Repository Maintenance Notes

- **GitHub repo maintainer:** Pipeline A owner (repo admin duties — PR reviews, branch hygiene, releases).
- `.gitignore` must exclude:
  ```
  data/raw/
  data/processed/
  *.edf
  __pycache__/
  .ipynb_checkpoints/
  ```
- Empty folders (e.g., `pipeline/01_preprocessing/evaluation/`, `results/figures/`) are tracked with `.gitkeep`.
- Every stage has exactly one shared `README.md` (the contract) and one shared `evaluation/` — changes to these require team agreement, not unilateral edits.

---

## 10. Getting Started

```bash
# 1. Clone
git clone https://github.com/<org>/Data-Speaks.git
cd Data-Speaks

# 2. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Dataset (not in Git — download separately)
# Place CHB-MIT under data/raw/physionet.org/ preserving the PhysioNet layout

# 4. Explore
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

Core dependencies: `mne`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pyedflib` (see `requirements.txt`).

---

## 11. Current Status & Roadmap

**Current stage:** Project architecture & preprocessing methodology design ✅ (repo scaffold, pipeline definitions, common policies finalized)

**Next steps:**

- [ ] Write `pipeline/01_preprocessing/README.md` (the formal common contract)
- [ ] Implement shared **Stage 0** (EDF integrity, annotation parsing, channel standardization, special-montage detection, QC, metadata generation)
- [ ] Dataset-wide audit → master `metadata.csv`
- [ ] Select ~6 representative benchmark recordings (+ 3 CS2 recordings as a separate special-case set)
- [ ] Implement candidate pipelines A–D on the benchmark
- [ ] Pilot comparison (waveform, PSD, artifact suppression, runtime, size)
- [ ] Freeze pipeline definitions → scale to the full dataset
- [ ] Proceed through stages 02–06 with the same candidate→evaluate→select cycle

---

## 12. References

- **Oppenheim & Schafer** — *Discrete-Time Signal Processing* (primary DSP theory)
- **Nunez & Srinivasan** — *Electric Fields of the Brain* (EEG-specific theory)
- **Goldberger et al.** — *PhysioBank, PhysioToolkit, and PhysioNet* (CHB-MIT source)
- **MNE-Python documentation** — practical EEG implementation
- Project reports → `reports/`

---

*Data-Speaks — a course project by the Mathematics and Computing team, IIT Dharwad.*
