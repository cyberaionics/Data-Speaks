# Data-Speaks: Multichannel EEG Mathematical Structure Analysis

**How does the mathematical structure of multichannel EEG change during epileptic seizures, and can these changes be used to characterize and distinguish ictal from interictal brain states?**

Data-Speaks is a computational neuroscience research pipeline built around the **CHB-MIT Scalp EEG Database**. Developed for the course *Mathematics for Data Science* at the **Indian Institute of Technology Dharwad (IIT Dharwad)**, Department of Mathematics and Computing.

> **Scope Disclaimer:** Educational and research-oriented project. Does **not** claim clinical usefulness or diagnostic validity.

---

## Table of Contents

1. [Research & Pipeline Overview](#1-research--pipeline-overview)
2. [Dataset & Local Layout](#2-dataset--local-layout)
3. [Repository Structure](#3-repository-structure)
4. [Completed Pipeline Stages & Scientific Methodology](#4-completed-pipeline-stages--scientific-methodology)
   - [Stage 0 — Standardization & Quality Control](#stage-0--standardization--quality-control)
   - [Pipeline C — Artifact Removal & Cleaning](#pipeline-c--artifact-removal--cleaning)
   - [Stage 02 — Segmentation](#stage-02--segmentation)
   - [Stage 03 — Feature Extraction](#stage-03--feature-extraction)
   - [Stage 04 — Dimensionality Reduction (PCA)](#stage-04--dimensionality-reduction-pca)
   - [Stage 05 — Unsupervised Clustering](#stage-05--unsupervised-clustering)
   - [Midterm 5-Pillar EDA](#midterm-5-pillar-eda)
   - [Stage 06 — Supervised Classification & Standalone Inference](#stage-06--supervised-classification--standalone-inference)
5. [Summary of Results & Empirical Validation](#5-summary-of-results--empirical-validation)
   - [PCA Variance Decomposition](#pca-variance-decomposition-72951-windows--442-features)
   - [Unsupervised Clustering Evaluation](#unsupervised-clustering-evaluation-110-pc-space)
   - [Supervised Classification Evaluation](#supervised-classification-evaluation-5-fold-groupkfold-by-recording_id)
   - [Standalone Inference Validation (chb01_03)](#standalone-inference-validation-chb01_03)
6. [How to Run the Code](#6-how-to-run-the-code)
7. [Repository Hygiene & Git Policies](#7-repository-hygiene--git-policies)
8. [References](#8-references)

---

## 1. Research & Pipeline Overview

The project investigates multichannel EEG dynamics during epileptic seizures by implementing a leakage-safe, reproducible data science pipeline:

```text
Raw EDF Data
  └── Stage 0: Channel Standardization (17 Bipolar Montage) & QC
       └── Pipeline C: Butterworth 1-40 Hz → ASR → FastICA → Z-Score
            └── Stage 02: 4.0s Sliding Windowing (50% Overlap, 256 Hz)
                 └── Stage 03: Feature Extraction (17 Ch x 26 Feats = 442 Features)
                      ├── Stage 04: Dimensionality Reduction (PCA to 95% Variance)
                      │    └── Stage 05: Unsupervised Clustering (K-Means & DBSCAN)
                      │         └── Midterm 5-Pillar EDA
                      └── Stage 06: Supervised Classification (GroupKFold CV)
                           └── Standalone EDF Inference (Annotation-free Seizure Detection)
```

Each stage follows strict principles:
- **No Data Leakage**: Scalers, PCA bases, and classifiers are fitted per-training-fold inside cross-validation without exposing held-out recording windows.
- **Labeling Policy**: Labels ($1 = \text{ictal}$, $0 = \text{interictal}$, $-1 = \text{ambiguous}$) are used strictly as supervised targets or post-hoc validation metrics, never during feature normalization or PCA transformation.
- **Grouped Validation**: Cross-validation uses `GroupKFold` on `recording_id` so windows from the same recording never cross the train/test split boundary.

---

## 2. Dataset & Local Layout

**CHB-MIT Scalp EEG Database** — [PhysioNet v1.0.0](https://physionet.org/content/chbmit/1.0.0/).
- Pediatric scalp EEG recordings with expert seizure start/end time annotations.
- Raw dataset resides locally under `data/raw/physionet.org/` (gitignored, raw EDFs are immutable).

### Subject CHB01 Metadata Summary
- **Recordings**: 42 EDF files (~40.55 total hours recorded)
- **Seizure Events**: 7 annotated seizure incidents total:
  - `chb01_03`: 2996s–3036s (40s)
  - `chb01_04`: 1467s–1494s (27s)
  - `chb01_15`: 1732s–1772s (40s)
  - `chb01_16`: 1015s–1066s (51s)
  - `chb01_18`: 1720s–1810s (90s)
  - `chb01_21`: 327s–420s (93s)
  - `chb01_26`: 1862s–1963s (101s)
- **Total Seizure Duration**: 442.0 seconds (7.37 minutes)
- **Mean Seizure Duration**: 63.14 seconds
- **Ictal Time Fraction**: 0.30% ($442\text{s} / 145,988\text{s}$)

---

## 3. Repository Structure

```text
Data-Speaks/
├── pipeline/
│   ├── 01_preprocessing/
│   │   ├── stage0.py                   ← Stage 0 standardization & parsing
│   │   ├── README.md
│   │   └── candidate_algorithms/
│   │       ├── pipeline_c.py           ← Butterworth -> ASR -> FastICA -> Z-score
│   │       ├── pipeline_c_compat.py    ← ASRpy / NumPy 2.x compatibility patch
│   │       ├── PIPELINE_A_README.md
│   │       └── PIPELINE_C_README.md
│   ├── 02_segmentation/
│   │   ├── segmentation.py             ← 4.0s windowing (50% overlap, 1024 samples)
│   │   └── README.md
│   ├── 03_feature_extraction/
│   │   ├── feature_extraction.py       ← 442 time/frequency/statistical features
│   │   └── README.md
│   ├── 04_dimensionality_reduction/
│   │   ├── dimensionality_reduction.py ← Feature isolation, StandardScaler, PCA
│   │   ├── test_dimensionality_reduction.py
│   │   └── README.md
│   ├── 05_clustering/
│   │   ├── clustering.py               ← K-Means (k=2..6) & DBSCAN clustering
│   │   └── README.md
│   ├── 06_classification/
│   │   ├── classification.py           ← GroupKFold CV, leakage-safe pipeline, metrics
│   │   ├── run_classification_chb01.py ← Classifier comparison & model saving runner
│   │   ├── inference.py                ← Standalone annotation-free EDF inference
│   │   └── README.md
│   ├── eda/
│   │   └── run_midterm_eda.py          ← Midterm 5-Pillar EDA generation
│   └── run_chb01_complete_batch.py    ← Master execution runner for patient chb01
├── models/
│   └── chb01_final_model/              ← Saved scaler, PCA, classifier & metadata
│       ├── scaler.pkl
│       ├── pca.pkl
│       ├── classifier.pkl
│       ├── feature_cols.pkl
│       └── metadata.json
├── results/
│   ├── figures/
│   │   ├── eda/                        ← 5-Pillar EDA figures (PNG)
│   │   ├── pca/                        ← Scree plots & PC1 vs PC2 scatter plots
│   │   ├── clustering/                 ← K-Means & DBSCAN cluster overlays
│   │   └── classification/             ← Model comparison & confusion matrices
│   └── tables/
│       ├── pipeline_c_chb01_metrics.csv
│       ├── chb01_pca_variance.csv      ← Explained variance per component
│       ├── chb01_pca_loadings.csv      ← Feature loading matrix
│       ├── chb01_clustering_metrics.csv← Silhouette, ARI, NMI scores
│       ├── chb01_clustered_windows.csv ← Window metadata + cluster assignments
│       ├── chb01_classification_metrics.csv      ← Summary metrics across models
│       ├── chb01_classification_fold_metrics.csv ← Per-fold CV breakdown
│       └── chb01_03_inference_predictions.csv    ← Standalone inference output
├── .gitignore
└── README.md
```

---

## 4. Completed Pipeline Stages & Scientific Methodology

### Stage 0 — Standardization & Quality Control
- Selects the canonical 17 bipolar EEG channel montage:
  `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`, `FP1-F3`, `F3-C3`, `C3-P3`, `P3-O1`, `FP2-F4`, `F4-C4`, `C4-P4`, `P4-O2`, `FP2-F8`, `F8-T8`, `T8-P8`, `P8-O2`, `FZ-CZ`.
- Removes non-EEG (ECG, VNS, dummy) channels.
- Flags artifacts (`FLAT`, `CLIPPED`, `HIGH_VARIANCE`).

### Pipeline C — Artifact Removal & Cleaning
- **Bandpass Filter**: 4th-order Butterworth zero-phase filter (1–40 Hz).
- **Artifact Subspace Reconstruction (ASR)**: Cutoff parameter $k=20.0$ on seizure-free calibration windows.
- **FastICA**: Removes eye-movement (EOG) and muscle artifacts based on proxy channel correlations.
- **Standardization**: Zero-mean, unit-variance scaling per channel.

### Stage 02 — Segmentation
- **Window Length**: 4.0 seconds (1024 samples at 256 Hz).
- **Stride**: 2.0 seconds (50% overlap).
- **Seizure Overlap Labeling**:
  - `label = 1` (Ictal): $\ge 50\%$ window overlap with annotated seizure interval.
  - `label = 0` (Interictal): $0\%$ overlap, outside boundary buffer.
  - `label = -1` (Ambiguous): $< 50\%$ overlap (excluded from classification).

### Stage 03 — Feature Extraction
- **26 Features per Channel** $\times 17$ channels = **442 numerical features per window**:
  - *Time-Domain*: Mean, variance, skewness, kurtosis, peak-to-peak amplitude, RMS, crest factor, zero-crossing rate, Hjorth activity/mobility/complexity.
  - *Frequency-Domain*: Welch PSD absolute/relative power in 5 canonical bands ($\delta$: 1–4Hz, $\theta$: 4–8Hz, $\alpha$: 8–12Hz, $\beta$: 12–30Hz, $\gamma$: 30–40Hz), spectral edge frequency (SEF95), spectral entropy, peak frequency.
- Extracted across **72,951 total windows** for patient `chb01`.

### Stage 04 — Dimensionality Reduction (PCA)
- Separates 7 metadata columns (`patient_id`, `recording_id`, `window_id`, `start_sec`, `end_sec`, `duration_sec`, `label`) from 442 EEG features.
- Fits `StandardScaler` and `PCA` basis across feature dimensions.
- 95% cumulative variance requires **110 principal components** (75.1% dimensionality reduction).

### Stage 05 — Unsupervised Clustering
- Executes **K-Means** ($k=2..6$) and **DBSCAN** ($\epsilon=12.0, \text{MinPts}=15$) on the 110-PC space.
- Computes Silhouette scores, ARI, and NMI. Demonstrates that unsupervised clustering groups by background physiological states rather than seizure labels due to extreme class imbalance ($99.70\%$ interictal vs $0.30\%$ ictal).

### Midterm 5-Pillar EDA
1. **Pillar 1: Clinical & Metadata EDA**: Class imbalance (0.30% ictal time) and seizure duration profile.
2. **Pillar 2: Time-Domain Statistical Moments**: Variance expansion and peak-to-peak amplitude surges during ictal windows.
3. **Pillar 3: Frequency-Domain Spectral Power**: Relative spectral band power distribution across the 5 canonical bands ($\theta$ increase $\Delta=+0.143$, $\delta$ share decrease $\Delta=-0.051$).
4. **Pillar 4: Time-Frequency STFT Spectrogram**: Short-time Fourier Transform transition across pre-ictal, ictal, and post-ictal states.
5. **Pillar 5: Cross-Channel Spatial Correlation**: Cross-channel Pearson correlation matrix as a synchrony proxy ($r=0.151$ interictal vs $r=0.192$ ictal).

### Stage 06 — Supervised Classification & Standalone Inference
- **Ambiguous Filtering**: Excludes 4 ambiguous windows (`label = -1`), leaving **72,947 usable windows** (72,721 interictal, 226 ictal).
- **Leakage Prevention**: Evaluates four models using 5-fold `GroupKFold` grouped by `recording_id`. Scaler, PCA (110 components), and classifiers are fitted inside each training split only.
- **Evaluated Classifiers**:
  1. **Linear SVM**: `LinearSVC(C=1.0, class_weight='balanced')` — computationally scalable linear formulation.
  2. **Logistic Regression**: `LogisticRegression(class_weight='balanced', max_iter=1000)`
  3. **Random Forest**: `RandomForestClassifier(n_estimators=100, class_weight='balanced')`
  4. **KNN**: `KNeighborsClassifier(n_neighbors=5, metric='euclidean')`
- **Model Selection Metric**: **PR-AUC (Average Precision)**, selected for extreme class imbalance. **Random Forest** achieved the top PR-AUC (**0.6310**) and was selected for the final model artifact.
- **Standalone Inference (`inference.py`)**: End-to-end pipeline that takes raw EDF input, executes Stage 0 → Pipeline C → Segmentation → Feature Extraction → saved Scaler → saved PCA → saved Classifier, outputting window-level predicted labels and probabilities without requiring ground-truth annotations.

---

## 5. Summary of Results & Empirical Validation

### PCA Variance Decomposition (72,951 Windows $\times$ 442 Features)

| Threshold | Principal Components Required | Feature Space Reduction |
|---|:---:|:---:|
| 80% Explained Variance | **35 PCs** | 92.1% reduction |
| 90% Explained Variance | **73 PCs** | 83.5% reduction |
| **95% Explained Variance** | **110 PCs** | **75.1% reduction** |
| 99% Explained Variance | **217 PCs** | 50.9% reduction |

### Unsupervised Clustering Evaluation (110-PC Space)

| Algorithm | Clusters ($K$) | Silhouette Score | Adjusted Rand Index (ARI) | Normalized Mutual Info (NMI) |
|---|:---:|:---:|:---:|:---:|
| **K-Means ($k=2$)** | 2 | **0.3142** | 0.0018 | 0.0031 |
| K-Means ($k=3$) | 3 | 0.2709 | 0.0022 | 0.0034 |
| K-Means ($k=4$) | 4 | 0.2315 | 0.0025 | 0.0040 |
| K-Means ($k=5$) | 5 | 0.2180 | 0.0028 | 0.0046 |
| K-Means ($k=6$) | 6 | 0.1954 | 0.0031 | 0.0049 |
| **DBSCAN** | 1 (118 noise) | 0.4105 | 0.0001 | 0.0012 |

### Supervised Classification Evaluation (5-Fold GroupKFold by `recording_id`)

| Model | PR-AUC (Primary) | F1 Score | Recall / Sensitivity | Specificity | ROC-AUC | Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Linear SVM** | 0.5725 ± 0.3077 | 0.4327 ± 0.2154 | 0.6548 ± 0.3102 | 0.9971 ± 0.0010 | 0.9597 ± 0.0331 | 0.9961 ± 0.0009 |
| **Logistic Regression** | 0.5730 ± 0.2985 | 0.4393 ± 0.2025 | **0.6592 ± 0.2793** | 0.9971 ± 0.0012 | **0.9653 ± 0.0277** | 0.9961 ± 0.0010 |
| **Random Forest (Selected)** | **0.6310 ± 0.3799** | 0.2858 ± 0.3361 | 0.2065 ± 0.2542 | **1.0000 ± 0.0000** | 0.9467 ± 0.0623 | 0.9974 ± 0.0021 |
| **KNN** | 0.5702 ± 0.4016 | **0.5390 ± 0.3687** | 0.4622 ± 0.3551 | 0.9999 ± 0.0001 | 0.8132 ± 0.1848 | **0.9984 ± 0.0011** |

> **Key Model Selection Finding:** Random Forest was selected as the top model based on mean PR-AUC (**0.6310**), which prioritizes precision-recall performance under severe class imbalance (0.30% positive class).

### Standalone Inference Validation (`chb01_03`)

The final model artifact was trained on 41 CHB01 recordings excluding held-out recording `chb01_03`. Running `inference.py` on `chb01_03` (1,799 windows) produced:
- **Artifact loading**: Successfully loaded pre-fitted `scaler.pkl`, `pca.pkl`, `classifier.pkl`, and `feature_cols.pkl` (0 refitting).
- **Predicted Labels**: 1,799 interictal, 0 binary ictal predictions at default 0.5 threshold.
- **Probability Trajectory**: Peak probability scores reached **0.38** specifically within the annotated seizure window (3022s–3026s), demonstrating discriminative ranking signal around the seizure event, but highlighting threshold sensitivity under high-imbalance generalization.

---

## 6. How to Run the Code

### Environment Setup

```powershell
# Clone the repository
git clone https://github.com/<org>/Data-Speaks.git
cd Data-Speaks
```

### Execution Options

#### Option A: Full Preprocessing & Feature Extraction Batch Pipeline (All 42 EDFs)
```powershell
C:\Users\Kavya\anaconda3\python.exe pipeline/run_chb01_complete_batch.py
```

#### Option B: Standalone Feature Analysis & EDA Passes
```powershell
# Run Pass 2 (PCA) on existing master feature matrix
C:\Users\Kavya\anaconda3\python.exe pipeline/run_chb01_complete_batch.py --pass2-only

# Run Pass 3 (Unsupervised Clustering) on existing PCA-reduced matrix
C:\Users\Kavya\anaconda3\python.exe pipeline/run_chb01_complete_batch.py --pass3-only

# Run Pass 4 (Midterm 5-Pillar EDA) on existing feature matrix
C:\Users\Kavya\anaconda3\python.exe pipeline/run_chb01_complete_batch.py --pass4-only
```

#### Option C: Stage 06 Supervised Classification
```powershell
C:\Users\Kavya\anaconda3\python.exe pipeline/06_classification/run_classification_chb01.py
```

#### Option D: Standalone EDF Inference
```powershell
# Run inference on held-out validation recording chb01_03
C:\Users\Kavya\anaconda3\python.exe pipeline/06_classification/inference.py --recording chb01_03

# Run inference on a new unannotated EDF file
C:\Users\Kavya\anaconda3\python.exe pipeline/06_classification/inference.py ^
    --edf path\to\recording.edf ^
    --recording my_recording_id ^
    --patient chb01 ^
    --no-validate
```

---

## 7. Repository Hygiene & Git Policies

- **Git Exclusions**: Raw EDF files (`data/raw/`), large intermediate window feature tables (`results/tables/chb01_all_features.csv`, `chb01_pca_reduced.csv`, `chb01_features_by_recording/`), binary model pickles (`*.pkl`), and log files are excluded in `.gitignore` to comply with GitHub file size limits (<100MB).
- **Tracked Artifacts**: All pipeline scripts, lightweight summary CSV tables (`chb01_classification_metrics.csv`, `chb01_pca_variance.csv`, `chb01_clustering_metrics.csv`), model metadata JSON (`metadata.json`), and generated high-resolution figures in `results/figures/` are tracked in version control.

---

## 8. References

- **Goldberger et al.** — *PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiological Signals* (CHB-MIT Database).
- **Oppenheim & Schafer** — *Discrete-Time Signal Processing* (DSP fundamentals).
- **Nunez & Srinivasan** — *Electric Fields of the Brain: The Neurophysics of EEG*.
- **MNE-Python Development Team** — *MNE Software for Processing Meg and EEG Data*.

---

*Data-Speaks — Course project by the Department of Mathematics and Computing, IIT Dharwad.*
