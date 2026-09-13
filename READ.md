# Data-Speaks — EEG Preprocessing Project Continuation README

## Purpose

This file is a continuation handoff for the Data-Speaks EEG project. Upload this README into a new ChatGPT conversation and tell the assistant to continue from the current state without repeating completed work.

## Repository

Repository: https://github.com/cyberaionics/Data-Speaks

Current Git branch:

```text
manish_FIR_ICA_ZSCORE
```

The branch is local and is not pushed by the finalization workflow. The final
working tree is expected to be clean after the documented commits are created.

Important GitHub detail: GitHub was previously showing the `master` branch. The user's completed Pipeline A/B work is on `manish_FIR_ICA_ZSCORE`. Always check the branch dropdown before inspecting files on GitHub.

---

# Project Goal

The project compares four EEG preprocessing pipelines fairly using the CHB-MIT scalp EEG dataset from PhysioNet.

The four candidate methods are:

```text
Pipeline A
FIR Band Pass → ICA → Z-score normalization → 256 Hz

Pipeline B
Butterworth Band Pass → Wavelet Denoising → Robust Scaling → Polyphase resampling

Pipeline C
Butterworth Band Pass → ASR + ICA → Per-channel standardization → Polyphase resampling

Pipeline D
Chebyshev Type II Bandpass → Robust Statistical artifact detection → Decimation
```

Common processing contract:

```text
EDF / Stage 0 standardized EEG
        ↓
Pipeline-specific preprocessing
        ↓
256 Hz common output
        ↓
4-second windows
50% overlap
1024 samples/window
512-sample stride
        ↓
Labels
        ↓
Same feature extractor
        ↓
Same downstream evaluation
```

The project must compare pipelines fairly. Do not declare a global winner from only one recording or one preliminary experiment.

---

# Dataset / Current Scope

Current dataset being used for development and benchmarking:

```text
CHB-MIT
Patient: chb02
```

There are 36 EDF recordings under:

```text
data/raw/physionet.org/chb02/
```

Three recordings contain seizures according to `chb02-summary.txt`:

```text
chb02_16.edf
  seizure: 130–212 sec

chb02_16+.edf
  seizure: 2972–3053 sec

chb02_19.edf
  seizure: 3369–3378 sec
```

The current CHB02 recordings contain 23 EEG channels at 256 Hz.

A duplicate `T8-P8` channel name is present in CHB02. MNE deterministically renames duplicates, e.g. `T8-P8-0` and `T8-P8-1`. We preserve both channels instead of arbitrarily deleting one.

---

# Environment

Python:

```text
3.12.14
```

MNE:

```text
1.12.1
```

A virtual environment is used:

```text
.venv/
```

The project has already been installed and tested successfully.

---

# Stage 0 — Common Standardization / QC

File:

```text
pipeline/preprocessing_common/stage0.py
```

Main functions:

```python
parse_seizure_annotations(...)
calculate_qc(...)
load_stage0(...)
metadata_to_dict(...)
```

`load_stage0()` responsibilities:

- EDF integrity/readability
- EEG-only channel selection
- preserve native sampling rate
- preserve bipolar montage
- parse seizure intervals from `chb02-summary.txt`
- basic QC
- clear `meas_date` to avoid invalid CHB02 EDF timestamps when writing FIF

Important API correction:

The correct entry point is:

```python
stage0.load_stage0(...)
```

NOT:

```python
load_and_validate_edf(...)
```

Stage 0 was successfully tested on CHB02 recordings including `chb02_01`, `chb02_16`, and `chb02_19`.

Stage 0 commit from earlier work:

```text
857fb19 Implement common Stage 0 EEG loading and QC
```

---

# Pipeline A — Completed

Pipeline A specification:

```text
FIR 1–40 Hz
→ ICA artifact removal
→ per-channel recording-level Z-score
→ 256 Hz
```

Main files:

```text
pipeline/01_preprocessing/candidate_algorithms/pipeline_a.py
pipeline/01_preprocessing/candidate_algorithms/filtering/fir_bandpass.py
pipeline/01_preprocessing/candidate_algorithms/artifact_removal/ica.py
pipeline/01_preprocessing/candidate_algorithms/normalization/zscore.py
pipeline/01_preprocessing/candidate_algorithms/resampling/resample_256.py
pipeline/01_preprocessing/candidate_algorithms/PIPELINE_A_README.md
```

Important Pipeline A parameters:

```text
FIR:
1–40 Hz
firwin
zero-phase

ICA:
FastICA
max 15 components, clamped to rank
random_state=42
EOG proxy = FP1-F7
EOG correlation threshold = 0.4
muscle threshold = 0.3

Normalization:
per-recording, per-channel Z-score

Resampling:
target 256 Hz
MNE polyphase when needed
```

Known limitation: FP1-F7 is used as an EOG proxy because Stage 0 removes EOG/ECG channels. Automatic ICA selection is heuristic and can potentially remove frontal EEG/seizure activity. Topomap plotting is not available because CHB02 lacks digitization/sensor positions.

Pipeline A end-to-end testing succeeded.

Pipeline A benchmark recordings already processed:

```text
chb02_01
chb02_16
```

Earlier benchmark runtime values (measured with the older process-memory approach) were approximately:

```text
chb02_01: 6.81 sec
chb02_16: 2.33 sec
```

Do NOT directly compare those old memory numbers with Pipeline B `tracemalloc` memory numbers; the measurement methods differ.

Pipeline A benchmark outputs include:

```text
data/processed/pipeline_A/chb02_01_raw.fif
data/processed/pipeline_A/chb02_16_raw.fif
results/benchmarks/pipeline_A_provenance.json
```

Pipeline A commit:

```text
be9d0a9 Implement Pipeline A FIR ICA Z-score preprocessing
```

A later commit included Pipeline B completion:

```text
ded9a2a Complete Pipeline B preprocessing and feature dataset
```

---

# Pipeline B — Completed

Pipeline B specification:

```text
Butterworth 1–40 Hz
→ Wavelet denoising
→ Robust scaling
→ polyphase resampling to 256 Hz
```

Main files:

```text
pipeline/01_preprocessing/candidate_algorithms/pipeline_b.py
pipeline/01_preprocessing/candidate_algorithms/filtering/butterworth_bandpass.py
pipeline/01_preprocessing/candidate_algorithms/artifact_removal/wavelet_denoising.py
pipeline/01_preprocessing/candidate_algorithms/normalization/robust_scaling.py
pipeline/01_preprocessing/candidate_algorithms/resampling/resample_256.py
```

There is also a directory:

```text
pipeline/01_preprocessing/candidate_algorithms/pipeline_b/
```

containing `__init__.py`.

The actual Pipeline B orchestrator is:

```text
pipeline/01_preprocessing/candidate_algorithms/pipeline_b.py
```

Pipeline B parameters currently implemented:

```text
Butterworth:
1–40 Hz
order = 4
zero-phase IIR

Wavelet:
db4
level = 5
soft universal threshold
noise sigma from MAD

Robust scaling:
center = median
scale = IQR (25th–75th percentile)

Resampling:
target = 256 Hz
MNE polyphase when needed
```

## Pipeline B filter validation

Real CHB02 PSD test on first EEG channel, 60 seconds:

```text
Below 1 Hz
Raw:      3.149520e-10
Filtered: 2.646209e-12

1–40 Hz
Raw:      1.601700e-11
Filtered: 1.479272e-11

Above 40 Hz
Raw:      4.963735e-13
Filtered: 2.051394e-14

Passband power ratio: 0.9236
Above-40Hz power ratio: 0.0413
```

This confirms strong out-of-band reduction and substantial passband preservation.

Test script:

```text
notebooks/25_test_butterworth_psd.py
```

## Pipeline B wavelet test

Real 60-second CHB02 test passed using `db4`, level 5, with finite output and a nonzero signal difference.

Test script:

```text
notebooks/26_test_wavelet_denoising.py
```

## Pipeline B robust scaling test

Real 60-second CHB02 test passed:

```text
Maximum absolute median: 0.0
IQR range: approximately 1.0
Finite output: True
```

Test script:

```text
notebooks/27_test_robust_scaling.py
```

## Pipeline B end-to-end test

60-second real EEG test passed:

```text
Input shape:  (23, 15361)
Output shape: (23, 15361)
Input sampling rate: 256.0
Output sampling rate: 256.0
Input channels: 23
Output channels: 23
Finite: True
```

All four stages appeared in provenance.

Test script:

```text
notebooks/28_test_pipeline_b.py
```

## Pipeline B benchmark

The benchmark was updated to use Stage 0 automatically and discover all 36 EDFs.

Benchmark script:

```text
notebooks/29_benchmark_pipeline_b.py
```

Final full CHB02 benchmark result:

```text
36 recordings processed
3 recordings with seizures
Total runtime: 29.056887915001425 sec
```

Seizure metadata correctly parsed:

```text
chb02_16+ → [(2972.0, 3053.0)]
chb02_16  → [(130.0, 212.0)]
chb02_19  → [(3369.0, 3378.0)]
```

Processed files are saved under:

```text
data/processed/pipeline_B/
```

The provenance file is:

```text
results/benchmarks/pipeline_B_provenance.json
```

The benchmark clears `meas_date` before saving FIF because the original CHB02 EDF measurement date can be outside the accepted FIF timestamp range. This does not modify EEG samples.

---

# Segmentation / Labeling — Completed and Shared

Windowing code:

```text
pipeline/02_segmentation/candidate_algorithms/windowing.py
```

Parameters:

```text
Window length: 4.0 sec
Overlap: 50%
Stride: 2.0 sec
Samples/window at 256 Hz: 1024
Stride at 256 Hz: 512 samples
Boundary buffer: 60 sec
```

Label rules used:

```text
1 = ictal if seizure overlap >= 50%
0 = interictal if no seizure overlap and >60 sec from seizure boundary
-1 = ambiguous if seizure overlap <50% OR within 60-sec seizure boundary buffer
```

Ambiguous windows are excluded from supervised feature datasets.

Important correction already made: duration should be computed from sample count:

```python
raw.n_times / raw.info["sfreq"]
```

rather than `raw.times[-1]`. This ensures the final complete window is not incorrectly dropped.

For `chb02_01` and `chb02_16`, the corrected common segmentation is:

```text
chb02_01: 1799 windows
chb02_16:  478 windows
----------------------
Total:     2277 windows
```

For the two-recording benchmark:

```text
Ambiguous: 62
Interictal: 2173
Ictal: 42
Supervised windows: 2215
```

Scripts:

```text
notebooks/16_segment_pipeline_a.py
notebooks/30_segment_pipeline_b.py
```

Window-label unit test:

```text
pipeline/02_segmentation/candidate_algorithms/test_windowing.py
```

The unit test passed.

---

# Feature Extraction — Completed

Feature directories:

```text
pipeline/03_feature_extraction/time_domain/
pipeline/03_feature_extraction/frequency_domain/
pipeline/03_feature_extraction/statistical/
```

Common feature extractor:

```text
pipeline/03_feature_extraction/feature_extractor.py
```

## Time-domain features

8 features/channel:

```text
mean
std
min
max
ptp
rms
mean_abs
zero_crossing_rate
```

With 23 channels:

```text
8 × 23 = 184 features
```

## Frequency-domain features

11 features/channel:

```text
delta_power
delta_relative_power
theta_power
theta_relative_power
alpha_power
alpha_relative_power
beta_power
beta_relative_power
gamma_power
gamma_relative_power
spectral_centroid
```

Bands:

```text
delta: 1–4 Hz
theta: 4–8 Hz
alpha: 8–13 Hz
beta: 13–30 Hz
gamma: 30–40 Hz
```

With 23 channels:

```text
11 × 23 = 253 features
```

## Statistical features

7 features/channel:

```text
median
mean_abs_deviation
median_abs_deviation
iqr
q05_q95_range
skewness
excess_kurtosis
```

With 23 channels:

```text
7 × 23 = 161 features
```

## Total

```text
8 + 11 + 7 = 26 features/channel
26 × 23 = 598 features/window
```

All three feature groups were tested on real Pipeline A EEG.

Integration test:

```text
notebooks/20_test_all_features_real.py
```

It passed with a 23×1024 real EEG window and 598 finite features.

---

# Feature Dataset — Pipeline A

Initial single-recording dataset:

```text
results/benchmarks/features_pipeline_A_chb02_16.csv
```

It had:

```text
416 rows
598 features
42 ictal
374 interictal
```

Validation passed:

```text
Shape: (416, 603)
Feature columns: 598
Missing: 0
Infinite: 0
Zero-variance: 0
```

The combined two-recording Pipeline A dataset was later created as:

```text
results/benchmarks/features_pipeline_A_chb02.csv
```

Final two-recording supervised dataset:

```text
2215 rows
598 features
42 ictal
2173 interictal
```

---

# Feature Dataset — Pipeline B

Two-recording Pipeline B dataset:

```text
results/benchmarks/features_pipeline_B_chb02.csv
```

Final two-recording supervised dataset:

```text
2215 rows
598 features
42 ictal
2173 interictal
```

Full CHB02 Pipeline B dataset:

```text
results/benchmarks/features_pipeline_B_chb02_full.csv
```

Final full CHB02 result:

```text
36 recordings
63,257 rows/windows
598 features per row
88 ictal
63,169 interictal
0 ambiguous rows in the saved supervised dataset
```

Full validation was run and passed.

Validation script:

```text
notebooks/40_validate_full_feature_dataset_pipeline_b.py
```

The full dataset was confirmed to have:

```text
36 recordings
598 feature columns
0 NaN
0 infinite values
0 zero-variance features
```

---

# A/B Dataset Alignment — Completed for Two-Recording Comparison

Script:

```text
notebooks/33_compare_pipeline_a_b_datasets.py
```

It passed:

```text
Pipeline A shape: (2215, 603)
Pipeline B shape: (2215, 603)

A feature count: 598
B feature count: 598

Feature columns identical: True
Window/label metadata identical: True

Pipeline A/B dataset alignment test passed.
```

This is a major fairness checkpoint.

---

# Classification Evaluation Status

A first attempt was made to train Logistic Regression on `chb02_01` and test on `chb02_16`.

That was invalid because `chb02_01` contains only interictal windows:

```text
Train: 1799 windows → all class 0
Test: 416 windows → 374 class 0 + 42 class 1
```

The classifier correctly failed because training contained only one class.

A second attempt with a 70/30 chronological split of `chb02_16` also failed for evaluation design reasons:

```text
Training: 291 windows → 249 class 0 + 42 class 1
Testing:  125 windows → all class 0
```

A proposed 60–280 sec test block showed:

```text
Train: 369 → all class 0
Test:   47 → 5 class 0 + 42 class 1
```

This is also invalid because the train set lacks class 1.

Conclusion:

**Do not claim a supervised classification result from the current two-recording setup.**

The appropriate next step is to design a valid evaluation using multiple seizure-containing recordings and ultimately multiple patients for a true patient-wise split. The user explicitly wants a fair comparison and should not use leakage-prone random window mixing merely to force a classifier result.

Current baseline classification script exists but should not be treated as final evaluation:

```text
notebooks/35_baseline_classification_ab.py
```

Additional split inspection:

```text
notebooks/36_inspect_split_options.py
notebooks/37_check_temporal_split.py
```

---

# Git Status / Important Branch Information

Current branch:

```text
manish_FIR_ICA_ZSCORE
```

GitHub remote branch is up to date.

The user previously committed the Pipeline B work and later committed the `.gitignore` change. Current `git status` was:

```text
On branch manish_FIR_ICA_ZSCORE
Your branch is up to date with 'origin/manish_FIR_ICA_ZSCORE'.
nothing to commit, working tree clean
```

`.gitignore` now includes:

```gitignore
# Experiment outputs
results/*.png
results/pipeline_a_test/

# Benchmark outputs
results/benchmarks/
```

Generated benchmark/data files are intentionally ignored and should NOT be committed:

```text
results/benchmarks/*.csv
results/benchmarks/*.json
```

Also ignored:

```text
data/raw/
data/processed/
*.edf
*.seizures
```

Do not ask the user to re-download data that already exists locally.

---

# Current Local Folder Structure

The important repository structure currently looks like:

```text
Data-Speaks/
├── pipeline/
│   ├── preprocessing_common/
│   │   ├── __init__.py
│   │   └── stage0.py
│   │
│   ├── 01_preprocessing/
│   │   ├── __init__.py
│   │   └── candidate_algorithms/
│   │       ├── artifact_removal/
│   │       │   ├── ica.py
│   │       │   └── wavelet_denoising.py
│   │       │
│   │       ├── filtering/
│   │       │   ├── fir_bandpass.py
│   │       │   └── butterworth_bandpass.py
│   │       │
│   │       ├── normalization/
│   │       │   ├── zscore.py
│   │       │   └── robust_scaling.py
│   │       │
│   │       ├── resampling/
│   │       │   └── resample_256.py
│   │       │
│   │       ├── PIPELINE_A_README.md
│   │       ├── pipeline_a.py
│   │       ├── pipeline_b.py
│   │       └── pipeline_b/
│   │           └── __init__.py
│   │
│   ├── 02_segmentation/
│   │   └── candidate_algorithms/
│   │       └── windowing.py
│   │
│   └── 03_feature_extraction/
│       ├── feature_extractor.py
│       ├── time_domain/
│       │   └── features.py
│       ├── frequency_domain/
│       │   └── features.py
│       └── statistical/
│           └── features.py
│
├── notebooks/
│   ├── 01_inspect_chb02.py
│   ├── 02_test_stage0.py
│   ├── 03_test_fir.py
│   ├── 04_test_fir_psd.py
│   ├── 05_test_ica.py
│   ├── 06_ica_diagnostics.py
│   ├── 07_evaluate_ica.py
│   ├── 08_compare_ica_exclusions.py
│   ├── 09_compare_ica_strategies.py
│   ├── 10_ica_seizure_test.py
│   ├── 13_test_pipeline_a.py
│   ├── 14_benchmark_pipeline_a.py
│   ├── 15_create_metadata.py
│   ├── 16_segment_pipeline_a.py
│   ├── 17_check_window_labels.py
│   ├── 18_test_time_features_real.py
│   ├── 19_test_frequency_features_real.py
│   ├── 20_test_all_features_real.py
│   ├── 21_build_feature_dataset_pipeline_a.py
│   ├── 22_validate_feature_dataset.py
│   ├── 23_build_full_feature_dataset_pipeline_a.py
│   ├── 24_test_butterworth.py
│   ├── 25_test_butterworth_psd.py
│   ├── 26_test_wavelet_denoising.py
│   ├── 27_test_robust_scaling.py
│   ├── 28_test_pipeline_b.py
│   ├── 29_benchmark_pipeline_b.py
│   ├── 30_segment_pipeline_b.py
│   ├── 31_build_feature_dataset_pipeline_b.py
│   ├── 32_validate_feature_dataset_pipeline_b.py
│   ├── 33_compare_pipeline_a_b_datasets.py
│   ├── 34_build_feature_dataset_pipeline_a_combined.py
│   ├── 35_baseline_classification_ab.py
│   ├── 36_inspect_split_options.py
│   ├── 37_check_temporal_split.py
│   ├── 38_list_chb02_recordings.py
│   ├── 39_build_full_feature_dataset_pipeline_b.py
│   └── 40_validate_full_feature_dataset_pipeline_b.py
│
├── data/
│   ├── raw/
│   │   └── physionet.org/chb02/
│   └── processed/
│       ├── pipeline_A/
│       └── pipeline_B/
│
└── results/
    └── benchmarks/   # ignored/generated outputs
```

Note: GitHub may show the older `master` contents where `pipeline_a.py` appears empty and `pipeline_b.py` is absent. That is NOT the working branch. On GitHub, select:

```text
manish_FIR_ICA_ZSCORE
```

before inspecting the code.

---

# Pipeline Separation Clarification

The project currently uses separate orchestrator files:

```text
pipeline_a.py = Method 1 / Pipeline A
pipeline_b.py = Method 2 / Pipeline B
```

Shared algorithm folders are used underneath them so code can be reused without changing the intended pipeline sequence.

This structure does not inherently change numerical output. Output only changes when the algorithm, parameters, order, or input data changes.

Do NOT reorganize working Pipeline A/B files unless there is a concrete reason; their behavior is already validated.

---

# Exact Next Task — Pipeline C

Pipeline C has NOT been implemented yet.

Target:

```text
Pipeline C
Butterworth Band Pass
→ ASR + ICA
→ Per-channel standardization
→ Polyphase resampling to 256 Hz
```

The user had just decided to start Pipeline C after committing Pipeline B.

The intended initial structure was:

```text
pipeline/01_preprocessing/candidate_algorithms/pipeline_c.py
pipeline/01_preprocessing/candidate_algorithms/artifact_removal/asr_ica.py
pipeline/01_preprocessing/candidate_algorithms/normalization/standardization.py
```

The shared, already-tested components that can be reused are:

```text
filtering/butterworth_bandpass.py
resampling/resample_256.py
```

The next implementation step should be **ASR first**, tested independently on real CHB02 EEG, then integrated with ICA, then standardization, then the `pipeline_c.py` orchestrator.

Do not skip validation or jump directly to full-dataset processing.

---

# Important Technical Lessons / Pitfalls

1. MNE duplicate channel warning for `T8-P8` is expected for this dataset. Preserve both deterministic renamed channels.

2. For Stage 0, use `load_stage0()`, not the nonexistent `load_and_validate_edf()`.

3. When dynamically importing modules that contain dataclasses, register the module in `sys.modules` before executing it. Several tests use:

```python
sys.modules[name] = module
spec.loader.exec_module(module)
```

4. `generate_windows()` returns a generator. Convert it to a list when `len()` is needed:

```python
windows = list(generate_windows(...))
```

5. `generate_windows()` does not provide `window_index`; create it with `enumerate()` when writing datasets.

6. Use:

```python
raw.n_times / raw.info["sfreq"]
```

for full sample-based recording duration, not `raw.times[-1]`, when deciding the number of complete windows.

7. The old Pipeline A memory benchmark and current Pipeline B `tracemalloc` memory benchmark are not directly comparable.

8. CHB02 EDF measurement dates can cause FIF save failures. Stage 0 already does:

```python
raw.set_meas_date(None)
```

9. Do not force a classifier split when the training or test data contains only one class. Current CHB02-only preliminary data is insufficient for a defensible patient-wise supervised evaluation.

10. Do not use random overlapping windows from the same recording as a casual fix for classification leakage.

---

# Useful Commands

Check branch:

```bash
git branch --show-current
```

Check Git status:

```bash
git status
```

Open project in Finder on macOS:

```bash
open ~/Data-Speaks
```

Open preprocessing candidate directory:

```bash
open ~/Data-Speaks/pipeline/01_preprocessing/candidate_algorithms/
```

Check pipeline files:

```bash
ls pipeline/01_preprocessing/candidate_algorithms/
```

Check all feature datasets:

```bash
ls -lh results/benchmarks/*features*
```

---

# Continuation Instruction for the Next ChatGPT Window

Paste/upload this README and tell the assistant:

> I am continuing the Data-Speaks EEG preprocessing project. Read this README completely and continue from the exact current state. Do not repeat completed Pipeline A or Pipeline B work. Pipeline A and Pipeline B are finished and validated. The next task is Pipeline C: Butterworth → ASR + ICA → per-channel standardization → polyphase 256 Hz. Work one step at a time with exact terminal commands and full code when a file must be replaced. Keep the comparison fair across all four pipelines.

