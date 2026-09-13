# Data-Speaks Final Verification

Acceptance audit date: 2026-09-13

## Repository

- Branch: `manish_FIR_ICA_ZSCORE`
- Latest pipeline implementation commit: `a739a47 Complete Pipeline D preprocessing and feature dataset`
- A/B/C/D implementation commits present: `be9d0a9`, `ded9a2a`, `c8b0040`, `a739a47`
- Working tree: clean after the finalization commits listed below.
- Forbidden tracked outputs: none. No EDF, FIF, generated CSV dataset, or benchmark output is tracked. Only `.gitkeep` files under results are tracked.
- `.gitignore` excludes raw data, processed data, EDF/FIF files, caches, benchmark outputs, and environments.

## Pipeline Status

| Pipeline | Components | End-to-End | Full CHB02 | Features | Integration |
|---|---|---|---|---|---|
| A | PASS | PASS | PASS | PASS | PASS |
| B | PASS | PASS | PASS | PASS | PASS |
| C | PASS | PASS | PASS | PASS | PASS |
| D | PASS | PASS | PASS | PASS | PASS |

All four fresh end-to-end checks used common Stage 0 input from `chb02_01.edf`.
The 60-second crop produced `(23, 15361)` at 256 Hz. Fresh runtimes were A
0.653 s, B 0.021 s, C 0.375 s, and D 0.028 s. Each output preserved channel
names/order, shape, sampling rate, finite values, and provenance.

Pipeline A's original test entry point had an import-path defect and was fixed
with a minimal repository-root bootstrap. The corrected A test passes. The
canonical A benchmark is now `notebooks/14_benchmark_pipeline_a.py`; it
discovers all recordings and does not hard-code a subset. The full canonical
run completed successfully for all 36 recordings.

## Common Contract

- Stage 0: shared loader and CHB-MIT summary parser; 23 EEG channels, deterministic channel order, seizure intervals from the common parser.
- Recordings: 36 EDF files discovered under `data/raw/physionet.org/chb02/`.
- Processed outputs: 36 FIF files and 36 provenance records for each A/B/C/D pipeline.
- Output contract: every stored provenance record reports 23 channels and 256 Hz.
- Segmentation: 4.0-second windows, 50% overlap, 1024 samples, 512-sample stride, duration `n_times / sfreq`.
- Labels: common ictal/interictal/ambiguous rules; ambiguous windows excluded from feature datasets.
- Seizure-bearing recordings: `chb02_16`, `chb02_16+`, and `chb02_19`; intervals match Stage 0 in all four provenance files.
- Feature schema: five metadata columns plus exactly 598 features, identical names and order.
- Window alignment: 63,257 common valid windows; only-in-A/B/C/D counts are all zero.

## Feature Quality

| Pipeline | Rows | Recordings | Features | NaN | Inf | Zero variance | Missing recordings |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 63,257 | 36 | 598 | 0 | 0 | 0 | 0 |
| B | 63,257 | 36 | 598 | 0 | 0 | 0 | 0 |
| C | 63,257 | 36 | 598 | 0 | 0 | 0 | 0 |
| D | 63,257 | 36 | 598 | 0 | 0 | 0 | 0 |

Labels are valid in every dataset: 63,169 interictal and 88 ictal. Feature
values differ across pipelines as expected.

## Level 1

The stored Level 1 evaluation uses the same 60-second `chb02_01` segment,
channels, seed 42, synthetic noise generation, 5% noise scale, Welch method,
1-40 Hz bands, and formulas for every pipeline.

| Pipeline | SNR improvement (dB) | Channel correlation | Clean runtime (s) | Noisy runtime (s) | Peak memory (MB) |
|---|---:|---:|---:|---:|---:|
| A | 0.405667 | 0.691284 | 0.093832 | 0.066930 | 828.875 |
| B | 4.173356 | 0.861298 | 0.022398 | 0.020237 | 834.720 |
| C | -2.657980 | 0.754035 | 0.379530 | 0.401407 | 846.516 |
| D | 3.862070 | 0.677441 | 0.029841 | 0.028609 | 846.594 |

Relative PSD preservation ratios are recorded per band in
`results/benchmarks/level1_evaluation.json`. Suspicious signals requiring
scientific caution include Pipeline D's near-zero gamma ratio (0.000213) and
Pipeline C's negative synthetic-noise SNR improvement.

## Downstream

- Dimensionality reduction: standardized PCA, 95% variance target, common fitting procedure, seed 42. Components: A 142, B 128, C 130, D 164.
- Clustering: common KMeans, `n_clusters=2`, `n_init=10`, seed 42, common silhouette evaluation. Silhouettes: A 0.293033, B 0.477125, C 0.445212, D 0.310665.
- Classification: common median imputation, standardization, balanced logistic regression, `max_iter=2000`, seed 42, same held-out recording `chb02_19` for the preliminary recording-wise comparison.
- Classification metrics are recorded for Accuracy, Balanced Accuracy, Precision, Recall, F1, and ROC-AUC in `common_evaluation.json`.
- Patient-wise validation: **NOT POSSIBLE**. All available recordings are from one patient, CHB02. The classification results are explicitly preliminary recording-wise results, not patient-wise generalization.

## Leakage

- Patient leakage: patient-wise experiment unavailable because only one patient is present; no patient-wise claim is made.
- Recording leakage: preliminary split holds out all windows from `chb02_19`; no windows from that recording are used for training.
- Window leakage: overlapping windows from the held-out recording are kept together; no train/test window split is performed within it.
- Label leakage: labels are taken from the common Stage 0 seizure intervals and common windowing code; feature extraction does not receive labels.
- Test-set fitting: the classifier pipeline fits imputation/scaling/classification on the training mask only. PCA and KMeans are exploratory full-dataset analyses and are not presented as held-out predictive evaluation.

## Limitations

- The dataset contains one patient, so patient-wise generalization cannot be established.
- The Level 1 comparison uses one 60-second recording segment and synthetic contamination, not all recordings or a clean ground-truth corpus.
- Full A was rerun through the canonical benchmark. Existing provenance verifies
full B/C/D processing; unchanged C/D full preprocessing was not unnecessarily
rerun during finalization.
- Runtime and memory values are environment-dependent.
- MNE reports duplicate EDF channel names (`T8-P8`) and deterministically renames them; channels were preserved rather than deleted.
- ICA emits a numerical-stability warning on some data because the configured component count is aggressive; the required tests still pass, but this warrants scientific review.

## Git

Verified recent history:

```text
a739a47 Complete Pipeline D preprocessing and feature dataset
c8b0040 Complete Pipeline C preprocessing and feature dataset
fdb0e27 Ignore benchmark outputs
ded9a2a Complete Pipeline B preprocessing and feature dataset
92a4998 Update CHB02 inspection script for file selection
09d3113 Ignore generated experiment results
be9d0a9 Implement Pipeline A FIR ICA Z-score preprocessing
857fb19 Implement common Stage 0 EEG loading and QC
```

No push was created. Source, evaluation, and documentation changes were
committed in logical commits; generated outputs remain ignored.

## FINAL STATUS

COMPLETE FOR CURRENT CHB02 SCOPE

All repository-side acceptance gates pass and the final working tree is clean.
Patient-wise generalization is NOT AVAILABLE WITH CURRENT DATA because every
recording belongs to CHB02; additional patient data is required for that
external scientific requirement.