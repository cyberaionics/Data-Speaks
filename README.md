# Data-Speaks — Approach 4 (Chebyshev Type II + Robust Statistical Artifact Detection)

EEG seizure-detection experiments on the CHB-MIT Scalp EEG Database, using a Chebyshev Type II bandpass filter, a statistical (non-ICA) artifact screen, and decimation, followed by a Random Forest classifier evaluated under leave-one-subject-out cross-validation.

This README documents how the raw data is interpreted, why each transformation is applied, how predictions are evaluated, and what the current results mean — covering both the single-recording `chb01` pilot used to design and verify the pipeline, and the dataset-wide cohort run that preprocesses the full patient population and carries on into classification.

## At a Glance

| Component | Current implementation |
|---|---|
| Dataset | CHB-MIT Scalp EEG Database |
| Approach | Chebyshev Type II bandpass + statistical artifact detection + decimation |
| Montage | 18-channel fixed bipolar montage |
| Windowing | 4 s windows, 50% overlap, 2 s step |
| Features | 126 per window |
| Classifier | Random Forest |
| Evaluation | Leave-One-Subject-Out (LOSO) |
| Current cohort | 18 subjects, 52 recordings, 76.16 h |
| Current benchmark | 72 annotated seizures |
| Status | Preprocessing + initial cohort classification implemented; model/feature expansion and full-cohort validation remain |

## Repository Structure
```
chbmit_pipeline_approach4/
├── README.md
├── requirements.txt
├── scripts/
│   ├── approach4_pipeline.py
│   ├── cohort_pipeline.py
│   ├── run_pipeline_chb01.py
│   └── make_notebook.py
├── notebooks/
│   └── Approach4_CHB01_Preprocessing_EDA.ipynb
├── tests/
│   └── test_cv_grouping.py
├── output/
│   ├── approach4_chb01_metrics.csv
│   └── cohort_model_evaluation_metrics.csv
├── plots/
│   ├── psd_chebyshev_comparison.png
│   ├── seizure_vs_normal_eeg.png
│   ├── artifact_detection_breakdown.png
│   ├── cohort_seizure_durations.png
│   └── cohort_model_roc_pr.png
└── processed_dataset/
```
### What each directory does:
| Directory / File | Purpose |
|---|---|
| `scripts/` | Preprocessing, cohort processing, notebook generation, and pipeline entry points |
| `notebooks/` | `chb01` preprocessing and EDA notebook |
| `tests/` | LOSO patient/recording leakage checks |
| `output/` | Generated CSV metrics and evaluation summaries |
| `plots/` | PSD, EEG, artifact, seizure-duration, and model evaluation figures |
| `processed_dataset/` | Generated cohort features, labels, IDs, and metadata |
| `docs/` | Planned final report and dashboard documentation |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## Scope
Two modules implement this approach, at two different scales:

- `scripts/approach4_pipeline.py` — a single-recording module, operating on one EDF at a time. It was used to design and verify the filtering, artifact-detection, and labeling logic on `chb01` before scaling up, and it backs the notebook deliverable (`notebooks/Approach4_CHB01_Preprocessing_EDA.ipynb`). It does not, on its own, preprocess the dataset.
- `scripts/cohort_pipeline.py` — runs the identical Chebyshev/decimation/artifact preprocessing across every populated patient in the dataset, not `chb01` alone, and in the same run continues on into feature extraction and Random Forest classification under leave-one-subject-out (LOSO) cross-validation.

The current implementation therefore answers two separate questions:

1. Does the Chebyshev Type II + decimation + statistical-artifact pipeline produce clean, correctly labeled 4-second epochs across the full patient cohort, and how much does it reduce data volume and suppress line noise?
2. Given labeled recordings pooled from the cohort, how well can a Random Forest classifier detect seizure epochs in a patient it has never seen, using LOSO cross-validation?

Question 2 is a patient-independent evaluation, not a patient-specific one: each validation fold withholds one entire patient's epochs from training. This is a harder and more clinically relevant test than training and testing on splits of the same patient, and the results below should be read with that in mind.

This is not yet a streaming or deployable detector. "Inference" here means prediction on held-out, already-labeled epochs after the shared preprocessing and feature-extraction steps described below.

## Project Status and Remaining Work

The preprocessing and initial classification pipeline are already implemented. The remaining work is focused on completing cohort coverage, strengthening the feature and model comparisons, validating the evaluation protocol, and preparing the final project outputs.

### Current implementation

- Dataset-wide preprocessing and feature extraction are implemented in `scripts/cohort_pipeline.py`.
- The current cohort covers 18 of the 24 CHB-MIT subjects and up to 3 recordings per patient.
- Random Forest classification with LOSO cross-validation is already running end-to-end on the current cohort.
- The `chb01` single-recording pipeline and notebook remain available as pilot/verification material.


## Environment

### Selecting Patient Subsets

The cohort pipeline can be adapted to a smaller patient set for experiments or debugging by configuring the selected patient IDs in `scripts/cohort_pipeline.py`. When no subset is specified, the pipeline can be run against the configured cohort.

### Running Multiple Configurations

The pipeline is designed to support repeated experiments with different feature sets, thresholds, or model settings. Keep each configuration's output clearly identified so that comparisons remain reproducible.

The project uses Python 3.10+ (tested on 3.14). Install dependencies with:

pip install -r requirements.txt

The core dependencies are NumPy, SciPy, pandas, MNE, scikit-learn, and Matplotlib (JupyterLab/nbformat/ipykernel are included for the notebook deliverable). See `requirements.txt` for exact version bounds.

## Data Layout
Raw patient data is expected outside the repository, at a location pointed to by the `CHBMIT_DIR` environment variable:

CHBMIT\_DIR/
chb01/
chb01\_01.edf
chb01\_03.edf
chb01-summary.txt
chb02/
...

Set it before running anything:

export CHBMIT\_DIR=/path/to/your/chbmit        # macOS/Linux
setx CHBMIT\_DIR "D:\path\to\your\chbmit"       # Windows

The EDF files contain multichannel scalp EEG. The patient summary (`chbXX-summary.txt`) contains the authoritative seizure start and end times for each EDF; this project parses that text file rather than the binary `.edf.seizures` sidecars, because the summary's timing fields are directly readable and auditable.

Raw EDF and summary files are not committed to the repository (large source data, not source code) — they stay wherever `CHBMIT_DIR` points on each machine.

`processed_dataset/` is written at the repository root (not inside `output/`) by `scripts/cohort_pipeline.py`. It is also not committed to Git; it is regenerated by re-running the cohort pipeline. See "Outputs" below for its contents.

## How the Raw Data Is Interpreted

1. EDF channels and montage Each EDF is loaded with MNE and matched against a fixed 18-channel bipolar montage (International 10-20 "double banana"), in this order:

FP1-F7, F7-T7, T7-P7, P7-O1,
FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2,
FP2-F8, F8-T8, T8-P8, P8-O2,
FZ-CZ, CZ-PZ

Matching first tries an exact, case/whitespace-normalized comparison, then falls back to prefix matching for duplicate-numbered channels (e.g., `chb01` contains both `T8-P8-0` and `T8-P8-1`); the candidate ending in `-0` is preferred, since it is the first occurrence in the EDF. A fixed montage, in a fixed order, is what makes feature columns comparable across recordings and across patients.

Unlike an approach that tolerates a partial montage, this pipeline requires all 18 target channels to resolve. If any one of the 18 cannot be matched, the file is skipped entirely (`load_standard_18` raises, the caller logs "Skipping \<file>" and moves on) rather than proceeding with fewer channels. This keeps feature width constant across the whole cohort, at the cost of silently dropping any recording with a genuinely missing electrode.

2. Filtering A Chebyshev Type II bandpass filter is applied:

order:              4
low cutoff:         0.5 Hz
high cutoff:        45 Hz
stopband ripple:    30 dB attenuation
implementation:     second-order sections (`scipy.signal.cheby2`), applied zero-phase via `sosfiltfilt`

Chebyshev Type II was chosen over Type I or Butterworth for its flat passband (no ripple distortion inside 0.5-45 Hz, where seizure morphology lives) combined with a steep stopband rolloff, which gives strong, fast attenuation of the 60 Hz US powerline hum without needing a separate notch filter. Zero-phase (forward-backward) filtering removes phase delay, so filtered waveforms stay time-aligned with the original seizure annotations. `plots/psd_chebyshev_comparison.png` shows the resulting suppression on a representative channel.

3. Artifact detection Each 4-second epoch is screened against four purely statistical, threshold-based rules — no ICA, no manual component review:

- Flatline: any channel variance below 1.0 uV^2 (detached or loose electrode).
- Lead pop / step artifact: any single-sample jump greater than 200 uV (cable bump, static discharge).
- Abnormal channel variance: any channel's variance exceeding 15x the epoch's median channel variance, provided that channel's variance also exceeds 4000 uV^2 (isolated electrode noise, not just relative to an already-quiet epoch).
- Extreme saturation: any channel's peak-to-peak amplitude exceeding 800 uV (hardware rail).

These thresholds are set above the amplitude range of genuine ictal discharges (which commonly run 200-900 uV and are rhythmic rather than single-sample jumps), so true seizures are not screened out even when they trip individual statistics. In the cohort pipeline, an epoch flagged by any of these four rules is dropped only if it is also labeled interictal; epochs labeled ictal are kept even if flagged, on the reasoning that a real seizure discharge should not be discarded because it looks "abnormal" by design. This is a deliberate trade-off: it protects seizure recall at the cost of a small number of true noisy ictal epochs remaining in the training data uncorrected.

4. Decimation Filtered signals are decimated from 256 Hz to 128 Hz using `scipy.signal.decimate` with `zero_phase=True`, which applies its own anti-aliasing lowpass before downsampling. This halves storage and downstream compute while still fully preserving the 0.5-45 Hz clinical band (new Nyquist frequency: 64 Hz). Measured on `chb01`, this and the filtering step together reduce a 40.44 MB raw file to roughly 15.82 MB processed — about a 60.9% size reduction — while running at 2,000-4,000x real-time speed (see the pilot benchmark table below).

No per-channel normalization or z-scoring step is applied before feature extraction. Variance and kurtosis features are therefore computed on raw uV-scale amplitudes; see Limitations.

## Segmentation and Labels
The decimated signal is split directly into 4.0-second windows with 50% overlap (2.0-second step) — there is no separate epoch-then-stack scheme; one window is one classification unit.

A window is labeled ictal (1) if it overlaps an annotated seizure interval by at least 50% of the window's duration (2.0 seconds); otherwise it is labeled interictal (0). There is currently no boundary/"unknown" class for windows that fall close to, but don't sufficiently overlap, a seizure edge — every window receives a hard binary label from the 50% rule alone. See Limitations.

## Features
The cohort pipeline (`scripts/cohort_pipeline.py`) computes a 126-dimensional feature vector per window: for each of the 18 channels, five Welch power-spectral-density band powers (Delta 0.5-4 Hz, Theta 4-8 Hz, Alpha 8-13 Hz, Beta 13-30 Hz, Gamma 30-45 Hz, integrated via the trapezoidal rule) plus variance and kurtosis, giving 18 x (5 + 1 + 1) = 126 features.

| Feature group | Calculation | Features |
|---|---|---:|
| Spectral band power | 5 bands × 18 channels | 90 |
| Variance | 1 per channel × 18 channels | 18 |
| Kurtosis | 1 per channel × 18 channels | 18 |
| **Total** | **18 × (5 + 1 + 1)** | **126** |

Cross-channel correlation, Hjorth parameters, spectral entropy, and zero-crossing rate are planned but not yet implemented — see ## Future Work

The next phase focuses on improving cohort coverage, feature richness, model comparison, evaluation reliability, and reproducibility.

| Priority | Planned work | Purpose |
|---|---|---|
| 1 | Extend to the remaining 6 CHB-MIT subjects and every recording per patient | Complete cohort coverage rather than using the current seizure-rich subset |
| 2 | Add spectral entropy, band-power ratios, Hjorth parameters, zero-crossing rate, and inter-channel correlation | Improve the representation beyond the current 126-feature set |
| 3 | Benchmark logistic regression, RBF-SVM, and gradient-boosted trees; optionally explore 1D-CNN/CNN-LSTM | Establish stronger model baselines and alternatives |
| 4 | Tune the classification threshold using the precision-recall trade-off | Reduce dependence on the default 0.5 threshold |
| 5 | Evaluate class-imbalance strategies such as controlled undersampling/SMOTE against `class_weight='balanced'` | Assess whether imbalance handling improves detection |
| 6 | Evaluate the full 24-patient cohort and full non-seizure recording hours | Obtain a more trustworthy false-alarm estimate |
| 7 | Report seizure-level sensitivity for recordings containing multiple seizures | Make event evaluation more clinically meaningful |
| 8 | Add PCA/t-SNE/UMAP and clustering (K-Means/DBSCAN) | Complement supervised modeling with structure discovery |
| 9 | Align windowing, labeling, and CV definitions with other approach branches | Enable fair cross-approach comparison |
| 10 | Build the final report and interactive results dashboard | Consolidate results into the final research presentation |


## References

1. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215-e220.
2. Shoeb, A. H. (2010). Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment. PhD Thesis, Massachusetts Institute of Technology.
3. Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7, 267.
