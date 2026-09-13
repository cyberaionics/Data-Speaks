# Pipeline C: Butterworth Bandpass → ASR → ICA → Per-Channel Z-Score → 256 Hz

Pipeline C is candidate preprocessing architecture C in the **Data-Speaks** framework. It evaluates the sequential compounding effects of high-order IIR filtering, Artifact Subspace Reconstruction (ASR), Independent Component Analysis (ICA), and robust standardization on continuous bipolar EEG from the CHB-MIT Scalp EEG database.

---

## 1. Algorithmic Pipeline Architecture

```
Stage 0 Raw (17 Core Bipolar Channels, 256 Hz)
       │
       ▼
1. Butterworth Bandpass Filter (1.0 – 40.0 Hz, Order 4, IIR)
       │
       ▼
2. Artifact Subspace Reconstruction (ASR)
       • Threshold Cutoff: 20.0
       • Calibration Window: 120s seizure-free segment (buffered ±60s from seizures)
       │
       ▼
3. Independent Component Analysis (FastICA)
       • Variance Retained: 99% (n_components=0.99)
       • EOG Proxy Channels: FP1-F7, FP2-F4 (automated EOG component rejection)
       │
       ▼
4. Per-Channel Standardization (Full-length Z-Score Normalization)
       • Zero-mean, unit-variance per electrode channel
       │
       ▼
5. Output Validation (Sampling Rate 256 Hz, Zero NaNs/Infs)
```

---

## ASRpy NumPy 2.x Compatibility

ASRpy 0.0.8 was developed for NumPy 1.x and has a scalar-conversion
incompatibility with NumPy 2.x.

To allow ASRpy to run in the current environment, the project includes
`pipeline/01_preprocessing/candidate_algorithms/pipeline_c_compat.py`.

The compatibility shim patches the affected scalar conversion at runtime
without modifying the installed ASRpy package in `site-packages`.

The patch does not change the ASR algorithm, parameters, thresholds, or
artifact-removal logic. It only ensures compatibility with NumPy 2.x.

The shim is applied before Pipeline C imports and runs ASR.

---

## 3. File Structure & Modules

- [`pipeline_c.py`](file:///C:/Users/Kavya/Data-Speaks/pipeline/01_preprocessing/candidate_algorithms/pipeline_c.py): Core functional implementation of Pipeline C (`run_pipeline_c`).
- [`pipeline_c_compat.py`](file:///C:/Users/Kavya/Data-Speaks/pipeline/01_preprocessing/candidate_algorithms/pipeline_c_compat.py): NumPy 2.x compatibility monkey-patch shim.
- [`pipeline_c_metrics.py`](file:///C:/Users/Kavya/Data-Speaks/pipeline/01_preprocessing/candidate_algorithms/pipeline_c_metrics.py): Comprehensive metric extraction suite covering Groups A through G.
- [`pipeline_c_plots.py`](file:///C:/Users/Kavya/Data-Speaks/pipeline/01_preprocessing/candidate_algorithms/pipeline_c_plots.py): Time-domain signal evolution and spectral diagnostics plotting routines.
- [`run_pipeline_c_chb01_03.py`](file:///C:/Users/Kavya/Data-Speaks/pipeline/01_preprocessing/candidate_algorithms/run_pipeline_c_chb01_03.py): Validation runner for single-record benchmark and verification.
- [`run_pipeline_c_chb01.py`](file:///C:/Users/Kavya/Data-Speaks/pipeline/01_preprocessing/candidate_algorithms/run_pipeline_c_chb01.py): Full cohort batch processing runner for patient `chb01`.
- [`stage0.py`](file:///C:/Users/Kavya/Data-Speaks/pipeline/01_preprocessing/stage0.py): Modularized Stage 0 standardization script extracted from `stage0.ipynb`.

---

## 4. Execution Instructions

Ensure the Python environment contains the required dependencies:
`mne`, `asrpy`, `scipy`, and `numpy`.

### Single Recording Validation (chb01_03)
From the repository root:
```powershell
python pipeline/01_preprocessing/candidate_algorithms/run_pipeline_c_chb01_03.py
```
Outputs:
- Figures in `results/figures/pipeline_c/`
- Summary JSON in `results/benchmarks/pipeline_C/chb01_03_summary.json`

### Full Patient Cohort Execution (chb01)
```powershell
python pipeline/01_preprocessing/candidate_algorithms/run_pipeline_c_chb01.py
```
Output:
- Summary evaluation metrics table: `results/tables/pipeline_c_chb01_metrics.csv`

---

## 5. Evaluation Metrics Extracted

1. **Structural & Integrity Metrics**: Output shape, channel preservation, sampling rate integrity, NaN/Inf counts.
2. **Runtime Breakdown**: Per-stage wall-clock latency (Butterworth, ASR calibration + transform, FastICA fitting + projection, Z-score).
3. **Data Preservation**: RMS difference, relative RMS change before vs. after cleaning.
4. **Spectral Metrics**: Welch PSD band power (Delta, Theta, Alpha, Beta, Gamma) on key representative channels (`FP1-F7`, `FZ-CZ`, `P8-O2`).
5. **ICA Artifact Rejection**: Component counts fit, variance explained, components excluded via frontal EOG proxies.
