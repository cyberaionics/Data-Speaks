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

## 2. ASRpy NumPy 2.x Compatibility Shim (`pipeline_c_compat.py`)

### Problem
`asrpy` 0.0.8 was authored for NumPy < 2.0. In NumPy 2.x, 0-d or 1-element arrays returned by `np.round(n * max_width)` in `asrpy.asr_utils.fit_eeg_distribution()` fail with:
`TypeError: only 0-dimensional arrays can be converted to Python scalars` when passed to `int()`.

### Crucial Implementation Detail
In `asrpy/asr.py`, `fit_eeg_distribution` is imported directly into the module namespace:
```python
from .asr_utils import fit_eeg_distribution
```
Furthermore, `clean_windows()` and `asr_calibrate()` are defined inside `asrpy.asr` and resolve `fit_eeg_distribution` from the module globals of `asrpy.asr`.
Therefore, patching `asrpy.asr_utils.fit_eeg_distribution` alone has **no effect** on `ASR.fit()`.

The monkey-patch in `pipeline_c_compat.py` explicitly replaces:
```python
asrpy.asr.fit_eeg_distribution = _fit_eeg_distribution_patched
```
This safely coerces scalar arrays via `int(float(np.round(...)))` without altering any algorithmic logic, thresholds, or modifying package files in `site-packages`.

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

Ensure you are using the Anaconda Python environment where `mne`, `asrpy`, `scipy`, and `numpy` are installed:

### Single Recording Validation (chb01_03)
```powershell
C:\Users\Kavya\anaconda3\python.exe C:\Users\Kavya\Data-Speaks\pipeline\01_preprocessing\candidate_algorithms\run_pipeline_c_chb01_03.py
```
Outputs:
- Figures in `results/figures/pipeline_c/`
- Summary JSON in `results/benchmarks/pipeline_C/chb01_03_summary.json`

### Full Patient Cohort Execution (chb01)
```powershell
C:\Users\Kavya\anaconda3\python.exe C:\Users\Kavya\Data-Speaks\pipeline\01_preprocessing\candidate_algorithms\run_pipeline_c_chb01.py
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
