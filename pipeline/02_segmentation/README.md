# Stage 02: Segmentation & Windowing

Stage 02 segments continuous 17-channel preprocessed EEG signals into uniform sliding time windows and assigns per-window ground-truth labels based on annotated seizure intervals.

---

## 1. Input & Output Contracts

### Input
- Continuous `mne.io.BaseRaw` object of preprocessed 17-channel bipolar EEG signals at $f_s = 256\text{ Hz}$ (produced by Pipeline C).
- Annotated seizure start/end intervals $(t_{\text{start}}, t_{\text{end}})$ in seconds.

### Parameters
- **`window_sec`**: 4.0 seconds ($1024$ samples at $256\text{ Hz}$).
- **`overlap`**: 0.50 ($50\%$ overlap / 2.0 second step / 512 samples).
- **`target_sfreq`**: 256.0 Hz.
- **`include_ambiguous`**: Boolean flag (`True` by default).

### Output (`SegmentationResult`)
- **`windows`**: 3D ndarray of shape `(n_windows, 17, 1024)`.
- **`metadata`**: List of per-window metadata dictionaries:
  - `patient_id`: Subject identifier (e.g. `chb01`).
  - `recording_id`: Recording identifier (e.g. `chb01_03`).
  - `window_id`: 0-indexed sequential window integer.
  - `start_sec`: Window start time in seconds from recording onset.
  - `end_sec`: Window end time in seconds from recording onset.
  - `duration_sec`: Window length in seconds ($4.0\text{ s}$).
  - `label`: Discrete state label ($1$, $0$, or $-1$).
- **`ch_names`**: Ordered list of 17 core bipolar channel names.

---

## 2. Seizure Overlap Labeling Policy

For each window $W_i = [t_{\text{start}}, t_{\text{end}}]$ and annotated seizure interval $I = [s_{\text{start}}, s_{\text{end}}]$:

$$\text{Overlap Ratio} = \frac{\text{length}(W_i \cap I)}{\text{length}(W_i)}$$

- **Ictal (`label = 1`)**: $\text{Overlap Ratio} \ge 0.50$ ($\ge 2.0\text{ s}$ of seizure activity within the 4.0 s window).
- **Interictal (`label = 0`)**: $\text{Overlap Ratio} = 0.0$.
- **Ambiguous / Transition (`label = -1`)**: $0.0 < \text{Overlap Ratio} < 0.50$. Ambiguous windows are excluded from supervised classification.

---

## 3. CHB01 Cohort Statistics

Across the 42 EDF recordings of Subject CHB01:
- **Total Windows Generated**: 72,951 windows (4.0s windows, 50% overlap).
- **Interictal Windows (`0`)**: 72,721 windows (99.68%).
- **Ictal Windows (`1`)**: 226 windows (0.31%).
- **Ambiguous Windows (`-1`)**: 4 windows (0.01%).

---
