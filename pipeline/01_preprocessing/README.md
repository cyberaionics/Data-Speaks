# 01 Preprocessing — Common Specification & Contract

## 1. Input Contract (Shared Stage 0 Standardized Output)
Every candidate pipeline receives data identical in structure, format, and metadata:
- **Format**: In-memory `mne.io.RawArray` or standardized structured dictionary / `.npy` array with metadata.
- **Channels**: Standard bipolar 10-20 montage EEG channels only (ECG, VNS, and dummy channels stripped).
- **Sampling Rate ($f_s$)**: Original recorded rate (typically 256 Hz) prior to pipeline resampling.
- **Reference**: Bipolar montage preserved; 3 CS2-referenced recordings routed via the unipolar branch.
- **Units**: Volts ($V$).

## 2. Pipeline Substep Contract
Each member's pipeline must implement the following sequential transformations:
1. **Band-Pass Filtering**: Target biological EEG band (e.g., 0.5–45 Hz or 1–40 Hz) + Notch filter (60 Hz power-line if applicable).
2. **Artifact Removal**: Denoiser implementation (ICA, Wavelet, ASR, or statistical thresholding).
3. **Normalization**: Scaling per channel across the temporal axis (Z-score or RobustScaler).
4. **Resampling / Decimation**: Target unified sampling rate of 256 Hz (or fixed decimation).

## 3. Output Contract
- **Shape**: `(n_channels, n_samples)`
- **Sampling Rate**: Unified 256 Hz across all pipelines.
- **Windowing Parameters**: 
  - Window length: 4.0 seconds (1024 samples at 256 Hz)
  - Overlap: 50% (2.0 seconds stride / 512 samples)
- **Labeling Scheme**:
  - `1` (Ictal): $\ge 50\%$ window overlap with annotated seizure interval.
  - `0` (Interictal): $0\%$ overlap and $> 60$ seconds away from any seizure boundary.
  - `-1` (Ambiguous / Boundary): Overlaps seizure $< 50\%$ or lies within the 60 s boundary buffer (excluded from supervised evaluation).

## 4. Evaluation Protocol
Pipelines are compared under identical conditions:
- **Level 1 (Intrinsic)**: Spectral power preservation (PSD preservation ratio), SNR improvement under synthetic noise, channel correlation, run-time ($s$), peak memory (MB).
- **Level 2 (Downstream)**: Identical downstream baseline (identical windowing, feature extraction, and patient-wise split classification).