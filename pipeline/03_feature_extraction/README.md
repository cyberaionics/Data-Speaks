# Stage 03: Feature Extraction

Stage 03 consumes the `SegmentationResult` produced by Stage 02 (`segmentation.py`) and generates a tabular feature representation (one row per 4-second EEG window) containing time-domain, Hjorth, and Welch spectral parameters across all 17 canonical bipolar channels.

---

## 1. Input & Output Contracts

### Input
- `SegmentationResult` from [`pipeline/02_segmentation/segmentation.py`](file:///C:/Users/Kavya/Data-Speaks/pipeline/02_segmentation/segmentation.py):
  - `windows`: 3D ndarray of shape `(n_windows, 17, 1024)` corresponding to 4-second windows sampled at 256 Hz with 50% overlap.
  - `metadata`: list of per-window metadata dictionaries (`patient_id`, `recording_id`, `window_id`, `start_sec`, `end_sec`, `duration_sec`, `label`).
  - `ch_names`: ordered list of 17 core bipolar channels.

### Output
- A `pandas.DataFrame` with shape `(n_windows, 449)`:
  - **7 Metadata columns** (preserved first): `patient_id`, `recording_id`, `window_id`, `start_sec`, `end_sec`, `duration_sec`, `label`.
  - **442 Numerical feature columns**: 17 channels $\times$ 26 features per channel, flattened into `<channel>_<feature_name>`.

---

## 2. Feature Definitions (26 Features per Channel)

### Time-Domain Statistics (11 Features)
Given a discrete 1D window signal $x[n]$ of length $N = 1024$ samples ($n = 0, \dots, N-1$):

1. **Mean**:
   $$\mu = \frac{1}{N}\sum_{n=0}^{N-1} x[n]$$
2. **Standard Deviation (std)**:
   $$\sigma = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} (x[n] - \mu)^2}$$
3. **Variance (var)**:
   $$\sigma^2 = \frac{1}{N}\sum_{n=0}^{N-1} (x[n] - \mu)^2$$
4. **Root Mean Square (rms)**:
   $$\text{RMS} = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x[n]^2}$$
5. **Skewness (skew)**: Fisher-Pearson coefficient of asymmetry ($0$ for flat/degenerate signals):
   $$\gamma_1 = \frac{\frac{1}{N}\sum (x[n] - \mu)^3}{\sigma^3}$$
6. **Kurtosis (kurtosis)**: Fisher excess kurtosis ($0$ for flat/degenerate signals):
   $$\gamma_2 = \frac{\frac{1}{N}\sum (x[n] - \mu)^4}{\sigma^4} - 3$$
7. **Minimum (min)**: $\min_n x[n]$
8. **Maximum (max)**: $\max_n x[n]$
9. **Peak-to-Peak (ptp)**: $\max_n x[n] - \min_n x[n]$
10. **Line Length**: Cumulative absolute length / waveform complexity:
    $$L = \sum_{n=1}^{N-1} |x[n] - x[n-1]|$$
11. **Zero-Crossing Rate (zero_crossing_rate)**: Fraction of sign transitions along the window:
    $$\text{ZCR} = \frac{1}{N-1}\sum_{n=1}^{N-1} \mathbb{I}(\text{sign}(x[n]) \neq \text{sign}(x[n-1]))$$

---

### Hjorth Parameters (3 Features)
Let $x'[n] = x[n] - x[n-1]$ and $x''[n] = x'[n] - x'[n-1]$:
12. **Hjorth Activity**:
    $$\text{Activity} = \text{var}(x) = \sigma_0^2$$
13. **Hjorth Mobility**:
    $$\text{Mobility} = \sqrt{\frac{\text{var}(x')}{\text{var}(x)}} = \frac{\sigma_1}{\sigma_0}$$
14. **Hjorth Complexity**:
    $$\text{Complexity} = \frac{\text{Mobility}(x')}{\text{Mobility}(x)} = \frac{\sigma_2 / \sigma_1}{\sigma_1 / \sigma_0}$$
*Note: If $\sigma_0^2 \le 10^{-12}$ or $\sigma_1^2 \le 10^{-12}$ (e.g. flat or constant signal), mobility and complexity safely evaluate to $0.0$.*

---

### Frequency-Domain & Spectral Features (12 Features)
Computed using Welch's Power Spectral Density (`scipy.signal.welch`) with 2-second sub-segments (`nperseg = min(len(x), int(sfreq * 2)) = 512 samples`):

- **Frequency Bands**:
  - **Delta**: $0.5 - 4.0\text{ Hz}$
  - **Theta**: $4.0 - 8.0\text{ Hz}$
  - **Alpha**: $8.0 - 13.0\text{ Hz}$
  - **Beta**: $13.0 - 30.0\text{ Hz}$
  - **Gamma**: $30.0 - 40.0\text{ Hz}$ *(bounded at 40 Hz to align with the 1–40 Hz bandpass filter)*

For each of the 5 bands:
15–19. **Absolute Band Power** (`<band>_power`): Numerical trapezoidal integration $\int_{f_{\text{low}}}^{f_{\text{high}}} P(f)\,df$.
20–24. **Relative Band Power** (`<band>_relpower`): $\frac{\text{Band Power}}{\text{Total Power}}$.
25. **Total Power (`total_power`)**: $\int_{0}^{f_{\text{Nyquist}}} P(f)\,df$.
26. **Spectral Edge Frequency 95% (`spectral_edge_95`)**: Frequency $f_{95}$ below which 95% of the total spectral power is concentrated:
    $$\int_0^{f_{95}} P(f)\,df = 0.95 \times \text{Total Power}$$

---

## 3. Robustness & Numerical Safety Guarantees

1. **No Silent NaNs/Infs**:
   - Zero-variance signals (flat electrodes, disconnected lines, or clipping) are detected.
   - Skewness and kurtosis return $0.0$ instead of `NaN`.
   - Divisors are strictly guarded with $\epsilon$ floors.
   - If any non-finite value is generated, `extract_features()` raises an informative `ValueError` rather than passing corrupted data downstream.
2. **Channel Ordering Safety**:
   - `SegmentationResult` explicitly records `ch_names`.
   - `extract_features(result, ch_names)` validates that `len(ch_names) == result.n_channels` and `ch_names == result.ch_names`. Channel ordering is strictly preserved and cannot be silently permuted.
3. **Metadata Integrity**:
   - Metadata columns (`patient_id`, `recording_id`, `window_id`, `start_sec`, `end_sec`, `duration_sec`, `label`) are placed at the beginning of the DataFrame. Labels ($0, 1, -1$) are never modified.

---

## 4. Usage Example

```python
from pipeline.02_segmentation.segmentation import segment_raw
from pipeline.03_feature_extraction.feature_extraction import extract_features

# 1. Segment preprocessed continuous EEG
seg_result = segment_raw(
    raw_clean,
    seizure_intervals=[(2996.0, 3036.0)],
    window_sec=4.0,
    overlap=0.50,
    patient_id="chb01",
    recording_id="chb01_03"
)

# 2. Extract features
features_df = extract_features(
    result=seg_result,
    ch_names=list(raw_clean.ch_names),
    sfreq=256.0
)

# 3. Inspect shape: (n_windows, 449)
print(features_df.shape)
```

---

## 5. Real Data Validation: `chb01_03`

Validation run command:
```powershell
C:\Users\Kavya\anaconda3\python.exe C:\Users\Kavya\Data-Speaks\pipeline\03_feature_extraction\run_feature_extraction_chb01_03.py
```

Results:
- **Input Recording**: `chb01_03.edf` (1 hour continuous EEG, 3600 seconds, 17 channels, seizure at 2996–3036s)
- **Preprocessed By**: Pipeline C (Butterworth 1–40 Hz $\rightarrow$ ASR $\rightarrow$ FastICA $\rightarrow$ Z-score $\rightarrow$ 256 Hz)
- **Number of Windows**: 1,799 windows
- **Number of Channels**: 17
- **Features per Channel**: 26
- **Total Feature Columns**: 442 numerical columns + 7 metadata columns = **449 columns**
- **Final DataFrame Shape**: `(1799, 449)`
- **NaNs / Infs**: 0 NaNs, 0 Infs
- **Label Distribution**:
  - Interictal (`0`): 1,778 windows
  - Ictal (`1`): 21 windows
  - Ambiguous (`-1`): 0 windows
- **Saved Output**: [`results/tables/chb01_03_features.csv`](file:///C:/Users/Kavya/Data-Speaks/results/tables/chb01_03_features.csv)
