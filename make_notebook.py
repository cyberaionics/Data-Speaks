import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title & Intro
cells.append(nbf.v4.new_markdown_cell("""# Course Assignment: Data Speaks!! (IIT Dharwad)
## EEG Seizure Analysis & Preprocessing on CHB-MIT Scalp EEG
### Approach 4: Chebyshev Type II Bandpass + Robust Statistical Artifact Detection + Decimation
**Author:** Avni  
**Target Subject:** `chb01`  
**Dataset:** CHB-MIT Scalp EEG Database (Boston Children's Hospital)

---

### Project Overview
This notebook implements **Approach 4** of the comparative EEG preprocessing pipeline evaluation:
1. **Common Standard Channels:** Extraction of 18 bipolar channels (International 10-20 Double Banana montage).
2. **Chebyshev Type II Bandpass Filter (0.5 - 45 Hz):** Zero-phase IIR filtering with maximally flat passband and 30 dB stopband attenuation (suppressing 60 Hz US powerline noise and DC drift).
3. **Decimation (256 Hz -> 128 Hz):** 2x downsampling with automatic anti-aliasing lowpass filtering (cutting storage & compute by 50%).
4. **Robust Statistical Artifact Detection:** Fast outlier detection using Flatline checks, lead pop discontinuities, and channel variance ratios.
5. **Windowing & Ground-Truth Labeling:** 4.0-second sliding windows (50% overlap) labeled against clinical seizure intervals from `chb01-summary.txt`.
"""))

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import scipy.signal as signal

# Import our modular pipeline module
import approach4_pipeline as ap4

print('Libraries loaded successfully!')
print(f'MNE Version: {mne.__version__}')
"""))

# Cell 2: Annotations
cells.append(nbf.v4.new_markdown_cell("""### 1. Parse Clinical Ground Truth Annotations"""))
cells.append(nbf.v4.new_code_cell("""summary_path = r'C:\\Users\\avani\\chbmit\\chb01\\chb01-summary.txt'
records = ap4.parse_summary_file(summary_path)

print(f'Found {len(records)} recordings in chb01.')
seizure_records = {k: v for k, v in records.items() if v['num_seizures'] > 0}
print(f'Recordings with seizures ({len(seizure_records)}):')
for k, v in seizure_records.items():
    print(f'  {k}: {v["num_seizures"]} seizure(s) at {v["seizure_intervals"]} seconds')
"""))

# Cell 3: Processing
cells.append(nbf.v4.new_markdown_cell("""### 2. Run Complete Pipeline on Representative Recording (`chb01_03.edf`)"""))
cells.append(nbf.v4.new_code_cell("""edf_path = r'C:\\Users\\avani\\chbmit\\chb01\\chb01_03.edf'

# Step A: Load standard 18 channels
raw_uV, orig_sfreq, ch_names = ap4.load_standard_channels(edf_path)
print(f'Loaded raw data: {raw_uV.shape[0]} channels, {raw_uV.shape[1]} samples at {orig_sfreq} Hz')

# Step B: Apply Chebyshev Type II Bandpass
filtered_data = ap4.apply_chebyshev2_bandpass(raw_uV, sfreq=orig_sfreq, lowcut=0.5, highcut=45.0, gstop=30)
print('Chebyshev Type II Bandpass applied successfully.')

# Step C: Decimate to 128 Hz
decimated_data = ap4.decimate_eeg(filtered_data, factor=2)
new_sfreq = orig_sfreq / 2
print(f'Decimated data shape: {decimated_data.shape} at {new_sfreq} Hz')

# Step D: Segment and detect artifacts
epochs, labels, artifacts, reasons, timestamps = ap4.segment_and_label(
    decimated_data, sfreq=new_sfreq, window_sec=4.0, step_sec=2.0,
    seizure_intervals=records['chb01_03.edf']['seizure_intervals']
)
print(f'Generated {len(epochs)} epochs (4s each).')
print(f'Seizure epochs: {np.sum(labels==1)}, Artifact epochs flagged: {np.sum(artifacts)}')
"""))

# Cell 4: PSD
cells.append(nbf.v4.new_markdown_cell("""### 3. Exploratory Data Analysis (EDA) Visualizations"""))
cells.append(nbf.v4.new_code_cell("""# Visualizing Power Spectral Density (PSD)
f_raw, psd_raw = signal.welch(raw_uV[0, :int(orig_sfreq * 300)], fs=orig_sfreq, nperseg=1024)
f_filt, psd_filt = signal.welch(filtered_data[0, :int(orig_sfreq * 300)], fs=orig_sfreq, nperseg=1024)

plt.figure(figsize=(10, 4.5))
plt.semilogy(f_raw, psd_raw, label='Raw EEG (Unfiltered)', color='gray', alpha=0.7)
plt.semilogy(f_filt, psd_filt, label='Chebyshev Type II (0.5 - 45 Hz)', color='teal', linewidth=2)
plt.axvline(60, color='crimson', linestyle='--', label='60 Hz Powerline Hum Suppressed')
plt.xlim([0, 90])
plt.title('Power Spectral Density (PSD): Raw vs. Chebyshev Type II Filter (FP1-F7)', fontweight='bold')
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'PSD ($\mu V^2 / Hz$)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
"""))

# Cell 5: Waveform
cells.append(nbf.v4.new_code_cell("""# Waveform: Normal (Interictal) vs Seizure (Ictal)
seizure_idx = np.where(labels == 1)[0][5]
normal_idx = np.where((labels == 0) & (~artifacts))[0][20]

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
t = np.linspace(0, 4, epochs.shape[-1])
channels_to_plot = [0, 4, 8, 12]

for i, ch_i in enumerate(channels_to_plot):
    axes[0].plot(t, epochs[normal_idx, ch_i] + i * 80, label=ch_names[ch_i])
    axes[1].plot(t, epochs[seizure_idx, ch_i] + i * 80, label=ch_names[ch_i])

axes[0].set_title('Normal Baseline EEG (Interictal)', fontweight='bold', color='navy')
axes[0].set_ylabel('Amplitude (uV)')
axes[0].grid(True, linestyle=':', alpha=0.5)

axes[1].set_title('Epileptic Seizure Discharge (Ictal Spike & Wave)', fontweight='bold', color='crimson')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude (uV)')
axes[1].grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()
"""))

# Cell 6: Metrics Table
cells.append(nbf.v4.new_markdown_cell("""### 4. Summary Metrics Table across chb01"""))
cells.append(nbf.v4.new_code_cell("""metrics_df = pd.read_csv(r'C:\\Users\\avani\\.gemini\\antigravity\\scratch\\chbmit_pipeline_approach4\\output\\approach4_chb01_metrics.csv')
metrics_df
"""))

nb['cells'] = cells

with open(r'C:\Users\avani\.gemini\antigravity\scratch\chbmit_pipeline_approach4\Approach4_CHB01_Preprocessing_EDA.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Jupyter Notebook successfully created at C:\\Users\\avani\\.gemini\\antigravity\\scratch\\chbmit_pipeline_approach4\\Approach4_CHB01_Preprocessing_EDA.ipynb')
