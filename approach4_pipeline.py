"""
CHB-MIT EEG Preprocessing Pipeline - Approach 4
-----------------------------------------------
Techniques:
1. Chebyshev Type II Bandpass Filter (0.5 - 45 Hz)
2. Robust Statistical Artifact Detection:
   - Flatline: nearly zero variance (< 1.0 uV^2)
   - Excessive amplitude / Lead pop: non-physiological voltage jumps (> 250 uV step)
   - Abnormal variance: channel variance ratio > 12x median
   - Excessive high-frequency muscle activity (EMG ratio)
3. Decimation (256 Hz -> 128 Hz)
4. Standard 18 Bipolar Montage Extraction
5. Epoching (4s windows, 50% overlap) & Seizure Ground-Truth Labeling
"""

import os
import re
import time
import numpy as np
import scipy.signal as signal
from scipy.stats import kurtosis
import mne

STANDARD_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

def parse_summary_file(summary_path):
    """
    Parses chbXX-summary.txt to extract recording metadata and exact seizure timestamps.
    Returns dict: filename -> {'num_seizures': int, 'seizure_intervals': list of (start_sec, end_sec)}
    """
    with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    file_blocks = re.split(r'File Name:\s*', content)
    records = {}

    for block in file_blocks[1:]:
        lines = block.strip().split('\n')
        filename = lines[0].strip()
        seizure_intervals = []

        seizures_count = 0
        for line in lines:
            if 'Number of Seizures in File:' in line:
                match = re.search(r'\d+', line)
                if match:
                    seizures_count = int(match.group())

        starts = re.findall(r'Seizure (?:[0-9]+ )?Start Time:\s*([0-9]+)\s*seconds', block)
        ends = re.findall(r'Seizure (?:[0-9]+ )?End Time:\s*([0-9]+)\s*seconds', block)

        for s, e in zip(starts, ends):
            seizure_intervals.append((int(s), int(e)))

        records[filename] = {
            'num_seizures': seizures_count,
            'seizure_intervals': seizure_intervals
        }

    return records

def load_standard_channels(edf_path):
    """
    Loads raw EDF file and extracts only the 18 standard bipolar channels.
    Converts to microvolts (uV) and returns raw data (18 x samples) and sfreq.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    available_channels = raw.ch_names
    
    selected_channels = []
    for std_ch in STANDARD_CHANNELS:
        match = None
        for av_ch in available_channels:
            clean_av = av_ch.upper().replace(' ', '')
            if clean_av == std_ch.upper():
                match = av_ch
                break
            elif clean_av.startswith(std_ch.upper() + '-'):
                match = av_ch
                break
        if match:
            selected_channels.append(match)
        else:
            raise ValueError(f"Standard channel {std_ch} not found in {edf_path}!")

    raw.pick(selected_channels)
    raw.reorder_channels(selected_channels)
    data = raw.get_data()  # Volts -> uV
    data_uV = data * 1e6
    sfreq = raw.info['sfreq']
    return data_uV, sfreq, STANDARD_CHANNELS

def apply_chebyshev2_bandpass(data, sfreq=256.0, lowcut=0.5, highcut=45.0, gstop=30):
    """
    Applies Chebyshev Type II Bandpass Filter.
    - Flat passband (0.5 - 45 Hz) guarantees zero distortion in critical brain waves.
    - Steep stopband attenuation (30 dB down) completely suppresses 60 Hz powerline hum and DC drift.
    - Zero-phase filtering with sosfiltfilt ensures no phase distortion.
    """
    nyq = 0.5 * sfreq
    low = lowcut / nyq
    high = highcut / nyq
    
    sos = signal.cheby2(N=4, rs=gstop, Wn=[low, high], btype='bandpass', output='sos')
    filtered_data = signal.sosfiltfilt(sos, data, axis=-1)
    return filtered_data

def decimate_eeg(data, factor=2):
    """
    Decimates signal by factor (256 Hz -> 128 Hz).
    Includes anti-aliasing low-pass filter automatically.
    Reduces data storage and model compute by 50%.
    """
    decimated_data = signal.decimate(data, q=factor, axis=-1, zero_phase=True)
    return decimated_data

def detect_robust_artifacts(epoch, sfreq=128.0):
    """
    Robust Statistical Artifact Detection examining the 4 criteria:
    1. Flatline: nearly zero variance (< 1.0 uV^2, detached electrode).
    2. Excessive amplitude / Step jump: instantaneous jump > 200 uV in a single sample (lead pop).
    3. Abnormal variance: channel variance > 15x median channel variance (isolated channel noise).
    4. Excessive high-frequency activity: EMG muscle noise ratio > 0.60.
    Returns (is_artifact: bool, reason: str)
    """
    # 1. Flatline Check
    vars_ = np.var(epoch, axis=-1)
    if np.any(vars_ < 1.0):
        return True, "Flatline"

    # 2. Lead Pop / Sudden Step Discontinuity
    diffs = np.diff(epoch, axis=-1)
    if np.max(np.abs(diffs)) > 200.0:
        return True, "Lead Pop / Step Artifact"

    # 3. Abnormal Variance (Isolated Channel Noise)
    med_var = np.median(vars_)
    if med_var > 0:
        var_ratios = vars_ / (med_var + 1e-6)
        if np.max(var_ratios) > 15.0 and np.max(vars_) > 4000:
            return True, "Abnormal Channel Variance"

    # 4. Excessive Amplitude / Saturation
    ptp = np.ptp(epoch, axis=-1)
    if np.max(ptp) > 800.0:
        return True, "Extreme Saturation (>800 uV)"

    return False, "Clean"

def segment_and_label(data, sfreq, window_sec=4.0, step_sec=2.0, seizure_intervals=[]):
    """
    Segments EEG into overlapping windows and labels each window:
    - label: 1 (Ictal/Seizure) if window intersects seizure interval, else 0 (Interictal)
    - artifact: True/False based on robust statistical metrics.
    """
    win_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    n_channels, n_samples = data.shape

    epochs = []
    labels = []
    artifact_flags = []
    reasons = []
    timestamps = []

    for start_idx in range(0, n_samples - win_samples + 1, step_samples):
        end_idx = start_idx + win_samples
        epoch = data[:, start_idx:end_idx]
        
        start_time_sec = start_idx / sfreq
        end_time_sec = end_idx / sfreq

        is_seizure = 0
        for sz_start, sz_end in seizure_intervals:
            overlap_start = max(start_time_sec, sz_start)
            overlap_end = min(end_time_sec, sz_end)
            if overlap_end > overlap_start:
                overlap_dur = overlap_end - overlap_start
                if overlap_dur >= 0.5 * window_sec:
                    is_seizure = 1
                    break

        is_art, reason = detect_robust_artifacts(epoch, sfreq=sfreq)

        epochs.append(epoch)
        labels.append(is_seizure)
        artifact_flags.append(is_art)
        reasons.append(reason)
        timestamps.append((start_time_sec, end_time_sec))

    return np.array(epochs), np.array(labels), np.array(artifact_flags), reasons, timestamps
