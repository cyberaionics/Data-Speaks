"""
Execution and Verification Script for Approach 4 on chb01
---------------------------------------------------------
Processes representative recordings, computes performance metrics,
and generates exploratory data analysis (EDA) plots for project deliverables.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# Add local directory to path
sys.path.append(r'C:\Users\avani\.gemini\antigravity\scratch\chbmit_pipeline_approach4')
import approach4_pipeline as ap4

DATA_DIR = r'C:\Users\avani\chbmit\chb01'
SUMMARY_PATH = os.path.join(DATA_DIR, 'chb01-summary.txt')
PLOTS_DIR = r'C:\Users\avani\.gemini\antigravity\scratch\chbmit_pipeline_approach4\plots'
OUTPUT_DIR = r'C:\Users\avani\.gemini\antigravity\scratch\chbmit_pipeline_approach4\output'


def run_pipeline():
    print("=" * 75)
    print("CHB-MIT DATASET: APPROACH 4 PIPELINE EXECUTION (Subject: chb01)")
    print("Chebyshev II Bandpass + Robust Statistical Artifacts + Decimation")
    print("=" * 75)

    # 1. Parse Summary Metadata
    print("\n[Step 1] Parsing Seizure Annotations from chb01-summary.txt...")
    summary_records = ap4.parse_summary_file(SUMMARY_PATH)
    print(f"Parsed {len(summary_records)} recording summaries.")
    for rec, details in summary_records.items():
        if details['num_seizures'] > 0:
            print(f"  -> {rec}: {details['num_seizures']} seizure(s) at {details['seizure_intervals']}")

    # Pick representative recordings:
    # chb01_03.edf (Seizure at 2996-3036s)
    # chb01_01.edf (Baseline normal recording, 0 seizures)
    # chb01_04.edf (Seizure at 1467-1494s)
    test_files = ['chb01_03.edf', 'chb01_01.edf', 'chb01_04.edf']
    
    results = []

    for fname in test_files:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"Warning: {fname} not found, skipping.")
            continue

        print(f"\n" + "-" * 65)
        print(f"Processing Recording: {fname}")
        print("-" * 65)

        raw_size_mb = os.path.getsize(fpath) / (1024 * 1024)
        seizure_info = summary_records.get(fname, {'num_seizures': 0, 'seizure_intervals': []})

        t0 = time.time()

        # Step 2: Load Standard 18 Channels
        raw_data, orig_sfreq, ch_names = ap4.load_standard_channels(fpath)
        duration_sec = raw_data.shape[1] / orig_sfreq
        t_load = time.time() - t0
        print(f"  [18 Standard Channels] Duration: {duration_sec:.1f}s ({duration_sec/3600:.2f} hrs), Orig Freq: {orig_sfreq} Hz")

        # Step 3: Chebyshev Type II Bandpass Filter (0.5 - 45 Hz)
        t_filt_start = time.time()
        filtered_data = ap4.apply_chebyshev2_bandpass(raw_data, sfreq=orig_sfreq, lowcut=0.5, highcut=45.0, gstop=30)
        t_filt = time.time() - t_filt_start
        print(f"  [Chebyshev II Filter] Completed in {t_filt:.3f}s (Speed: {duration_sec/t_filt:.1f}x real-time)")

        # Step 4: Decimation (256 Hz -> 128 Hz)
        t_dec_start = time.time()
        decimated_data = ap4.decimate_eeg(filtered_data, factor=2)
        new_sfreq = orig_sfreq / 2
        t_dec = time.time() - t_dec_start
        print(f"  [Decimation 256Hz -> 128Hz] New shape: {decimated_data.shape}, Completed in {t_dec:.3f}s")

        # Step 5: Windowing & Robust Statistical Artifact Detection
        t_win_start = time.time()
        epochs, labels, artifacts, reasons, timestamps = ap4.segment_and_label(
            decimated_data, sfreq=new_sfreq, window_sec=4.0, step_sec=2.0,
            seizure_intervals=seizure_info['seizure_intervals']
        )
        t_win = time.time() - t_win_start

        total_epochs = len(labels)
        seizure_epochs = int(np.sum(labels == 1))
        normal_epochs = total_epochs - seizure_epochs
        artifact_count = int(np.sum(artifacts))
        clean_count = total_epochs - artifact_count
        artifact_pct = (artifact_count / total_epochs) * 100 if total_epochs > 0 else 0

        total_proc_time = t_filt + t_dec + t_win

        print(f"  [Epoching & Artifacts] Total: {total_epochs} (4s windows, 50% overlap)")
        print(f"  [Seizure Epochs] {seizure_epochs} ictal vs {normal_epochs} interictal")
        print(f"  [Artifacts Flagged] {artifact_count} / {total_epochs} ({artifact_pct:.2f}%)")
        print(f"  [Total Preprocessing Time] {total_proc_time:.3f} seconds for 1 hour of EEG!")

        results.append({
            'recording_id': fname,
            'duration_hours': round(duration_sec / 3600, 2),
            'raw_size_mb': round(raw_size_mb, 2),
            'processed_size_mb': round(raw_size_mb * (18 / 23) * 0.5, 2),
            'orig_sfreq': orig_sfreq,
            'final_sfreq': new_sfreq,
            'seizure_count': seizure_info['num_seizures'],
            'total_epochs': total_epochs,
            'seizure_epochs': seizure_epochs,
            'clean_epochs': clean_count,
            'artifact_epochs': artifact_count,
            'artifact_pct': round(artifact_pct, 2),
            'filter_time_sec': round(t_filt, 3),
            'decimate_time_sec': round(t_dec, 3),
            'total_time_sec': round(total_proc_time, 3),
            'speedup_vs_realtime': round(duration_sec / total_proc_time, 1)
        })

        # Generate plots on chb01_03 (primary seizure recording)
        if fname == 'chb01_03.edf':
            print("\n[Step 6] Generating Publication-Quality EDA Plots for chb01_03...")
            generate_plots(raw_data, filtered_data, decimated_data, orig_sfreq, epochs, labels, artifacts, reasons, ch_names)

    # Save summary dataframe
    df_metrics = pd.DataFrame(results)
    metrics_path = os.path.join(OUTPUT_DIR, 'approach4_chb01_metrics.csv')
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics summary to: {metrics_path}")

    return df_metrics


def generate_plots(raw_data, filtered_data, decimated_data, sfreq, epochs, labels, artifacts, reasons, ch_names):
    # -------------------------------------------------------------
    # Plot 1: Power Spectral Density (PSD) - Raw vs Chebyshev II
    # -------------------------------------------------------------
    ch_idx = 0  # FP1-F7
    f_raw, psd_raw = signal.welch(raw_data[ch_idx, :int(sfreq * 300)], fs=sfreq, nperseg=1024)
    f_filt, psd_filt = signal.welch(filtered_data[ch_idx, :int(sfreq * 300)], fs=sfreq, nperseg=1024)

    plt.figure(figsize=(10, 5), dpi=200)
    plt.semilogy(f_raw, psd_raw, label='Raw EEG (Unfiltered)', color='#7f7f7f', alpha=0.7)
    plt.semilogy(f_filt, psd_filt, label='Chebyshev Type II Filtered (0.5 - 45 Hz)', color='#008080', linewidth=2)
    plt.axvline(60, color='#d62728', linestyle='--', linewidth=1.5, label='60 Hz Powerline Hum Suppressed (>30 dB)')
    plt.axvspan(0.5, 45, color='#e0f3f8', alpha=0.4, label='Clinical EEG Band (Passband)')
    plt.xlim([0, 90])
    plt.title('Power Spectral Density (PSD): Raw vs. Chebyshev Type II Filter\nChannel: FP1-F7 (chb01_03)', fontsize=13, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=11)
    plt.ylabel(r'Power Spectral Density ($\mu V^2 / Hz$)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    psd_path = os.path.join(PLOTS_DIR, 'psd_chebyshev_comparison.png')
    plt.savefig(psd_path)
    plt.close()
    print(f"  -> Generated: {psd_path}")

    # -------------------------------------------------------------
    # Plot 2: Waveform Comparison: Interictal (Normal) vs Ictal (Seizure)
    # -------------------------------------------------------------
    seizure_indices = np.where(labels == 1)[0]
    normal_indices = np.where((labels == 0) & (~artifacts))[0]

    if len(seizure_indices) > 0 and len(normal_indices) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, dpi=200)
        disp_channels = [0, 4, 8, 12]  # FP1-F7, FP1-F3, FP2-F4, FP2-F8
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # Normal Epoch
        norm_ep = epochs[normal_indices[20]]
        t_axis = np.linspace(0, 4, norm_ep.shape[1])
        for idx, c_i in enumerate(disp_channels):
            axes[0].plot(t_axis, norm_ep[c_i] + idx * 90, color=colors[idx], label=ch_names[c_i])
            axes[0].text(-0.2, idx * 90, ch_names[c_i], fontsize=9, verticalalignment='center', fontweight='bold')
        axes[0].set_title('A. Baseline Normal EEG (Interictal State)', fontsize=12, fontweight='bold', color='#002b36')
        axes[0].set_ylabel('Amplitude Offset (uV)')
        axes[0].set_xlim([0, 4])
        axes[0].grid(True, linestyle=':', alpha=0.5)

        # Seizure Epoch
        sz_ep = epochs[seizure_indices[5]]
        for idx, c_i in enumerate(disp_channels):
            axes[1].plot(t_axis, sz_ep[c_i] + idx * 90, color=colors[idx])
            axes[1].text(-0.2, idx * 90, ch_names[c_i], fontsize=9, verticalalignment='center', fontweight='bold')
        axes[1].set_title('B. Epileptic Seizure Discharge (Ictal State - High-Amplitude Rhythmic Waves)', fontsize=12, fontweight='bold', color='#b58900')
        axes[1].set_xlabel('Time within Window (seconds)', fontsize=11)
        axes[1].set_ylabel('Amplitude Offset (uV)')
        axes[1].set_xlim([0, 4])
        axes[1].grid(True, linestyle=':', alpha=0.5)

        plt.suptitle('Representative EEG Waveforms: Normal vs. Seizure (Subject: chb01)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        waveform_path = os.path.join(PLOTS_DIR, 'seizure_vs_normal_eeg.png')
        plt.savefig(waveform_path)
        plt.close()
        print(f"  -> Generated: {waveform_path}")

    # -------------------------------------------------------------
    # Plot 3: Artifact Detection Breakdown
    # -------------------------------------------------------------
    reason_counts = pd.Series(reasons).value_counts()
    plt.figure(figsize=(7, 4), dpi=200)
    colors_bars = ['#2ca02c' if 'Clean' in k else '#d62728' for k in reason_counts.index]
    bars = plt.bar(reason_counts.index, reason_counts.values, color=colors_bars, width=0.5)
    plt.title('Statistical Artifact Detection Results (chb01_03 - 1,799 Epochs)', fontsize=12, fontweight='bold')
    plt.ylabel('Epoch Count', fontsize=11)
    plt.xticks(rotation=10, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f"{yval} ({yval/len(epochs)*100:.1f}%)", ha='center', va='bottom', fontsize=9)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.ylim([0, max(reason_counts.values) * 1.15])
    plt.tight_layout()
    art_path = os.path.join(PLOTS_DIR, 'artifact_detection_breakdown.png')
    plt.savefig(art_path)
    plt.close()
    print(f"  -> Generated: {art_path}")


if __name__ == '__main__':
    df = run_pipeline()
    print("\n" + "=" * 75)
    print("FINAL SUMMARY METRICS TABLE:")
    print("=" * 75)
    print(df.to_string(index=False))
