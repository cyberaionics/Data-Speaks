"""
Midterm 5-Pillar Exploratory Data Analysis (EDA) on CHB-MIT (chb01 / chb01_03).

Pillars:
1. Clinical & Metadata EDA (Class imbalance, seizure durations, cohort demographics)
2. Time-Domain Statistical EDA (Amplitude KDEs, variance/kurtosis/PTP moments, stacked waveforms)
3. Frequency-Domain / Spectral EDA (5 canonical brain bands: Delta, Theta, Alpha, Beta, Gamma)
4. Time-Frequency EDA (STFT Spectrogram transition across pre-ictal, ictal, post-ictal)
5. Spatial & Cross-Channel Synchrony (17x17 Pearson correlation matrix heatmaps)

Outputs saved to results/figures/eda/
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\Kavya\Data-Speaks")
PREPROC_DIR = REPO_ROOT / "pipeline" / "01_preprocessing"
CANDIDATE_C_DIR = PREPROC_DIR / "candidate_algorithms"
sys.path.insert(0, str(PREPROC_DIR))
sys.path.insert(0, str(CANDIDATE_C_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
import seaborn as sns

from stage0 import run_stage0, DATASET_DIR

EDA_FIG_DIR = REPO_ROOT / "results" / "figures" / "eda"
EDA_FIG_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# PILLAR 1: Clinical & Metadata EDA
# ===========================================================================

def run_pillar_1():
    print("--- Running Pillar 1: Clinical & Metadata EDA ---")
    data_dir = DATASET_DIR
    summary_file = data_dir / "chb01" / "chb01-summary.txt"
    subject_info_file = data_dir / "SUBJECT-INFO"

    # 1. Parse chb01 seizure annotations
    text = summary_file.read_text()
    seizures = re.findall(
        r"Seizure\s+\d*\s*Start\s+Time:\s*(\d+)\s*seconds.*?Seizure\s+\d*\s*End\s+Time:\s*(\d+)\s*seconds",
        text,
        re.DOTALL,
    )
    durations = [int(end) - int(start) for start, end in seizures]
    total_seizure_sec = sum(durations)

    # In chb01, there are ~40-42 hours of recording (~144,000 to 151,200 sec)
    # Total EDF files count: 42
    n_files = len(list((data_dir / "chb01").glob("*.edf")))
    approx_total_sec = n_files * 3600.0
    seizure_pct = (total_seizure_sec / approx_total_sec) * 100
    normal_pct = 100.0 - seizure_pct

    # 2. Parse SUBJECT-INFO
    sub_lines = [l.strip() for l in subject_info_file.read_text().splitlines() if l.strip()]
    header = sub_lines[0].split("\t")
    records = []
    for line in sub_lines[1:]:
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) >= 3:
            records.append({"case": parts[0], "gender": parts[1], "age": float(parts[2])})
    demog_df = pd.DataFrame(records)

    # Plot Pillar 1
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1])

    # Subplot 1: Extreme Class Imbalance Bar Chart
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(["Normal State", "Seizure State"], [approx_total_sec / 3600, total_seizure_sec / 3600], color=["#1f77b4", "crimson"], alpha=0.85)
    ax1.set_yscale("log")
    ax1.set_ylabel("Total Duration (Hours, log-scale)", fontsize=10)
    ax1.set_title(f"Severe Class Imbalance\n(Seizures: {seizure_pct:.2f}% of Total Time)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, which="both")
    ax1.text(0, approx_total_sec / 3600, f"~{n_files} hrs\n(99.71%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.text(1, total_seizure_sec / 3600, f"{total_seizure_sec/60:.1f} mins\n(0.29%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 2: Seizure Duration Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(1, len(durations) + 1), durations, color="coral", edgecolor="black", alpha=0.85)
    ax2.axhline(np.mean(durations), color="red", linestyle="--", label=f"Mean: {np.mean(durations):.1f}s")
    ax2.set_xlabel("Seizure Incident (#)", fontsize=10)
    ax2.set_ylabel("Duration (seconds)", fontsize=10)
    ax2.set_title("chb01 Seizure Duration Distribution (7 Seizures)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    for i, d in enumerate(durations):
        ax2.text(i + 1, d + 2, f"{d}s", ha="center", va="bottom", fontsize=8)

    # Subplot 3: Pediatric Demographics
    ax3 = fig.add_subplot(gs[0, 2])
    gender_counts = demog_df["gender"].value_counts()
    ax3.pie(gender_counts, labels=[f"Female ({gender_counts.get('F',0)})", f"Male ({gender_counts.get('M',0)})"], autopct="%1.1f%%", colors=["#ff9999", "#66b3ff"], startangle=140)
    ax3.set_title(f"CHB-MIT Pediatric Cohort\n(N={len(demog_df)}, Age: {demog_df['age'].min():.1f}-{demog_df['age'].max():.1f} yrs)", fontsize=11, fontweight="bold")

    plt.suptitle("Pillar 1: Clinical & Metadata Exploratory Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = EDA_FIG_DIR / "pillar1_clinical_summary.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ===========================================================================
# PILLAR 2: Time-Domain Statistical EDA
# ===========================================================================

def run_pillar_2(raw_std, features_df):
    print("--- Running Pillar 2: Time-Domain Statistical EDA ---")
    # 1. Amplitude Distribution (Raw in microvolts)
    # Seizure interval for chb01_03 is 2996.0 - 3036.0
    sfreq = float(raw_std.info["sfreq"])
    data = raw_std.get_data() * 1e6  # convert Volts to microvolts (uV)
    ch_idx = 0  # FP1-F7 (frontal channel strongly reflecting ictal activity)
    ch_name = raw_std.ch_names[ch_idx]

    normal_samples = data[ch_idx, int(2900 * sfreq):int(2980 * sfreq)]
    seizure_samples = data[ch_idx, int(2996 * sfreq):int(3036 * sfreq)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Subplot 1: Amplitude Histogram & KDE
    ax1 = axes[0]
    sns.kdeplot(normal_samples, ax=ax1, color="#1f77b4", label=f"Normal State (±50 μV)", fill=True, alpha=0.3, linewidth=1.5)
    sns.kdeplot(seizure_samples, ax=ax1, color="crimson", label=f"Seizure State (±200-400 μV)", fill=True, alpha=0.3, linewidth=1.5)
    ax1.set_xlim(-400, 400)
    ax1.set_xlabel("Voltage Amplitude (μV)", fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=10)
    ax1.set_title(f"Voltage Amplitude Distribution ({ch_name})", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Statistical Moments Comparison
    ax2 = axes[1]
    moments = ["Variance", "Kurtosis", "PTP Amplitude"]
    # Compute average from features_df across channels
    interictal_df = features_df[features_df["label"] == 0]
    ictal_df = features_df[features_df["label"] == 1]

    norm_vals = [
        interictal_df[f"{ch_name}_var"].mean(),
        interictal_df[f"{ch_name}_kurtosis"].mean(),
        interictal_df[f"{ch_name}_ptp"].mean(),
    ]
    seiz_vals = [
        ictal_df[f"{ch_name}_var"].mean(),
        ictal_df[f"{ch_name}_kurtosis"].mean(),
        ictal_df[f"{ch_name}_ptp"].mean(),
    ]

    x = np.arange(len(moments))
    w = 0.35
    ax2.bar(x - w/2, norm_vals, width=w, label="Normal", color="#1f77b4", alpha=0.85)
    ax2.bar(x + w/2, seiz_vals, width=w, label="Seizure", color="crimson", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(moments, fontsize=10)
    ax2.set_title("Statistical Moments Comparison", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Normalized Value / Arbitrary Units", fontsize=10)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Multi-Channel Stacked Waveforms at Seizure Onset
    ax3 = axes[2]
    t_start, t_end = 2992.0, 3012.0
    s_idx, e_idx = int(t_start * sfreq), int(t_end * sfreq)
    t_axis = np.linspace(t_start, t_end, e_idx - s_idx)

    # Stack top 8 channels
    n_show = 8
    spacing = 150  # uV
    for i in range(n_show):
        sig = data[i, s_idx:e_idx] + (n_show - 1 - i) * spacing
        ax3.plot(t_axis, sig, color="black", linewidth=0.7)
        ax3.text(t_start - 0.5, (n_show - 1 - i) * spacing, raw_std.ch_names[i], ha="right", va="center", fontsize=8)

    ax3.axvspan(2996.0, t_end, color="crimson", alpha=0.15, label="Ictal State")
    ax3.axvline(2996.0, color="crimson", linestyle="--", linewidth=1.2, label="Onset (2996s)")
    ax3.set_xlabel("Time (seconds)", fontsize=10)
    ax3.set_title("Multichannel Stacked Waveforms (Onset)", fontsize=11, fontweight="bold")
    ax3.set_yticks([])
    ax3.legend(loc="upper right", fontsize=8)
    ax3.set_xlim(t_start, t_end)
    ax3.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("Pillar 2: Time-Domain Statistical Exploratory Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = EDA_FIG_DIR / "pillar2_time_domain_stats.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ===========================================================================
# PILLAR 3: Frequency-Domain / Spectral EDA
# ===========================================================================

def run_pillar_3(features_df):
    print("--- Running Pillar 3: Frequency-Domain / Spectral EDA ---")
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    band_names = ["Delta (0.5-4Hz)", "Theta (4-8Hz)", "Alpha (8-13Hz)", "Beta (13-30Hz)", "Gamma (30-40Hz)"]

    interictal_df = features_df[features_df["label"] == 0]
    ictal_df = features_df[features_df["label"] == 1]

    # Calculate mean relative band power across all 17 channels
    normal_rel = []
    seiz_rel = []
    for b in bands:
        cols = [c for c in features_df.columns if c.endswith(f"_{b}_relpower")]
        normal_rel.append(interictal_df[cols].mean().mean() * 100)
        seiz_rel.append(ictal_df[cols].mean().mean() * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of 5 bands
    x = np.arange(len(bands))
    w = 0.35
    ax1.bar(x - w/2, normal_rel, width=w, label="Normal State", color="#1f77b4", alpha=0.85)
    ax1.bar(x + w/2, seiz_rel, width=w, label="Seizure State", color="crimson", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(band_names, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("Relative Spectral Power (%)", fontsize=10)
    ax1.set_title("5 Canonical Brain Waves Power Distribution", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Add ratio change comparison
    delta_theta_normal = (normal_rel[0] + normal_rel[1])
    delta_theta_seiz = (seiz_rel[0] + seiz_rel[1])
    ax2.bar(["Normal State", "Seizure State"], [delta_theta_normal, delta_theta_seiz], color=["#1f77b4", "crimson"], alpha=0.85, width=0.45)
    ax2.set_ylabel("Delta + Theta Combined Power (%)", fontsize=10)
    ax2.set_title("Dominance of Low-Frequency Slow Rhythms (Delta + Theta)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.text(0, delta_theta_normal + 1, f"{delta_theta_normal:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax2.text(1, delta_theta_seiz + 1, f"{delta_theta_seiz:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Pillar 3: Frequency-Domain Spectral Exploratory Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = EDA_FIG_DIR / "pillar3_spectral_band_distribution.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ===========================================================================
# PILLAR 4: Time-Frequency EDA (Spectrogram / STFT)
# ===========================================================================

def run_pillar_4(raw_std):
    print("--- Running Pillar 4: Time-Frequency EDA (Spectrogram / STFT) ---")
    sfreq = float(raw_std.info["sfreq"])
    # 3-minute window spanning pre-ictal (2900s) -> onset (2996s) -> post-ictal (3080s)
    t_start, t_end = 2930.0, 3070.0
    s_idx, e_idx = int(t_start * sfreq), int(t_end * sfreq)

    data = raw_std.get_data()
    ch_idx = 0  # FP1-F7
    ch_name = raw_std.ch_names[ch_idx]
    sig = data[ch_idx, s_idx:e_idx]

    f, t_spec, Sxx = spectrogram(sig, fs=sfreq, nperseg=int(sfreq * 2), noverlap=int(sfreq * 1.5))
    t_spec_abs = t_spec + t_start

    # Limit to 0-40 Hz
    mask = f <= 40.0
    f_sub = f[mask]
    Sxx_sub = 10 * np.log10(Sxx[mask, :] + 1e-12)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1, 2]})

    # Waveform
    t_sig = np.linspace(t_start, t_end, len(sig))
    ax1.plot(t_sig, sig * 1e6, color="#1f77b4", linewidth=0.8)
    ax1.axvspan(2996.0, 3036.0, color="crimson", alpha=0.2, label="Seizure Interval (2996-3036s)")
    ax1.axvline(2996.0, color="crimson", linestyle="--", label="Seizure Onset")
    ax1.set_ylabel("Amplitude (μV)", fontsize=10)
    ax1.set_title(f"Waveform & STFT Spectrogram Transition ({ch_name})", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Spectrogram Heatmap
    im = ax2.pcolormesh(t_spec_abs, f_sub, Sxx_sub, shading="gouraud", cmap="inferno")
    ax2.axvline(2996.0, color="cyan", linestyle="--", linewidth=1.5, label="Seizure Onset (2996s)")
    ax2.axvline(3036.0, color="cyan", linestyle=":", linewidth=1.5, label="Seizure Offset (3036s)")
    ax2.set_ylabel("Frequency (Hz)", fontsize=10)
    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)

    cbar = fig.colorbar(im, ax=ax2, orientation="horizontal", pad=0.2, aspect=40)
    cbar.set_label("Power Spectral Density (dB)", fontsize=9)

    plt.suptitle("Pillar 4: Time-Frequency STFT Spectrogram Analysis", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = EDA_FIG_DIR / "pillar4_stft_spectrogram.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ===========================================================================
# PILLAR 5: Spatial & Cross-Channel Synchrony
# ===========================================================================

def run_pillar_5(raw_std):
    print("--- Running Pillar 5: Spatial & Cross-Channel Synchrony ---")
    sfreq = float(raw_std.info["sfreq"])
    data = raw_std.get_data()
    ch_names = list(raw_std.ch_names)

    # 1. Normal window (e.g. 2900 - 2940s, 40 seconds)
    norm_data = data[:, int(2900 * sfreq):int(2940 * sfreq)]
    corr_norm = np.corrcoef(norm_data)

    # 2. Seizure window (2996 - 3036s, 40 seconds)
    seiz_data = data[:, int(2996 * sfreq):int(3036 * sfreq)]
    corr_seiz = np.corrcoef(seiz_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Heatmap Normal
    sns.heatmap(corr_norm, ax=ax1, xticklabels=ch_names, yticklabels=ch_names, cmap="coolwarm", vmin=-1.0, vmax=1.0, cbar_kws={"label": "Pearson Correlation (r)"})
    ax1.set_title("Interictal Normal State (Regional Independence)", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="x", rotation=90, labelsize=8)
    ax1.tick_params(axis="y", rotation=0, labelsize=8)

    # Heatmap Seizure
    sns.heatmap(corr_seiz, ax=ax2, xticklabels=ch_names, yticklabels=ch_names, cmap="coolwarm", vmin=-1.0, vmax=1.0, cbar_kws={"label": "Pearson Correlation (r)"})
    ax2.set_title("Ictal State (Widespread Hyper-Synchronization)", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="x", rotation=90, labelsize=8)
    ax2.tick_params(axis="y", rotation=0, labelsize=8)

    # Quantify mean synchrony
    upper_tri = np.triu_indices(len(ch_names), k=1)
    mean_corr_norm = np.mean(corr_norm[upper_tri])
    mean_corr_seiz = np.mean(corr_seiz[upper_tri])

    fig.text(0.5, 0.01, f"Mean Cross-Channel Correlation: Normal r = {mean_corr_norm:.3f}  -->  Seizure r = {mean_corr_seiz:.3f}", ha="center", fontsize=10, fontweight="bold", bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

    plt.suptitle("Pillar 5: Spatial & Cross-Channel Connectivity EDA (17x17 Correlation)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = EDA_FIG_DIR / "pillar5_spatial_correlation_heatmaps.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ===========================================================================
# MAIN RUNNER
# ===========================================================================

def main():
    print("=" * 75)
    print("STARTING MIDTERM 5-PILLAR EDA")
    print("=" * 75)

    # Load chb01_03 raw and features
    edf_path = DATASET_DIR / "chb01" / "chb01_03.edf"
    summary_path = DATASET_DIR / "chb01" / "chb01-summary.txt"
    s0 = run_stage0(edf_path, summary_path, "chb01", "chb01_03")

    features_path = REPO_ROOT / "results" / "tables" / "chb01_03_features.csv"
    features_df = pd.read_csv(features_path)

    run_pillar_1()
    run_pillar_2(s0.raw, features_df)
    run_pillar_3(features_df)
    run_pillar_4(s0.raw)
    run_pillar_5(s0.raw)

    print("\n" + "=" * 75)
    print("ALL 5 EDA PILLARS COMPLETED SUCCESSFULLY!")
    print(f"Figures saved in: {EDA_FIG_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
