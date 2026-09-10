"""
DATA-SPEAKS — Pipeline C Visual Validation Plots
================================================
Generates publication-quality validation figures for Pipeline C on chb01_03:

Figure 1: Time-domain 4-panel comparison (raw, filtered, ASR-cleaned, final)
          focused on the seizure onset transition (2990–3045s).
Figure 2: Multi-panel diagnostic comparison:
          - PSD comparison (before vs after Pipeline C) for key channels
          - ICA component topographies / properties or variance explained
          - Stage-by-stage RMS comparison
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

FIG_DIR = Path(r"C:\Users\Kavya\Data-Speaks\results\figures\pipeline_c")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_time_domain_comparison(
    raw_stage0,
    raw_filtered,
    raw_asr,
    raw_final,
    t_start: float = 2990.0,
    t_end: float = 3045.0,
    channels: list[str] | None = None,
    save_path: Path | None = None,
):
    """
    Plot time-domain comparison of a selected window across stages.
    Highlights seizure region (seizure starts at 2996s, ends at 3036s for chb01_03).
    """
    if channels is None:
        channels = ["FP1-F7", "FZ-CZ", "P8-O2"]
    
    # Filter available channels
    channels = [ch for ch in channels if ch in raw_stage0.ch_names]
    
    sfreq = float(raw_stage0.info["sfreq"])
    start_idx = int(t_start * sfreq)
    end_idx = int(t_end * sfreq)
    times = np.linspace(t_start, t_end, end_idx - start_idx)

    stages = [
        ("Stage 0 (Standardized)", raw_stage0),
        ("Butterworth Bandpass (1-40 Hz)", raw_filtered),
        ("ASR Cleaned (cutoff=20)", raw_asr),
        ("Pipeline C Final (ICA + Z-Score)", raw_final),
    ]

    fig, axes = plt.subplots(len(stages), 1, figsize=(14, 10), sharex=True, sharey=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, (title, raw) in zip(axes, stages):
        data = raw.get_data(picks=channels, start=start_idx, stop=end_idx)
        for i, ch in enumerate(channels):
            ax.plot(times, data[i], label=ch, color=colors[i % len(colors)], linewidth=0.9, alpha=0.85)
        
        # Shade seizure interval (2996 - 3036 s)
        ax.axvspan(2996.0, 3036.0, color="crimson", alpha=0.15, label="Seizure (2996-3036s)" if ax == axes[0] else "")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel("Amplitude" if "Final" in title else "μV / V", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        if ax == axes[0]:
            ax.legend(loc="upper right", ncol=len(channels) + 1, fontsize=9)

    axes[-1].set_xlabel("Time (seconds)", fontsize=10)
    fig.suptitle("Pipeline C: chb01_03 Time-Domain Signal Evolution (Seizure Window)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_file = save_path or (FIG_DIR / "chb01_03_time_domain_comparison.png")
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved time-domain comparison to {out_file}")
    return out_file


def plot_spectral_and_diagnostics(
    raw_stage0,
    raw_final,
    ica,
    timings: dict,
    channels: list[str] | None = None,
    save_path: Path | None = None,
):
    """
    Plot PSD before vs after Pipeline C, along with stage timings and ICA summary.
    """
    if channels is None:
        channels = ["FP1-F7", "FZ-CZ", "P8-O2"]
    channels = [ch for ch in channels if ch in raw_stage0.ch_names]

    sfreq = float(raw_stage0.info["sfreq"])
    data_in = raw_stage0.get_data(picks=channels)
    data_out = raw_final.get_data(picks=channels)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    # Subplot 1: PSD Before vs After
    ax_psd = fig.add_subplot(gs[0, :])
    for i, ch in enumerate(channels):
        freqs_in, psd_in = welch(data_in[i], fs=sfreq, nperseg=int(sfreq * 4))
        freqs_out, psd_out = welch(data_out[i], fs=sfreq, nperseg=int(sfreq * 4))

        # Normalize or show in dB
        ax_psd.semilogy(freqs_in, psd_in, linestyle="--", alpha=0.7, label=f"{ch} (Input Stage 0)")
        ax_psd.semilogy(freqs_out, psd_out, linestyle="-", linewidth=1.2, label=f"{ch} (Pipeline C Final)")

    ax_psd.set_xlim(0, 60)
    ax_psd.axvline(1.0, color="gray", linestyle=":", label="1 Hz / 40 Hz filter bounds")
    ax_psd.axvline(40.0, color="gray", linestyle=":")
    ax_psd.set_title("Power Spectral Density (PSD) Before vs After Pipeline C", fontsize=11, fontweight="bold")
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=10)
    ax_psd.set_ylabel("Power Spectral Density (V²/Hz)", fontsize=10)
    ax_psd.legend(loc="upper right", ncol=2, fontsize=8)
    ax_psd.grid(True, which="both", linestyle="--", alpha=0.5)

    # Subplot 2: Timing breakdown
    ax_time = fig.add_subplot(gs[1, 0])
    stage_keys = ["butterworth_sec", "asr_sec", "ica_sec", "standardization_sec"]
    labels = ["Butterworth", "ASR", "ICA", "Z-Score"]
    times = [timings.get(k, 0.0) for k in stage_keys]
    bars = ax_time.bar(labels, times, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], alpha=0.85)
    ax_time.set_ylabel("Runtime (seconds)", fontsize=10)
    ax_time.set_title(f"Runtime Breakdown (Total: {timings.get('total_pipeline_c_sec', 0.0):.1f}s)", fontsize=11, fontweight="bold")
    ax_time.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, t in zip(bars, times):
        ax_time.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{t:.1f}s", ha="center", va="bottom", fontsize=9)

    # Subplot 3: ICA Components Summary
    ax_ica = fig.add_subplot(gs[1, 1])
    n_fit = ica.n_components_
    n_exc = len(ica.exclude)
    n_kept = n_fit - n_exc
    ax_ica.pie(
        [n_kept, n_exc],
        labels=[f"Retained ({n_kept})", f"Excluded EOG ({n_exc})"],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#2ca02c", "#d62728"],
        explode=(0, 0.1) if n_exc > 0 else (0, 0),
        textprops={"fontsize": 10}
    )
    ax_ica.set_title(f"ICA Component Decomposition ({n_fit} total components)", fontsize=11, fontweight="bold")

    fig.suptitle("Pipeline C: chb01_03 Diagnostic & Performance Summary", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_file = save_path or (FIG_DIR / "chb01_03_spectral_diagnostics.png")
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved spectral and diagnostics plot to {out_file}")
    return out_file
