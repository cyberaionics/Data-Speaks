from pathlib import Path
import importlib.util

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne

from pipeline.preprocessing_common.stage0 import load_stage0


DATA_DIR = Path("data/raw/physionet.org/chb02")
SUMMARY_FILE = DATA_DIR / "chb02-summary.txt"


def load_fir_function():
    fir_path = (
        Path("pipeline")
        / "01_preprocessing"
        / "candidate_algorithms"
        / "filtering"
        / "fir_bandpass.py"
    )

    spec = importlib.util.spec_from_file_location(
        "fir_bandpass",
        fir_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {fir_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.apply_fir_bandpass


def main():
    apply_fir_bandpass = load_fir_function()

    raw, _ = load_stage0(
        DATA_DIR / "chb02_01.edf",
        SUMMARY_FILE,
    )

    # Use only the first 60 seconds for this test.
    raw.crop(tmin=0.0, tmax=60.0)

    filtered = apply_fir_bandpass(
        raw,
        l_freq=1.0,
        h_freq=40.0,
    )

    channel = 0

    print("Computing PSD...")

    raw_psd = raw.compute_psd(
        method="welch",
        fmin=0.1,
        fmax=100.0,
        picks=[channel],
        verbose=False,
    )

    filtered_psd = filtered.compute_psd(
        method="welch",
        fmin=0.1,
        fmax=100.0,
        picks=[channel],
        verbose=False,
    )

    raw_power = raw_psd.get_data()[0]
    filtered_power = filtered_psd.get_data()[0]

    frequencies = raw_psd.freqs

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.semilogy(
        frequencies,
        raw_power,
        label="Raw",
    )

    ax.semilogy(
        frequencies,
        filtered_power,
        label="FIR 1–40 Hz",
    )

    ax.axvline(
        1.0,
        linestyle="--",
        label="1 Hz cutoff",
    )

    ax.axvline(
        40.0,
        linestyle="--",
        label="40 Hz cutoff",
    )

    ax.set_xlim(0.1, 100)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title(
        f"Power Spectral Density — {raw.ch_names[channel]}"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    output_path = Path("results") / "fir_psd_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"PSD figure saved to: {output_path}")

    # Simple quantitative checks.
    low_band = frequencies < 1.0
    eeg_band = (frequencies >= 1.0) & (frequencies <= 40.0)
    high_band = frequencies > 40.0

    raw_low = raw_power[low_band].mean()
    filtered_low = filtered_power[low_band].mean()

    raw_eeg = raw_power[eeg_band].mean()
    filtered_eeg = filtered_power[eeg_band].mean()

    raw_high = raw_power[high_band].mean()
    filtered_high = filtered_power[high_band].mean()

    print("\nMean PSD:")
    print(f"Below 1 Hz:")
    print(f"  Raw:      {raw_low:.6e}")
    print(f"  Filtered: {filtered_low:.6e}")

    print(f"\n1–40 Hz:")
    print(f"  Raw:      {raw_eeg:.6e}")
    print(f"  Filtered: {filtered_eeg:.6e}")

    print(f"\nAbove 40 Hz:")
    print(f"  Raw:      {raw_high:.6e}")
    print(f"  Filtered: {filtered_high:.6e}")


if __name__ == "__main__":
    main()
