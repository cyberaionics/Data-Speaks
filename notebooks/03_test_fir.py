from pathlib import Path
import importlib.util

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

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

    edf_file = DATA_DIR / "chb02_01.edf"

    print("Loading Stage 0...")

    raw, metadata = load_stage0(
        edf_file,
        SUMMARY_FILE,
    )

    print("Stage 0 complete.")

    # Keep only the first 60 seconds for this FIR test.
    raw.crop(tmin=0.0, tmax=60.0)

    print(f"Samples used: {raw.n_times}")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")

    print("\nApplying FIR 1-40 Hz...")

    filtered = apply_fir_bandpass(
        raw,
        l_freq=1.0,
        h_freq=40.0,
    )

    print("FIR filtering complete.")

    channel_index = 0

    raw_data = raw.get_data(
        picks=[channel_index]
    )[0]

    filtered_data = filtered.get_data(
        picks=[channel_index]
    )[0]

    times = raw.times

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
    )

    axes[0].plot(times, raw_data)
    axes[0].set_title(
        f"Raw EEG — {raw.ch_names[channel_index]}"
    )
    axes[0].set_ylabel("Amplitude (V)")

    axes[1].plot(times, filtered_data)
    axes[1].set_title(
        f"FIR 1–40 Hz — {filtered.ch_names[channel_index]}"
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (V)")

    plt.tight_layout()

    output_path = Path("results") / "fir_raw_vs_filtered.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
