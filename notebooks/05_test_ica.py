from pathlib import Path
import importlib.util

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pipeline.preprocessing_common.stage0 import load_stage0


DATA_DIR = Path("data/raw/physionet.org/chb02")
SUMMARY_FILE = DATA_DIR / "chb02-summary.txt"


def load_function(file_path, function_name):
    spec = importlib.util.spec_from_file_location(
        function_name,
        file_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, function_name)


def main():
    fir_function = load_function(
        Path("pipeline")
        / "01_preprocessing"
        / "candidate_algorithms"
        / "filtering"
        / "fir_bandpass.py",
        "apply_fir_bandpass",
    )

    ica_function = load_function(
        Path("pipeline")
        / "01_preprocessing"
        / "candidate_algorithms"
        / "artifact_removal"
        / "ica.py",
        "apply_ica_artifact_removal",
    )

    print("Loading Stage 0...")

    raw, _ = load_stage0(
        DATA_DIR / "chb02_01.edf",
        SUMMARY_FILE,
    )

    print("Stage 0 complete.")

    # Only use the first 60 seconds for this test.
    raw.crop(tmin=0.0, tmax=60.0)

    print("\nApplying FIR 1-40 Hz...")

    filtered = fir_function(
        raw,
        l_freq=1.0,
        h_freq=40.0,
    )

    print("FIR complete.")

    print("\nApplying ICA...")

    cleaned, provenance = ica_function(
        filtered,
        n_components=15,
        random_state=42,
        eog_ch="FP1-F7",
        eog_threshold=0.4,
        muscle_threshold=0.3,
    )

    print("ICA complete.")

    print("\n===== ICA RESULTS =====")
    print(
        f"Estimated EEG rank: "
        f"{provenance['estimated_eeg_rank']}"
    )
    print(
        f"Requested components: "
        f"{provenance['requested_n_components']}"
    )
    print(
        f"Actual components: "
        f"{provenance['actual_n_components']}"
    )
    print(
        f"EOG components: "
        f"{provenance['eog_indices']}"
    )
    print(
        f"Muscle components: "
        f"{provenance['muscle_indices']}"
    )
    print(
        f"Excluded components: "
        f"{provenance['excluded_components']}"
    )

    # Compare one channel before and after ICA.
    channel = 0

    before = filtered.get_data(
        picks=[channel]
    )[0]

    after = cleaned.get_data(
        picks=[channel]
    )[0]

    times = filtered.times

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
    )

    axes[0].plot(times, before)
    axes[0].set_title(
        f"FIR-filtered EEG — {filtered.ch_names[channel]}"
    )
    axes[0].set_ylabel("Amplitude (V)")

    axes[1].plot(times, after)
    axes[1].set_title(
        f"After ICA — {cleaned.ch_names[channel]}"
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (V)")

    plt.tight_layout()

    output_path = Path("results") / "fir_ica_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()
