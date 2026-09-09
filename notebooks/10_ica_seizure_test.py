from pathlib import Path
import importlib.util

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np

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

    raw, metadata = load_stage0(
        DATA_DIR / "chb02_16.edf",
        SUMMARY_FILE,
    )

    print("\n===== RECORDING =====")
    print(f"Recording: {metadata.recording_id}")
    print(f"Duration: {metadata.duration_sec:.2f} sec")
    print(f"Sampling rate: {metadata.sampling_rate_hz} Hz")
    print(f"Seizure intervals: {metadata.seizure_intervals}")

    # Analyze the region containing the seizure.
    raw.crop(tmin=100.0, tmax=240.0)

    filtered = fir_function(
        raw,
        l_freq=1.0,
        h_freq=40.0,
    )

    print("\nFitting ICA...")

    rank = mne.compute_rank(
        filtered,
        rank="info",
        verbose=False,
    )

    eeg_rank = int(rank["eeg"])

    n_components = min(
        15,
        eeg_rank,
        len(filtered.ch_names),
    )

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="fastica",
        random_state=42,
        max_iter="auto",
    )

    ica.fit(
        filtered,
        picks="eeg",
        verbose=False,
    )

    print("ICA fitting complete.")

    eog_indices, eog_scores = ica.find_bads_eog(
        filtered,
        ch_name="FP1-F7",
        threshold=0.4,
        verbose=False,
    )

    muscle_indices, muscle_scores = ica.find_bads_muscle(
        filtered,
        threshold=0.3,
        verbose=False,
    )

    eog_indices = [int(i) for i in eog_indices]
    muscle_indices = [int(i) for i in muscle_indices]

    eog_scores = np.asarray(eog_scores)
    muscle_scores = np.asarray(muscle_scores)

    print("\n===== ICA COMPONENT SCORES =====")

    print(
        f"{'Comp':>4} "
        f"{'EOG':>10} "
        f"{'Muscle':>10} "
        f"{'EOG?':>6} "
        f"{'Muscle?':>8}"
    )

    for i in range(n_components):
        print(
            f"{i:>4} "
            f"{eog_scores[i]:>10.4f} "
            f"{muscle_scores[i]:>10.4f} "
            f"{'YES' if i in eog_indices else 'NO':>6} "
            f"{'YES' if i in muscle_indices else 'NO':>8}"
        )

    sources = ica.get_sources(filtered).get_data()

    print("\n===== COMPONENT SEIZURE-WINDOW ENERGY =====")

    # Original seizure interval is 130–212 s.
    # Our cropped recording now starts at 100 s,
    # so seizure samples correspond to 30–112 s.
    seizure_start = 30.0
    seizure_end = 112.0

    start_sample = int(seizure_start * filtered.info["sfreq"])
    end_sample = int(seizure_end * filtered.info["sfreq"])

    seizure_sources = sources[:, start_sample:end_sample]

    for i in range(n_components):
        energy = np.sqrt(
            np.mean(seizure_sources[i] ** 2)
        )

        print(
            f"Component {i:02d}: "
            f"RMS={energy:.6e}"
        )

    # Plot ICA components over the seizure-containing period.
    times = filtered.times

    fig, axes = plt.subplots(
        n_components,
        1,
        figsize=(14, 2 * n_components),
        sharex=True,
    )

    if n_components == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(times, sources[i])

        if i in eog_indices:
            label = "EOG candidate"

        elif i in muscle_indices:
            label = "Muscle candidate"

        else:
            label = "Not selected"

        ax.set_ylabel(f"IC {i}")
        ax.set_title(label)

    axes[-1].set_xlabel("Time (s)")

    fig.suptitle(
        "ICA components around seizure — chb02_16"
    )

    plt.tight_layout()

    output_path = (
        Path("results")
        / "ica_seizure_components.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
