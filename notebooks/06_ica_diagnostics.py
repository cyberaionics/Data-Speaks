from pathlib import Path
import importlib.util

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import mne

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

    raw, _ = load_stage0(
        DATA_DIR / "chb02_01.edf",
        SUMMARY_FILE,
    )

    raw.crop(tmin=0.0, tmax=60.0)

    filtered = fir_function(
        raw,
        l_freq=1.0,
        h_freq=40.0,
    )

    rank = mne.compute_rank(
        filtered,
        rank="info",
        verbose=False,
    )

    eeg_rank = rank["eeg"]

    n_components = min(
        15,
        int(eeg_rank),
        len(filtered.ch_names),
    )

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="fastica",
        random_state=42,
        max_iter="auto",
    )

    print("Fitting ICA...")
    ica.fit(
        filtered,
        picks="eeg",
        verbose=False,
    )

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

    eog_scores = np.asarray(eog_scores)
    muscle_scores = np.asarray(muscle_scores)

    print("\n===== COMPONENT SUMMARY =====")
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

    print("\nEOG selected:")
    print([int(i) for i in eog_indices])

    print("\nMuscle selected:")
    print([int(i) for i in muscle_indices])

    # Get ICA source signals for the 60-second test.
    sources = ica.get_sources(filtered).get_data()

    print("\n===== COMPONENT STANDARD DEVIATIONS =====")

    for i in range(n_components):
        print(
            f"Component {i:02d}: "
            f"std={np.std(sources[i]):.6e}"
        )

    # Plot the first 15 component time series.
    fig, axes = plt.subplots(
        n_components,
        1,
        figsize=(14, 2 * n_components),
        sharex=True,
    )

    if n_components == 1:
        axes = [axes]

    times = filtered.times

    for i, ax in enumerate(axes):
        ax.plot(times, sources[i])
        ax.set_ylabel(f"IC {i}")

    axes[-1].set_xlabel("Time (s)")

    fig.suptitle(
        "ICA component time series — chb02_01 (first 60 s)"
    )

    plt.tight_layout()

    output = Path("results") / "ica_component_timeseries.png"

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
