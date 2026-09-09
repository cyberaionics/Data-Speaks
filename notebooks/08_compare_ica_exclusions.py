from pathlib import Path
import importlib.util

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


def calculate_metrics(before, after):
    difference = after - before

    return {
        "mean_abs_change": float(np.mean(np.abs(difference))),
        "rms_change": float(np.sqrt(np.mean(difference ** 2))),
        "max_abs_change": float(np.max(np.abs(difference))),
        "before_rms": float(np.sqrt(np.mean(before ** 2))),
        "after_rms": float(np.sqrt(np.mean(after ** 2))),
        "remaining_rms_percent": float(
            100 * np.sqrt(np.mean(after ** 2))
            / np.sqrt(np.mean(before ** 2))
        ),
    }


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

    # Fit ICA exactly as Pipeline A specifies.
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

    eog_indices = [int(i) for i in eog_indices]
    muscle_indices = [int(i) for i in muscle_indices]

    exclusion_sets = {
        "none": [],
        "eog_only": sorted(set(eog_indices)),
        "muscle_only": sorted(set(muscle_indices)),
        "eog_plus_muscle": sorted(
            set(eog_indices + muscle_indices)
        ),
    }

    before = filtered.get_data()

    print("\n===== ICA EXCLUSION COMPARISON =====")

    for name, exclusions in exclusion_sets.items():

        test_ica = mne.preprocessing.ICA(
            n_components=n_components,
            method="fastica",
            random_state=42,
            max_iter="auto",
        )

        # Reuse the already-fitted ICA solution rather than
        # fitting ICA four separate times.
        test_ica = ica.copy()
        test_ica.exclude = exclusions

        cleaned = filtered.copy()

        test_ica.apply(
            cleaned,
            verbose=False,
        )

        after = cleaned.get_data()

        metrics = calculate_metrics(
            before,
            after,
        )

        print(f"\n{name}")
        print(f"Excluded components: {exclusions}")
        print(
            f"Mean absolute change: "
            f"{metrics['mean_abs_change']:.6e} V"
        )
        print(
            f"RMS change: "
            f"{metrics['rms_change']:.6e} V"
        )
        print(
            f"Maximum absolute change: "
            f"{metrics['max_abs_change']:.6e} V"
        )
        print(
            f"Before RMS: "
            f"{metrics['before_rms']:.6e} V"
        )
        print(
            f"After RMS: "
            f"{metrics['after_rms']:.6e} V"
        )
        print(
            f"Remaining RMS: "
            f"{metrics['remaining_rms_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()
