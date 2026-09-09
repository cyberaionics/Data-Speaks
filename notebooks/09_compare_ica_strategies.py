from pathlib import Path

import importlib.util

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


def calculate_metrics(before, after):
    difference = after - before

    before_rms = np.sqrt(np.mean(before ** 2))
    after_rms = np.sqrt(np.mean(after ** 2))

    return {
        "mean_abs_change": np.mean(np.abs(difference)),
        "rms_change": np.sqrt(np.mean(difference ** 2)),
        "max_abs_change": np.max(np.abs(difference)),
        "before_rms": before_rms,
        "after_rms": after_rms,
        "remaining_rms_percent": 100.0 * after_rms / before_rms,
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

    # First 60 seconds for controlled validation.
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

    eog_indices, _ = ica.find_bads_eog(
        filtered,
        ch_name="FP1-F7",
        threshold=0.4,
        verbose=False,
    )

    muscle_indices, _ = ica.find_bads_muscle(
        filtered,
        threshold=0.3,
        verbose=False,
    )

    eog_indices = [int(i) for i in eog_indices]
    muscle_indices = [int(i) for i in muscle_indices]

    strategies = {
        "none": [],
        "strong_eog_only": [0, 1],
        "muscle_only": sorted(set(muscle_indices)),
        "automatic_eog": sorted(set(eog_indices)),
        "automatic_eog_plus_muscle": sorted(
            set(eog_indices + muscle_indices)
        ),
    }

    before = filtered.get_data()

    print("\n===== ICA STRATEGY COMPARISON =====")

    for name, exclusions in strategies.items():

        test_ica = ica.copy()
        test_ica.exclude = exclusions

        cleaned = filtered.copy()

        if exclusions:
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

    print("\n===== DETECTOR OUTPUT =====")
    print("Automatic EOG:", eog_indices)
    print("Automatic muscle:", muscle_indices)


if __name__ == "__main__":
    main()
