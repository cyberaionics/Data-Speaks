from pathlib import Path
import importlib.util

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

    ica_function = load_function(
        Path("pipeline")
        / "01_preprocessing"
        / "candidate_algorithms"
        / "artifact_removal"
        / "ica.py",
        "apply_ica_artifact_removal",
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

    cleaned, provenance = ica_function(
        filtered,
        n_components=15,
        random_state=42,
        eog_ch="FP1-F7",
        eog_threshold=0.4,
        muscle_threshold=0.3,
    )

    before = filtered.get_data()
    after = cleaned.get_data()

    difference = after - before

    print("\n===== ICA CHANGE METRICS =====")

    print(
        f"Mean absolute change: "
        f"{np.mean(np.abs(difference)):.6e} V"
    )

    print(
        f"RMS change: "
        f"{np.sqrt(np.mean(difference ** 2)):.6e} V"
    )

    print(
        f"Maximum absolute change: "
        f"{np.max(np.abs(difference)):.6e} V"
    )

    print(
        f"FIR RMS: "
        f"{np.sqrt(np.mean(before ** 2)):.6e} V"
    )

    print(
        f"ICA-cleaned RMS: "
        f"{np.sqrt(np.mean(after ** 2)):.6e} V"
    )

    print("\n===== COMPONENTS =====")

    print(
        "EOG candidates:",
        [int(i) for i in provenance["eog_indices"]],
    )

    print(
        "Muscle candidates:",
        [int(i) for i in provenance["muscle_indices"]],
    )

    print(
        "Excluded:",
        [int(i) for i in provenance["excluded_components"]],
    )


if __name__ == "__main__":
    main()
