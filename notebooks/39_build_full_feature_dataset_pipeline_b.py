import csv
import importlib.util
import sys
from pathlib import Path

import mne
import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


# =========================================================
# Load common modules
# =========================================================

feature_module = load_module(
    "feature_extractor",
    "pipeline/03_feature_extraction/feature_extractor.py",
)

windowing_module = load_module(
    "windowing",
    "pipeline/02_segmentation/candidate_algorithms/windowing.py",
)

stage0 = load_module(
    "stage0",
    "pipeline/preprocessing_common/stage0.py",
)


# =========================================================
# Paths
# =========================================================

RAW_DIR = Path(
    "data/raw/physionet.org/chb02"
)

SUMMARY_PATH = (
    RAW_DIR / "chb02-summary.txt"
)

INPUT_DIR = Path(
    "data/processed/pipeline_B"
)

OUTPUT_DIR = Path(
    "results/benchmarks"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_PATH = (
    OUTPUT_DIR
    / "features_pipeline_B_chb02_full.csv"
)


# =========================================================
# Discover processed Pipeline B recordings
# =========================================================

processed_files = sorted(
    INPUT_DIR.glob("*_raw.fif")
)

print(
    "Pipeline B processed recordings found:",
    len(processed_files),
)

assert len(processed_files) == 36, (
    f"Expected 36 processed recordings, "
    f"found {len(processed_files)}"
)


# =========================================================
# Dataset storage
# =========================================================

rows = []
feature_names = None


# =========================================================
# Process every recording
# =========================================================

for recording_number, input_path in enumerate(
    processed_files,
    start=1,
):

    recording_id = input_path.stem.replace(
        "_raw",
        "",
    )

    print("\n" + "=" * 70)
    print(
        f"Processing {recording_number}/"
        f"{len(processed_files)}: "
        f"{recording_id}"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # Read processed EEG
    # -----------------------------------------------------

    raw = mne.io.read_raw_fif(
        input_path,
        preload=False,
        verbose=False,
    )

    sfreq = float(
        raw.info["sfreq"]
    )

    n_channels = len(
        raw.ch_names
    )

    duration_sec = (
        raw.n_times / sfreq
    )

    print(
        "Sampling rate:",
        sfreq,
    )

    print(
        "Channels:",
        n_channels,
    )

    print(
        "Duration:",
        duration_sec,
    )

    # -----------------------------------------------------
    # Obtain seizure intervals from Stage 0 parser
    # -----------------------------------------------------

    edf_path = (
        RAW_DIR
        / f"{recording_id}.edf"
    )

    if not edf_path.exists():
        raise FileNotFoundError(
            f"Original EDF not found for "
            f"{recording_id}: {edf_path}"
        )

    _, metadata = stage0.load_stage0(
        edf_path,
        summary_file=SUMMARY_PATH,
    )

    seizure_intervals = (
        metadata.seizure_intervals
    )

    print(
        "Seizure intervals:",
        seizure_intervals,
    )

    # -----------------------------------------------------
    # Validate processed recording
    # -----------------------------------------------------

    assert sfreq == 256.0
    assert n_channels == 23

    # -----------------------------------------------------
    # Generate windows
    # -----------------------------------------------------

    windows = list(
        windowing_module.generate_windows(
            duration_sec=duration_sec,
            sampling_rate=sfreq,
            seizure_intervals=seizure_intervals,
        )
    )

    print(
        "Total windows:",
        len(windows),
    )

    valid_count = 0
    ambiguous_count = 0

    # -----------------------------------------------------
    # Extract features
    # -----------------------------------------------------

    for window_index, window_info in enumerate(
        windows
    ):

        label = window_info["label"]

        # Exclude ambiguous windows.
        if label == -1:
            ambiguous_count += 1
            continue

        start_sec = (
            window_info["start_sec"]
        )

        end_sec = (
            window_info["end_sec"]
        )

        start_sample = int(
            round(start_sec * sfreq)
        )

        stop_sample = (
            start_sample + 1024
        )

        # Read only this 4-second window.
        window_data = raw.get_data(
            start=start_sample,
            stop=stop_sample,
        )

        if window_data.shape != (
            n_channels,
            1024,
        ):
            print(
                "Skipping malformed window:",
                window_index,
                window_data.shape,
            )
            continue

        # -------------------------------------------------
        # Extract the common 598 features.
        # -------------------------------------------------

        features = (
            feature_module.extract_features(
                window_data,
                sfreq,
            )
        )

        if feature_names is None:
            feature_names = list(
                features.keys()
            )

        if len(features) != 598:
            raise RuntimeError(
                f"Expected 598 features, "
                f"got {len(features)} "
                f"for {recording_id}, "
                f"window {window_index}"
            )

        if not all(
            np.isfinite(value)
            for value in features.values()
        ):
            raise RuntimeError(
                f"Non-finite feature detected "
                f"for {recording_id}, "
                f"window {window_index}"
            )

        # -------------------------------------------------
        # Build row.
        # -------------------------------------------------

        row = {
            "recording_id": recording_id,
            "window_index": window_index,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "label": label,
        }

        row.update(features)

        rows.append(row)

        valid_count += 1

        if valid_count % 100 == 0:
            print(
                f"Valid windows processed: "
                f"{valid_count}"
            )

    print(
        "Valid windows:",
        valid_count,
    )

    print(
        "Ambiguous windows:",
        ambiguous_count,
    )


# =========================================================
# Build CSV columns
# =========================================================

if feature_names is None:
    raise RuntimeError(
        "No features were extracted."
    )


fieldnames = [
    "recording_id",
    "window_index",
    "start_sec",
    "end_sec",
    "label",
] + feature_names


# =========================================================
# Save dataset
# =========================================================

with open(
    OUTPUT_PATH,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


# =========================================================
# Final statistics
# =========================================================

ictal = sum(
    row["label"] == 1
    for row in rows
)

interictal = sum(
    row["label"] == 0
    for row in rows
)

ambiguous = sum(
    row["label"] == -1
    for row in rows
)


print("\n" + "=" * 70)
print("Pipeline B full CHB02 feature dataset complete")
print("=" * 70)

print(
    "Recordings processed:",
    len(processed_files),
)

print(
    "Rows:",
    len(rows),
)

print(
    "Features per row:",
    len(feature_names),
)

print(
    "Ictal:",
    ictal,
)

print(
    "Interictal:",
    interictal,
)

print(
    "Ambiguous excluded:",
    ambiguous,
)

print(
    "Output:",
    OUTPUT_PATH,
)


# =========================================================
# Validation
# =========================================================

assert len(processed_files) == 36
assert len(feature_names) == 598
assert len(rows) > 0
assert ictal > 0
assert interictal > 0
assert ambiguous == 0

print(
    "\nPipeline B full feature dataset "
    "creation passed."
)
