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


# ---------------------------------------------------------
# Load common modules
# ---------------------------------------------------------

feature_module = load_module(
    "feature_extractor",
    "pipeline/03_feature_extraction/feature_extractor.py",
)

windowing_module = load_module(
    "windowing",
    "pipeline/02_segmentation/candidate_algorithms/windowing.py",
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_DIR = Path("data/processed/pipeline_B")
OUTPUT_DIR = Path("results/benchmarks")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RECORDINGS = {
    "chb02_01": {
        "file": "chb02_01_raw.fif",
        "seizures": [],
    },
    "chb02_16": {
        "file": "chb02_16_raw.fif",
        "seizures": [(130.0, 212.0)],
    },
}


OUTPUT_PATH = (
    OUTPUT_DIR
    / "features_pipeline_B_chb02.csv"
)


# ---------------------------------------------------------
# Build dataset
# ---------------------------------------------------------

rows = []
feature_names = None


for recording_id, info in RECORDINGS.items():

    input_path = INPUT_DIR / info["file"]

    print("\n" + "=" * 60)
    print("Processing:", recording_id)
    print("=" * 60)

    raw = mne.io.read_raw_fif(
        input_path,
        preload=False,
        verbose=False,
    )

    sfreq = float(raw.info["sfreq"])
    n_channels = len(raw.ch_names)

    duration_sec = raw.n_times / sfreq

    print("Sampling rate:", sfreq)
    print("Channels:", n_channels)
    print("Duration:", duration_sec)

    windows = list(
        windowing_module.generate_windows(
            duration_sec=duration_sec,
            sampling_rate=sfreq,
            seizure_intervals=info["seizures"],
        )
    )

    print("Total windows:", len(windows))

    valid_count = 0

    for window_index, window_info in enumerate(windows):

        label = window_info["label"]

        # Exclude ambiguous windows.
        if label == -1:
            continue

        start_sec = window_info["start_sec"]

        start_sample = int(
            round(start_sec * sfreq)
        )

        stop_sample = (
            start_sample
            + int(round(4.0 * sfreq))
        )

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

        features = feature_module.extract_features(
            window_data,
            sfreq,
        )

        if feature_names is None:
            feature_names = list(
                features.keys()
            )

        if len(features) != 598:
            raise RuntimeError(
                f"Expected 598 features, got {len(features)}"
            )

        if not all(
            np.isfinite(value)
            for value in features.values()
        ):
            raise RuntimeError(
                f"Non-finite feature detected "
                f"in {recording_id}, window {window_index}"
            )

        row = {
            "recording_id": recording_id,
            "window_index": window_index,
            "start_sec": window_info["start_sec"],
            "end_sec": window_info["end_sec"],
            "label": label,
        }

        row.update(features)

        rows.append(row)

        valid_count += 1

        if valid_count % 100 == 0:
            print(
                f"Processed {valid_count} valid windows..."
            )

    print("Valid windows:", valid_count)


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

fieldnames = [
    "recording_id",
    "window_index",
    "start_sec",
    "end_sec",
    "label",
] + feature_names


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


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

ictal = sum(
    row["label"] == 1
    for row in rows
)

interictal = sum(
    row["label"] == 0
    for row in rows
)


print("\n" + "=" * 60)
print("Pipeline B feature dataset complete")
print("=" * 60)

print("Rows:", len(rows))
print("Features per row:", len(feature_names))
print("Ictal:", ictal)
print("Interictal:", interictal)
print("Output:", OUTPUT_PATH)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

assert len(feature_names) == 598
assert len(rows) == 2215
assert ictal == 42
assert interictal == 2173

print("\nPipeline B feature dataset test passed.")
