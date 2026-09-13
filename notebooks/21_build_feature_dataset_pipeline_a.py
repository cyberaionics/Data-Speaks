import csv
import importlib.util
from pathlib import Path

import mne
import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

recording_id = "chb02_16"

processed_path = (
    "data/processed/pipeline_A/chb02_16_raw.fif"
)

output_dir = Path("results/benchmarks")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = (
    output_dir
    / "features_pipeline_A_chb02_16.csv"
)

SEIZURE_INTERVALS = [(130.0, 212.0)]


# ---------------------------------------------------------
# Load processed EEG
# ---------------------------------------------------------

raw = mne.io.read_raw_fif(
    processed_path,
    preload=True,
    verbose=False,
)

sfreq = float(raw.info["sfreq"])
n_channels = len(raw.ch_names)

print("Recording:", recording_id)
print("Sampling rate:", sfreq)
print("Channels:", n_channels)
print("Samples:", raw.n_times)


# ---------------------------------------------------------
# Generate windows
# ---------------------------------------------------------

windows = list(
    windowing_module.generate_windows(
        duration_sec=raw.n_times / sfreq,
        sampling_rate=sfreq,
        seizure_intervals=SEIZURE_INTERVALS,
    )
)

print("\nTotal windows:", len(windows))


# ---------------------------------------------------------
# Build feature rows
# ---------------------------------------------------------

rows = []

feature_names = None

for index, window_info in enumerate(windows):

    label = window_info["label"]

    # Skip ambiguous windows
    if label == -1:
        continue

    start_sec = window_info["start_sec"]
    end_sec = window_info["end_sec"]

    start_sample = int(round(start_sec * sfreq))
    stop_sample = start_sample + int(round(4.0 * sfreq))

    window = raw.get_data(
        start=start_sample,
        stop=stop_sample,
    )

    # Safety check
    if window.shape != (n_channels, 1024):
        continue

    features = feature_module.extract_features(
        window,
        sfreq,
    )

    if feature_names is None:
        feature_names = list(features.keys())

    row = {
        "recording_id": recording_id,
        "window_index": index,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "label": label,
    }

    row.update(features)

    rows.append(row)

    if (len(rows) % 50) == 0:
        print(
            f"Processed {len(rows)} valid windows..."
        )


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
    output_path,
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

labels = [row["label"] for row in rows]

ictal_count = labels.count(1)
interictal_count = labels.count(0)

print("\nDataset created:")
print("Output:", output_path)
print("Rows:", len(rows))
print("Features per row:", len(feature_names))
print("Ictal windows:", ictal_count)
print("Interictal windows:", interictal_count)

assert len(rows) > 0
assert len(feature_names) == 598
assert ictal_count > 0
assert interictal_count > 0

print("\nPipeline A feature dataset test passed.")
