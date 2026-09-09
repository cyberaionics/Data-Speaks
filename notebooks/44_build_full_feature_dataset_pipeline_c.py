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


features_module = load_module("feature_extractor", "pipeline/03_feature_extraction/feature_extractor.py")
windowing = load_module("windowing", "pipeline/02_segmentation/candidate_algorithms/windowing.py")
stage0 = load_module("stage0", "pipeline/preprocessing_common/stage0.py")

RAW_DIR = Path("data/raw/physionet.org/chb02")
SUMMARY_PATH = RAW_DIR / "chb02-summary.txt"
INPUT_DIR = Path("data/processed/pipeline_C")
OUTPUT_PATH = Path("results/benchmarks/features_pipeline_C_chb02_full.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

processed_files = sorted(INPUT_DIR.glob("*_raw.fif"))
assert len(processed_files) == 36, f"Expected 36 processed recordings, found {len(processed_files)}"

rows = []
feature_names = None
for recording_number, input_path in enumerate(processed_files, start=1):
    recording_id = input_path.stem.removesuffix("_raw")
    print(f"Processing {recording_number}/{len(processed_files)}: {recording_id}")
    raw = mne.io.read_raw_fif(input_path, preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"])
    n_channels = len(raw.ch_names)
    assert sfreq == 256.0
    assert n_channels == 23
    _, metadata = stage0.load_stage0(RAW_DIR / f"{recording_id}.edf", summary_file=SUMMARY_PATH)
    windows = list(windowing.generate_windows(raw.n_times / sfreq, sfreq, metadata.seizure_intervals))

    for window_index, window_info in enumerate(windows):
        if window_info["label"] == -1:
            continue
        start_sample = int(round(window_info["start_sec"] * sfreq))
        window_data = raw.get_data(start=start_sample, stop=start_sample + 1024)
        if window_data.shape != (n_channels, 1024):
            raise RuntimeError(f"Malformed window for {recording_id}: {window_data.shape}")
        features = features_module.extract_features(window_data, sfreq)
        if len(features) != 598 or not all(np.isfinite(value) for value in features.values()):
            raise RuntimeError(f"Invalid features for {recording_id}, window {window_index}")
        if feature_names is None:
            feature_names = list(features)
        elif list(features) != feature_names:
            raise RuntimeError("Feature column order changed between windows")
        rows.append(
            {
                "recording_id": recording_id,
                "window_index": window_index,
                "start_sec": window_info["start_sec"],
                "end_sec": window_info["end_sec"],
                "label": window_info["label"],
                **features,
            }
        )

if feature_names is None:
    raise RuntimeError("No features were extracted.")
fieldnames = ["recording_id", "window_index", "start_sec", "end_sec", "label", *feature_names]
with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Pipeline C feature dataset complete")
print("Rows:", len(rows))
print("Features per row:", len(feature_names))
print("Ictal:", sum(row["label"] == 1 for row in rows))
print("Interictal:", sum(row["label"] == 0 for row in rows))
