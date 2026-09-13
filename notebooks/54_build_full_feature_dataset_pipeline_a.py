import csv
import importlib.util
import os
import sys
from pathlib import Path

import mne
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/physionet.org/chb02"
SUMMARY_PATH = RAW_DIR / "chb02-summary.txt"
INPUT_DIR = ROOT / "data/processed/pipeline_A"
OUTPUT_PATH = ROOT / "results/benchmarks/features_pipeline_A_chb02_full.csv"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


features_module = load_module(
    "feature_extractor_a_full", ROOT / "pipeline/03_feature_extraction/feature_extractor.py"
)
windowing = load_module(
    "windowing_a_full", ROOT / "pipeline/02_segmentation/candidate_algorithms/windowing.py"
)
stage0 = load_module("stage0_a_full", ROOT / "pipeline/preprocessing_common/stage0.py")


def main():
    processed_files = sorted(INPUT_DIR.glob("*_raw.fif"))
    expected_files = sorted(RAW_DIR.glob("*.edf"))
    if len(processed_files) != len(expected_files):
        raise RuntimeError(
            f"Expected one A output per EDF: {len(expected_files)} EDFs, "
            f"found {len(processed_files)} FIFs"
        )

    feature_names = None
    row_count = 0
    ictal_count = 0
    interictal_count = 0
    temporary_path = OUTPUT_PATH.with_suffix(".tmp.csv")
    writer = None
    handle = None
    for input_path in processed_files:
        recording_id = input_path.stem.removesuffix("_raw")
        raw = mne.io.read_raw_fif(input_path, preload=False, verbose=False)
        sfreq = float(raw.info["sfreq"])
        if sfreq != 256.0 or len(raw.ch_names) != 23:
            raise RuntimeError(f"Invalid A output contract for {recording_id}")
        if not np.isfinite(raw.get_data()).all():
            raise RuntimeError(f"Non-finite A output for {recording_id}")

        _, metadata = stage0.load_stage0(
            RAW_DIR / f"{recording_id}.edf", summary_file=SUMMARY_PATH
        )
        windows = windowing.generate_windows(
            raw.n_times / sfreq, sfreq, metadata.seizure_intervals
        )
        for window_info in windows:
            if window_info["label"] == -1:
                continue
            start_sample = window_info["start_sample"]
            window_data = raw.get_data(start=start_sample, stop=start_sample + 1024)
            if window_data.shape != (23, 1024):
                raise RuntimeError(f"Malformed A window for {recording_id}: {window_data.shape}")
            features = features_module.extract_features(window_data, sfreq)
            if len(features) != 598 or not all(np.isfinite(value) for value in features.values()):
                raise RuntimeError(f"Invalid A features for {recording_id}, window {window_info['window_id']}")
            names = list(features)
            if feature_names is None:
                feature_names = names
                temporary_path.parent.mkdir(parents=True, exist_ok=True)
                handle = temporary_path.open("w", newline="")
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["recording_id", "window_index", "start_sec", "end_sec", "label"] + feature_names,
                )
                writer.writeheader()
            elif names != feature_names:
                raise RuntimeError("Pipeline A feature column order changed")
            writer.writerow(
                {
                    "recording_id": recording_id,
                    "window_index": window_info["window_id"],
                    "start_sec": window_info["start_sec"],
                    "end_sec": window_info["end_sec"],
                    "label": window_info["label"],
                    **features,
                }
            )
            row_count += 1
            if window_info["label"] == 1:
                ictal_count += 1
            else:
                interictal_count += 1

    if feature_names is None:
        raise RuntimeError("No Pipeline A features were extracted")
    handle.close()
    os.replace(temporary_path, OUTPUT_PATH)
    print("Pipeline A full feature dataset complete")
    print("Rows:", row_count)
    print("Features per row:", len(feature_names))
    print("Ictal:", ictal_count)
    print("Interictal:", interictal_count)


if __name__ == "__main__":
    main()