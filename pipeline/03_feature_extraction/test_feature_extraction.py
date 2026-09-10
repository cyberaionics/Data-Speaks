"""
Tests and sanity checks for Stage 03 Feature Extraction.
Verifies:
- Feature extraction on synthetic windows with various signal dynamics (normal, flat, sinusoidal, noisy)
- Preservation of metadata columns
- Correctness of channel prefixing
- Absence of NaNs and Infs
- Zero-variance and flat window handling
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add directories
STAGE03_DIR = Path(__file__).resolve().parent
REPO_ROOT = STAGE03_DIR.parents[1]
SEG_DIR = REPO_ROOT / "pipeline" / "02_segmentation"
sys.path.insert(0, str(STAGE03_DIR))
sys.path.insert(0, str(SEG_DIR))

import numpy as np
import pandas as pd
from segmentation import SegmentationResult
from feature_extraction import extract_features, BANDS


def test_synthetic_feature_extraction():
    print("--- Running test_synthetic_feature_extraction ---")
    n_windows = 5
    n_channels = 3
    n_samples = 1024
    ch_names = ["FP1-F7", "FZ-CZ", "P8-O2"]

    # Construct synthetic data:
    # win 0: standard normal noise
    # win 1: pure sine wave
    # win 2: completely flat (all zeros)
    # win 3: constant offset
    # win 4: linear trend
    windows = np.zeros((n_windows, n_channels, n_samples), dtype=np.float64)
    t = np.linspace(0, 4.0, n_samples, endpoint=False)

    windows[0] = np.random.RandomState(42).randn(n_channels, n_samples)
    for c in range(n_channels):
        windows[1, c] = np.sin(2 * np.pi * (10 + c * 2) * t)
    windows[2] = 0.0  # flat
    windows[3] = 5.0  # constant offset
    for c in range(n_channels):
        windows[4, c] = np.linspace(0, 1.0 + c, n_samples)

    metadata = [
        {"patient_id": "chb01", "recording_id": "test_rec", "window_id": i, "start_sec": i * 2.0, "end_sec": i * 2.0 + 4.0, "duration_sec": 4.0, "label": 1 if i == 1 else 0}
        for i in range(n_windows)
    ]

    seg_res = SegmentationResult(
        windows=windows,
        metadata=metadata,
        ch_names=ch_names,
    )

    df = extract_features(seg_res, ch_names, sfreq=256.0)

    # 1. Check row count
    assert len(df) == n_windows, f"Expected {n_windows} rows, got {len(df)}"

    # 2. Check metadata columns preserved
    expected_meta = ["patient_id", "recording_id", "window_id", "start_sec", "end_sec", "duration_sec", "label"]
    for col in expected_meta:
        assert col in df.columns, f"Missing metadata column {col}"
        assert list(df[col]) == [m[col] for m in metadata], f"Mismatch in metadata column {col}"

    # 3. Check channel feature naming and count
    features_per_channel = [
        "mean", "std", "var", "rms", "skew", "kurtosis", "min", "max", "ptp",
        "line_length", "zero_crossing_rate", "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
        "delta_power", "delta_relpower",
        "theta_power", "theta_relpower",
        "alpha_power", "alpha_relpower",
        "beta_power", "beta_relpower",
        "gamma_power", "gamma_relpower",
        "total_power", "spectral_edge_95"
    ]
    assert len(features_per_channel) == 26, f"Expected 26 features per channel, got {len(features_per_channel)}"

    for ch in ch_names:
        for feat in features_per_channel:
            col_name = f"{ch}_{feat}"
            assert col_name in df.columns, f"Missing expected feature column {col_name}"

    total_feature_cols = n_channels * len(features_per_channel)
    expected_total_cols = len(expected_meta) + total_feature_cols
    assert df.shape[1] == expected_total_cols, f"Expected {expected_total_cols} columns, got {df.shape[1]}"

    # 4. Check for NaNs and Infs
    num_cols = [c for c in df.columns if c not in ["patient_id", "recording_id"]]
    nan_count = int(df[num_cols].isna().sum().sum())
    inf_count = int(np.isinf(df[num_cols].to_numpy()).sum())

    assert nan_count == 0, f"Found {nan_count} NaNs in feature table!"
    assert inf_count == 0, f"Found {inf_count} Infs in feature table!"

    # 5. Check flat signal handling (win 2 and win 3)
    for ch in ch_names:
        assert df.loc[2, f"{ch}_var"] == 0.0
        assert df.loc[2, f"{ch}_hjorth_mobility"] == 0.0
        assert df.loc[2, f"{ch}_hjorth_complexity"] == 0.0
        assert df.loc[3, f"{ch}_var"] == 0.0
        assert df.loc[3, f"{ch}_hjorth_mobility"] == 0.0

    print("All synthetic feature extraction tests passed successfully!")


def test_channel_order_safety():
    print("--- Running test_channel_order_safety ---")
    n_windows = 2
    n_channels = 2
    n_samples = 1024
    windows = np.random.randn(n_windows, n_channels, n_samples)
    metadata = [{"window_id": i, "label": 0} for i in range(n_windows)]

    seg_res = SegmentationResult(
        windows=windows,
        metadata=metadata,
        ch_names=["CH_A", "CH_B"],
    )

    # Passing mismatched channel names should raise ValueError
    try:
        extract_features(seg_res, ch_names=["CH_B", "CH_A"], sfreq=256.0)
        assert False, "Should have raised ValueError on mismatched ch_names"
    except ValueError as e:
        print(f"Correctly caught channel mismatch error: {e}")

    # Passing wrong number of channels should raise ValueError
    try:
        extract_features(seg_res, ch_names=["CH_A"], sfreq=256.0)
        assert False, "Should have raised ValueError on wrong channel count"
    except ValueError as e:
        print(f"Correctly caught channel count error: {e}")

    print("All channel order safety tests passed successfully!")


if __name__ == "__main__":
    test_synthetic_feature_extraction()
    test_channel_order_safety()
