from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\Kavya\Data-Speaks")
PREPROC_DIR = REPO_ROOT / "pipeline" / "01_preprocessing"
CANDIDATE_C_DIR = PREPROC_DIR / "candidate_algorithms"
SEG_DIR = REPO_ROOT / "pipeline" / "02_segmentation"
FEAT_DIR = REPO_ROOT / "pipeline" / "03_feature_extraction"

sys.path.insert(0, str(PREPROC_DIR))
sys.path.insert(0, str(CANDIDATE_C_DIR))
sys.path.insert(0, str(SEG_DIR))
sys.path.insert(0, str(FEAT_DIR))

from stage0 import run_stage0, DATASET_DIR
from pipeline_c import run_pipeline_c
from segmentation import segment_raw
from feature_extraction import extract_features
import numpy as np
import pandas as pd


def main():
    print("=" * 70)
    print("RUNNING FEATURE EXTRACTION ON REAL CHB-MIT RECORDING chb01_03")
    print("=" * 70)

    edf_path = DATASET_DIR / "chb01" / "chb01_03.edf"
    summary_path = DATASET_DIR / "chb01" / "chb01-summary.txt"

    if not edf_path.exists():
        print(f"Error: EDF not found at {edf_path}")
        sys.exit(1)

    t0 = time.time()
    # 1. Stage 0
    print("\n[Step 1/3] Running Stage 0 Standardization...")
    s0_res = run_stage0(edf_path, summary_path, patient_id="chb01", recording_id="chb01_03")
    print(f"  Stage 0 complete: {len(s0_res.channels_kept)} channels, duration {s0_res.duration_sec:.1f}s")
    print(f"  Seizure intervals: {[f'{iv.start_sec}-{iv.end_sec}s' for iv in s0_res.seizure_intervals]}")

    # 2. Pipeline C
    print("\n[Step 2/3] Running Pipeline C Preprocessing...")
    c_out = run_pipeline_c(s0_res.raw, s0_res.seizure_intervals, asr_cutoff=20.0)
    raw_clean = c_out["final"]
    print("  Pipeline C complete.")

    # 3. Segmentation
    print("\n[Step 3/3] Running 4-second 50% Overlap Segmentation...")
    seiz_tuples = [(iv.start_sec, iv.end_sec) for iv in s0_res.seizure_intervals]
    seg_res = segment_raw(
        raw_clean,
        seizure_intervals=seiz_tuples,
        window_sec=4.0,
        overlap=0.50,
        target_sfreq=256.0,
        include_ambiguous=True,
        patient_id="chb01",
        recording_id="chb01_03",
    )
    print(f"  Segmentation complete: {seg_res.n_windows} windows, {seg_res.n_channels} channels, {seg_res.n_samples} samples per window.")

    # 4. Feature Extraction
    print("\n[Feature Extraction] Extracting features for all windows...")
    t_feat_start = time.time()
    features_df = extract_features(seg_res, ch_names=list(raw_clean.ch_names), sfreq=256.0)
    t_feat_elapsed = time.time() - t_feat_start
    print(f"  Feature extraction finished in {t_feat_elapsed:.2f}s")

    # 5. Validation & Reporting
    meta_cols = [c for c in ["patient_id", "recording_id", "window_id", "start_sec", "end_sec", "duration_sec", "label"] if c in features_df.columns]
    feat_cols = [c for c in features_df.columns if c not in meta_cols]

    n_windows = len(features_df)
    n_channels = seg_res.n_channels
    n_features_total = len(feat_cols)
    feats_per_channel = n_features_total // n_channels

    nan_count = int(features_df[feat_cols].isna().sum().sum())
    inf_count = int(np.isinf(features_df[feat_cols].to_numpy()).sum())
    label_counts = features_df["label"].value_counts().to_dict()

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY: chb01_03 FEATURES")
    print("=" * 70)
    print(f"Number of windows:        {n_windows}")
    print(f"Number of channels:       {n_channels}")
    print(f"Features per channel:     {feats_per_channel}")
    print(f"Total feature columns:    {n_features_total}")
    print(f"Total columns (inc meta): {features_df.shape[1]}")
    print(f"Final DataFrame shape:    {features_df.shape}")
    print(f"Total NaNs:               {nan_count}")
    print(f"Total Infs:               {inf_count}")
    print(f"Label counts:             {label_counts}  (0=interictal, 1=ictal, -1=ambiguous)")

    print("\nSample feature values (Window 0):")
    sample_cols = [
        "FP1-F7_mean", "FP1-F7_std", "FP1-F7_rms", "FP1-F7_skew", "FP1-F7_kurtosis",
        "FP1-F7_hjorth_mobility", "FP1-F7_delta_power", "FP1-F7_delta_relpower", "FP1-F7_spectral_edge_95",
        "FZ-CZ_mean", "FZ-CZ_rms", "FZ-CZ_total_power",
        "P8-O2_rms", "P8-O2_spectral_edge_95",
    ]
    for col in sample_cols:
        if col in features_df.columns:
            print(f"  {col:<30}: {features_df.loc[0, col]:.6f}")

    # 6. Save results to results/tables/chb01_03_features.csv
    tables_dir = REPO_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "chb01_03_features.csv"
    features_df.to_csv(out_csv, index=False)
    print(f"\nSaved feature table to: {out_csv}")
    print(f"Total script runtime: {time.time() - t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
