"""
Stage 06: Standalone inference.

Processes an EDF through:
Stage 0 -> Pipeline C -> segmentation -> feature extraction
-> saved scaler -> saved PCA -> saved classifier.

No seizure annotation is required for the prediction step.
For chb01_03, known annotations are used only for post-hoc validation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

for p in [
    REPO_ROOT / "pipeline" / "01_preprocessing",
    REPO_ROOT / "pipeline" / "01_preprocessing" / "candidate_algorithms",
    REPO_ROOT / "pipeline" / "02_segmentation",
    REPO_ROOT / "pipeline" / "03_feature_extraction",
]:
    sys.path.insert(0, str(p))

MODELS_DIR = REPO_ROOT / "models" / "chb01_final_model"
DATASET_DIR = REPO_ROOT / "data" / "CHB-MIT"
TABLES_DIR = REPO_ROOT / "results" / "tables"


def load_model_artifacts():
    required = [
        "scaler.pkl",
        "pca.pkl",
        "classifier.pkl",
        "feature_cols.pkl",
        "metadata.json",
    ]

    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {missing}. "
            "Run run_classification_chb01.py first."
        )

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    pca = joblib.load(MODELS_DIR / "pca.pkl")
    clf = joblib.load(MODELS_DIR / "classifier.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Loaded model: {metadata['model_name']}")
    print(f"PCA components: {metadata['n_pca_components']}")

    return scaler, pca, clf, feature_cols, metadata


def run_edf_pipeline(
    edf_path: Path,
    recording_id: str,
    patient_id: str = "chb01",
) -> pd.DataFrame:
    """Run Stage 0 -> Pipeline C -> segmentation -> features."""
    from pipeline_c_compat import apply_asrpy_numpy2_patch
    apply_asrpy_numpy2_patch()

    from stage0 import run_stage0
    from pipeline_c import run_pipeline_c
    from segmentation import segment_raw
    from feature_extraction import extract_features

    summary_file = edf_path.parent / f"{patient_id}-summary.txt"

    print(f"\nProcessing {edf_path.name}...")

    s0 = run_stage0(
        edf_path,
        summary_file,
        patient_id,
        recording_id,
    )

    c_out = run_pipeline_c(
        s0.raw,
        s0.seizure_intervals,
        asr_cutoff=20.0,
    )
    clean_raw = c_out["final"]

    seizure_tuples = [
        (interval.start_sec, interval.end_sec)
        for interval in s0.seizure_intervals
    ]

    segments = segment_raw(
        clean_raw,
        seizure_intervals=seizure_tuples,
        window_sec=4.0,
        overlap=0.50,
        target_sfreq=256.0,
        include_ambiguous=True,
        patient_id=patient_id,
        recording_id=recording_id,
    )

    features_df = extract_features(
        segments,
        ch_names=list(clean_raw.ch_names),
        sfreq=256.0,
    )

    print(f"Extracted feature matrix: {features_df.shape}")
    return features_df


def run_inference(
    features_df: pd.DataFrame,
    scaler,
    pca,
    clf,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Apply saved preprocessing and classifier."""
    meta_cols = [
        "patient_id", "recording_id", "window_id",
        "start_sec", "end_sec", "duration_sec", "label",
    ]

    present_meta = [c for c in meta_cols if c in features_df.columns]
    out = features_df[present_meta].copy()

    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} expected features. "
            f"Example: {missing[:5]}"
        )

    X = features_df[feature_cols].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("Inference features contain NaN or Inf.")

    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    y_pred = clf.predict(X_pca)

    if hasattr(clf, "predict_proba"):
        score = clf.predict_proba(X_pca)[:, 1]
        score_name = "probability_ictal"
    elif hasattr(clf, "decision_function"):
        score = clf.decision_function(X_pca)
        score_name = "decision_score_ictal"
    else:
        score = np.full(len(y_pred), np.nan)
        score_name = "model_score"

    out["predicted_label"] = y_pred
    out[score_name] = score

    return out


def compare_against_annotations(
    predictions_df: pd.DataFrame,
    known_intervals: list[tuple[float, float]],
) -> None:
    """Optional post-hoc validation against known seizure annotations."""
    if not known_intervals:
        return

    print("\n--- Post-hoc annotation comparison ---")

    for start_s, end_s in known_intervals:
        mask = (
            (predictions_df["start_sec"] >= start_s)
            & (predictions_df["start_sec"] <= end_s)
        )
        sub = predictions_df[mask]

        if len(sub) == 0:
            print(f"{start_s}-{end_s}s: no windows found")
            continue

        n_ictal = int((sub["predicted_label"] == 1).sum())
        print(
            f"{start_s}-{end_s}s: "
            f"{n_ictal}/{len(sub)} windows predicted ictal "
            f"({100 * n_ictal / len(sub):.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf", type=str)
    parser.add_argument("--recording", default="chb01_03")
    parser.add_argument("--patient", default="chb01")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    scaler, pca, clf, feature_cols, metadata = load_model_artifacts()

    if args.edf:
        edf_path = Path(args.edf)
    else:
        edf_path = DATASET_DIR / args.patient / f"{args.recording}.edf"

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF not found: {edf_path}")

    t0 = time.time()

    features_df = run_edf_pipeline(
        edf_path,
        args.recording,
        args.patient,
    )

    predictions = run_inference(
        features_df,
        scaler,
        pca,
        clf,
        feature_cols,
    )

    n = len(predictions)
    n_ictal = int((predictions["predicted_label"] == 1).sum())

    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY")
    print("=" * 70)
    print(f"Recording:          {args.recording}")
    print(f"Model:              {metadata['model_name']}")
    print(f"Windows:            {n}")
    print(f"Predicted ictal:    {n_ictal} ({100*n_ictal/n:.1f}%)")
    print(f"Predicted interictal: {n-n_ictal} ({100*(n-n_ictal)/n:.1f}%)")
    print(f"Runtime:            {time.time()-t0:.1f}s")

    if (
        not args.no_validate
        and args.recording == "chb01_03"
    ):
        compare_against_annotations(
            predictions,
            [(2996.0, 3036.0)],
        )

    if args.output_csv:
        output = Path(args.output_csv)
    else:
        output = TABLES_DIR / f"{args.recording}_inference_predictions.csv"

    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
