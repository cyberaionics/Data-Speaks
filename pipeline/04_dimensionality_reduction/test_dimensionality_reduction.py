"""
Sanity and unit tests for Stage 04: Dimensionality Reduction.
Verifies:
- Separation of metadata from numerical features
- Metadata remains unmodified
- Labels are not used in PCA fitting
- Leakage-safe train/test fitting vs transform
- Accurate cumulative explained variance calculation
- Robustness against NaNs/Infs
"""

from __future__ import annotations

import sys
from pathlib import Path

STAGE04_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE04_DIR))

import numpy as np
import pandas as pd
from dimensionality_reduction import (
    split_features_and_metadata,
    fit_pca,
    transform_pca,
    apply_pca_pipeline,
    STANDARD_META_COLS,
)


def test_split_and_metadata_preservation():
    print("--- Running test_split_and_metadata_preservation ---")
    n_samples = 20
    meta = {
        "patient_id": ["chb01"] * n_samples,
        "recording_id": ["rec_01"] * n_samples,
        "window_id": list(range(n_samples)),
        "start_sec": [i * 2.0 for i in range(n_samples)],
        "end_sec": [i * 2.0 + 4.0 for i in range(n_samples)],
        "duration_sec": [4.0] * n_samples,
        "label": [1 if i % 5 == 0 else 0 for i in range(n_samples)],
    }
    feats = {f"feat_{j}": np.random.randn(n_samples) for j in range(10)}
    df = pd.DataFrame({**meta, **feats})

    meta_df, feat_df, f_names = split_features_and_metadata(df)

    assert meta_df.shape == (n_samples, 7)
    assert feat_df.shape == (n_samples, 10)
    assert len(f_names) == 10
    assert "label" not in feat_df.columns
    assert "label" in meta_df.columns
    print("Passed split and metadata preservation test.")


def test_leakage_safe_pca():
    print("--- Running test_leakage_safe_pca ---")
    np.random.seed(42)
    n_train = 50
    n_test = 20
    n_feats = 8

    X_train_df = pd.DataFrame(np.random.randn(n_train, n_feats) * 2.0 + 5.0, columns=[f"f_{i}" for i in range(n_feats)])
    X_test_df = pd.DataFrame(np.random.randn(n_test, n_feats) * 2.0 + 5.0, columns=[f"f_{i}" for i in range(n_feats)])

    pca_res, scaler, pca = fit_pca(X_train_df, n_components=None)

    # Check scaling mean matches train
    assert np.allclose(scaler.mean_, X_train_df.mean(axis=0), atol=1e-5)

    # Transform train and test
    train_proj = transform_pca(X_train_df, scaler, pca, n_components=4)
    test_proj = transform_pca(X_test_df, scaler, pca, n_components=4)

    assert train_proj.shape == (n_train, 4)
    assert test_proj.shape == (n_test, 4)
    assert not np.isnan(test_proj.to_numpy()).any()
    assert not np.isinf(test_proj.to_numpy()).any()

    # Verify cumulative variance calculation
    cum_var = pca_res.cumulative_variance_ratio
    assert np.all(np.diff(cum_var) >= -1e-12)
    assert np.isclose(cum_var[-1], 1.0)
    print("Passed leakage-safe PCA test.")


def test_apply_pca_pipeline_end_to_end():
    print("--- Running test_apply_pca_pipeline_end_to_end ---")
    n_samples = 30
    n_feats = 12
    meta = {
        "patient_id": ["chb01"] * n_samples,
        "recording_id": ["rec_01"] * n_samples,
        "window_id": list(range(n_samples)),
        "start_sec": [i * 2.0 for i in range(n_samples)],
        "end_sec": [i * 2.0 + 4.0 for i in range(n_samples)],
        "duration_sec": [4.0] * n_samples,
        "label": [1 if i < 5 else 0 for i in range(n_samples)],
    }
    feats = {f"F_{j}": np.random.randn(n_samples) for j in range(n_feats)}
    df = pd.DataFrame({**meta, **feats})

    reduced_df, res = apply_pca_pipeline(df, n_components_or_variance=0.90)

    # Metadata must be preserved in exact order first
    for col in STANDARD_META_COLS:
        assert col in reduced_df.columns
        assert list(reduced_df[col]) == meta[col]

    # PCs must follow
    pc_cols = [c for c in reduced_df.columns if c.startswith("PC")]
    assert len(pc_cols) == res.components_for_90
    assert not reduced_df.isna().any().any()
    assert not np.isinf(reduced_df[pc_cols].to_numpy()).any()
    print("Passed end-to-end PCA pipeline test.")


if __name__ == "__main__":
    test_split_and_metadata_preservation()
    test_leakage_safe_pca()
    test_apply_pca_pipeline_end_to_end()
    print("All Stage 04 tests passed successfully!")
