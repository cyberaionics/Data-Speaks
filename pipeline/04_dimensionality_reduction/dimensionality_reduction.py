"""
Stage 04: Dimensionality Reduction for the Data-Speaks EEG project.

Implements Principal Component Analysis (PCA) with strict feature/metadata separation,
leakage-safe fitting, feature standardization, variance tracking, and projection.

Key design principles:
- Feature/metadata isolation: metadata (and labels) are never included in PCA transformations.
- Leakage-safe: fits StandardScaler and PCA on training data only (when train/test split is supplied).
- Comprehensive variance metrics: tracks individual and cumulative explained variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

STANDARD_META_COLS = [
    "patient_id",
    "recording_id",
    "window_id",
    "start_sec",
    "end_sec",
    "duration_sec",
    "label",
]


@dataclass
class PCAResult:
    """Container for fitted PCA model, scaler, and variance summaries."""
    pca: PCA
    scaler: StandardScaler
    feature_names: list[str]
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    components_for_80: int
    components_for_90: int
    components_for_95: int
    components_for_99: int

    def summary_dataframe(self) -> pd.DataFrame:
        """Return DataFrame with PC variance metrics."""
        n_comps = len(self.explained_variance_ratio)
        return pd.DataFrame({
            "component": [f"PC{i+1}" for i in range(n_comps)],
            "explained_variance": self.pca.explained_variance_,
            "explained_variance_ratio": self.explained_variance_ratio,
            "cumulative_variance_ratio": self.cumulative_variance_ratio,
        })

    def get_loadings(self) -> pd.DataFrame:
        """Return DataFrame of feature loadings (coefficients) for all components."""
        n_comps = len(self.explained_variance_ratio)
        cols = [f"PC{i+1}" for i in range(n_comps)]
        return pd.DataFrame(self.pca.components_.T, index=self.feature_names, columns=cols)


def split_features_and_metadata(
    df: pd.DataFrame,
    metadata_cols: Sequence[str] = STANDARD_META_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Separate metadata columns from numerical EEG feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input table from Stage 03.
    metadata_cols : Sequence[str]
        List of metadata column names.

    Returns
    -------
    meta_df : pd.DataFrame
        DataFrame containing only metadata columns present in df.
    features_df : pd.DataFrame
        DataFrame containing strictly numerical features.
    feature_names : list[str]
        List of feature column names.
    """
    present_meta = [c for c in metadata_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in present_meta]

    if not feature_cols:
        raise ValueError("No feature columns found after splitting metadata.")

    meta_df = df[present_meta].copy()
    features_df = df[feature_cols].copy()

    # Validate that features are finite numerical values
    if not np.isfinite(features_df.to_numpy()).all():
        raise ValueError("Input feature matrix contains NaN or Inf values.")

    return meta_df, features_df, feature_cols


def fit_pca(
    features_df: pd.DataFrame,
    n_components: Optional[int | float] = None,
    random_state: int = 42,
) -> tuple[PCAResult, StandardScaler, PCA]:
    """
    Standardize features and fit PCA without labels or metadata.

    Parameters
    ----------
    features_df : pd.DataFrame
        Numerical features DataFrame (no metadata, no labels).
    n_components : int or float, optional
        Number of components or variance fraction. If None, fits all components.
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    result : PCAResult
        Container with fitted models and explained variance statistics.
    scaler : StandardScaler
        Fitted scaler.
    pca : PCA
        Fitted PCA model.
    """
    feature_names = list(features_df.columns)
    X = features_df.to_numpy(dtype=np.float64)

    # 1. Standardize (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Fit PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_scaled)

    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    # Determine component counts for key variance thresholds
    def _find_k(threshold: float) -> int:
        idx = np.searchsorted(cum_evr, threshold)
        return int(idx + 1) if idx < len(cum_evr) else len(cum_evr)

    k_80 = _find_k(0.80)
    k_90 = _find_k(0.90)
    k_95 = _find_k(0.95)
    k_99 = _find_k(0.99)

    res = PCAResult(
        pca=pca,
        scaler=scaler,
        feature_names=feature_names,
        explained_variance_ratio=evr,
        cumulative_variance_ratio=cum_evr,
        components_for_80=k_80,
        components_for_90=k_90,
        components_for_95=k_95,
        components_for_99=k_99,
    )
    return res, scaler, pca


def transform_pca(
    features_df: pd.DataFrame,
    scaler: StandardScaler,
    pca: PCA,
    n_components: Optional[int] = None,
) -> pd.DataFrame:
    """
    Transform feature DataFrame using pre-fitted scaler and PCA.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix (columns must match scaler training features).
    scaler : StandardScaler
        Fitted scaler.
    pca : PCA
        Fitted PCA model.
    n_components : int, optional
        Number of top components to retain. If None, uses all fitted components.

    Returns
    -------
    pd.DataFrame
        Transformed PC coordinates with columns ['PC1', 'PC2', ...].
    """
    X = features_df.to_numpy(dtype=np.float64)
    X_scaled = scaler.transform(X)
    X_proj = pca.transform(X_scaled)

    if n_components is not None:
        X_proj = X_proj[:, :n_components]

    cols = [f"PC{i+1}" for i in range(X_proj.shape[1])]
    return pd.DataFrame(X_proj, columns=cols, index=features_df.index)


def apply_pca_pipeline(
    df: pd.DataFrame,
    n_components_or_variance: Optional[int | float] = None,
    metadata_cols: Sequence[str] = STANDARD_META_COLS,
    random_state: int = 42,
) -> tuple[pd.DataFrame, PCAResult]:
    """
    End-to-end unsupervised exploratory PCA pipeline on a single dataset.
    Splits metadata, standardizes features, fits PCA, transforms, and rejoins metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Full table containing metadata and features.
    n_components_or_variance : int or float, optional
        Number of components or variance threshold to retain in the reduced table.
        If None, retains all fitted components.
    metadata_cols : Sequence[str]
        List of metadata column names.

    Returns
    -------
    reduced_df : pd.DataFrame
        DataFrame with original metadata columns followed by retained PC columns.
    pca_result : PCAResult
        Variance metrics, loadings, and fitted models.
    """
    meta_df, features_df, _ = split_features_and_metadata(df, metadata_cols=metadata_cols)

    pca_result, scaler, pca = fit_pca(
        features_df=features_df,
        n_components=None,  # Fit full spectrum to assess all variance thresholds
        random_state=random_state,
    )

    # Determine how many PCs to export
    if isinstance(n_components_or_variance, float) and 0 < n_components_or_variance <= 1.0:
        k_export = np.searchsorted(pca_result.cumulative_variance_ratio, n_components_or_variance) + 1
        k_export = min(k_export, len(pca_result.explained_variance_ratio))
    elif isinstance(n_components_or_variance, int) and n_components_or_variance > 0:
        k_export = min(n_components_or_variance, len(pca_result.explained_variance_ratio))
    else:
        k_export = len(pca_result.explained_variance_ratio)

    pc_df = transform_pca(features_df, scaler, pca, n_components=k_export)

    reduced_df = pd.concat([meta_df.reset_index(drop=True), pc_df.reset_index(drop=True)], axis=1)
    return reduced_df, pca_result
