"""
Stage 05: Unsupervised Clustering for the Data-Speaks EEG project.

Implements K-Means and DBSCAN clustering on dimensionality-reduced EEG representations
(e.g., PCA components). Evaluates unsupervised cluster purity, silhouette scores,
contingency against ground truth labels (post-hoc), and cluster characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

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
class ClusteringEvaluation:
    algorithm: str
    n_clusters: int
    silhouette: float
    ari: float
    nmi: float
    contingency_table: pd.DataFrame
    cluster_labels: np.ndarray

    def summary_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "n_clusters": self.n_clusters,
            "silhouette_score": round(self.silhouette, 4),
            "adjusted_rand_index": round(self.ari, 4),
            "normalized_mutual_info": round(self.nmi, 4),
        }


def extract_features_and_meta(
    df: pd.DataFrame,
    meta_cols: Sequence[str] = STANDARD_META_COLS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Separate metadata, feature matrix X, and ground-truth labels y.
    """
    present_meta = [c for c in meta_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in present_meta]

    meta_df = df[present_meta].copy()
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=int) if "label" in df.columns else np.zeros(len(df))

    return meta_df, X, y


def run_kmeans(
    X: np.ndarray,
    n_clusters: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans]:
    """
    Fit K-Means clustering unsupervised on feature matrix X.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return labels, km


def run_dbscan(
    X: np.ndarray,
    eps: float = 5.0,
    min_samples: int = 10,
) -> tuple[np.ndarray, DBSCAN]:
    """
    Fit DBSCAN density-based clustering unsupervised on feature matrix X.
    Noise points are marked as -1.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels, db


def evaluate_clusters(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    algorithm_name: str,
) -> ClusteringEvaluation:
    """
    Compute silhouette, ARI, NMI, and contingency table comparing
    clusters to ground-truth seizure labels (post-hoc evaluation only).
    """
    unique_clusters = set(cluster_labels)
    # Ignore noise cluster (-1 in DBSCAN) for count if desired, but report unique
    n_clusters = len(unique_clusters - {-1}) if -1 in unique_clusters else len(unique_clusters)

    # Silhouette requires at least 2 distinct clusters and not all points in their own cluster
    if 1 < len(unique_clusters) < len(X):
        sil = float(silhouette_score(X, cluster_labels))
    else:
        sil = 0.0

    ari = float(adjusted_rand_score(true_labels, cluster_labels))
    nmi = float(normalized_mutual_info_score(true_labels, cluster_labels))

    # Build contingency table
    df_eval = pd.DataFrame({"cluster": cluster_labels, "true_label": true_labels})
    contingency = pd.crosstab(df_eval["cluster"], df_eval["true_label"], rownames=["Cluster"], colnames=["True Label"])

    return ClusteringEvaluation(
        algorithm=algorithm_name,
        n_clusters=n_clusters,
        silhouette=sil,
        ari=ari,
        nmi=nmi,
        contingency_table=contingency,
        cluster_labels=cluster_labels,
    )
