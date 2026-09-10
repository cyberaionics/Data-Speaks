"""
Run Stage 05 Unsupervised Clustering on chb01_03 PCA-reduced representation.

Loads chb01_03_pca_reduced.csv (100 PCs at 95% variance) and executes:
1. K-Means (k=2 and elbow evaluation k=2..6)
2. DBSCAN (evaluating density-based outlier detection)
3. Cluster evaluations: Silhouette score, Adjusted Rand Index (ARI), Normalized Mutual Info (NMI)
4. Contingency tables comparing unsupervised clusters against ground-truth labels
5. Visualizations saved to results/figures/clustering/
6. Summary table saved to results/tables/chb01_03_clustering_metrics.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\Kavya\Data-Speaks")
STAGE05_DIR = REPO_ROOT / "pipeline" / "05_clustering"
sys.path.insert(0, str(STAGE05_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clustering import (
    extract_features_and_meta,
    run_kmeans,
    run_dbscan,
    evaluate_clusters,
    STANDARD_META_COLS,
)

CLUST_FIG_DIR = REPO_ROOT / "results" / "figures" / "clustering"
CLUST_FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = REPO_ROOT / "results" / "tables"


def plot_kmeans_clusters(reduced_df, km_labels, true_labels, save_path: Path):
    """Plot PC1 vs PC2 colored by K-Means clusters with true ictal markers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. K-Means Clusters
    scatter1 = ax1.scatter(reduced_df["PC1"], reduced_df["PC2"], c=km_labels, cmap="tab10", alpha=0.6, s=30)
    ax1.set_xlabel("PC1 (23.2% Variance)", fontsize=10)
    ax1.set_ylabel("PC2 (16.1% Variance)", fontsize=10)
    ax1.set_title("Unsupervised K-Means Clusters (k=2)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(scatter1, ax=ax1, label="Cluster Assignment")

    # 2. Ground-Truth Overlay
    interictal = reduced_df[true_labels == 0]
    ictal = reduced_df[true_labels == 1]
    ax2.scatter(interictal["PC1"], interictal["PC2"], color="#1f77b4", alpha=0.4, s=25, label="Normal (0)")
    ax2.scatter(ictal["PC1"], ictal["PC2"], color="crimson", alpha=0.9, s=60, marker="^", label="Seizure (1)")
    ax2.set_xlabel("PC1 (23.2% Variance)", fontsize=10)
    ax2.set_ylabel("PC2 (16.1% Variance)", fontsize=10)
    ax2.set_title("Ground-Truth Class Labels", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Stage 05: K-Means Clustering vs Ground-Truth (chb01_03)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved K-Means cluster figure to {save_path}")


def plot_dbscan_clusters(reduced_df, db_labels, true_labels, save_path: Path):
    """Plot DBSCAN clusters and noise points."""
    fig, ax = plt.subplots(figsize=(9, 6))

    unique_labels = set(db_labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0.6, 0.6, 0.6, 0.5]
            label = "Noise / Outliers (-1)"
            marker = "x"
            size = 35
        else:
            label = f"Cluster {k}"
            marker = "o"
            size = 30

        class_member_mask = (db_labels == k)
        xy = reduced_df[class_member_mask]
        ax.scatter(xy["PC1"], xy["PC2"], color=col, label=label, marker=marker, s=size, alpha=0.7)

    # Highlight true ictal windows
    ictal_df = reduced_df[true_labels == 1]
    ax.scatter(ictal_df["PC1"], ictal_df["PC2"], facecolors="none", edgecolors="crimson", s=90, linewidth=1.5, label="True Seizures")

    ax.set_xlabel("PC1 (23.2% Variance)", fontsize=10)
    ax.set_ylabel("PC2 (16.1% Variance)", fontsize=10)
    ax.set_title("DBSCAN Density Clustering & Outlier Detection (chb01_03)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved DBSCAN cluster figure to {save_path}")


def main():
    print("=" * 75)
    print("STAGE 05: UNSUPERVISED CLUSTERING ON chb01_03 PCA REPRESENTATION")
    print("=" * 75)

    pca_path = TABLES_DIR / "chb01_03_pca_reduced.csv"
    if not pca_path.exists():
        print(f"Error: PCA reduced file not found at {pca_path}")
        sys.exit(1)

    print(f"Loading PCA representation from {pca_path}...")
    df = pd.read_csv(pca_path)
    meta_df, X, y = extract_features_and_meta(df)
    n_samples, n_pcs = X.shape
    print(f"Loaded: {n_samples} windows, {n_pcs} PCs (95% cumulative variance)")
    print(f"Ground-truth class distribution: Normal={sum(y==0)}, Seizure={sum(y==1)}")

    eval_summaries = []

    # 1. K-Means (k=2)
    print("\n[1/3] Running K-Means (k=2)...")
    km_labels_2, km2 = run_kmeans(X, n_clusters=2, random_state=42)
    eval_km2 = evaluate_clusters(X, km_labels_2, y, "K-Means (k=2)")
    eval_summaries.append(eval_km2.summary_dict())
    print("  K-Means (k=2) Contingency Table (Rows=Cluster, Cols=True Label):")
    print(eval_km2.contingency_table)
    print(f"  Silhouette: {eval_km2.silhouette:.4f}, ARI: {eval_km2.ari:.4f}, NMI: {eval_km2.nmi:.4f}")

    # Plot K-Means
    plot_kmeans_clusters(
        df,
        km_labels_2,
        y,
        save_path=CLUST_FIG_DIR / "kmeans_k2_clusters.png",
    )

    # 2. Elbow & Silhouette search for k in 2..6
    print("\n[2/3] Evaluating K-Means for k=2..6...")
    inertias = []
    silhouettes = []
    k_range = list(range(2, 7))
    for k in k_range:
        labels, model = run_kmeans(X, n_clusters=k, random_state=42)
        inertias.append(model.inertia_)
        sil = evaluate_clusters(X, labels, y, f"K-Means (k={k})")
        silhouettes.append(sil.silhouette)
        if k > 2:
            eval_summaries.append(sil.summary_dict())

    # Elbow and Silhouette curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(k_range, inertias, marker="o", color="#1f77b4", linewidth=1.5)
    ax1.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax1.set_ylabel("Inertia (WCSS)", fontsize=10)
    ax1.set_title("K-Means Elbow Curve", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(k_range, silhouettes, marker="s", color="crimson", linewidth=1.5)
    ax2.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax2.set_ylabel("Silhouette Score", fontsize=10)
    ax2.set_title("Silhouette Score vs k", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    elbow_path = CLUST_FIG_DIR / "kmeans_elbow_silhouette.png"
    plt.savefig(elbow_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved elbow and silhouette curves to {elbow_path}")

    # 3. DBSCAN
    print("\n[3/3] Running DBSCAN (Density-Based Outlier Detection)...")
    # In 100-dimensional PCA space, distances are relatively large; eps around 8.0-12.0
    db_labels, db = run_dbscan(X, eps=10.0, min_samples=8)
    eval_db = evaluate_clusters(X, db_labels, y, "DBSCAN (eps=10, min_samples=8)")
    eval_summaries.append(eval_db.summary_dict())
    print("  DBSCAN Contingency Table (Rows=Cluster [-1 is Noise], Cols=True Label):")
    print(eval_db.contingency_table)
    print(f"  Silhouette: {eval_db.silhouette:.4f}, ARI: {eval_db.ari:.4f}, NMI: {eval_db.nmi:.4f}")

    plot_dbscan_clusters(
        df,
        db_labels,
        y,
        save_path=CLUST_FIG_DIR / "dbscan_clusters.png",
    )

    # 4. Save metrics summary
    summary_df = pd.DataFrame(eval_summaries)
    metrics_csv = TABLES_DIR / "chb01_03_clustering_metrics.csv"
    summary_df.to_csv(metrics_csv, index=False)
    print(f"\n[Results] Saved clustering evaluation metrics to: {metrics_csv}")
    print(summary_df.to_string(index=False))

    # Save clustered labels table
    df_clustered = df[STANDARD_META_COLS].copy()
    df_clustered["kmeans_k2"] = km_labels_2
    df_clustered["dbscan"] = db_labels
    clustered_csv = TABLES_DIR / "chb01_03_clustered_windows.csv"
    df_clustered.to_csv(clustered_csv, index=False)
    print(f"[Results] Saved window cluster assignments to: {clustered_csv}")

    print("\n" + "=" * 75)
    print("STAGE 05 CLUSTERING COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
