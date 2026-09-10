"""
Run Stage 04: Dimensionality Reduction on real CHB-MIT recording chb01_03.

Performs:
1. Load features from results/tables/chb01_03_features.csv (1799 rows, 449 cols).
2. Separate metadata (7 columns) from numerical EEG features (442 columns).
3. Standardize features using StandardScaler.
4. Fit full PCA on standardized features (unsupervised, no labels used).
5. Calculate and report cumulative explained variance for 80%, 90%, 95%, 99%.
6. Project data and export PCA reduced table with preserved metadata.
7. Save explained variance summary CSV.
8. Generate publication-quality visualizations:
   - Scree plot + cumulative explained variance curve
   - PC1 vs PC2 scatter plot color-coded by interictal vs ictal
   - Top feature loadings on PC1 and PC2
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\Kavya\Data-Speaks")
STAGE04_DIR = REPO_ROOT / "pipeline" / "04_dimensionality_reduction"
sys.path.insert(0, str(STAGE04_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dimensionality_reduction import (
    split_features_and_metadata,
    fit_pca,
    transform_pca,
)


def plot_variance_curves(pca_res, save_path: Path):
    """Plot Scree plot and Cumulative Explained Variance curve."""
    evr = pca_res.explained_variance_ratio * 100
    cum_evr = pca_res.cumulative_variance_ratio * 100
    n_comps = len(evr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Scree Plot (top 30 components)
    k_scree = min(30, n_comps)
    x_scree = np.arange(1, k_scree + 1)
    ax1.bar(x_scree, evr[:k_scree], color="#1f77b4", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.plot(x_scree, evr[:k_scree], color="navy", marker="o", markersize=4, linewidth=1.5)
    ax1.set_xlabel("Principal Component", fontsize=10)
    ax1.set_ylabel("Explained Variance Ratio (%)", fontsize=10)
    ax1.set_title(f"Scree Plot (Top {k_scree} Components)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Cumulative Explained Variance Curve
    x_all = np.arange(1, n_comps + 1)
    ax2.plot(x_all, cum_evr, color="crimson", linewidth=2.0, label="Cumulative Variance")

    # Mark threshold lines
    thresholds = [
        (80, pca_res.components_for_80, "#2ca02c"),
        (90, pca_res.components_for_90, "#ff7f0e"),
        (95, pca_res.components_for_95, "#9467bd"),
        (99, pca_res.components_for_99, "#d62728"),
    ]
    for pct, k, col in thresholds:
        ax2.axhline(pct, color=col, linestyle=":", alpha=0.8)
        ax2.axvline(k, color=col, linestyle="--", alpha=0.8, label=f"{pct}%: {k} PCs")
        ax2.plot(k, pct, marker="o", color=col, markersize=5)

    ax2.set_xlabel("Number of Principal Components", fontsize=10)
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=10)
    ax2.set_title("Cumulative Explained Variance Curve", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.set_xlim(1, n_comps)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("PCA Variance Decomposition — chb01_03 (442 Features)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved variance curves to {save_path}")


def plot_pc_scatter(reduced_df: pd.DataFrame, pca_res, save_path: Path):
    """Plot PC1 vs PC2 projection color-coded by class label."""
    evr1 = pca_res.explained_variance_ratio[0] * 100
    evr2 = pca_res.explained_variance_ratio[1] * 100

    fig, ax = plt.subplots(figsize=(9, 7))

    interictal = reduced_df[reduced_df["label"] == 0]
    ictal = reduced_df[reduced_df["label"] == 1]
    ambiguous = reduced_df[reduced_df["label"] == -1]

    # Plot interictal points
    ax.scatter(
        interictal["PC1"],
        interictal["PC2"],
        c="#1f77b4",
        alpha=0.45,
        s=25,
        label=f"Interictal (label=0, n={len(interictal)})",
        edgecolors="none",
    )

    # Plot ictal points on top with prominent styling
    if len(ictal) > 0:
        ax.scatter(
            ictal["PC1"],
            ictal["PC2"],
            c="crimson",
            alpha=0.9,
            s=60,
            marker="^",
            label=f"Ictal (label=1, n={len(ictal)})",
            edgecolors="black",
            linewidths=0.7,
        )

    # Plot ambiguous points if any
    if len(ambiguous) > 0:
        ax.scatter(
            ambiguous["PC1"],
            ambiguous["PC2"],
            c="gray",
            alpha=0.6,
            s=35,
            marker="s",
            label=f"Ambiguous (label=-1, n={len(ambiguous)})",
            edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({evr1:.2f}% Variance)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"PC2 ({evr2:.2f}% Variance)", fontsize=11, fontweight="bold")
    ax.set_title("Exploratory Unsupervised PCA: PC1 vs PC2 Projection (chb01_03)", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)

    note_text = (
        "Note: PCA is completely unsupervised (fitted without labels).\n"
        "This projection displays primary directions of overall signal variance."
    )
    plt.annotate(
        note_text,
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved PC1 vs PC2 scatter plot to {save_path}")


def plot_top_loadings(loadings_df: pd.DataFrame, save_path: Path, top_k: int = 10):
    """Plot top absolute loadings on PC1 and PC2."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # PC1
    top_pc1 = loadings_df["PC1"].abs().nlargest(top_k).index
    pc1_vals = loadings_df.loc[top_pc1, "PC1"].sort_values()
    colors1 = ["crimson" if v < 0 else "#1f77b4" for v in pc1_vals]
    ax1.barh(range(len(pc1_vals)), pc1_vals, color=colors1, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.set_yticks(range(len(pc1_vals)))
    ax1.set_yticklabels(pc1_vals.index, fontsize=9)
    ax1.set_xlabel("Loading Coefficient", fontsize=10)
    ax1.set_title(f"Top {top_k} Feature Loadings on PC1", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # PC2
    top_pc2 = loadings_df["PC2"].abs().nlargest(top_k).index
    pc2_vals = loadings_df.loc[top_pc2, "PC2"].sort_values()
    colors2 = ["crimson" if v < 0 else "#1f77b4" for v in pc2_vals]
    ax2.barh(range(len(pc2_vals)), pc2_vals, color=colors2, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(range(len(pc2_vals)))
    ax2.set_yticklabels(pc2_vals.index, fontsize=9)
    ax2.set_xlabel("Loading Coefficient", fontsize=10)
    ax2.set_title(f"Top {top_k} Feature Loadings on PC2", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Feature Loadings on Primary Principal Components — chb01_03", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved feature loadings plot to {save_path}")


def main():
    print("=" * 75)
    print("STAGE 04: DIMENSIONALITY REDUCTION (PCA) — REAL RECORDING chb01_03")
    print("=" * 75)

    features_path = REPO_ROOT / "results" / "tables" / "chb01_03_features.csv"
    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        sys.exit(1)

    print(f"\n[1/5] Loading features from {features_path}...")
    df = pd.read_csv(features_path)
    print(f"  Loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 1. Separate metadata and numerical features
    print("\n[2/5] Isolating metadata from features...")
    meta_df, features_df, feature_names = split_features_and_metadata(df)
    n_windows, n_features = features_df.shape
    print(f"  Metadata columns ({meta_df.shape[1]}): {list(meta_df.columns)}")
    print(f"  Numerical EEG features ({n_features}): 17 channels x 26 features.")
    assert n_features == 442, f"Expected 442 features, got {n_features}"
    assert n_windows == 1799, f"Expected 1799 rows, got {n_windows}"

    # 2. Standardize and Fit PCA
    print("\n[3/5] Standardizing features and fitting full PCA...")
    pca_res, scaler, pca = fit_pca(features_df, n_components=None, random_state=42)
    n_components_fitted = len(pca_res.explained_variance_ratio)
    print(f"  PCA fitted {n_components_fitted} principal components.")

    # 3. Report Cumulative Explained Variance Thresholds
    print("\n" + "-" * 60)
    print("CUMULATIVE EXPLAINED VARIANCE ANALYSIS:")
    print("-" * 60)
    print(f"  PC1 Explained Variance:  {pca_res.explained_variance_ratio[0]*100:6.2f}%")
    print(f"  PC2 Explained Variance:  {pca_res.explained_variance_ratio[1]*100:6.2f}%")
    print(f"  Top 2 PCs Combined:      {(pca_res.explained_variance_ratio[0] + pca_res.explained_variance_ratio[1])*100:6.2f}%")
    print(f"  Top 5 PCs Combined:      {pca_res.cumulative_variance_ratio[4]*100:6.2f}%")
    print(f"  Top 10 PCs Combined:     {pca_res.cumulative_variance_ratio[9]*100:6.2f}%")
    print()
    print(f"  Components for 80% Variance: {pca_res.components_for_80:3d} PCs (out of {n_features}) -> {(1 - pca_res.components_for_80/n_features)*100:.1f}% reduction")
    print(f"  Components for 90% Variance: {pca_res.components_for_90:3d} PCs (out of {n_features}) -> {(1 - pca_res.components_for_90/n_features)*100:.1f}% reduction")
    print(f"  Components for 95% Variance: {pca_res.components_for_95:3d} PCs (out of {n_features}) -> {(1 - pca_res.components_for_95/n_features)*100:.1f}% reduction")
    print(f"  Components for 99% Variance: {pca_res.components_for_99:3d} PCs (out of {n_features}) -> {(1 - pca_res.components_for_99/n_features)*100:.1f}% reduction")
    print("-" * 60)

    # 4. Transform and save outputs
    print("\n[4/5] Projecting data and saving tabular outputs...")
    # Project to 95% variance components as the primary candidate representation
    k_95 = pca_res.components_for_95
    pc_proj_df = transform_pca(features_df, scaler, pca, n_components=k_95)

    reduced_df = pd.concat([meta_df.reset_index(drop=True), pc_proj_df.reset_index(drop=True)], axis=1)

    tables_dir = REPO_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    reduced_csv = tables_dir / "chb01_03_pca_reduced.csv"
    reduced_df.to_csv(reduced_csv, index=False)
    print(f"  Saved reduced feature table (95% variance, {k_95} PCs) to: {reduced_csv}")
    print(f"    Shape: {reduced_df.shape} (7 metadata + {k_95} PC columns)")

    # Save variance summary table
    var_summary_df = pca_res.summary_dataframe()
    var_csv = tables_dir / "chb01_03_pca_variance.csv"
    var_summary_df.to_csv(var_csv, index=False)
    print(f"  Saved full PCA variance table to: {var_csv}")

    # Top feature loadings
    loadings_df = pca_res.get_loadings()
    loadings_csv = tables_dir / "chb01_03_pca_loadings.csv"
    loadings_df.to_csv(loadings_csv)
    print(f"  Saved feature loadings table to: {loadings_csv}")

    # 5. Visualizations
    print("\n[5/5] Generating publication figures...")
    figures_dir = REPO_ROOT / "results" / "figures" / "dimensionality_reduction"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Scree and cumulative variance
    plot_variance_curves(
        pca_res,
        save_path=figures_dir / "pca_explained_variance_scree.png",
    )

    # PC1 vs PC2 scatter plot
    pc_2d_df = pd.concat([meta_df.reset_index(drop=True), transform_pca(features_df, scaler, pca, n_components=2)], axis=1)
    plot_pc_scatter(
        pc_2d_df,
        pca_res,
        save_path=figures_dir / "pca_scatter_pc1_pc2.png",
    )

    # Top loadings
    plot_top_loadings(
        loadings_df,
        save_path=figures_dir / "pca_top_loadings_pc1_pc2.png",
        top_k=10,
    )

    # Validation Checks
    print("\n" + "=" * 75)
    print("VALIDATION SUMMARY")
    print("=" * 75)
    nan_count = int(reduced_df.isna().sum().sum())
    inf_count = int(np.isinf(reduced_df.iloc[:, 7:].to_numpy()).sum())
    print(f"Input features used:             {n_features} (from 17 channels)")
    print(f"Output row count:                {len(reduced_df)} (must match input 1799)")
    print(f"NaN count in reduced table:      {nan_count}")
    print(f"Inf count in reduced table:      {inf_count}")
    print(f"Metadata preserved:              {list(reduced_df.columns[:7])}")
    print(f"Metadata matches original:       {reduced_df[list(meta_df.columns)].equals(meta_df)}")
    print(f"Class distribution:              {reduced_df['label'].value_counts().to_dict()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
