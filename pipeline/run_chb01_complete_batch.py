"""
DATA-SPEAKS — Complete Cohort Processing & Analysis Runner for chb01 (42 Recordings)
=====================================================================================
Processes all 42 EDF recordings for patient chb01 through:
  Stage 0 -> Pipeline C -> Segmentation -> Feature Extraction
  -> Unified映 Dimensionality Reduction (PCA)
  -> Unsupervised Clustering (K-Means & DBSCAN)
  -> Comprehensive Midterm 5-Pillar EDA

Design constraints:
- Strictly unsupervised PCA and clustering (labels used solely for post-hoc analysis).
- Memory-safe: raw continuous EEG and window arrays are freed per recording.
- Incremental checkpointing: per-recording features are saved to disk so any
  interrupted run can resume without recomputing completed recordings.
- Robust error handling: single recording failure is logged without aborting the batch.
"""

from __future__ import annotations

import gc
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROC_DIR = REPO_ROOT / "pipeline" / "01_preprocessing"
CANDIDATE_C_DIR = PREPROC_DIR / "candidate_algorithms"
SEG_DIR = REPO_ROOT / "pipeline" / "02_segmentation"
FEAT_DIR = REPO_ROOT / "pipeline" / "03_feature_extraction"
PCA_DIR = REPO_ROOT / "pipeline" / "04_dimensionality_reduction"
CLUST_DIR = REPO_ROOT / "pipeline" / "05_clustering"
EDA_DIR = REPO_ROOT / "pipeline" / "eda"

for p in [PREPROC_DIR, CANDIDATE_C_DIR, SEG_DIR, FEAT_DIR, PCA_DIR, CLUST_DIR, EDA_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import spectrogram

# Pipeline module imports
from pipeline_c_compat import apply_asrpy_numpy2_patch
apply_asrpy_numpy2_patch()

from stage0 import run_stage0, DATASET_DIR
from pipeline_c import run_pipeline_c
from segmentation import segment_raw
from feature_extraction import extract_features
from dimensionality_reduction import (
    split_features_and_metadata,
    fit_pca,
    transform_pca,
    STANDARD_META_COLS,
)
from clustering import (
    extract_features_and_meta,
    run_kmeans,
    run_dbscan,
    evaluate_clusters,
)

# Output directories
TABLES_DIR = REPO_ROOT / "results" / "tables"
RECS_FEAT_DIR = TABLES_DIR / "chb01_features_by_recording"
EDA_FIG_DIR = REPO_ROOT / "results" / "figures" / "eda"
CLUST_FIG_DIR = REPO_ROOT / "results" / "figures" / "clustering"
PCA_FIG_DIR = REPO_ROOT / "results" / "figures" / "pca"

for d in [TABLES_DIR, RECS_FEAT_DIR, EDA_FIG_DIR, CLUST_FIG_DIR, PCA_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# PASS 1: Sequential Extraction (Stage 0 -> Pipeline C -> Seg -> Features)
# ===========================================================================

def run_pass1_feature_extraction(edf_files: list[Path], summary_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 75)
    print(f"PASS 1: SEQUENTIAL FEATURE EXTRACTION FOR {len(edf_files)} RECORDINGS")
    print("=" * 75)

    all_dfs = []
    failed_recs = []

    for idx, edf_path in enumerate(edf_files, start=1):
        rec_id = edf_path.stem
        rec_csv = RECS_FEAT_DIR / f"{rec_id}_features.csv"

        # Check if already computed
        if rec_csv.exists() and rec_csv.stat().st_size > 1000:
            try:
                df_cached = pd.read_csv(rec_csv)
                if len(df_cached) > 0 and df_cached.shape[1] == 449:
                    all_dfs.append(df_cached)
                    print(f"[{idx:02d}/{len(edf_files)}] {rec_id}: Loaded from cache ({len(df_cached)} windows, {df_cached.shape[1]} cols)")
                    continue
            except Exception as e:
                print(f"[{idx:02d}/{len(edf_files)}] {rec_id}: Corrupt cache ({e}), recomputing...")

        print(f"[{idx:02d}/{len(edf_files)}] {rec_id}: Processing...")
        t_start = time.time()

        try:
            # 1. Stage 0
            s0_res = run_stage0(
                edf_path=edf_path,
                summary_path=summary_path,
                patient_id="chb01",
                recording_id=rec_id,
            )

            # 2. Pipeline C
            c_out = run_pipeline_c(
                raw=s0_res.raw,
                seizure_intervals=s0_res.seizure_intervals,
                asr_cutoff=20.0,
            )
            raw_clean = c_out["final"]

            # 3. Segmentation (4s, 50% overlap, 256 Hz)
            seiz_tuples = [(iv.start_sec, iv.end_sec) for iv in s0_res.seizure_intervals]
            seg_res = segment_raw(
                raw_clean,
                seizure_intervals=seiz_tuples,
                window_sec=4.0,
                overlap=0.50,
                target_sfreq=256.0,
                include_ambiguous=True,
                patient_id="chb01",
                recording_id=rec_id,
            )

            # 4. Feature Extraction (17 ch x 26 feats = 442 numerical feats + 7 meta)
            feats_df = extract_features(
                seg_res,
                ch_names=list(raw_clean.ch_names),
                sfreq=256.0,
            )

            # Save per-recording incremental checkpoint
            feats_df.to_csv(rec_csv, index=False)
            all_dfs.append(feats_df)

            t_elapsed = time.time() - t_start
            n_ictal = int((feats_df["label"] == 1).sum())
            n_ambig = int((feats_df["label"] == -1).sum())
            print(f"  --> {rec_id} SUCCESS in {t_elapsed:.1f}s | Windows: {len(feats_df)} (ictal={n_ictal}, ambig={n_ambig}) | Saved to disk")

            # Clean memory immediately
            del s0_res, c_out, raw_clean, seg_res, feats_df
            gc.collect()

        except Exception as e:
            t_elapsed = time.time() - t_start
            err_trace = traceback.format_exc()
            print(f"  --> {rec_id} FAILED after {t_elapsed:.1f}s: {e}")
            failed_recs.append({"recording_id": rec_id, "error": str(e), "traceback": err_trace})
            gc.collect()

    if failed_recs:
        print(f"\n[WARNING] {len(failed_recs)} recording(s) failed:")
        for fr in failed_recs:
            print(f"  - {fr['recording_id']}: {fr['error']}")
        # Save failures log
        fail_df = pd.DataFrame(failed_recs)
        fail_df.to_csv(TABLES_DIR / "chb01_failed_recordings.csv", index=False)

    # Concatenate all features
    print("\nConcatenating features across all successful recordings...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    master_feat_csv = TABLES_DIR / "chb01_all_features.csv"
    combined_df.to_csv(master_feat_csv, index=False)
    print(f"Master feature matrix saved to {master_feat_csv}")
    print(f"Total Cohort Windows: {len(combined_df)}")
    print(f"Total Matrix Shape:   {combined_df.shape}")
    print(f"Cohort Label Counts:  {combined_df['label'].value_counts().to_dict()}")

    return combined_df


# ===========================================================================
# PASS 2: Unified Dimensionality Reduction (PCA)
# ===========================================================================

def run_pass2_pca(master_df: pd.DataFrame):
    print("\n" + "=" * 75)
    print("PASS 2: UNIFIED DIMENSIONALITY REDUCTION (PCA) ACROSS ALL RECORDINGS")
    print("=" * 75)

    meta_df, feats_df, feat_names = split_features_and_metadata(master_df)
    print(f"Separated {len(STANDARD_META_COLS)} metadata columns from {len(feat_names)} numerical EEG features.")

    # Fit ONE StandardScaler and ONE PCA model unsupervised across all recordings.
    # fit_pca() returns a 3-tuple: (PCAResult, StandardScaler, PCA)
    print("Fitting unified StandardScaler and PCA (unsupervised, no labels used)...")
    fit_result = fit_pca(feats_df, random_state=42)

    # Guard: ensure we correctly unpack the tuple (prevents future regression)
    if isinstance(fit_result, tuple):
        pca_res, scaler, pca_model = fit_result
    else:
        raise TypeError(
            f"fit_pca() returned {type(fit_result).__name__!r} but expected a 3-tuple "
            "(PCAResult, StandardScaler, PCA). Check dimensionality_reduction.py."
        )
    assert hasattr(pca_res, "components_for_80"), (
        "fit_pca()[0] is not a PCAResult — wrong tuple position?"
    )

    print("\n" + "-" * 50)
    print("CUMULATIVE EXPLAINED VARIANCE THRESHOLDS:")
    print(f"  80% Variance: {pca_res.components_for_80} components")
    print(f"  90% Variance: {pca_res.components_for_90} components")
    print(f"  95% Variance: {pca_res.components_for_95} components")
    print(f"  99% Variance: {pca_res.components_for_99} components")
    print("-" * 50)

    # Save variance summary
    summary_df = pca_res.summary_dataframe()
    summary_csv = TABLES_DIR / "chb01_pca_variance.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved variance metrics to {summary_csv}")

    # Project to 95% variance components.
    # transform_pca(features_df, scaler, pca, n_components) -> PC DataFrame (no meta).
    k95 = pca_res.components_for_95
    print(f"\nProjecting cohort feature matrix to {k95} components (95% variance)...")
    pc_proj_df = transform_pca(feats_df, scaler, pca_model, n_components=k95)
    # Re-attach metadata columns (reset both indices to ensure correct row alignment)
    reduced_df = pd.concat(
        [meta_df.reset_index(drop=True), pc_proj_df.reset_index(drop=True)], axis=1
    )
    reduced_csv = TABLES_DIR / "chb01_pca_reduced.csv"
    reduced_df.to_csv(reduced_csv, index=False)
    print(f"Saved reduced PCA representation ({reduced_df.shape}) to {reduced_csv}")

    # Save feature loadings for top components
    loadings_df = pca_res.get_loadings()
    loadings_csv = TABLES_DIR / "chb01_pca_loadings.csv"
    loadings_df.to_csv(loadings_csv)
    print(f"Saved feature loadings to {loadings_csv}")

    # Visualization: Scree plot & Cumulative curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    evr = pca_res.explained_variance_ratio * 100
    cum_evr = pca_res.cumulative_variance_ratio * 100

    k_scree = min(30, len(evr))
    x_scree = np.arange(1, k_scree + 1)
    ax1.bar(x_scree, evr[:k_scree], color="#1f77b4", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.plot(x_scree, evr[:k_scree], color="navy", marker="o", markersize=4, linewidth=1.5)
    ax1.set_xlabel("Principal Component", fontsize=10)
    ax1.set_ylabel("Explained Variance Ratio (%)", fontsize=10)
    ax1.set_title(f"Cohort Scree Plot (Top {k_scree} Components)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    x_all = np.arange(1, len(cum_evr) + 1)
    ax2.plot(x_all, cum_evr, color="crimson", linewidth=2.0, label="Cumulative Variance")
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
    ax2.set_title("Cohort Cumulative Explained Variance Curve", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"PCA Variance Decomposition — Entire Patient chb01 Cohort ({len(master_df)} Windows)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    scree_fig_path = PCA_FIG_DIR / "chb01_pca_variance_curves.png"
    plt.savefig(scree_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cohort variance curves to {scree_fig_path}")

    # Visualization: PC1 vs PC2 Projection
    fig, ax = plt.subplots(figsize=(10, 7))
    interictal = reduced_df[reduced_df["label"] == 0]
    ictal = reduced_df[reduced_df["label"] == 1]
    ambiguous = reduced_df[reduced_df["label"] == -1]

    ax.scatter(interictal["PC1"], interictal["PC2"], c="#1f77b4", alpha=0.25, s=15, label=f"Interictal (0) [n={len(interictal)}]")
    if len(ambiguous) > 0:
        ax.scatter(ambiguous["PC1"], ambiguous["PC2"], c="#ff7f0e", alpha=0.5, s=25, label=f"Ambiguous (-1) [n={len(ambiguous)}]")
    ax.scatter(ictal["PC1"], ictal["PC2"], c="crimson", alpha=0.9, s=55, marker="^", edgecolors="black", linewidths=0.5, label=f"Ictal (1) [n={len(ictal)}]")

    evr1 = pca_res.explained_variance_ratio[0] * 100
    evr2 = pca_res.explained_variance_ratio[1] * 100
    ax.set_xlabel(f"PC1 ({evr1:.1f}% Variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({evr2:.1f}% Variance)", fontsize=11)
    ax.set_title(f"Unsupervised PCA Projection: PC1 vs PC2 (All chb01 Recordings, N={len(reduced_df)})", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    pc_scatter_path = PCA_FIG_DIR / "chb01_pc1_vs_pc2_scatter.png"
    plt.tight_layout()
    plt.savefig(pc_scatter_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PC1 vs PC2 scatter plot to {pc_scatter_path}")

    return reduced_df, pca_res


def run_pass2_only():
    """
    Standalone entry-point for PASS 2 (PCA) only.
    Loads the already-saved chb01_all_features.csv master feature matrix
    and runs unified dimensionality reduction without touching the 42 EDFs.
    """
    print("=" * 80)
    print("DATA-SPEAKS — PASS 2 ONLY: PCA ON EXISTING chb01_all_features.csv")
    print("=" * 80)

    master_feat_csv = TABLES_DIR / "chb01_all_features.csv"
    if not master_feat_csv.exists():
        raise FileNotFoundError(
            f"Master feature matrix not found at {master_feat_csv}.\n"
            "Run the full batch (pass 1) first to generate it."
        )

    print(f"Loading master feature matrix from {master_feat_csv} ...")
    master_df = pd.read_csv(master_feat_csv)
    print(f"Loaded: {master_df.shape[0]} rows x {master_df.shape[1]} columns")

    # Sanity check
    assert master_df.shape[1] == 449, (
        f"Expected 449 columns (7 meta + 442 features), got {master_df.shape[1]}"
    )
    assert "label" in master_df.columns, "Missing 'label' column in feature matrix"
    n_nan = int(master_df.select_dtypes("number").isna().sum().sum())
    n_inf = int(np.isinf(master_df.select_dtypes("number").values).sum())
    print(f"Pre-PCA NaN count: {n_nan}  |  Inf count: {n_inf}")
    if n_nan > 0 or n_inf > 0:
        print("[WARNING] NaN/Inf detected in feature matrix — PCA may be affected.")

    t_start = time.time()
    reduced_df, pca_res = run_pass2_pca(master_df)
    elapsed = time.time() - t_start

    print("\n" + "=" * 80)
    print("PASS 2 COMPLETE — VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"  Runtime:                {elapsed:.1f}s")
    print(f"  Components for 80% var: {pca_res.components_for_80}")
    print(f"  Components for 90% var: {pca_res.components_for_90}")
    print(f"  Components for 95% var: {pca_res.components_for_95}")
    print(f"  Components for 99% var: {pca_res.components_for_99}")
    pc_cols = [c for c in reduced_df.columns if c.startswith("PC")]
    print(f"  Reduced matrix shape:   {reduced_df.shape}  ({len(pc_cols)} PC columns)")
    out_nan = int(reduced_df[pc_cols].isna().sum().sum())
    out_inf = int(np.isinf(reduced_df[pc_cols].values).sum())
    print(f"  Output NaN count:       {out_nan}")
    print(f"  Output Inf count:       {out_inf}")
    print(f"\n  Output files:")
    for f_path in [
        TABLES_DIR / "chb01_pca_variance.csv",
        TABLES_DIR / "chb01_pca_reduced.csv",
        TABLES_DIR / "chb01_pca_loadings.csv",
        PCA_FIG_DIR / "chb01_pca_variance_curves.png",
        PCA_FIG_DIR / "chb01_pc1_vs_pc2_scatter.png",
    ]:
        status = "OK" if f_path.exists() else "MISSING"
        print(f"    [{status}] {f_path.name}")
    print("=" * 80)
    return reduced_df, pca_res


def test_fit_pca_return_type():
    """
    Regression test: verifies that fit_pca returns a 3-tuple (PCAResult, scaler, pca)
    and that pca_result has the expected variance-threshold attributes.
    Run this before executing PASS 2 on real data to catch API changes early.
    """
    import pandas as pd
    import numpy as np
    from dimensionality_reduction import fit_pca, PCAResult
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA as SklearnPCA

    np.random.seed(0)
    dummy = pd.DataFrame(np.random.randn(50, 10), columns=[f"f{i}" for i in range(10)])
    result = fit_pca(dummy, random_state=0)

    assert isinstance(result, tuple) and len(result) == 3, (
        f"fit_pca must return a 3-tuple, got {type(result)}"
    )
    pca_res, scaler, pca_model = result
    assert isinstance(pca_res, PCAResult), f"First element must be PCAResult, got {type(pca_res)}"
    assert isinstance(scaler, StandardScaler), f"Second element must be StandardScaler, got {type(scaler)}"
    assert isinstance(pca_model, SklearnPCA), f"Third element must be sklearn PCA, got {type(pca_model)}"
    for attr in ["components_for_80", "components_for_90", "components_for_95", "components_for_99"]:
        assert hasattr(pca_res, attr), f"PCAResult missing attribute '{attr}'"
    print("test_fit_pca_return_type PASSED")


# PASS 3: Cohort Unsupervised Clustering
# ===========================================================================

def run_pass3_clustering(reduced_df: pd.DataFrame):
    print("\n" + "=" * 75)
    print("PASS 3: UNSUPERVISED CLUSTERING ACROSS ENTIRE COHORT")
    print("=" * 75)

    meta_df, X, y = extract_features_and_meta(reduced_df)
    print(f"Feature matrix for clustering: {X.shape}")

    metrics_list = []

    # 1. K-Means (k=2 through 6)
    print("\nRunning K-Means for k=2..6...")
    best_km_labels = None
    for k in range(2, 7):
        km_labels, km_model = run_kmeans(X, n_clusters=k, random_state=42)
        if k == 2:
            best_km_labels = km_labels

        eval_res = evaluate_clusters(X, km_labels, y, algorithm_name=f"K-Means (k={k})")
        m = eval_res.summary_dict()
        m["inertia"] = round(float(km_model.inertia_), 2)
        metrics_list.append(m)

        print(f"  k={k}: Silhouette={m['silhouette_score']:.4f} | ARI={m['adjusted_rand_index']:.4f} | NMI={m['normalized_mutual_info']:.4f}")
        if k == 2:
            print("  Contingency Table (k=2):")
            print(eval_res.contingency_table)

    # 2. DBSCAN
    print("\nRunning DBSCAN density clustering...")
    db_labels, db_model = run_dbscan(X, eps=12.0, min_samples=15)
    db_eval = evaluate_clusters(X, db_labels, y, algorithm_name="DBSCAN (eps=12.0, min_samples=15)")
    db_m = db_eval.summary_dict()
    db_m["inertia"] = np.nan
    metrics_list.append(db_m)
    print(f"  DBSCAN: Clusters={db_m['n_clusters']} | Silhouette={db_m['silhouette_score']:.4f} | ARI={db_m['adjusted_rand_index']:.4f} | NMI={db_m['normalized_mutual_info']:.4f}")
    print("  DBSCAN Contingency Table:")
    print(db_eval.contingency_table)

    # Save metrics table
    metrics_df = pd.DataFrame(metrics_list)
    clust_metrics_csv = TABLES_DIR / "chb01_clustering_metrics.csv"
    metrics_df.to_csv(clust_metrics_csv, index=False)
    print(f"\nSaved clustering metrics to {clust_metrics_csv}")

    # Save window cluster assignments
    clustered_windows = meta_df.copy()
    clustered_windows["kmeans_k2"] = best_km_labels
    clustered_windows["dbscan_cluster"] = db_labels
    clust_win_csv = TABLES_DIR / "chb01_clustered_windows.csv"
    clustered_windows.to_csv(clust_win_csv, index=False)
    print(f"Saved window cluster assignments to {clust_win_csv}")

    # Visualization: K-Means k=2 vs Ground-Truth
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    scatter1 = ax1.scatter(reduced_df["PC1"], reduced_df["PC2"], c=best_km_labels, cmap="tab10", alpha=0.3, s=20)
    ax1.set_xlabel("PC1", fontsize=10)
    ax1.set_ylabel("PC2", fontsize=10)
    ax1.set_title("Unsupervised K-Means Clusters (k=2)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(scatter1, ax=ax1, label="Cluster Assignment")

    interictal = reduced_df[y == 0]
    ictal = reduced_df[y == 1]
    ax2.scatter(interictal["PC1"], interictal["PC2"], color="#1f77b4", alpha=0.25, s=20, label="Normal / Interictal (0)")
    ax2.scatter(ictal["PC1"], ictal["PC2"], color="crimson", alpha=0.9, s=50, marker="^", edgecolors="black", linewidths=0.5, label="Seizure / Ictal (1)")
    ax2.set_xlabel("PC1", fontsize=10)
    ax2.set_ylabel("PC2", fontsize=10)
    ax2.set_title("Ground-Truth Class Labels (All chb01 Recordings)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Cohort Unsupervised K-Means vs Ground-Truth Labels", fontsize=13, fontweight="bold")
    plt.tight_layout()
    km_fig_path = CLUST_FIG_DIR / "chb01_kmeans_vs_labels.png"
    plt.savefig(km_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved K-Means cluster plot to {km_fig_path}")

    # Visualization: DBSCAN Clusters
    fig, ax = plt.subplots(figsize=(10, 7))
    unique_labels = set(db_labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k_val, col in zip(unique_labels, colors):
        if k_val == -1:
            col = [0.6, 0.6, 0.6, 0.2]
            label = "Noise / Outliers (-1)"
            marker = "x"
            size = 20
        else:
            label = f"Cluster {k_val}"
            marker = "o"
            size = 25

        class_member_mask = (db_labels == k_val)
        xy = reduced_df[class_member_mask]
        ax.scatter(xy["PC1"], xy["PC2"], color=col, label=label, marker=marker, s=size, alpha=0.4)

    ictal_df = reduced_df[y == 1]
    ax.scatter(ictal_df["PC1"], ictal_df["PC2"], facecolors="none", edgecolors="crimson", s=70, linewidth=1.5, label=f"True Seizures (n={len(ictal_df)})")
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_title("Cohort DBSCAN Density Clustering & Outlier Detection", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    db_fig_path = CLUST_FIG_DIR / "chb01_dbscan_clustering.png"
    plt.tight_layout()
    plt.savefig(db_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved DBSCAN cluster plot to {db_fig_path}")


# ===========================================================================
# PASS 3 STANDALONE: load pca_reduced.csv and cluster
# ===========================================================================

def run_pass3_only():
    """
    Standalone entry-point for PASS 3 (Clustering) only.
    Loads the already-saved chb01_pca_reduced.csv and runs K-Means + DBSCAN.
    """
    print("=" * 80)
    print("DATA-SPEAKS — PASS 3 ONLY: CLUSTERING ON EXISTING chb01_pca_reduced.csv")
    print("=" * 80)

    reduced_csv = TABLES_DIR / "chb01_pca_reduced.csv"
    if not reduced_csv.exists():
        raise FileNotFoundError(
            f"PCA-reduced matrix not found at {reduced_csv}.\n"
            "Run --pass2-only first to generate it."
        )

    print(f"Loading PCA-reduced matrix from {reduced_csv} ...")
    reduced_df = pd.read_csv(reduced_csv)
    print(f"Loaded: {reduced_df.shape[0]} rows x {reduced_df.shape[1]} columns")

    t_start = time.time()
    run_pass3_clustering(reduced_df)
    elapsed = time.time() - t_start

    print("\n" + "=" * 80)
    print("PASS 3 COMPLETE — VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"  Runtime: {elapsed:.1f}s")
    for f_path in [
        TABLES_DIR / "chb01_clustering_metrics.csv",
        TABLES_DIR / "chb01_clustered_windows.csv",
        CLUST_FIG_DIR / "chb01_kmeans_vs_labels.png",
        CLUST_FIG_DIR / "chb01_dbscan_clustering.png",
    ]:
        status = "OK" if f_path.exists() else "MISSING"
        print(f"    [{status}] {f_path.name}")
    print("=" * 80)


# ===========================================================================
# PASS 4 STANDALONE: load all_features.csv and run EDA
# ===========================================================================

def run_pass4_only():
    """
    Standalone entry-point for PASS 4 (5-Pillar EDA) only.
    Loads the already-saved chb01_all_features.csv and generates all EDA plots.
    Note: Pillar 4 & 5 re-run Pipeline C on chb01_03.edf only (small, 2.5 min).
    """
    print("=" * 80)
    print("DATA-SPEAKS — PASS 4 ONLY: 5-PILLAR EDA ON EXISTING FEATURE MATRIX")
    print("=" * 80)

    master_feat_csv = TABLES_DIR / "chb01_all_features.csv"
    if not master_feat_csv.exists():
        raise FileNotFoundError(
            f"Master feature matrix not found at {master_feat_csv}.\n"
            "Run the full batch (pass 1) first to generate it."
        )

    print(f"Loading master feature matrix from {master_feat_csv} ...")
    master_df = pd.read_csv(master_feat_csv)
    print(f"Loaded: {master_df.shape[0]} rows x {master_df.shape[1]} columns")

    t_start = time.time()
    run_pass4_eda(master_df)
    elapsed = time.time() - t_start

    print("\n" + "=" * 80)
    print("PASS 4 COMPLETE — VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"  Runtime: {elapsed:.1f}s")
    for f_path in [
        EDA_FIG_DIR / "pillar1_clinical_metadata.png",
        EDA_FIG_DIR / "pillar2_time_domain_stats.png",
        EDA_FIG_DIR / "pillar3_spectral_bands.png",
        EDA_FIG_DIR / "pillar4_spectrogram_transition.png",
        EDA_FIG_DIR / "pillar5_spatial_synchrony.png",
    ]:
        status = "OK" if f_path.exists() else "MISSING"
        print(f"    [{status}] {f_path.name}")
    print("=" * 80)


# ===========================================================================
# PASS 4: Midterm 5-Pillar EDA
# ===========================================================================

def run_pass4_eda(master_df: pd.DataFrame):
    print("\n" + "=" * 75)
    print("PASS 4: COMPREHENSIVE MIDTERM 5-PILLAR EDA (SCIENTIFICALLY AUDITED)")
    print("=" * 75)

    # -------------------------------------------------------------
    # Pillar 1: Clinical & Metadata EDA (Exact CHB01 Metadata)
    # -------------------------------------------------------------
    print("--- Generating Pillar 1: Clinical & Metadata EDA ---")
    summary_file = DATASET_DIR / "chb01" / "chb01-summary.txt"
    subject_info_file = DATASET_DIR / "SUBJECT-INFO"

    text = summary_file.read_text()
    seizures = re.findall(
        r"Seizure\s+\d*\s*Start\s+Time:\s*(\d+)\s*seconds.*?Seizure\s+\d*\s*End\s+Time:\s*(\d+)\s*seconds",
        text,
        re.DOTALL,
    )
    durations = [int(end) - int(start) for start, end in seizures]
    total_seizure_sec = float(sum(durations))
    mean_seizure_sec = float(np.mean(durations))

    # Calculate exact total recording duration across all 42 EDF files in CHB01
    metrics_csv = TABLES_DIR / "pipeline_c_chb01_metrics.csv"
    if metrics_csv.exists():
        df_m = pd.read_csv(metrics_csv)
        total_rec_sec = float(df_m["duration_sec"].sum())
    else:
        # Fallback to sum of window durations (72951 windows * 2s step = 145902s)
        total_rec_sec = float(len(master_df) * 2.0)

    seizure_pct = (total_seizure_sec / total_rec_sec) * 100.0

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1])

    # Subplot 1: Patient CHB01 Specific Class Imbalance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(["Interictal Baseline", "Ictal Events"], [total_rec_sec / 3600.0, total_seizure_sec / 3600.0], color=["#1f77b4", "crimson"], alpha=0.85)
    ax1.set_yscale("log")
    ax1.set_ylabel("Total Duration (Hours, log-scale)", fontsize=10)
    ax1.set_title(f"Patient CHB01 Class Imbalance\n(Ictal: {seizure_pct:.2f}% of Recorded Time)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, which="both")
    ax1.text(0, total_rec_sec / 3600.0, f"{total_rec_sec/3600.0:.1f} hrs\n({100.0-seizure_pct:.2f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.text(1, total_seizure_sec / 3600.0, f"{total_seizure_sec/60.0:.1f} mins\n({seizure_pct:.2f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 2: Patient CHB01 Specific Seizure Durations
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(1, len(durations) + 1), durations, color="coral", edgecolor="black", alpha=0.85)
    ax2.axhline(mean_seizure_sec, color="red", linestyle="--", label=f"CHB01 Mean: {mean_seizure_sec:.1f}s")
    ax2.set_xlabel("Seizure Incident (#)", fontsize=10)
    ax2.set_ylabel("Duration (seconds)", fontsize=10)
    ax2.set_title(f"Patient CHB01 Seizure Duration Profile ({len(durations)} Events, Total {total_seizure_sec:.0f}s)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    for i, d in enumerate(durations):
        ax2.text(i + 1, d + 1, f"{d}s", ha="center", va="bottom", fontsize=9)

    # Subplot 3: Contextual CHB-MIT Pediatric Cohort Age Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    sub_lines = [l.strip() for l in subject_info_file.read_text().splitlines() if l.strip()]
    records = []
    for line in sub_lines[1:]:
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) >= 3:
            records.append({"case": parts[0], "gender": parts[1], "age": float(parts[2])})
    demog_df = pd.DataFrame(records)
    ax3.hist(demog_df["age"], bins=10, color="teal", edgecolor="black", alpha=0.7)
    ax3.axvline(11.0, color="crimson", linewidth=2, linestyle="--", label="Subject CHB01 (11 yr F)")
    ax3.set_xlabel("Age (years)", fontsize=10)
    ax3.set_ylabel("Patient Count", fontsize=10)
    ax3.set_title("Broader CHB-MIT Pediatric Cohort\nDemographics (Contextual N=24)", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Pillar 1: Clinical & Metadata EDA (Subject CHB01 Record Profile & Cohort Context)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p1_path = EDA_FIG_DIR / "pillar1_clinical_metadata.png"
    plt.savefig(p1_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pillar 1 to {p1_path}")

    # -------------------------------------------------------------
    # Pillar 2: Time-Domain Statistical EDA
    # -------------------------------------------------------------
    print("--- Generating Pillar 2: Time-Domain Statistical EDA ---")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ictal_mask = (master_df["label"] == 1)
    interictal_mask = (master_df["label"] == 0)

    # 1. Variance Shift
    var_cols = [c for c in master_df.columns if "_var" in c]
    if var_cols:
        ictal_var = master_df.loc[ictal_mask, var_cols].mean(axis=1)
        interictal_var = master_df.loc[interictal_mask, var_cols].mean(axis=1)

        sns.kdeplot(np.log10(interictal_var + 1e-6), ax=axes[0], color="#1f77b4", fill=True, alpha=0.4, label="Interictal")
        sns.kdeplot(np.log10(ictal_var + 1e-6), ax=axes[0], color="crimson", fill=True, alpha=0.4, label="Ictal")
        axes[0].set_xlabel("Mean Channel Variance (log10)", fontsize=10)
        axes[0].set_title("Observed Variance Shift (Ictal vs Interictal)", fontsize=11, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Kurtosis Distribution
    kurt_cols = [c for c in master_df.columns if "_kurtosis" in c]
    if kurt_cols:
        ictal_kurt = master_df.loc[ictal_mask, kurt_cols].mean(axis=1)
        interictal_kurt = master_df.loc[interictal_mask, kurt_cols].mean(axis=1)

        sns.kdeplot(interictal_kurt.clip(-2, 10), ax=axes[1], color="#1f77b4", fill=True, alpha=0.4, label=f"Interictal (Mean: {interictal_kurt.mean():.2f})")
        sns.kdeplot(ictal_kurt.clip(-2, 10), ax=axes[1], color="crimson", fill=True, alpha=0.4, label=f"Ictal (Mean: {ictal_kurt.mean():.2f})")
        axes[1].set_xlabel("Mean Channel Kurtosis", fontsize=10)
        axes[1].set_title("Kurtosis Distribution Across Windows", fontsize=11, fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Peak-to-Peak (PTP) Amplitude
    ptp_cols = [c for c in master_df.columns if "_ptp" in c]
    if ptp_cols:
        ictal_ptp = master_df.loc[ictal_mask, ptp_cols].mean(axis=1)
        interictal_ptp = master_df.loc[interictal_mask, ptp_cols].mean(axis=1)

        sns.kdeplot(interictal_ptp, ax=axes[2], color="#1f77b4", fill=True, alpha=0.4, label="Interictal")
        sns.kdeplot(ictal_ptp, ax=axes[2], color="crimson", fill=True, alpha=0.4, label="Ictal")
        axes[2].set_xlabel("Mean Peak-to-Peak Amplitude", fontsize=10)
        axes[2].set_title("Peak-to-Peak Amplitude Distribution", fontsize=11, fontweight="bold")
        axes[2].legend()
        axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Pillar 2: Time-Domain Statistical Moments (Observed Feature Distributions for Patient CHB01)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p2_path = EDA_FIG_DIR / "pillar2_time_domain_stats.png"
    plt.savefig(p2_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pillar 2 to {p2_path}")

    # -------------------------------------------------------------
    # Pillar 3: Frequency-Domain / Spectral EDA
    # -------------------------------------------------------------
    print("--- Generating Pillar 3: Frequency-Domain Spectral EDA ---")
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    band_records = []

    for b in bands:
        b_cols = [c for c in master_df.columns if f"_{b}_rel" in c]
        if b_cols:
            mean_inter = master_df.loc[interictal_mask, b_cols].mean().mean()
            mean_ictal = master_df.loc[ictal_mask, b_cols].mean().mean()
            band_records.append({"Band": b.capitalize(), "Interictal": mean_inter, "Ictal": mean_ictal})

    band_df = pd.DataFrame(band_records)
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(band_df))
    width = 0.35

    ax.bar(x_pos - width / 2, band_df["Interictal"], width, label="Interictal Baseline", color="#1f77b4", alpha=0.85)
    ax.bar(x_pos + width / 2, band_df["Ictal"], width, label="Ictal Windows", color="crimson", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(band_df["Band"], fontsize=11)
    ax.set_ylabel("Mean Relative Band Power", fontsize=11)
    ax.set_title("Pillar 3: Observed Relative Spectral Band Power Distribution (Patient CHB01 Record Set)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    for i in x_pos:
        diff = band_df["Ictal"][i] - band_df["Interictal"][i]
        sym = "▲" if diff > 0 else "▼"
        col = "darkred" if diff > 0 else "navy"
        ax.text(i, max(band_df["Interictal"][i], band_df["Ictal"][i]) + 0.015, f"{sym} {abs(diff):.3f}", ha="center", fontsize=9, fontweight="bold", color=col)

    plt.tight_layout()
    p3_path = EDA_FIG_DIR / "pillar3_spectral_bands.png"
    plt.savefig(p3_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pillar 3 to {p3_path}")

    # -------------------------------------------------------------
    # Pillar 4: Time-Frequency Spectrogram
    # -------------------------------------------------------------
    print("--- Generating Pillar 4: Time-Frequency Spectrogram ---")
    edf_03 = DATASET_DIR / "chb01" / "chb01_03.edf"
    sum_03 = DATASET_DIR / "chb01" / "chb01-summary.txt"
    s0_03 = run_stage0(edf_03, sum_03, "chb01", "chb01_03")
    c_03 = run_pipeline_c(s0_03.raw, s0_03.seizure_intervals, asr_cutoff=20.0)
    clean_03 = c_03["final"]

    sfreq = 256.0
    start_sample = int(2950 * sfreq)
    stop_sample = int(3080 * sfreq)
    target_ch = "FP1-F7" if "FP1-F7" in clean_03.ch_names else clean_03.ch_names[0]
    ch_idx = clean_03.ch_names.index(target_ch)
    data_segment = clean_03.get_data(picks=[ch_idx], start=start_sample, stop=stop_sample)[0]

    f, t_spec, Sxx = spectrogram(data_segment, fs=sfreq, nperseg=512, noverlap=384)
    t_global = t_spec + 2950

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.pcolormesh(t_global, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="inferno")
    ax.set_ylim(0, 40)
    ax.axvline(2996, color="cyan", linestyle="--", linewidth=2, label="Annotated Seizure Start (2996s)")
    ax.axvline(3036, color="cyan", linestyle="-.", linewidth=2, label="Annotated Seizure End (3036s)")
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    ax.set_xlabel("Time (seconds in chb01_03)", fontsize=11)
    ax.set_title(f"Pillar 4: Short-Time Fourier Transform (STFT) Spectrogram Transition ({target_ch} Channel, chb01_03)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Power Spectral Density (dB/Hz)", fontsize=10)

    plt.tight_layout()
    p4_path = EDA_FIG_DIR / "pillar4_spectrogram_transition.png"
    plt.savefig(p4_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pillar 4 to {p4_path}")

    # -------------------------------------------------------------
    # Pillar 5: Spatial & Cross-Channel Synchrony
    # -------------------------------------------------------------
    print("--- Generating Pillar 5: Cross-Channel Synchrony ---")
    inter_data = clean_03.get_data(start=int(1000 * sfreq), stop=int(1040 * sfreq))
    ictal_data = clean_03.get_data(start=int(2996 * sfreq), stop=int(3036 * sfreq))

    corr_inter = np.corrcoef(inter_data)
    corr_ictal = np.corrcoef(ictal_data)

    triu_idx = np.triu_indices(17, k=1)
    mean_corr_inter = float(np.mean(corr_inter[triu_idx]))
    mean_corr_ictal = float(np.mean(corr_ictal[triu_idx]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(corr_inter, ax=ax1, cmap="vlag", vmin=-1, vmax=1, xticklabels=clean_03.ch_names, yticklabels=clean_03.ch_names, cbar=False)
    ax1.set_title(f"Interictal Cross-Channel Correlation\n(Mean Off-Diagonal Pearson r = {mean_corr_inter:.3f})", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="x", rotation=90, labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)

    im2 = sns.heatmap(corr_ictal, ax=ax2, cmap="vlag", vmin=-1, vmax=1, xticklabels=clean_03.ch_names, yticklabels=clean_03.ch_names)
    ax2.set_title(f"Ictal Cross-Channel Correlation\n(Mean Off-Diagonal Pearson r = {mean_corr_ictal:.3f})", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="x", rotation=90, labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)

    plt.suptitle("Pillar 5: Cross-Channel Pearson Correlation Matrices (Cross-Channel Synchrony Proxy)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    p5_path = EDA_FIG_DIR / "pillar5_spatial_synchrony.png"
    plt.savefig(p5_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pillar 5 to {p5_path}")

    # Free memory
    for var in ["s0_03", "c_03", "clean_03"]:
        if var in locals():
            del locals()[var]
    gc.collect()




# ===========================================================================
# MASTER PIPELINE ENTRY POINT
# ===========================================================================

def main():
    print("=" * 80)
    print("DATA-SPEAKS — FULL PIPELINE EXECUTION FOR ALL 42 chb01 RECORDINGS")
    print("=" * 80)

    chb01_dir = DATASET_DIR / "chb01"
    summary_path = chb01_dir / "chb01-summary.txt"

    if not chb01_dir.exists():
        print(f"Error: Dataset directory not found at {chb01_dir}")
        sys.exit(1)

    edf_files = sorted(list(chb01_dir.glob("*.edf")))
    print(f"Found {len(edf_files)} EDF files in {chb01_dir}")

    t_total_start = time.time()

    # Pass 1: Extraction
    master_features_df = run_pass1_feature_extraction(edf_files, summary_path)

    # Pass 2: PCA
    reduced_df, pca_res = run_pass2_pca(master_features_df)

    # Pass 3: Clustering
    run_pass3_clustering(reduced_df)

    # Pass 4: 5-Pillar EDA
    run_pass4_eda(master_features_df)

    t_total_elapsed = time.time() - t_total_start
    print("\n" + "=" * 80)
    print(f"ALL 42 RECORDINGS COMPLETED IN {t_total_elapsed / 60:.2f} MINUTES")
    print("=" * 80)


if __name__ == "__main__":
    # --pass2-only : resume from saved chb01_all_features.csv, skip EDF re-processing
    # --pass3-only : run clustering on existing chb01_pca_reduced.csv
    # --pass4-only : run 5-pillar EDA on existing chb01_all_features.csv
    # --test       : run regression test for fit_pca return type, then exit
    if "--test" in sys.argv:
        test_fit_pca_return_type()
        print("All regression tests passed.")
        sys.exit(0)
    elif "--pass2-only" in sys.argv:
        run_pass2_only()
    elif "--pass3-only" in sys.argv:
        run_pass3_only()
    elif "--pass4-only" in sys.argv:
        run_pass4_only()
    else:
        main()

