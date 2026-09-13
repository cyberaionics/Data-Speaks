"""
Stage 06: CHB01 classification runner.

Runs four lightweight classifiers with leakage-safe GroupKFold:
Linear SVM, Logistic Regression, Random Forest, KNN.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipeline" / "06_classification"))

from classification import (
    load_and_prepare,
    run_grouped_cv,
    select_best_model,
    fit_final_model,
    N_FOLDS,
    N_PCA_COMPONENTS,
    RANDOM_STATE,
)

FEATURES_CSV = REPO_ROOT / "results" / "tables" / "chb01_all_features.csv"
TABLES_DIR = REPO_ROOT / "results" / "tables"
FIG_DIR = REPO_ROOT / "results" / "figures" / "classification"
MODELS_DIR = REPO_ROOT / "models" / "chb01_final_model"

for d in [TABLES_DIR, FIG_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def plot_model_comparison(summary_df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("pr_auc_mean", "PR-AUC"),
        ("f1_mean", "F1"),
        ("recall_sensitivity_mean", "Recall/Sensitivity"),
    ]

    models = summary_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, label) in enumerate(metrics):
        vals = summary_df[metric].fillna(0).to_numpy()
        errs = summary_df[metric.replace("_mean", "_std")].fillna(0).to_numpy()

        bars = ax.bar(
            x + i * width,
            vals,
            width,
            label=label,
            yerr=errs,
            capsize=4,
        )

        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (GroupKFold mean ± std)")
    ax.set_title(
        "Stage 06: CHB01 Classifier Comparison\n"
        "Best model selected by PR-AUC"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(fold_df: pd.DataFrame, out_path: Path) -> None:
    models = fold_df["model"].unique()
    fig, axes = plt.subplots(
        1, len(models), figsize=(4 * len(models), 4)
    )

    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        sub = fold_df[fold_df["model"] == model_name]

        cm = np.array([
            [sub["tn"].sum(), sub["fp"].sum()],
            [sub["fn"].sum(), sub["tp"].sum()],
        ])

        ax.imshow(cm)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: 0", "Pred: 1"])
        ax.set_yticklabels(["True: 0", "True: 1"])
        ax.set_title(model_name)

        for (r, c), value in np.ndenumerate(cm):
            ax.text(c, r, str(value), ha="center", va="center")

    fig.suptitle("Aggregated Confusion Matrices — GroupKFold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    t0 = time.time()

    print("=" * 70)
    print("DATA-SPEAKS — STAGE 06: CLASSIFICATION (CHB01)")
    print("=" * 70)
    print(f"Features: {FEATURES_CSV}")
    print(f"GroupKFold: {N_FOLDS} | PCA: {N_PCA_COMPONENTS}")

    meta_df, X, feature_cols = load_and_prepare(str(FEATURES_CSV))

    print("\nRunning classification...")
    fold_df, summary_df = run_grouped_cv(
        meta_df,
        X,
        feature_cols,
        n_folds=N_FOLDS,
        n_pca=N_PCA_COMPONENTS,
    )

    fold_out = TABLES_DIR / "chb01_classification_fold_metrics.csv"
    summary_out = TABLES_DIR / "chb01_classification_metrics.csv"

    fold_df.to_csv(fold_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print("\nSUMMARY")
    print(summary_df[
        [
            "model",
            "pr_auc_mean", "pr_auc_std",
            "f1_mean", "f1_std",
            "recall_sensitivity_mean",
            "roc_auc_mean",
            "accuracy_mean",
        ]
    ].to_string(index=False))

    plot_model_comparison(
        summary_df,
        FIG_DIR / "model_comparison.png",
    )
    plot_confusion_matrices(
        fold_df,
        FIG_DIR / "confusion_matrices.png",
    )

    best_model = select_best_model(summary_df)

    # Hold out chb01_03 so inference is genuinely on an unseen recording.
    holdout = "chb01_03"
    print(f"\nFitting final {best_model} model excluding {holdout}...")

    scaler, pca, clf, saved_features = fit_final_model(
        meta_df,
        X,
        feature_cols,
        clf_name=best_model,
        exclude_recording=holdout,
        n_pca=N_PCA_COMPONENTS,
    )

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(pca, MODELS_DIR / "pca.pkl")
    joblib.dump(clf, MODELS_DIR / "classifier.pkl")
    joblib.dump(saved_features, MODELS_DIR / "feature_cols.pkl")

    metadata = {
        "model_name": best_model,
        "n_pca_components": int(pca.n_components_),
        "n_feature_cols": len(saved_features),
        "trained_on": "chb01 all recordings except chb01_03",
        "inference_validation_holdout": holdout,
        "random_state": RANDOM_STATE,
        "labels": {"0": "interictal", "1": "ictal"},
        "selection_metric": "pr_auc",
        "score_type": (
            "decision_function for LinearSVM; "
            "probability for probabilistic classifiers"
        ),
    }

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved model artifacts:")
    for name in [
        "scaler.pkl",
        "pca.pkl",
        "classifier.pkl",
        "feature_cols.pkl",
        "metadata.json",
    ]:
        print(f"  [{'OK' if (MODELS_DIR / name).exists() else 'MISSING'}] {name}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    print("=" * 70)
    print("STAGE 06 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
