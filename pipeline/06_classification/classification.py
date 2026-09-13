"""
Stage 06: Classification — Data-Speaks EEG Project.

Fast, leakage-safe CHB01 classification:
- Linear SVM
- Logistic Regression
- Random Forest
- KNN

Important:
- Ambiguous windows (label=-1) are excluded.
- GroupKFold is done by recording_id.
- StandardScaler and PCA are fitted inside each training fold only.
- PR-AUC is used for model selection because ictal windows are highly imbalanced.
- Linear SVM uses decision_function scores (not probabilities).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

META_COLS = [
    "patient_id", "recording_id", "window_id",
    "start_sec", "end_sec", "duration_sec", "label",
]

N_PCA_COMPONENTS = 110
N_FOLDS = 5
RANDOM_STATE = 42


def build_classifiers() -> dict:
    """Return four practical classifiers."""
    return {
        "LinearSVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=5000,
        ),
        "LR": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "RF": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            metric="euclidean",
            n_jobs=-1,
        ),
    }


def load_and_prepare(
    features_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load features and remove ambiguous windows."""
    df = pd.read_csv(features_csv)
    original_len = len(df)

    df = df[df["label"] != -1].reset_index(drop=True)
    excluded = original_len - len(df)

    print(
        f"Loaded {original_len} windows. Excluded {excluded} ambiguous. "
        f"Remaining: {len(df)} | Ictal: {(df['label'] == 1).sum()} | "
        f"Interictal: {(df['label'] == 0).sum()}"
    )

    present_meta = [c for c in META_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in present_meta]

    meta_df = df[present_meta].copy()
    X = df[feature_cols].copy()

    if len(feature_cols) != 442:
        raise ValueError(f"Expected 442 EEG features, got {len(feature_cols)}")
    if X.isnull().any().any():
        raise ValueError("NaN values found in feature matrix")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Inf values found in feature matrix")

    return meta_df, X, feature_cols


def _score_model(clf, X_test: np.ndarray) -> np.ndarray:
    """Return a continuous ictal score for ROC-AUC/PR-AUC."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X_test)[:, 1]
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X_test)
    return clf.predict(X_test).astype(float)


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    fold: int,
    clf_name: str,
    recording_ids: list[str],
) -> dict:
    """Compute metrics for one held-out fold."""
    n_test = len(y_true)
    n_ictal = int(y_true.sum())
    n_interictal = int((y_true == 0).sum())

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    acc = accuracy_score(y_true, y_pred)

    if n_ictal == 0:
        prec = rec = f1 = roc_auc = pr_auc = np.nan
    else:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if y_score is not None:
            roc_auc = roc_auc_score(y_true, y_score)
            pr_auc = average_precision_score(y_true, y_score)
        else:
            roc_auc = pr_auc = np.nan

    return {
        "fold": fold,
        "model": clf_name,
        "test_recordings": ";".join(sorted(set(recording_ids))),
        "n_test_windows": n_test,
        "n_ictal_windows": n_ictal,
        "n_interictal_windows": n_interictal,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4) if not np.isnan(prec) else np.nan,
        "recall_sensitivity": round(rec, 4) if not np.isnan(rec) else np.nan,
        "specificity": round(specificity, 4) if not np.isnan(specificity) else np.nan,
        "f1": round(f1, 4) if not np.isnan(f1) else np.nan,
        "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else np.nan,
        "pr_auc": round(pr_auc, 4) if not np.isnan(pr_auc) else np.nan,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def run_grouped_cv(
    meta_df: pd.DataFrame,
    X: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int = N_FOLDS,
    n_pca: int = N_PCA_COMPONENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leakage-safe GroupKFold classification.

    Every recording_id stays entirely inside either train or test.
    """
    del feature_cols  # kept in API for compatibility

    y = meta_df["label"].to_numpy()
    groups = meta_df["recording_id"].to_numpy()
    X_np = X.to_numpy()

    classifiers = build_classifiers()
    gkf = GroupKFold(n_splits=n_folds)
    all_fold_metrics: list[dict] = []

    for clf_name, clf in classifiers.items():
        print(f"\n{'=' * 60}")
        print(f"Classifier: {clf_name}")
        print(f"{'=' * 60}")

        for fold_idx, (train_idx, test_idx) in enumerate(
            gkf.split(X_np, y, groups), 1
        ):
            train_recs = list(set(groups[train_idx]))
            test_recs = list(set(groups[test_idx]))

            if set(train_recs) & set(test_recs):
                raise RuntimeError("LEAKAGE: recording appears in train and test")

            X_train, X_test = X_np[train_idx], X_np[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if y_train.sum() == 0:
                print(f"  Fold {fold_idx}: skipped — no ictal training windows.")
                continue

            # Fit preprocessing ONLY on training data.
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            actual_n_pca = min(
                n_pca,
                X_train_scaled.shape[1],
                X_train_scaled.shape[0] - 1,
            )
            pca = SklearnPCA(
                n_components=actual_n_pca,
                random_state=RANDOM_STATE,
            )
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            clf.fit(X_train_pca, y_train)
            y_pred = clf.predict(X_test_pca)
            y_score = _score_model(clf, X_test_pca)

            print(
                f"  Fold {fold_idx}: test={len(y_test)} "
                f"(ictal={int(y_test.sum())}) | "
                f"recs={','.join(sorted(test_recs)[:3])}"
                f"{'...' if len(test_recs) > 3 else ''}"
            )

            all_fold_metrics.append(
                compute_fold_metrics(
                    y_test,
                    y_pred,
                    y_score,
                    fold_idx,
                    clf_name,
                    test_recs,
                )
            )

    fold_df = pd.DataFrame(all_fold_metrics)

    metric_cols = [
        "accuracy", "precision", "recall_sensitivity",
        "specificity", "f1", "roc_auc", "pr_auc",
    ]
    summary_rows = []

    for clf_name in classifiers:
        sub = fold_df[fold_df["model"] == clf_name]
        row = {"model": clf_name}

        for metric in metric_cols:
            vals = sub[metric].dropna()
            row[f"{metric}_mean"] = round(vals.mean(), 4) if len(vals) else np.nan
            row[f"{metric}_std"] = round(vals.std(), 4) if len(vals) > 1 else np.nan

        row["n_folds_with_positives"] = int(
            sub["n_ictal_windows"].gt(0).sum()
        )
        summary_rows.append(row)

    return fold_df, pd.DataFrame(summary_rows)


def select_best_model(summary_df: pd.DataFrame) -> str:
    """Select the model with the highest mean PR-AUC."""
    sub = summary_df.dropna(subset=["pr_auc_mean"])

    if len(sub) == 0:
        sub = summary_df.dropna(subset=["f1_mean"])
        metric = "f1_mean"
    else:
        metric = "pr_auc_mean"

    if len(sub) == 0:
        raise RuntimeError("No valid classifier results were produced.")

    best_idx = sub[metric].idxmax()
    best_model = str(sub.loc[best_idx, "model"])
    print(
        f"\nBest model: {best_model} "
        f"({metric}={sub.loc[best_idx, metric]:.4f})"
    )
    return best_model


def fit_final_model(
    meta_df: pd.DataFrame,
    X: pd.DataFrame,
    feature_cols: list[str],
    clf_name: str,
    exclude_recording: Optional[str] = None,
    n_pca: int = N_PCA_COMPONENTS,
):
    """Fit scaler + PCA + selected classifier for later inference."""
    y = meta_df["label"].to_numpy()
    groups = meta_df["recording_id"].to_numpy()
    X_np = X.to_numpy()

    if exclude_recording:
        mask = groups != exclude_recording
        X_train_np = X_np[mask]
        y_train = y[mask]
        print(
            f"Final model: training on {mask.sum()} windows "
            f"(excluded {exclude_recording})"
        )
    else:
        X_train_np = X_np
        y_train = y
        print(f"Final model: training on {len(y_train)} windows")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_np)

    actual_n_pca = min(n_pca, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca = SklearnPCA(
        n_components=actual_n_pca,
        random_state=RANDOM_STATE,
    )
    X_pca = pca.fit_transform(X_scaled)

    classifiers = build_classifiers()
    if clf_name not in classifiers:
        raise ValueError(f"Unknown classifier: {clf_name}")

    clf = classifiers[clf_name]
    clf.fit(X_pca, y_train)

    print(
        f"Final model fitted: {clf.__class__.__name__}, "
        f"PCA={actual_n_pca}"
    )
    return scaler, pca, clf, feature_cols
