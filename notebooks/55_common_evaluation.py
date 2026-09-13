import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results/benchmarks"
DATASETS = {
    name: RESULTS_DIR / f"features_pipeline_{name}_chb02_full.csv"
    for name in "ABCD"
}
METADATA_COLUMNS = ["recording_id", "window_index", "start_sec", "end_sec", "label"]
SEED = 42


def load_datasets():
    datasets = {name: pd.read_csv(path) for name, path in DATASETS.items()}
    reference = datasets["A"]
    for name, dataset in datasets.items():
        if list(dataset.columns) != list(reference.columns):
            raise RuntimeError(f"Feature schema mismatch for Pipeline {name}")
        if not dataset[METADATA_COLUMNS].equals(reference[METADATA_COLUMNS]):
            raise RuntimeError(f"Window metadata mismatch for Pipeline {name}")
    return datasets


def classification_result(dataset, features):
    labels = dataset["label"].to_numpy()
    recordings = dataset["recording_id"].to_numpy()
    unique_recordings = sorted(np.unique(recordings))
    positive_recordings = [recording for recording in unique_recordings if ((recordings == recording) & (labels == 1)).any()]
    if len(set(dataset["recording_id"])) == 1:
        return {
            "status": "unavailable",
            "reason": "Only one patient and one patient identifier are available; patient-wise generalization cannot be estimated.",
            "patient_count": 1,
        }
    if len(positive_recordings) < 2:
        return {
            "status": "unavailable",
            "reason": "Fewer than two recordings contain ictal windows for a defensible grouped evaluation.",
            "patient_count": 1,
        }

    test_recording = positive_recordings[-1]
    train_mask = recordings != test_recording
    test_mask = ~train_mask
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        ]
    )
    model.fit(features[train_mask], labels[train_mask])
    predictions = model.predict(features[test_mask])
    probabilities = model.predict_proba(features[test_mask])[:, 1]
    result = {
        "status": "preliminary_recording_wise",
        "test_recording": test_recording,
        "accuracy": accuracy_score(labels[test_mask], predictions),
        "balanced_accuracy": balanced_accuracy_score(labels[test_mask], predictions),
        "precision": precision_score(labels[test_mask], predictions, zero_division=0),
        "recall": recall_score(labels[test_mask], predictions, zero_division=0),
        "f1": f1_score(labels[test_mask], predictions, zero_division=0),
    }
    result["roc_auc"] = roc_auc_score(labels[test_mask], probabilities) if len(np.unique(labels[test_mask])) == 2 else None
    return result


def main():
    datasets = load_datasets()
    summary = {
        "seed": SEED,
        "patient_wise_status": "unavailable: CHB02 is one patient; no patient-wise split is possible",
        "pipelines": {},
    }
    comparison_rows = []
    for name, dataset in datasets.items():
        feature_frame = dataset.drop(columns=METADATA_COLUMNS)
        features = feature_frame.to_numpy(dtype=float)
        pca = PCA(n_components=0.95, svd_solver="full", random_state=SEED)
        reduced = pca.fit_transform(StandardScaler().fit_transform(features))
        sample = reduced[:: max(1, len(reduced) // 10000)]
        clusters = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit_predict(sample)
        clustering = {
            "algorithm": "KMeans",
            "n_clusters": 2,
            "sample_count": len(sample),
            "silhouette": float(silhouette_score(sample, clusters)),
        }
        classification = classification_result(dataset, features)
        summary["pipelines"][name] = {
            "rows": len(dataset),
            "recordings": int(dataset["recording_id"].nunique()),
            "ictal": int((dataset["label"] == 1).sum()),
            "interictal": int((dataset["label"] == 0).sum()),
            "ambiguous_excluded": 0,
            "pca": {
                "algorithm": "standardized PCA",
                "variance_target": 0.95,
                "components": int(pca.n_components_),
                "explained_variance": float(pca.explained_variance_ratio_.sum()),
            },
            "clustering": clustering,
            "classification": classification,
        }
        comparison_rows.append({
            "pipeline": name,
            "rows": len(dataset),
            "recordings": dataset["recording_id"].nunique(),
            "pca_components": pca.n_components_,
            "pca_explained_variance": pca.explained_variance_ratio_.sum(),
            "silhouette": clustering["silhouette"],
            "classification_status": classification["status"],
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "common_evaluation.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.DataFrame(comparison_rows).to_csv(RESULTS_DIR / "common_evaluation_summary.csv", index=False)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()