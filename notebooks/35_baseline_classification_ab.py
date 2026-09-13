import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


A_PATH = "results/benchmarks/features_pipeline_A_chb02.csv"
B_PATH = "results/benchmarks/features_pipeline_B_chb02.csv"


METADATA_COLUMNS = [
    "recording_id",
    "window_index",
    "start_sec",
    "end_sec",
    "label",
]


def load_dataset(path):
    df = pd.read_csv(path)

    feature_columns = [
        col
        for col in df.columns
        if col not in METADATA_COLUMNS
    ]

    X = df[feature_columns]
    y = df["label"]

    return df, X, y


def run_experiment(name, df, X, y):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # -----------------------------------------------------
    # Preliminary time-based split
    #
    # Use chb02_16 only because it contains both classes.
    #
    # Training:
    #   first 70% of valid windows
    #
    # Testing:
    #   final 30% of valid windows
    #
    # The same exact split is used for A and B.
    # -----------------------------------------------------

    recording_mask = df["recording_id"] == "chb02_16"

    recording_df = (
        df.loc[recording_mask]
        .sort_values("start_sec")
        .reset_index(drop=True)
    )

    split_index = int(len(recording_df) * 0.70)

    train_df = recording_df.iloc[:split_index]
    test_df = recording_df.iloc[split_index:]

    train_indices = train_df.index
    test_indices = test_df.index

    X_recording = X.loc[
        recording_mask
    ].reset_index(drop=True)

    y_recording = y.loc[
        recording_mask
    ].reset_index(drop=True)

    X_train = X_recording.loc[train_indices]
    X_test = X_recording.loc[test_indices]

    y_train = y_recording.loc[train_indices]
    y_test = y_recording.loc[test_indices]

    print("Recording:", "chb02_16")
    print("Training windows:", len(X_train))
    print("Testing windows:", len(X_test))

    print(
        "Training labels:",
        y_train.value_counts().sort_index().to_dict(),
    )

    print(
        "Testing labels:",
        y_test.value_counts().sort_index().to_dict(),
    )

    # Make sure the split actually contains both classes.
    if y_train.nunique() < 2:
        raise RuntimeError(
            "Training split contains only one class. "
            "Choose a split containing both ictal and interictal windows."
        )

    if y_test.nunique() < 2:
        raise RuntimeError(
            "Testing split contains only one class. "
            "Choose a split containing both ictal and interictal windows."
        )

    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print("\nResults:")

    print(
        f"Accuracy:          {accuracy:.4f}"
    )

    print(
        f"Balanced accuracy: {balanced_accuracy:.4f}"
    )

    print(
        f"F1 score:          {f1:.4f}"
    )

    print(
        f"ROC-AUC:           {roc_auc:.4f}"
    )

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return {
        "pipeline": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

a_df, a_X, a_y = load_dataset(A_PATH)
b_df, b_X, b_y = load_dataset(B_PATH)


# ---------------------------------------------------------
# Verify dataset alignment
# ---------------------------------------------------------

assert a_df[METADATA_COLUMNS].equals(
    b_df[METADATA_COLUMNS]
)

assert list(a_X.columns) == list(b_X.columns)

print("A/B alignment verified.")


# ---------------------------------------------------------
# Run identical baseline
# ---------------------------------------------------------

a_result = run_experiment(
    "Pipeline A",
    a_df,
    a_X,
    a_y,
)

b_result = run_experiment(
    "Pipeline B",
    b_df,
    b_X,
    b_y,
)


# ---------------------------------------------------------
# Compare
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Pipeline A vs Pipeline B")
print("=" * 60)

print(
    f"Balanced accuracy: "
    f"A={a_result['balanced_accuracy']:.4f}, "
    f"B={b_result['balanced_accuracy']:.4f}"
)

print(
    f"F1 score:          "
    f"A={a_result['f1']:.4f}, "
    f"B={b_result['f1']:.4f}"
)

print(
    f"ROC-AUC:           "
    f"A={a_result['roc_auc']:.4f}, "
    f"B={b_result['roc_auc']:.4f}"
)

print(
    "\nPreliminary chb02 recording-wise "
    "time-based comparison complete."
)
