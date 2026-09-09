import pandas as pd
import numpy as np


PATH = (
    "results/benchmarks/"
    "features_pipeline_B_chb02_full.csv"
)


METADATA_COLUMNS = [
    "recording_id",
    "window_index",
    "start_sec",
    "end_sec",
    "label",
]


df = pd.read_csv(PATH)


print("Shape:", df.shape)

print("\nRecordings:")
print(df["recording_id"].nunique())

print("\nWindows per recording:")
print(
    df["recording_id"]
    .value_counts()
    .sort_index()
)

print("\nLabels:")
print(
    df["label"]
    .value_counts()
    .sort_index()
)


features = df.drop(
    columns=METADATA_COLUMNS
)


print("\nFeature columns:")
print(features.shape[1])


print("\nMissing values:")
print(
    features.isna().sum().sum()
)


print("\nInfinite values:")
print(
    np.isinf(
        features.to_numpy()
    ).sum()
)


print("\nZero-variance features:")
print(
    (features.std() == 0).sum()
)


print("\nFeature names unique:")
print(
    len(features.columns)
    == len(set(features.columns))
)


assert df["recording_id"].nunique() == 36
assert features.shape[1] == 598
assert df["label"].isin([0, 1]).all()
assert features.isna().sum().sum() == 0
assert np.isinf(features.to_numpy()).sum() == 0
assert (features.std() == 0).sum() == 0
assert len(features.columns) == len(set(features.columns))


print("\nFull Pipeline B feature dataset validation passed.")
