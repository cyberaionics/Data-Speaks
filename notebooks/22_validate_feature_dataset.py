import pandas as pd
import numpy as np


path = "results/benchmarks/features_pipeline_A_chb02_16.csv"

df = pd.read_csv(path)

print("Shape:", df.shape)

print("\nLabels:")
print(df["label"].value_counts().sort_index())

print("\nMissing values:")
print(df.isna().sum().sum())

print("\nInfinite values:")
numeric = df.select_dtypes(include=[np.number])
print(np.isinf(numeric.to_numpy()).sum())

print("\nFeature statistics:")
features = df.drop(
    columns=[
        "recording_id",
        "window_index",
        "start_sec",
        "end_sec",
        "label",
    ]
)

print("Feature columns:", features.shape[1])

stds = features.std()

print("Zero-variance features:", (stds == 0).sum())

print("Features with NaN:", features.isna().any().sum())

print("\nClass distribution:")
print(
    df["label"]
    .value_counts(normalize=True)
    .sort_index()
)

print("\nValidation passed.")
