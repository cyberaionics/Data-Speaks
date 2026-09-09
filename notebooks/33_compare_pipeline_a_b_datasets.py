import pandas as pd


A_PATH = "results/benchmarks/features_pipeline_A_chb02.csv"
B_PATH = "results/benchmarks/features_pipeline_B_chb02.csv"


a = pd.read_csv(A_PATH)
b = pd.read_csv(B_PATH)


metadata_columns = [
    "recording_id",
    "window_index",
    "start_sec",
    "end_sec",
    "label",
]


a_features = [
    col for col in a.columns
    if col not in metadata_columns
]

b_features = [
    col for col in b.columns
    if col not in metadata_columns
]


print("Pipeline A shape:", a.shape)
print("Pipeline B shape:", b.shape)

print("\nA feature count:", len(a_features))
print("B feature count:", len(b_features))

print("\nFeature columns identical:", a_features == b_features)

assert len(a_features) == 598
assert len(b_features) == 598
assert a_features == b_features


a_keys = a[metadata_columns].sort_values(
    ["recording_id", "window_index"]
).reset_index(drop=True)

b_keys = b[metadata_columns].sort_values(
    ["recording_id", "window_index"]
).reset_index(drop=True)


print(
    "Window/label metadata identical:",
    a_keys.equals(b_keys),
)

assert a_keys.equals(b_keys)

print("\nPipeline A/B dataset alignment test passed.")
