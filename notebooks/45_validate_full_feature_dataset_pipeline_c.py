import numpy as np
import pandas as pd


PATH = "results/benchmarks/features_pipeline_C_chb02_full.csv"
METADATA_COLUMNS = ["recording_id", "window_index", "start_sec", "end_sec", "label"]

df = pd.read_csv(PATH)
features = df.drop(columns=METADATA_COLUMNS)

assert df["recording_id"].nunique() == 36
assert features.shape[1] == 598
assert df["label"].isin([0, 1]).all()
assert df.isna().sum().sum() == 0
assert np.isinf(features.to_numpy()).sum() == 0
assert (features.std() == 0).sum() == 0
assert len(features.columns) == len(set(features.columns))

print("Full Pipeline C feature dataset validation passed.")
print("Shape:", df.shape)
print("Labels:", df["label"].value_counts().sort_index().to_dict())
