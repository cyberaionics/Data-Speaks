import numpy as np
import pandas as pd


METADATA_COLUMNS = ["recording_id", "window_index", "start_sec", "end_sec", "label"]
d = pd.read_csv("results/benchmarks/features_pipeline_D_chb02_full.csv")
b = pd.read_csv("results/benchmarks/features_pipeline_B_chb02_full.csv")
features = d.drop(columns=METADATA_COLUMNS)
assert d["recording_id"].nunique() == 36
assert features.shape[1] == 598
assert d["label"].isin([0, 1]).all()
assert d.isna().sum().sum() == 0
assert np.isinf(features.to_numpy()).sum() == 0
assert (features.std() == 0).sum() == 0
assert len(features.columns) == len(set(features.columns))
assert list(d.columns) == list(b.columns)
assert d[METADATA_COLUMNS].equals(b[METADATA_COLUMNS])
print("Full Pipeline D feature dataset validation passed.")
print("Shape:", d.shape)
print("Labels:", d["label"].value_counts().sort_index().to_dict())
print("Feature columns and window/label metadata match Pipeline B.")
