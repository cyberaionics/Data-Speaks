import pandas as pd


PATH = "results/benchmarks/features_pipeline_A_chb02.csv"

df = pd.read_csv(PATH)

df = df[
    df["recording_id"] == "chb02_16"
].sort_values("start_sec").reset_index(drop=True)


# ---------------------------------------------------------
# Temporal test block
# ---------------------------------------------------------

TEST_START = 60.0
TEST_END = 280.0


test_mask = (
    (df["start_sec"] >= TEST_START)
    & (df["end_sec"] <= TEST_END)
)

train_mask = ~test_mask


train = df[train_mask]
test = df[test_mask]


print("Train windows:", len(train))
print("Test windows:", len(test))

print("\nTrain labels:")
print(
    train["label"]
    .value_counts()
    .sort_index()
)

print("\nTest labels:")
print(
    test["label"]
    .value_counts()
    .sort_index()
)

print("\nTest time range:")
print(
    test["start_sec"].min(),
    "to",
    test["end_sec"].max(),
)


assert train["label"].nunique() == 2
assert test["label"].nunique() == 2

print("\nTemporal split contains both classes in train and test.")
