import pandas as pd


PATH = "results/benchmarks/features_pipeline_A_chb02.csv"

df = pd.read_csv(PATH)

df = df[df["recording_id"] == "chb02_16"].copy()
df = df.sort_values("start_sec").reset_index(drop=True)

print("Total valid windows:", len(df))

print("\nLabel counts:")
print(df["label"].value_counts().sort_index())

print("\nFirst 20 windows:")
print(
    df[
        [
            "window_index",
            "start_sec",
            "end_sec",
            "label",
        ]
    ].head(20).to_string(index=False)
)

print("\nWindows around seizure:")
around = df[
    (df["start_sec"] >= 100)
    & (df["start_sec"] <= 240)
]

print(
    around[
        [
            "window_index",
            "start_sec",
            "end_sec",
            "label",
        ]
    ].to_string(index=False)
)

print("\nLast 20 windows:")
print(
    df[
        [
            "window_index",
            "start_sec",
            "end_sec",
            "label",
        ]
    ].tail(20).to_string(index=False)
)
