import pandas as pd


FILE = "results/benchmarks/window_metadata_pipeline_A.csv"

df = pd.read_csv(FILE)

recording = df[df["recording_id"] == "chb02_16"].copy()

near_seizure = recording[
    (
        (recording["start_sec"] >= 60)
        & (recording["start_sec"] <= 280)
    )
]

print(
    near_seizure[
        [
            "window_id",
            "start_sec",
            "end_sec",
            "label",
        ]
    ].to_string(index=False)
)
