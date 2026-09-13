from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results/benchmarks"
METADATA_COLUMNS = ["recording_id", "window_index", "start_sec", "end_sec", "label"]


def main():
    datasets = {
        name: pd.read_csv(RESULTS_DIR / f"features_pipeline_{name}_chb02_full.csv")
        for name in "ABCD"
    }
    reference = datasets["A"]
    for name, dataset in datasets.items():
        assert list(dataset.columns) == list(reference.columns), name
        assert dataset[METADATA_COLUMNS].equals(reference[METADATA_COLUMNS]), name
        assert dataset.iloc[:, 5:].notna().all().all(), name
        assert dataset.iloc[:, 5:].nunique().gt(0).all(), name
    for name in "BCD":
        assert not datasets[name].iloc[:, 5:].equals(reference.iloc[:, 5:]), name
    print("Cross-pipeline contract passed")
    print("Shape:", reference.shape)
    print("Recordings:", reference["recording_id"].nunique())
    print("Features:", len(reference.columns) - len(METADATA_COLUMNS))


if __name__ == "__main__":
    main()