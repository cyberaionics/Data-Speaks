from pathlib import Path
import csv
import importlib.util

import mne


PROCESSED_DIR = Path("data/processed/pipeline_A")
OUTPUT_FILE = Path(
    "results/benchmarks/window_metadata_pipeline_A.csv"
)

RECORDINGS = [
    "chb02_01",
    "chb02_16",
]


def load_windowing():
    path = (
        Path("pipeline")
        / "02_segmentation"
        / "candidate_algorithms"
        / "windowing.py"
    )

    spec = importlib.util.spec_from_file_location(
        "windowing_module",
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.generate_windows


def main():
    generate_windows = load_windowing()

    rows = []

    for recording_id in RECORDINGS:
        fif_path = (
            PROCESSED_DIR
            / f"{recording_id}_raw.fif"
        )

        if not fif_path.exists():
            raise FileNotFoundError(fif_path)

        seizure_intervals = {
            "chb02_01": [],
            "chb02_16": [(130.0, 212.0)],
        }[recording_id]

        print("=" * 70)
        print(f"Segmenting: {recording_id}")

        raw = mne.io.read_raw_fif(
            fif_path,
            preload=False,
            verbose=False,
        )

        sfreq = float(raw.info["sfreq"])
        duration = raw.n_times / raw.info["sfreq"]
        print(f"Sampling rate: {sfreq} Hz")
        print(f"Duration: {duration:.2f} sec")

        windows = list(
            generate_windows(
                duration,
                sfreq,
                seizure_intervals,
            )
        )

        label_counts = {
            1: 0,
            0: 0,
            -1: 0,
        }

        for window in windows:
            label_counts[window["label"]] += 1

            rows.append(
                {
                    "patient_id": "chb02",
                    "recording_id": recording_id,
                    "window_id": window["window_id"],
                    "start_sample": window["start_sample"],
                    "end_sample": window["end_sample"],
                    "start_sec": window["start_sec"],
                    "end_sec": window["end_sec"],
                    "label": window["label"],
                    "seizure_intervals":
                        ";".join(
                            f"{start}-{end}"
                            for start, end
                            in seizure_intervals
                        ),
                    "preprocessing_method":
                        "pipeline_A_fir_ica_zscore_256",
                }
            )

        print(f"Total windows: {len(windows)}")
        print(f"Ictal (1): {label_counts[1]}")
        print(f"Interictal (0): {label_counts[0]}")
        print(f"Ambiguous (-1): {label_counts[-1]}")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "patient_id",
        "recording_id",
        "window_id",
        "start_sample",
        "end_sample",
        "start_sec",
        "end_sec",
        "label",
        "seizure_intervals",
        "preprocessing_method",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 70)
    print(f"Created: {OUTPUT_FILE}")
    print(f"Total windows: {len(rows)}")


if __name__ == "__main__":
    main()
