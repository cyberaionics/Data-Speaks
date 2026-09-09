import csv
import importlib.util
import sys
from pathlib import Path

import mne


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


# Load the common windowing implementation.
windowing = load_module(
    "windowing",
    "pipeline/02_segmentation/candidate_algorithms/windowing.py",
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

INPUT_DIR = Path("data/processed/pipeline_B")
OUTPUT_DIR = Path("results/benchmarks")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Recordings
# ---------------------------------------------------------

RECORDINGS = {
    "chb02_01": {
        "file": "chb02_01_raw.fif",
        "seizures": [],
    },
    "chb02_16": {
        "file": "chb02_16_raw.fif",
        "seizures": [(130.0, 212.0)],
    },
}


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

output_path = (
    OUTPUT_DIR
    / "window_metadata_pipeline_B.csv"
)


rows = []


# ---------------------------------------------------------
# Process recordings
# ---------------------------------------------------------

for recording_id, info in RECORDINGS.items():

    input_path = INPUT_DIR / info["file"]

    print("\n" + "=" * 60)
    print("Recording:", recording_id)
    print("=" * 60)

    # Load processed Pipeline B data.
    raw = mne.io.read_raw_fif(
        input_path,
        preload=False,
        verbose=False,
    )

    sfreq = float(raw.info["sfreq"])
    duration_sec = raw.n_times / sfreq

    print("Sampling rate:", sfreq)
    print("Channels:", len(raw.ch_names))
    print("Duration:", duration_sec)

    # Generate windows using the common segmentation code.
    windows = list(
        windowing.generate_windows(
            duration_sec=duration_sec,
            sampling_rate=sfreq,
            seizure_intervals=info["seizures"],
        )
    )

    print("Total windows:", len(windows))

    # Count labels.
    label_counts = {
        -1: 0,
        0: 0,
        1: 0,
    }

    # generate_windows() does not provide window_index,
    # so enumerate() creates it here.
    for window_index, window in enumerate(windows):

        label = window["label"]

        label_counts[label] += 1

        rows.append(
            {
                "recording_id": recording_id,
                "window_index": window_index,
                "start_sec": window["start_sec"],
                "end_sec": window["end_sec"],
                "label": label,
            }
        )

    print("Ambiguous:", label_counts[-1])
    print("Interictal:", label_counts[0])
    print("Ictal:", label_counts[1])

    # Safety checks.
    assert sfreq == 256.0
    assert len(raw.ch_names) == 23
    assert len(windows) > 0


# ---------------------------------------------------------
# Write CSV
# ---------------------------------------------------------

fieldnames = [
    "recording_id",
    "window_index",
    "start_sec",
    "end_sec",
    "label",
]


with open(
    output_path,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


# ---------------------------------------------------------
# Overall summary
# ---------------------------------------------------------

ambiguous = sum(
    row["label"] == -1
    for row in rows
)

interictal = sum(
    row["label"] == 0
    for row in rows
)

ictal = sum(
    row["label"] == 1
    for row in rows
)


print("\n" + "=" * 60)
print("Pipeline B segmentation complete")
print("=" * 60)

print("Total windows:", len(rows))
print("Ambiguous:", ambiguous)
print("Interictal:", interictal)
print("Ictal:", ictal)
print("Output:", output_path)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

assert len(rows) == 2277
assert ambiguous == 62
assert interictal == 2173
assert ictal == 42

print("\nPipeline B segmentation test passed.")
