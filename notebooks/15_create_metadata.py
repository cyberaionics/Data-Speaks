from __future__ import annotations

from pathlib import Path
import csv
import re


DATA_DIR = Path("data/raw/physionet.org/chb02")
SUMMARY_FILE = DATA_DIR / "chb02-summary.txt"
OUTPUT_FILE = Path("results/benchmarks/metadata.csv")


def main():
    text = SUMMARY_FILE.read_text(encoding="utf-8")
    lines = text.splitlines()

    rows = []

    current = None
    seizure_intervals = []

    for line in lines:
        line = line.strip()

        if line.startswith("File Name:"):
            if current is not None:
                current["seizure_intervals"] = seizure_intervals
                rows.append(current)

            filename = line.split(":", 1)[1].strip()

            current = {
                "patient_id": "chb02",
                "recording_id": Path(filename).stem,
                "filename": filename,
                "duration_sec": "",
                "channels": 23,
                "sampling_rate_hz": 256,
                "seizure_presence": 0,
                "seizure_intervals": "",
                "preprocessing_method":
                    "pipeline_A_fir_ica_zscore_256",
                "output_location": "",
            }

            seizure_intervals = []

        elif current is not None and line.startswith("File End Time:"):
            # Duration is available from the EDF itself.
            pass

        elif current is not None and line.startswith(
            "Number of Seizures in File:"
        ):
            count = int(
                line.split(":", 1)[1].strip()
            )
            current["seizure_presence"] = int(count > 0)

        elif current is not None and line.startswith(
            "Seizure Start Time:"
        ):
            start = float(
                re.search(
                    r"[-+]?\d*\.?\d+",
                    line.split(":", 1)[1],
                ).group()
            )
            seizure_intervals.append(
                [start, None]
            )

        elif current is not None and line.startswith(
            "Seizure End Time:"
        ):
            end = float(
                re.search(
                    r"[-+]?\d*\.?\d+",
                    line.split(":", 1)[1],
                ).group()
            )

            if not seizure_intervals:
                raise ValueError(
                    f"End time found before start time "
                    f"for {current['filename']}"
                )

            seizure_intervals[-1][1] = end

    if current is not None:
        current["seizure_intervals"] = seizure_intervals
        rows.append(current)

    # Add actual duration and output location from the EDFs.
    for row in rows:
        edf_path = DATA_DIR / row["filename"]

        import mne

        raw = mne.io.read_raw_edf(
            edf_path,
            preload=False,
            verbose=False,
        )

        row["duration_sec"] = round(
            float(raw.times[-1]),
            3,
        )

        row["output_location"] = str(
            Path("data/processed/pipeline_A")
            / f"{row['recording_id']}_raw.fif"
        )

        # Convert intervals into a compact string.
        intervals = row["seizure_intervals"]

        row["seizure_intervals"] = ";".join(
            f"{start:.3f}-{end:.3f}"
            for start, end in intervals
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "patient_id",
        "recording_id",
        "filename",
        "duration_sec",
        "channels",
        "sampling_rate_hz",
        "seizure_presence",
        "seizure_intervals",
        "preprocessing_method",
        "output_location",
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

    print(f"Created: {OUTPUT_FILE}")
    print(f"Recordings: {len(rows)}")

    seizure_count = sum(
        row["seizure_presence"]
        for row in rows
    )

    print(
        f"Recordings with seizures: "
        f"{seizure_count}"
    )


if __name__ == "__main__":
    main()
