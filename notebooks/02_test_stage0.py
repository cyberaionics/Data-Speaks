from pathlib import Path
import sys

from pipeline.preprocessing_common.stage0 import load_stage0

DATA_DIR = Path("data/raw/physionet.org/chb02")
SUMMARY_FILE = DATA_DIR / "chb02-summary.txt"


def main() -> None:
    recordings = [
        "chb02_01.edf",
        "chb02_16.edf",
        "chb02_19.edf",
    ]

    for filename in recordings:
        print("\n" + "=" * 70)
        print(f"TESTING: {filename}")
        print("=" * 70)

        raw, metadata = load_stage0(
            DATA_DIR / filename,
            SUMMARY_FILE,
        )

        print(f"Recording ID:    {metadata.recording_id}")
        print(f"Duration:        {metadata.duration_sec:.2f} sec")
        print(f"Sampling rate:   {metadata.sampling_rate_hz} Hz")
        print(f"Channels:        {metadata.n_channels}")
        print(f"Samples:         {metadata.n_samples}")
        print(f"Seizures:        {metadata.seizure_intervals}")
        print(f"Data shape:      {raw.get_data().shape}")

        flatline_count = sum(
            channel_qc["flatline"]
            for channel_qc in metadata.qc["channels"].values()
        )

        print(f"Flatline channels: {flatline_count}")
        print(f"NaN present:       {metadata.qc['has_nan']}")
        print(f"Inf present:       {metadata.qc['has_inf']}")


if __name__ == "__main__":
    main()
