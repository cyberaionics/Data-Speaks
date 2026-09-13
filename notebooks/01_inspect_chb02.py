from pathlib import Path
import sys

import mne


DATA_DIR = Path("data/raw/physionet.org/chb02")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python notebooks/01_inspect_chb02.py <edf_filename>"
        )

    edf_name = sys.argv[1]
    edf_file = DATA_DIR / edf_name

    if not edf_file.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_file}")

    print("=" * 60)
    print("CHB-MIT EEG INSPECTION")
    print("=" * 60)
    print(f"Loading: {edf_file}")

    raw = mne.io.read_raw_edf(
        edf_file,
        preload=False,
        verbose=False,
    )

    print("\n--- Recording ---")
    print(f"File:               {edf_file.name}")
    print(f"Duration:           {raw.times[-1]:.2f} seconds")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Number of channels: {len(raw.ch_names)}")

    print("\n--- Channels ---")
    for i, name in enumerate(raw.ch_names, start=1):
        print(f"{i:02d}: {name}")

    print("\n--- Channel types ---")
    for name, channel_type in zip(
        raw.ch_names,
        raw.get_channel_types(),
    ):
        print(f"{name:15s} -> {channel_type}")


if __name__ == "__main__":
    main()