from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import mne


@dataclass
class Stage0Metadata:
    recording_id: str
    source_file: str
    sampling_rate_hz: float
    n_channels: int
    n_samples: int
    duration_sec: float
    channel_names: list[str]
    seizure_intervals: list[tuple[float, float]]
    qc: dict[str, Any]


def parse_seizure_annotations(
    summary_file: Path,
    recording_name: str,
) -> list[tuple[float, float]]:
    """
    Parse seizure intervals for one recording from the CHB-MIT summary file.

    Expected summary format contains:
        File Name: chb02_16.edf
        Number of Seizures in File: 1
        Seizure Start Time: 130 seconds
        Seizure End Time: 212 seconds
    """

    if not summary_file.exists():
        raise FileNotFoundError(
            f"Summary file not found: {summary_file}"
        )

    lines = summary_file.read_text().splitlines()

    intervals: list[tuple[float, float]] = []

    current_file: str | None = None
    n_seizures = 0
    pending_start: float | None = None

    for raw_line in lines:
        line = raw_line.strip()

        if line.startswith("File Name:"):
            current_file = line.split(":", 1)[1].strip()
            n_seizures = 0
            pending_start = None
            continue

        if current_file != recording_name:
            continue

        if line.startswith("Number of Seizures in File:"):
            n_seizures = int(line.split(":", 1)[1].strip())
            continue

        if n_seizures == 0:
            continue

        if line.startswith("Seizure Start Time:"):
            value = line.split(":", 1)[1].strip()
            pending_start = float(value.split()[0])
            continue

        if line.startswith("Seizure End Time:"):
            value = line.split(":", 1)[1].strip()
            end_time = float(value.split()[0])

            if pending_start is None:
                raise ValueError(
                    f"Found seizure end without start time "
                    f"for {recording_name}"
                )

            if end_time <= pending_start:
                raise ValueError(
                    f"Invalid seizure interval for {recording_name}: "
                    f"{pending_start} -> {end_time}"
                )

            intervals.append((pending_start, end_time))
            pending_start = None

    return intervals


def calculate_qc(raw: mne.io.BaseRaw) -> dict[str, Any]:
    """
    Calculate basic Stage-0 quality-control flags.

    This function flags potential issues but does not remove data.
    """

    data = raw.get_data()

    qc: dict[str, Any] = {
        "has_nan": bool(np.isnan(data).any()),
        "has_inf": bool(np.isinf(data).any()),
        "channels": {},
    }

    for index, channel_name in enumerate(raw.ch_names):
        x = data[index]

        variance = float(np.var(x))
        std = float(np.std(x))
        peak = float(np.max(np.abs(x)))

        qc["channels"][channel_name] = {
            "variance": variance,
            "std": std,
            "peak_abs_volts": peak,
            "flatline": bool(std < 1e-12),
            "non_finite": bool(
                np.isnan(x).any() or np.isinf(x).any()
            ),
        }

    return qc


def load_stage0(
    edf_file: str | Path,
    summary_file: str | Path | None = None,
) -> tuple[mne.io.Raw, Stage0Metadata]:
    """
    Load and standardize one CHB-MIT EDF recording.

    Stage 0 responsibilities:
    - EDF integrity/readability
    - EEG channel selection
    - preserve recorded sampling rate
    - preserve bipolar montage
    - parse seizure intervals
    - basic QC

    Pipeline-specific preprocessing is NOT performed here.
    """

    edf_path = Path(edf_file)

    if not edf_path.exists():
        raise FileNotFoundError(
            f"EDF file not found: {edf_path}"
        )

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False,
    )

    # Keep EEG channels only.
    eeg_picks = mne.pick_types(
        raw.info,
        eeg=True,
        meg=False,
        ecg=False,
        eog=False,
        emg=False,
        misc=False,
        stim=False,
        exclude=[],
    )

    if len(eeg_picks) == 0:
        raise ValueError(
            f"No EEG channels found in {edf_path.name}"
        )

    raw.pick(eeg_picks)

    # MNE automatically makes duplicate names unique when loading EDF,
    # for example T8-P8 -> T8-P8-0 / T8-P8-1.
    # We preserve those deterministic names rather than deleting channels.
    raw.set_meas_date(None)

    sampling_rate = float(raw.info["sfreq"])

    if sampling_rate <= 0:
        raise ValueError(
            f"Invalid sampling rate: {sampling_rate}"
        )

    if not np.isfinite(sampling_rate):
        raise ValueError("Sampling rate is not finite.")

    data = raw.get_data()

    if not np.isfinite(data).all():
        raise ValueError(
            f"Non-finite EEG values found in {edf_path.name}"
        )

    recording_id = edf_path.stem

    seizure_intervals: list[tuple[float, float]] = []

    if summary_file is not None:
        seizure_intervals = parse_seizure_annotations(
            Path(summary_file),
            edf_path.name,
        )

    qc = calculate_qc(raw)

    metadata = Stage0Metadata(
        recording_id=recording_id,
        source_file=str(edf_path),
        sampling_rate_hz=sampling_rate,
        n_channels=len(raw.ch_names),
        n_samples=raw.n_times,
        duration_sec=float(raw.times[-1]),
        channel_names=list(raw.ch_names),
        seizure_intervals=seizure_intervals,
        qc=qc,
    )

    return raw, metadata


def metadata_to_dict(metadata: Stage0Metadata) -> dict[str, Any]:
    """Convert Stage-0 metadata to a normal dictionary."""

    return asdict(metadata)
