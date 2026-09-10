from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
import mne
import pandas as pd

WINDOW_SEC = 4.0
OVERLAP = 0.50
TARGET_SFREQ = 256.0

WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_SFREQ)
STEP_SAMPLES = int(WINDOW_SAMPLES * (1.0 - OVERLAP))


@dataclass
class SegmentationResult:
    windows: np.ndarray
    metadata: list[dict]
    ch_names: list[str] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return self.windows.shape[0]

    @property
    def n_channels(self) -> int:
        return self.windows.shape[1]

    @property
    def n_samples(self) -> int:
        return self.windows.shape[2]


def label_window(
    start_sec: float,
    end_sec: float,
    seizure_intervals: Sequence[tuple[float, float]],
    overlap_threshold: float = 0.50,
) -> int:
    window_duration = end_sec - start_sec
    if window_duration <= 0:
        raise ValueError("Window duration must be positive.")

    max_overlap = 0.0
    for seizure_start, seizure_end in seizure_intervals:
        overlap_start = max(start_sec, seizure_start)
        overlap_end = min(end_sec, seizure_end)
        overlap = max(0.0, overlap_end - overlap_start)
        max_overlap = max(max_overlap, overlap)

    if max_overlap == 0:
        return 0

    overlap_fraction = max_overlap / window_duration
    if overlap_fraction >= overlap_threshold:
        return 1

    return -1


def segment_raw(
    raw: mne.io.BaseRaw,
    seizure_intervals: Sequence[tuple[float, float]] | None = None,
    window_sec: float = WINDOW_SEC,
    overlap: float = OVERLAP,
    target_sfreq: float = TARGET_SFREQ,
    include_ambiguous: bool = True,
    patient_id: str | None = None,
    recording_id: str | None = None,
) -> SegmentationResult:
    if seizure_intervals is None:
        seizure_intervals = []

    sfreq = float(raw.info["sfreq"])
    if not np.isclose(sfreq, target_sfreq):
        raise ValueError(f"Expected sampling rate {target_sfreq} Hz, got {sfreq} Hz.")

    if not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1.")

    if window_sec <= 0:
        raise ValueError("window_sec must be positive.")

    window_samples = int(round(window_sec * sfreq))
    step_samples = int(round(window_samples * (1.0 - overlap)))
    if step_samples <= 0:
        raise ValueError("Overlap produces a zero/negative step size.")

    ch_names = list(raw.ch_names)
    n_channels = len(ch_names)
    n_samples_total = raw.n_times
    if n_samples_total < window_samples:
        raise ValueError("Recording is shorter than one complete segmentation window.")

    data = raw.get_data()
    if not np.isfinite(data).all():
        raise ValueError("Input Raw contains NaN or Inf values.")

    if patient_id is None:
        patient_id = raw.info.get("subject_info", {}).get("his_id", None)
    if recording_id is None:
        recording_id = raw.info.get("description", None)

    windows = []
    metadata = []
    window_id = 0

    for start_sample in range(0, n_samples_total - window_samples + 1, step_samples):
        stop_sample = start_sample + window_samples
        start_sec = start_sample / sfreq
        end_sec = stop_sample / sfreq

        window = data[:, start_sample:stop_sample]
        label = label_window(
            start_sec=start_sec,
            end_sec=end_sec,
            seizure_intervals=seizure_intervals,
        )

        if label == -1 and not include_ambiguous:
            continue

        windows.append(window)
        metadata.append(
            {
                "patient_id": patient_id,
                "recording_id": recording_id,
                "window_id": window_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": end_sec - start_sec,
                "label": label,
            }
        )
        window_id += 1

    windows_array = np.stack(windows, axis=0)
    expected_samples = int(round(window_sec * target_sfreq))

    assert windows_array.ndim == 3
    assert windows_array.shape[1] == n_channels
    assert windows_array.shape[2] == expected_samples

    return SegmentationResult(
        windows=windows_array,
        metadata=metadata,
        ch_names=ch_names,
    )


def segmentation_dataframe(result: SegmentationResult) -> pd.DataFrame:
    return pd.DataFrame(result.metadata)
