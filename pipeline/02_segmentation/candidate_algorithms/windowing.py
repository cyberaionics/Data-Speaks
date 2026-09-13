from __future__ import annotations

from typing import Sequence

import numpy as np


WINDOW_SEC = 4.0
OVERLAP = 0.50
STRIDE_SEC = WINDOW_SEC * (1.0 - OVERLAP)
BOUNDARY_BUFFER_SEC = 60.0


def calculate_window_label(
    window_start: float,
    window_end: float,
    seizure_intervals: Sequence[tuple[float, float]],
) -> int:
    """
    Apply the project's common seizure-labeling rule.

    1 = ictal:
        seizure overlap >= 50% of window duration

    0 = interictal:
        zero seizure overlap AND more than 60 seconds
        from every seizure boundary

    -1 = ambiguous:
        seizure overlap < 50%, OR within 60 seconds
        of a seizure boundary
    """

    window_duration = window_end - window_start

    if window_duration <= 0:
        raise ValueError("Window duration must be positive.")

    max_overlap = 0.0

    for seizure_start, seizure_end in seizure_intervals:
        overlap_start = max(window_start, seizure_start)
        overlap_end = min(window_end, seizure_end)

        if overlap_end > overlap_start:
            overlap = overlap_end - overlap_start
            max_overlap = max(max_overlap, overlap)

    overlap_fraction = max_overlap / window_duration

    if overlap_fraction >= 0.50:
        return 1

    if max_overlap > 0.0:
        return -1

    # No overlap: check distance to seizure boundaries.
    for seizure_start, seizure_end in seizure_intervals:
        distance_to_start = abs(window_end - seizure_start)
        distance_to_end = abs(window_start - seizure_end)

        # Window lies within 60 seconds before/after a seizure boundary.
        if (
            distance_to_start <= BOUNDARY_BUFFER_SEC
            or distance_to_end <= BOUNDARY_BUFFER_SEC
        ):
            return -1

    return 0


def generate_windows(
    duration_sec: float,
    sampling_rate: float,
    seizure_intervals: Sequence[tuple[float, float]],
):
    """
    Generate 4-second windows with 50% overlap.
    """

    window_samples = int(round(WINDOW_SEC * sampling_rate))
    stride_samples = int(round(STRIDE_SEC * sampling_rate))

    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("Invalid sampling rate.")

    n_samples = int(np.floor(duration_sec * sampling_rate))

    window_id = 0
    start_sample = 0

    while start_sample + window_samples <= n_samples:
        end_sample = start_sample + window_samples

        start_sec = start_sample / sampling_rate
        end_sec = end_sample / sampling_rate

        label = calculate_window_label(
            start_sec,
            end_sec,
            seizure_intervals,
        )

        yield {
            "window_id": window_id,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "label": label,
        }

        window_id += 1
        start_sample += stride_samples
