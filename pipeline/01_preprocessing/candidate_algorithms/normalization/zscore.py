from __future__ import annotations

from typing import Any

import numpy as np
import mne


def apply_zscore(
    raw: mne.io.BaseRaw,
) -> tuple[mne.io.Raw, dict[str, Any]]:
    """
    Apply recording-level, per-channel Z-score normalization.

    For each EEG channel:

        z = (x - mean) / standard_deviation

    Statistics are calculated across the complete recording,
    not separately for individual windows.
    """

    data = raw.get_data()

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D EEG data, got shape {data.shape}"
        )

    if not np.isfinite(data).all():
        raise ValueError(
            "Input EEG contains NaN or Inf values."
        )

    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)

    if not np.all(np.isfinite(means)):
        raise ValueError("Non-finite channel mean detected.")

    if not np.all(np.isfinite(stds)):
        raise ValueError("Non-finite channel standard deviation detected.")

    # Prevent division by zero for a truly flat channel.
    # Stage 0 should already report such channels through QC.
    flat_channels = np.where(stds <= np.finfo(float).eps)[0]

    if len(flat_channels) > 0:
        names = [
            raw.ch_names[int(index)]
            for index in flat_channels
        ]

        raise ValueError(
            "Cannot Z-score flat channels: "
            + ", ".join(names)
        )

    normalized_data = (
        data - means[:, np.newaxis]
    ) / stds[:, np.newaxis]

    normalized = mne.io.RawArray(
        normalized_data,
        raw.info.copy(),
        first_samp=raw.first_samp,
    )

    provenance: dict[str, Any] = {
        "method": "per-channel recording-level Z-score",
        "axis": "time",
        "window_normalization": False,
        "channel_means_before": means.tolist(),
        "channel_stds_before": stds.tolist(),
        "channel_means_after": np.mean(
            normalized_data,
            axis=1,
        ).tolist(),
        "channel_stds_after": np.std(
            normalized_data,
            axis=1,
        ).tolist(),
    }

    return normalized, provenance
