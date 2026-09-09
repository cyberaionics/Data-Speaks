from __future__ import annotations

from typing import Any

import mne
import numpy as np


def apply_per_channel_standardization(
    raw: mne.io.BaseRaw,
) -> tuple[mne.io.Raw, dict[str, Any]]:
    """Apply recording-level per-channel zero-mean, unit-variance scaling."""
    data = raw.get_data()
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("Standardization input must be finite 2-D EEG data.")

    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    if np.any(std <= np.finfo(float).eps):
        bad_channels = np.flatnonzero(std[:, 0] <= np.finfo(float).eps).tolist()
        raise ValueError(f"Cannot standardize zero-variance channels: {bad_channels}")

    standardized = mne.io.RawArray(
        (data - mean) / std,
        raw.info.copy(),
        verbose=False,
    )
    provenance: dict[str, Any] = {
        "method": "per_channel_standardization",
        "center": "mean",
        "scale": "standard_deviation",
        "scope": "recording",
    }
    return standardized, provenance
