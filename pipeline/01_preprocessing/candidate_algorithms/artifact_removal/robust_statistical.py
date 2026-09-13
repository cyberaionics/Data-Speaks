from __future__ import annotations

from typing import Any

import mne
import numpy as np
from scipy.ndimage import median_filter


def apply_robust_statistical_artifact_detection(
    raw: mne.io.BaseRaw,
    window_seconds: float = 0.5,
    z_threshold: float = 6.0,
) -> tuple[mne.io.Raw, dict[str, Any]]:
    """Suppress sample outliers using a per-channel Hampel-style MAD rule.

    A local median provides the replacement value for samples whose robust
    z-score exceeds ``z_threshold``. The method never drops channels or time
    points, so the common shape and windowing contracts remain unchanged.
    """
    if window_seconds <= 0 or z_threshold <= 0:
        raise ValueError("window_seconds and z_threshold must be positive.")
    data = raw.get_data()
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("Input must be finite 2-D EEG data.")

    window_samples = int(round(window_seconds * float(raw.info["sfreq"])))
    if window_samples % 2 == 0:
        window_samples += 1
    if window_samples < 3:
        window_samples = 3

    cleaned_data = data.copy()
    artifact_counts = []
    for channel_index in range(data.shape[0]):
        signal = data[channel_index]
        local_median = median_filter(signal, size=window_samples, mode="nearest")
        local_mad = median_filter(np.abs(signal - local_median), size=window_samples, mode="nearest")
        robust_scale = 1.4826 * local_mad
        finite_scale = robust_scale[robust_scale > np.finfo(float).eps]
        fallback_scale = float(np.median(finite_scale)) if finite_scale.size else np.finfo(float).eps
        robust_scale = np.maximum(robust_scale, fallback_scale)
        artifact_mask = np.abs(signal - local_median) / robust_scale > z_threshold
        cleaned_data[channel_index, artifact_mask] = local_median[artifact_mask]
        artifact_counts.append(int(np.sum(artifact_mask)))

    cleaned = mne.io.RawArray(cleaned_data, raw.info.copy(), verbose=False)
    provenance: dict[str, Any] = {
        "method": "local_median_MAD_hampel",
        "window_seconds": window_seconds,
        "window_samples": window_samples,
        "z_threshold": z_threshold,
        "mad_to_sigma_factor": 1.4826,
        "artifact_samples_per_channel": artifact_counts,
        "artifact_samples_total": int(sum(artifact_counts)),
        "replacement": "local_median",
    }
    return cleaned, provenance
