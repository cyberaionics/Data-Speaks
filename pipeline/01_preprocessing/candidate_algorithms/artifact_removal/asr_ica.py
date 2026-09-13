from __future__ import annotations

from typing import Any

import mne
import numpy as np
from meegkit.asr import ASR


def apply_asr(
    raw: mne.io.BaseRaw,
    cutoff: float = 5.0,
    calibration_seconds: float = 60.0,
    window_length: float = 0.5,
    window_overlap: float = 0.66,
) -> tuple[mne.io.Raw, dict[str, Any]]:
    """Clean a recording with Artifact Subspace Reconstruction (ASR).

    ASR is calibrated independently for each recording using its initial,
    post-bandpass segment.  ``meegkit`` selects clean sub-windows from that
    segment with robust statistics before estimating reconstruction thresholds.
    Keeping calibration within a recording avoids mixing patients or montages.

    The input must already have been high-pass filtered; Pipeline C supplies
    the shared 1--40 Hz Butterworth-filtered Stage-0 EEG.
    """
    if len(raw.ch_names) < 2:
        raise ValueError("ASR requires at least two EEG channels.")
    if cutoff <= 0:
        raise ValueError("ASR cutoff must be positive.")
    if calibration_seconds < 30:
        raise ValueError("ASR calibration requires at least 30 seconds.")

    data = raw.get_data()
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("ASR input must be finite 2-D EEG data.")

    sfreq = float(raw.info["sfreq"])
    calibration_samples = min(
        int(round(calibration_seconds * sfreq)),
        data.shape[1],
    )
    minimum_samples = int(round(30.0 * sfreq))
    if calibration_samples < minimum_samples:
        raise ValueError(
            "Recording is too short for ASR calibration: "
            f"need at least {minimum_samples} samples, got {calibration_samples}."
        )

    calibration_data = data[:, :calibration_samples]
    asr = ASR(
        sfreq=sfreq,
        cutoff=cutoff,
        win_len=window_length,
        win_overlap=window_overlap,
        method="euclid",
        estimator="scm",
    )
    clean_calibration, retained_mask = asr.fit(calibration_data)
    cleaned_data = asr.transform(data)

    if cleaned_data.shape != data.shape or not np.isfinite(cleaned_data).all():
        raise RuntimeError("ASR produced invalid output.")

    cleaned = mne.io.RawArray(
        cleaned_data,
        raw.info.copy(),
        verbose=False,
    )
    provenance: dict[str, Any] = {
        "method": "ASR",
        "implementation": "meegkit.asr.ASR",
        "cutoff": cutoff,
        "calibration_seconds_requested": calibration_seconds,
        "calibration_seconds_used": calibration_samples / sfreq,
        "calibration_samples": calibration_samples,
        "calibration_clean_samples": int(clean_calibration.shape[1]),
        "calibration_retained_fraction": float(np.mean(retained_mask)),
        "window_length_seconds": window_length,
        "window_overlap": window_overlap,
        "method_metric": "euclid",
        "covariance_estimator": "scm",
    }
    return cleaned, provenance
