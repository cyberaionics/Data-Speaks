"""
DATA-SPEAKS — Pipeline C
========================
Butterworth band-pass → ASR → ICA → per-channel standardization → 256 Hz

Input:  Stage 0 standardized MNE Raw (17-channel, 256 Hz, bipolar montage)
Output: Cleaned MNE Raw + intermediate stages + calibration info + ICA object

Parameters
----------
Butterworth:        1–40 Hz, order 4 (IIR)
ASR:                cutoff=20.0, seizure-aware calibration
ICA:                FastICA, n_components=0.99 (99% variance retained)
EOG proxy:          FP1-F7, FP2-F4 (no dedicated EOG in CHB-MIT)
Standardization:    per-channel z-score
Output sfreq:       256 Hz

See PIPELINE_C_README.md for full documentation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mne
import numpy as np
from asrpy import ASR

# Apply NumPy 2.x compatibility patch before any asrpy usage.
# See pipeline_c_compat.py for the exact incompatibility and fix.
from pipeline_c_compat import apply_asrpy_numpy2_patch

apply_asrpy_numpy2_patch()


# ============================================================
# ASR CALIBRATION
# ============================================================

@dataclass
class CalibrationSegment:
    start_sec: float
    stop_sec: float
    reason: str


def select_calibration_segment(
    raw: mne.io.BaseRaw,
    seizure_intervals: list,
    min_duration_sec: float = 60.0,
    seizure_buffer_sec: float = 60.0,
) -> CalibrationSegment:
    """
    Select the best seizure-free segment for ASR calibration.

    Strategy: find the largest contiguous region that is at least
    seizure_buffer_sec away from every seizure boundary.  Cap
    calibration at 2*min_duration_sec (120 s by default) to avoid
    excessive memory use.  Raise ValueError if no segment of at least
    min_duration_sec exists.
    """

    total_duration = raw.times[-1]

    # --------------------------------------------------------
    # Seizure + buffer = forbidden regions
    # --------------------------------------------------------

    forbidden = []

    for iv in seizure_intervals:
        start = max(
            0.0,
            iv.start_sec - seizure_buffer_sec,
        )

        end = min(
            total_duration,
            iv.end_sec + seizure_buffer_sec,
        )

        forbidden.append((start, end))

    forbidden.sort()

    # --------------------------------------------------------
    # Merge overlapping forbidden regions
    # --------------------------------------------------------

    merged = []

    for start, end in forbidden:

        if merged and start <= merged[-1][1]:
            merged[-1] = (
                merged[-1][0],
                max(merged[-1][1], end),
            )

        else:
            merged.append((start, end))

    # --------------------------------------------------------
    # Find allowed regions
    # --------------------------------------------------------

    allowed = []
    cursor = 0.0

    for start, end in merged:

        if start > cursor:
            allowed.append((cursor, start))

        cursor = max(cursor, end)

    if cursor < total_duration:
        allowed.append(
            (cursor, total_duration)
        )

    if not allowed:
        raise ValueError(
            "No seizure-free calibration region available."
        )

    # --------------------------------------------------------
    # Choose largest allowed region
    # --------------------------------------------------------

    largest = max(
        allowed,
        key=lambda x: x[1] - x[0],
    )

    region_len = largest[1] - largest[0]

    if region_len < min_duration_sec:
        raise ValueError(
            f"Largest seizure-free region is only "
            f"{region_len:.1f}s; need at least "
            f"{min_duration_sec}s."
        )

    # Use at most 120 seconds
    duration = min(
        region_len,
        2 * min_duration_sec,
    )

    start = largest[0]
    stop = start + duration

    if seizure_intervals:
        reason = (
            f"largest seizure-free region; "
            f"{seizure_buffer_sec}s seizure buffer"
        )
    else:
        reason = "no seizures in recording"

    return CalibrationSegment(
        start_sec=start,
        stop_sec=stop,
        reason=reason,
    )


# ============================================================
# BUTTERWORTH BAND-PASS
# ============================================================

def butterworth_bandpass(
    raw: mne.io.BaseRaw,
    low_hz: float = 1.0,
    high_hz: float = 40.0,
    order: int = 4,
) -> mne.io.BaseRaw:
    """
    Apply a 4th-order Butterworth IIR band-pass filter (1–40 Hz).

    Uses MNE's IIR filter path with method='iir' and ftype='butter'.
    Operates on a copy — does not modify the input Raw.
    """

    raw_filtered = raw.copy()

    raw_filtered.filter(
        l_freq=low_hz,
        h_freq=high_hz,
        method="iir",
        iir_params={
            "order": order,
            "ftype": "butter",
        },
        verbose="ERROR",
    )

    return raw_filtered


# ============================================================
# ASR
# ============================================================

def apply_asr(
    raw: mne.io.BaseRaw,
    calibration: CalibrationSegment,
    cutoff: float = 20.0,
) -> mne.io.BaseRaw:
    """
    Calibrate and apply ASR (Artifact Subspace Reconstruction).

    Calibration is performed on the preselected seizure-free segment.
    The NumPy 2.x compatibility patch must already be applied before
    this function is called (pipeline_c.py applies it at import time).
    """

    sfreq = float(raw.info["sfreq"])

    start_sample = int(calibration.start_sec * sfreq)
    stop_sample = int(calibration.stop_sec * sfreq)

    asr = ASR(
        sfreq=sfreq,
        cutoff=cutoff,
    )

    asr.fit(
        raw,
        start=start_sample,
        stop=stop_sample,
    )

    cleaned = asr.transform(raw)

    return cleaned


# ============================================================
# ICA
# ============================================================

def apply_ica(
    raw: mne.io.BaseRaw,
    n_components: float = 0.99,
    random_state: int = 42,
) -> tuple[mne.io.BaseRaw, mne.preprocessing.ICA]:
    """
    Fit ICA (FastICA, 99% variance) and remove EOG-like components.

    CHB-MIT has no dedicated EOG channel.  FP1-F7 and FP2-F4 are
    used as EOG proxies because they capture frontal activity where
    eye-blink artefacts dominate.  This is a heuristic that may
    also flag legitimate frontal EEG activity — see PIPELINE_C_README.md.
    """

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        method="fastica",
    )

    ica.fit(
        raw,
        verbose="ERROR",
    )

    # CHB-MIT has no dedicated EOG channel.
    # Use frontal EEG channels as an EOG proxy.
    eog_proxy_channels = [
        ch
        for ch in ["FP1-F7", "FP2-F4"]
        if ch in raw.ch_names
    ]

    bad_components = []

    if eog_proxy_channels:

        eog_indices, _ = ica.find_bads_eog(
            raw,
            ch_name=eog_proxy_channels,
            verbose="ERROR",
        )

        bad_components = list(eog_indices)

    ica.exclude = bad_components

    raw_clean = raw.copy()

    ica.apply(
        raw_clean,
        verbose="ERROR",
    )

    return raw_clean, ica


# ============================================================
# PER-CHANNEL STANDARDIZATION
# ============================================================

def standardize_per_channel(
    raw: mne.io.BaseRaw,
) -> mne.io.BaseRaw:
    """
    Z-score standardize each channel independently over the full
    recording duration.

    mean(channel) → ≈ 0
    std(channel)  → ≈ 1

    Channels with std == 0 (flat) are left unchanged (divisor set to 1).
    """

    raw_std = raw.copy()

    data = raw_std.get_data()

    mean = data.mean(
        axis=1,
        keepdims=True,
    )

    std = data.std(
        axis=1,
        keepdims=True,
    )

    std[std == 0] = 1.0

    raw_std._data = (data - mean) / std

    return raw_std


# ============================================================
# FULL PIPELINE C
# ============================================================

def run_pipeline_c(
    raw: mne.io.BaseRaw,
    seizure_intervals: list,
    asr_cutoff: float = 20.0,
    butterworth_low: float = 1.0,
    butterworth_high: float = 40.0,
    butterworth_order: int = 4,
    ica_n_components: float = 0.99,
    ica_random_state: int = 42,
) -> dict:
    """
    Run the full Pipeline C sequence with per-stage timing.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Stage 0 standardized 17-channel 256-Hz Raw.
    seizure_intervals : list[SeizureInterval]
        Seizure intervals from Stage 0 (used to avoid calibrating on
        ictal data).
    asr_cutoff : float
        ASR threshold (default 20.0 — unchanged from the notebook).
    butterworth_low, butterworth_high, butterworth_order
        Butterworth filter parameters.
    ica_n_components, ica_random_state
        ICA parameters.

    Returns
    -------
    dict with keys:
        calibration        CalibrationSegment
        filtered           Raw after Butterworth
        asr_cleaned        Raw after ASR
        ica                ICA object (fitted, with .exclude set)
        ica_cleaned        Raw after ICA
        final              Raw after standardization (and resampling
                           if needed)
        timings            dict of per-stage wall-clock seconds
    """

    t_pipeline_start = time.perf_counter()

    # --------------------------------------------------------
    # 1. Select ASR calibration region
    # --------------------------------------------------------

    calibration = select_calibration_segment(
        raw,
        seizure_intervals,
    )

    # --------------------------------------------------------
    # 2. Butterworth band-pass
    # --------------------------------------------------------

    t0 = time.perf_counter()
    raw_filtered = butterworth_bandpass(
        raw,
        low_hz=butterworth_low,
        high_hz=butterworth_high,
        order=butterworth_order,
    )
    t_butterworth = time.perf_counter() - t0

    # --------------------------------------------------------
    # 3. ASR
    # --------------------------------------------------------

    t0 = time.perf_counter()
    raw_asr = apply_asr(
        raw_filtered,
        calibration,
        cutoff=asr_cutoff,
    )
    t_asr = time.perf_counter() - t0

    # --------------------------------------------------------
    # 4. ICA
    # --------------------------------------------------------

    t0 = time.perf_counter()
    raw_ica, ica = apply_ica(
        raw_asr,
        n_components=ica_n_components,
        random_state=ica_random_state,
    )
    t_ica = time.perf_counter() - t0

    # --------------------------------------------------------
    # 5. Per-channel standardization
    # --------------------------------------------------------

    t0 = time.perf_counter()
    raw_final = standardize_per_channel(raw_ica)
    t_std = time.perf_counter() - t0

    # --------------------------------------------------------
    # 6. Ensure 256 Hz
    # --------------------------------------------------------

    if not abs(raw_final.info["sfreq"] - 256.0) < 1e-9:
        raw_final = raw_final.copy()
        raw_final.resample(256, verbose="ERROR")

    t_total = time.perf_counter() - t_pipeline_start

    return {
        "calibration": calibration,
        "filtered": raw_filtered,
        "asr_cleaned": raw_asr,
        "ica": ica,
        "ica_cleaned": raw_ica,
        "final": raw_final,
        "timings": {
            "butterworth_sec": t_butterworth,
            "asr_sec": t_asr,
            "ica_sec": t_ica,
            "standardization_sec": t_std,
            "total_pipeline_c_sec": t_total,
        },
    }
