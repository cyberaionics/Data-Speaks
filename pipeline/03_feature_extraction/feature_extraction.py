"""
Stage 03: Feature extraction for the Data-Speaks EEG project.

Consumes SegmentationResult from segmentation.py and extracts comprehensive
time-domain, Hjorth, and frequency-domain (Welch PSD) features for each channel
in each 4-second window.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# Ensure segmentation module can be imported
SEG_DIR = Path(__file__).resolve().parent.parent / "02_segmentation"
if str(SEG_DIR) not in sys.path:
    sys.path.insert(0, str(SEG_DIR))

from segmentation import SegmentationResult

# ---------------------------------------------------------------------------
# Frequency band definitions (Hz)
# Upper gamma is bounded at 40 Hz to match Pipeline C bandpass filter (1-40 Hz)
# ---------------------------------------------------------------------------
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Per-channel feature computation
# ---------------------------------------------------------------------------

def _time_domain_features(sig: np.ndarray) -> dict[str, float]:
    """
    Compute time-domain statistical and Hjorth parameters for a 1D signal.
    Guards against zero-variance / flat / degenerate signals without producing NaNs.
    """
    n_samples = len(sig)
    mean_val = float(np.mean(sig))
    std_val = float(np.std(sig))
    var0 = float(np.var(sig))
    rms_val = float(np.sqrt(np.mean(sig ** 2)))

    min_val = float(np.min(sig))
    max_val = float(np.max(sig))
    ptp_val = max_val - min_val

    diff1 = np.diff(sig)
    diff2 = np.diff(diff1)
    line_length = float(np.sum(np.abs(diff1)))

    # Zero-crossing rate: fraction of sign changes along the signal
    zero_crossings = int(np.sum(np.diff(np.signbit(sig)) != 0))
    zcr = zero_crossings / (n_samples - 1) if n_samples > 1 else 0.0

    # Skewness and kurtosis: if constant or flat, scipy returns NaN/0.0, safeguard here
    if var0 <= 1e-12:
        skew_val = 0.0
        kurt_val = 0.0
    else:
        s = skew(sig)
        k = kurtosis(sig)
        skew_val = float(s) if np.isfinite(s) else 0.0
        kurt_val = float(k) if np.isfinite(k) else 0.0

    # Hjorth parameters
    var1 = float(np.var(diff1)) if len(diff1) > 0 else 0.0
    var2 = float(np.var(diff2)) if len(diff2) > 0 else 0.0

    if var0 > 1e-12:
        mobility = float(np.sqrt(var1 / var0))
    else:
        mobility = 0.0

    if mobility > 1e-12 and var1 > 1e-12:
        mobility_diff1 = float(np.sqrt(var2 / var1))
        complexity = mobility_diff1 / mobility
    else:
        complexity = 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "var": var0,
        "rms": rms_val,
        "skew": skew_val,
        "kurtosis": kurt_val,
        "min": min_val,
        "max": max_val,
        "ptp": ptp_val,
        "line_length": line_length,
        "zero_crossing_rate": zcr,
        "hjorth_activity": var0,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }


def _frequency_domain_features(sig: np.ndarray, sfreq: float) -> dict[str, float]:
    """
    Compute Welch PSD band powers and 95% spectral edge frequency.
    """
    n_samples = len(sig)
    nperseg = min(n_samples, int(round(sfreq * 2)))

    freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg)
    total_power = float(_TRAPZ(psd, freqs))

    features: dict[str, float] = {}

    for band_name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.any():
            band_power = float(_TRAPZ(psd[mask], freqs[mask]))
        else:
            band_power = 0.0

        features[f"{band_name}_power"] = band_power
        features[f"{band_name}_relpower"] = (
            band_power / total_power if total_power > 1e-14 else 0.0
        )

    # Total power
    features["total_power"] = total_power

    # Spectral edge frequency (95%)
    if total_power > 1e-14 and np.sum(psd) > 0:
        cumulative = np.cumsum(psd) / np.sum(psd)
        idx = np.searchsorted(cumulative, 0.95)
        idx = min(idx, len(freqs) - 1)
        features["spectral_edge_95"] = float(freqs[idx])
    else:
        features["spectral_edge_95"] = 0.0

    return features


def _window_features(window: np.ndarray, ch_names: list[str], sfreq: float) -> dict[str, float]:
    """
    Extract all features for one multichannel window, flattened into '<channel>_<feature>'.
    """
    row: dict[str, float] = {}
    for i, ch in enumerate(ch_names):
        sig = window[i]
        td = _time_domain_features(sig)
        fd = _frequency_domain_features(sig, sfreq)
        for k, v in td.items():
            row[f"{ch}_{k}"] = v
        for k, v in fd.items():
            row[f"{ch}_{k}"] = v
    return row


# ---------------------------------------------------------------------------
# Main Orchestration API
# ---------------------------------------------------------------------------

def extract_features(
    result: SegmentationResult,
    ch_names: list[str],
    sfreq: float = 256.0,
) -> pd.DataFrame:
    """
    Extract comprehensive features for each window in a SegmentationResult.

    Parameters
    ----------
    result : SegmentationResult
        Segmentation output containing windows array of shape
        (n_windows, n_channels, n_samples) and window metadata list.
    ch_names : list[str]
        Ordered list of channel names corresponding to the channel axis
        of result.windows.
    sfreq : float, optional
        Sampling frequency in Hz, default 256.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with metadata columns followed by <channel>_<feature> columns.
    """
    if not isinstance(result, SegmentationResult):
        raise TypeError(f"Expected SegmentationResult, got {type(result)}")

    if sfreq <= 0:
        raise ValueError(f"sfreq must be positive, got {sfreq}")

    if result.n_windows == 0:
        metadata_cols = [
            "patient_id", "recording_id", "window_id",
            "start_sec", "end_sec", "duration_sec", "label"
        ]
        return pd.DataFrame(columns=metadata_cols)

    if result.n_channels != len(ch_names):
        raise ValueError(
            f"SegmentationResult has {result.n_channels} channels, but ch_names "
            f"has {len(ch_names)} entries ({ch_names}). Channel count and order must match."
        )

    # Validate channel naming consistency with SegmentationResult if available
    if getattr(result, "ch_names", None):
        if result.ch_names != ch_names:
            raise ValueError(
                f"Channel names passed ({ch_names}) do not match the channel names "
                f"recorded in SegmentationResult ({result.ch_names})."
            )

    feature_rows: list[dict[str, float]] = []
    for i in range(result.n_windows):
        window = result.windows[i]
        row = _window_features(window, ch_names, sfreq)
        feature_rows.append(row)

    features_df = pd.DataFrame(feature_rows)

    # Ensure no NaNs or Infs in numerical features
    non_finite_mask = ~np.isfinite(features_df.to_numpy())
    if np.any(non_finite_mask):
        n_invalid = int(np.sum(non_finite_mask))
        raise ValueError(
            f"Feature extraction generated {n_invalid} non-finite values (NaN/Inf). "
            f"Check signal conditioning or division guards."
        )

    # Construct metadata DataFrame preserving standard columns first
    metadata_df = pd.DataFrame(result.metadata)

    # Enforce standard metadata column order if present
    standard_meta_cols = [
        "patient_id",
        "recording_id",
        "window_id",
        "start_sec",
        "end_sec",
        "duration_sec",
        "label",
    ]
    meta_order = [c for c in standard_meta_cols if c in metadata_df.columns]
    other_meta = [c for c in metadata_df.columns if c not in standard_meta_cols]
    metadata_df = metadata_df[meta_order + other_meta]

    out_df = pd.concat([metadata_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    return out_df
