from __future__ import annotations

from typing import Any

import mne
import numpy as np


def apply_chebyshev_type_ii_bandpass(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    order: int = 4,
    stopband_attenuation_db: float = 40.0,
) -> tuple[mne.io.Raw, dict[str, Any]]:
    """Apply a zero-phase Chebyshev Type II EEG band-pass filter."""
    if l_freq <= 0 or h_freq <= l_freq:
        raise ValueError("Require 0 < l_freq < h_freq.")
    if order < 1 or stopband_attenuation_db <= 0:
        raise ValueError("Filter order and stopband attenuation must be positive.")
    if h_freq >= float(raw.info["sfreq"]) / 2:
        raise ValueError("h_freq must be below the Nyquist frequency.")

    filtered = raw.copy()
    filtered.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params={
            "ftype": "cheby2",
            "order": order,
            "rs": stopband_attenuation_db,
        },
        phase="zero",
        verbose=False,
    )
    if not np.isfinite(filtered.get_data()).all():
        raise RuntimeError("Chebyshev Type II filter produced non-finite data.")
    return filtered, {
        "method": "Chebyshev Type II IIR",
        "phase": "zero",
        "l_freq": l_freq,
        "h_freq": h_freq,
        "order": order,
        "stopband_attenuation_db": stopband_attenuation_db,
    }
