from __future__ import annotations

import mne


def apply_fir_bandpass(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
) -> mne.io.Raw:
    """
    Apply a zero-phase FIR band-pass filter to EEG data.

    Parameters
    ----------
    raw:
        Stage-0 standardized EEG recording.
    l_freq:
        Lower cutoff frequency in Hz.
    h_freq:
        Upper cutoff frequency in Hz.

    Returns
    -------
    mne.io.Raw
        Filtered copy of the input recording.
    """

    if l_freq <= 0:
        raise ValueError("l_freq must be greater than 0 Hz.")

    if h_freq <= l_freq:
        raise ValueError("h_freq must be greater than l_freq.")

    sfreq = float(raw.info["sfreq"])

    if h_freq >= sfreq / 2:
        raise ValueError(
            f"h_freq={h_freq} Hz must be below Nyquist "
            f"frequency ({sfreq / 2:.2f} Hz)."
        )

    filtered = raw.copy()

    filtered.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose=False,
    )

    return filtered
