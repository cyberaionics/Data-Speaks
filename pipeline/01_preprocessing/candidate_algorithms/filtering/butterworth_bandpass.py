import mne


def apply_butterworth_bandpass(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    order: int = 4,
) -> mne.io.BaseRaw:
    """
    Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Standardized EEG recording.
    l_freq : float
        Lower cutoff frequency in Hz.
    h_freq : float
        Upper cutoff frequency in Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    mne.io.BaseRaw
        Filtered EEG recording.
    """
    if l_freq <= 0:
        raise ValueError("l_freq must be positive")

    if h_freq <= l_freq:
        raise ValueError("h_freq must be greater than l_freq")

    if order < 1:
        raise ValueError("order must be at least 1")

    sfreq = float(raw.info["sfreq"])

    if h_freq >= sfreq / 2:
        raise ValueError(
            "h_freq must be below the Nyquist frequency"
        )

    filtered = raw.copy()

    filtered.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params={
            "ftype": "butter",
            "order": order,
        },
        phase="zero",
        verbose=False,
    )

    return filtered
