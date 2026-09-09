import numpy as np
import mne


def apply_robust_scaling(
    raw: mne.io.BaseRaw,
) -> tuple[mne.io.BaseRaw, dict]:
    """
    Apply per-channel robust scaling using the median and IQR.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        EEG recording.

    Returns
    -------
    scaled : mne.io.BaseRaw
        Robust-scaled EEG.
    provenance : dict
        Processing information.
    """
    data = raw.get_data()

    if data.ndim != 2:
        raise ValueError(
            "EEG data must have shape (n_channels, n_samples)"
        )

    if not np.isfinite(data).all():
        raise ValueError(
            "Input EEG contains non-finite values"
        )

    median = np.median(data, axis=1, keepdims=True)

    q25 = np.percentile(
        data,
        25,
        axis=1,
        keepdims=True,
    )

    q75 = np.percentile(
        data,
        75,
        axis=1,
        keepdims=True,
    )

    iqr = q75 - q25

    if np.any(iqr <= 0):
        bad_channels = np.where(
            iqr[:, 0] <= 0
        )[0].tolist()

        raise ValueError(
            f"Channels with zero IQR: {bad_channels}"
        )

    scaled_data = (
        data - median
    ) / iqr

    scaled = mne.io.RawArray(
        scaled_data,
        raw.info.copy(),
        verbose=False,
    )

    provenance = {
        "method": "robust_scaling",
        "center": "median",
        "scale": "IQR",
        "q25": 25.0,
        "q75": 75.0,
    }

    return scaled, provenance
